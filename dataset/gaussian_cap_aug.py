import copy
import os
from collections import defaultdict

import numpy as np
import torch
import torchvision
from func_timeout import FunctionTimedOut, func_timeout
from loguru import logger
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
from tqdm import tqdm


class GausCapAug:
    def __init__(
        self,
        patch_size=512,
        n_objects_range=[3, 8],
        gt_img_root=None,
        gt_txt_path=None,
        low_img_root=None,
        edge_img_root=None,
        gt_edge_img_root=None,
        dataset_type=None,
        use_naive=False,
    ):
        self.patch_size = patch_size
        self.n_objects_range = n_objects_range
        self.gt_img_root = gt_img_root
        self.gt_txt_path = gt_txt_path

        self.low_img_root = low_img_root
        self.edge_img_root = edge_img_root
        self.gt_edge_img_root = gt_edge_img_root

        self.dataset_type = dataset_type

        self.use_naive = use_naive

        # hardcoded minimum for sampled h and w, and w_h_ratio
        self.min_h = 32
        self.min_w = 32
        self.w_h_ratio = 1.5

        # Building gt_img_dict
        # calculate the normal distributions of w and h
        # split must be 'gt' or 'low'
        all_w, all_h, self.all_gt_img_keys, self.gt_img_dict = self.get_img_dict(
            self.gt_img_root,
            self.gt_txt_path,
            "gt",
            name="gt_img_dict",
            add_wh=True,
            add_keys=True,
            dataset_type=self.dataset_type,
        )

        if len(all_w) > 0 and len(all_h) > 0 and len(all_w) == len(all_h):
            self.mean_h = np.mean(all_h)
            self.std_h = np.std(all_h)
            self.mean_w = np.mean(all_w)
            self.std_w = np.std(all_w)
        else:
            raise ValueError("Error calculating mean and std for w and h.")

        _, _, _, self.gt_edge_img_dict = self.get_img_dict(
            self.gt_edge_img_root,
            self.gt_txt_path,
            "gt",
            name="gt_edge_img_dict",
            dataset_type=self.dataset_type,
        )

        _, _, self.all_low_img_keys, self.low_img_dict = self.get_img_dict(
            self.low_img_root,
            self.gt_txt_path,
            "low",
            name="low_img_dict",
            add_wh=False,
            add_keys=True,
            dataset_type=self.dataset_type,
        )

        _, _, _, self.low_edge_img_dict = self.get_img_dict(
            self.edge_img_root,
            self.gt_txt_path,
            "low",
            name="low_edge_img_dict",
            dataset_type=self.dataset_type,
        )

        gt_txt_count = 0
        for v in self.gt_img_dict.values():
            temp_boxes = v["bbox"]
            gt_txt_count += len(temp_boxes)

        logger.info(
            f"Enabled CAP of {self.dataset_type} with gt_image count: {len(self.gt_img_dict)} and gt_text count: {gt_txt_count}"
        )

    def get_img_dict(
        self,
        img_root,
        txt_path,
        split,
        name=None,
        add_wh=False,
        add_keys=False,
        dataset_type=None,
    ):
        all_txts = os.listdir(txt_path)
        img_keys = []
        all_w = []
        all_h = []
        img_dict = defaultdict(lambda: defaultdict(list))

        for txt_name in tqdm(all_txts, desc=f"Building {name}"):
            img_id = self.get_img_id(txt_name, dataset_type)
            current_txt_path = os.path.join(txt_path, txt_name)
            if split == "gt":
                img_paths = [
                    self.get_img_path(
                        img_root, img_id, split=split, dataset_type=dataset_type
                    )
                ]
            elif split == "low":
                img_paths, exposures = self.get_img_path(
                    img_root, img_id, split=split, dataset_type=dataset_type
                )

            for idx, img_path in enumerate(img_paths):
                widths, heights, polys, cropped_texts, ids = self.read_txt(
                    current_txt_path, img_path, dataset_type
                )

                if split == "gt":
                    new_img_id = img_id
                elif split == "low":
                    new_img_id = img_id + "_" + exposures[idx]

                img_dict[new_img_id]["width"] = widths
                img_dict[new_img_id]["height"] = heights
                img_dict[new_img_id]["bbox"] = polys
                img_dict[new_img_id]["id"] = ids
                img_dict[new_img_id]["img"] = cropped_texts

                if add_wh:
                    all_w.extend(widths)
                    all_h.extend(heights)

                if add_keys:
                    for text_id in ids:
                        if split == "gt":
                            img_keys.append(f"{img_id}_{text_id}")
                        elif split == "low":
                            img_keys.append(f"{img_id}_{text_id}_{exposures[idx]}")

        return all_w, all_h, img_keys, img_dict

    def get_img_id(self, txt_name, dataset_type):
        if dataset_type == "icdar15" or dataset_type == "lol":
            return os.path.basename(txt_name).replace(".txt", "").replace("gt_img_", "")
        else:
            return (
                os.path.basename(txt_name)
                .replace(".txt", "")
                .replace("gt_img_", "")[:5]
            )

    def get_img_path(self, img_folder, img_id, split="gt", dataset_type=None):
        if split == "gt":
            real_img_path = None

            if dataset_type == "sony" or dataset_type == "fuji":
                real_img_name = f"{img_id}_00_10s.png"
                real_img_path = os.path.join(img_folder, real_img_name)
                if not os.path.isfile(real_img_path):
                    real_img_name = f"{img_id}_00_30s.png"
                    real_img_path = os.path.join(img_folder, real_img_name)
            elif dataset_type == "icdar15":
                real_img_name = f"{img_id}.jpg"
                real_img_path = os.path.join(img_folder, real_img_name)
            elif dataset_type == "lol":
                real_img_name = f"{img_id}.png"
                real_img_path = os.path.join(img_folder, real_img_name)

            return real_img_path

        elif split == "low":
            real_img_paths = []
            exposures = []

            if dataset_type == "sony" or dataset_type == "fuji":
                for exposure in ["0.1", "0.04", "0.033"]:
                    real_img_name = f"{img_id}_00_{exposure}s.png"
                    real_img_path = os.path.join(img_folder, real_img_name)
                    if os.path.isfile(real_img_path):
                        real_img_paths.append(real_img_path)
                        exposures.append(exposure)
            elif dataset_type == "icdar15":
                real_img_name = f"{img_id}.jpg"
                real_img_path = os.path.join(img_folder, real_img_name)
                if os.path.isfile(real_img_path):
                    real_img_paths.append(real_img_path)
                    exposures.append("1")
            elif dataset_type == "lol":
                real_img_name = f"{img_id}.png"
                real_img_path = os.path.join(img_folder, real_img_name)
                if os.path.isfile(real_img_path):
                    real_img_paths.append(real_img_path)
                    exposures.append("1")

            return real_img_paths, exposures

    def get_min_max_xy(self, poly):
        xs = poly[0::2]
        ys = poly[1::2]
        minx = int(min(xs))
        maxx = int(max(xs))
        miny = int(min(ys))
        maxy = int(max(ys))
        return minx, miny, maxx, maxy

    def convert_poly(self, poly, rescale=False, minx=-1, miny=-1, return_format=None):
        new_poly = []
        xs = poly[0::2]
        ys = poly[1::2]
        for x, y in zip(xs, ys):
            if rescale and minx != -1 and miny != -1:
                x = x - minx
                y = y - miny
            # return in this format for pillow
            if return_format == "raw":
                new_poly.append(x)
                new_poly.append(y)
            else:
                new_poly.append((x, y))
        return new_poly

    def read_crop_image(self, image_to_crop, polys):
        pc_alphas = []
        for poly in polys:
            minx, miny, maxx, maxy = self.get_min_max_xy(poly)
            box_size = (maxx - minx, maxy - miny)
            new_polygon = self.convert_poly(poly, rescale=True, minx=minx, miny=miny)

            pc_alpha = Image.new("RGBA", box_size)
            maskIm = Image.new("L", box_size, 0)
            ImageDraw.Draw(maskIm).polygon(new_polygon, fill=255)

            if image_to_crop is not None:
                poly_crop = image_to_crop.crop((minx, miny, maxx, maxy))
            else:
                poly_crop = Image.new("RGB", box_size, (0, 0, 0))
            assert pc_alpha.size == maskIm.size

            pc_alpha.paste(poly_crop, (0, 0), mask=maskIm)
            pc_alphas.append(pc_alpha)

        return pc_alphas

    def read_txt(self, txt_path, img_path, dataset_type):
        assert os.path.isfile(txt_path), f"Invalid text file path: {txt_path}"

        widths = []
        heights = []
        polys = []
        ids = []
        cropped_texts = []

        with open(txt_path, mode="r", encoding="utf-8-sig") as txt:
            lines = txt.readlines()
            if len(lines) > 0:
                for idx, line in enumerate(lines):
                    line = line.strip().split(",")
                    poly = [int(x) for x in line[:8]]
                    assert len(poly) == 8, (
                        f"Invalid poly count at {txt_path}, only {len(poly)} coords"
                    )

                    if (
                        dataset_type == "sony"
                        or dataset_type == "fuji"
                        or dataset_type == "lol"
                    ):
                        cat = line[-1]
                        assert cat == "###" or cat == "Text", (
                            f"Invalid category at {txt_path}, {cat}"
                        )
                    elif dataset_type == "icdar15":
                        cats = line[8:]
                        cat = "".join(x for x in cats)

                    if "###" not in cat:
                        assert Polygon(
                            [
                                (poly[0], poly[1]),
                                (poly[2], poly[3]),
                                (poly[4], poly[5]),
                                (poly[6], poly[7]),
                            ]
                        ).is_valid, f"Invalid polygon found at {txt_path}: {poly}"

                        # assuming all coords are in the order of l,r,t,b
                        # then we can calculate the width and height as below
                        width = Point(poly[0], poly[1]).distance(
                            Point(poly[2], poly[3])
                        )
                        height = Point((poly[2], poly[3])).distance(
                            Point(poly[4], poly[5])
                        )

                        widths.append(width)
                        heights.append(height)
                        polys.append(poly)
                        ids.append(idx)

                if len(polys) > 0:
                    pil_image = Image.open(img_path)
                    cropped_texts = self.read_crop_image(pil_image, polys)

        return widths, heights, polys, cropped_texts, ids

    def gen_syn_boxes(
        self, current_img_id, boxes_in_patch, ids_in_patch, max_box_in_patch
    ):
        iou_threshold = 0
        exit_condition = True
        while exit_condition:
            all_boxes = copy.deepcopy(boxes_in_patch)
            all_crop_ids = copy.deepcopy(ids_in_patch)
            scores = torch.Tensor(np.ones(max_box_in_patch))
            i = len(all_boxes)
            while i < max_box_in_patch:
                # this is to randomly sample crop of images that are not exist in the current patch
                # randomly selects a crop
                rand_idx = np.random.randint(
                    low=0, high=len(self.all_gt_img_keys) - 1, size=1
                )[0]
                rand_sel_full_id = self.all_gt_img_keys[rand_idx]
                rand_sel_img_id = self.all_gt_img_keys[rand_idx].split("_")[0]
                rand_sel_crop_id = self.all_gt_img_keys[rand_idx].split("_")[-1]

                if (
                    rand_sel_img_id != current_img_id
                    and rand_sel_full_id not in all_crop_ids
                ):
                    # randomly samples x,y,w,h
                    x = np.random.uniform(low=0, high=self.patch_size, size=1)[0]
                    y = np.random.uniform(low=0, high=self.patch_size, size=1)[0]
                    sampled_w = np.random.normal(self.mean_w, self.std_w, 1)[0]
                    sampled_h = np.random.normal(self.mean_h, self.std_h, 1)[0]

                    actual_list_id = self.gt_img_dict[rand_sel_img_id]["id"].index(
                        int(rand_sel_crop_id)
                    )
                    actual_w = self.gt_img_dict[rand_sel_img_id]["width"][
                        actual_list_id
                    ]
                    actual_h = self.gt_img_dict[rand_sel_img_id]["height"][
                        actual_list_id
                    ]
                    if (
                        0 <= x + sampled_w <= self.patch_size
                        and 0 <= y + sampled_h <= self.patch_size
                    ):
                        if (
                            actual_w >= self.min_w
                            and actual_h >= self.min_h
                            and sampled_w >= self.min_w
                            and sampled_h >= self.min_h
                        ):
                            if sampled_w >= sampled_h * self.w_h_ratio:
                                all_boxes.append([x, y, x + sampled_w, y + sampled_h])
                                all_crop_ids.append(rand_sel_full_id + "_syn")
                                i += 1

            remaining_boxes = torchvision.ops.nms(
                boxes=torch.Tensor(all_boxes),
                scores=scores,
                iou_threshold=iou_threshold,
            )
            if len(remaining_boxes) == max_box_in_patch:
                exit_condition = False

        return remaining_boxes, all_boxes, all_crop_ids

    def gen_syn_boxes_naive(
        self, current_img_id, boxes_in_patch, ids_in_patch, max_box_in_patch
    ):
        all_boxes = copy.deepcopy(boxes_in_patch)
        all_crop_ids = copy.deepcopy(ids_in_patch)
        i = len(all_boxes)
        while i == max_box_in_patch:
            # this is to randomly sample crop of images that are not exist in the current patch
            # randomly selects a crop
            rand_idx = np.random.randint(
                low=0, high=len(self.all_gt_img_keys) - 1, size=1
            )[0]
            rand_sel_full_id = self.all_gt_img_keys[rand_idx]
            rand_sel_img_id = self.all_gt_img_keys[rand_idx].split("_")[0]
            rand_sel_crop_id = self.all_gt_img_keys[rand_idx].split("_")[-1]

            if (
                rand_sel_img_id != current_img_id
                and rand_sel_full_id not in all_crop_ids
            ):
                # randomly samples x,y,w,h
                x = np.random.uniform(low=0, high=self.patch_size, size=1)[0]
                y = np.random.uniform(low=0, high=self.patch_size, size=1)[0]

                actual_list_id = self.gt_img_dict[rand_sel_img_id]["id"].index(
                    int(rand_sel_crop_id)
                )
                actual_w = self.gt_img_dict[rand_sel_img_id]["width"][actual_list_id]
                actual_h = self.gt_img_dict[rand_sel_img_id]["height"][actual_list_id]
                if (
                    0 <= x + actual_w <= self.patch_size
                    and 0 <= y + actual_h <= self.patch_size
                ):
                    all_boxes.append([x, y, x + actual_w, y + actual_h])
                    all_crop_ids.append(rand_sel_full_id + "_syn")
                    i += 1

        all_boxes_ids = []
        for idx in range(i):
            all_boxes_ids.append(idx)

        return all_boxes_ids, all_boxes, all_crop_ids

    def __call__(
        self,
        low_image,
        gt_image,
        edge_image,
        gt_edge_image,
        x_start,
        y_start,
        gt_txt_name,
    ):
        max_box_in_patch = np.random.randint(
            low=self.n_objects_range[0], high=self.n_objects_range[1], size=1
        )[0]
        # based on given boxes, we decide which boxes are still in the patch
        self.x_start = x_start
        self.y_start = y_start

        # to filter out the out-of-box boxes
        patch_polygon = Polygon(
            [
                [self.x_start, self.y_start],
                [self.x_start + self.patch_size, self.y_start],
                [self.x_start + self.patch_size, self.y_start + self.patch_size],
                [self.x_start, self.y_start + self.patch_size],
            ]
        )
        assert patch_polygon.is_valid, f"Invalid patch_polygon found at {patch_polygon}"

        # to read the text file for boxes in this image
        current_img_id = self.get_img_id(gt_txt_name, self.dataset_type)
        current_gt_img_dict = self.gt_img_dict[current_img_id]
        current_polys = current_gt_img_dict["bbox"]
        current_widths = current_gt_img_dict["width"]
        current_heights = current_gt_img_dict["height"]
        current_ids = current_gt_img_dict["id"]

        boxes_in_patch = []
        ids_in_patch = []

        for idx, coords in enumerate(current_polys):
            width = current_widths[idx]
            height = current_heights[idx]
            min_point = Point(coords[0], coords[1])
            max_point = Point(coords[0] + width, coords[1] + height)
            if not patch_polygon.contains(min_point) and not patch_polygon.contains(
                max_point
            ):
                continue
            else:
                ori_poly = Polygon(self.convert_poly(coords, rescale=False))
                # check intersection area with patch polygon
                intersect_area = patch_polygon.intersection(ori_poly).area
                if intersect_area >= self.min_w * self.min_h:
                    ori_minx, ori_miny, ori_maxx, ori_maxy = self.get_min_max_xy(coords)
                    new_coords = self.convert_poly(
                        coords,
                        rescale=True,
                        minx=ori_minx,
                        miny=ori_miny,
                        return_format="raw",
                    )
                    minx, miny, maxx, maxy = self.get_min_max_xy(new_coords)
                    boxes_in_patch.append([minx, miny, maxx, maxy])
                    ids_in_patch.append(f"{current_img_id}_{current_ids[idx]}")

        if max_box_in_patch - len(boxes_in_patch) > 0:
            try:
                if self.use_naive:
                    remaining_boxes, all_boxes, all_crop_ids = func_timeout(
                        5,
                        self.gen_syn_boxes_naive,
                        args=(
                            current_img_id,
                            boxes_in_patch,
                            ids_in_patch,
                            max_box_in_patch,
                        ),
                    )
                else:
                    remaining_boxes, all_boxes, all_crop_ids = func_timeout(
                        5,
                        self.gen_syn_boxes,
                        args=(
                            current_img_id,
                            boxes_in_patch,
                            ids_in_patch,
                            max_box_in_patch,
                        ),
                    )

                # convert the images to pillow for pasting
                low_image = Image.fromarray(np.uint8(low_image))
                gt_image = Image.fromarray(np.uint8(gt_image))
                edge_image = Image.fromarray(np.uint8(edge_image))
                gt_edge_image = Image.fromarray(np.uint8(gt_edge_image))

                for idx in remaining_boxes:
                    box = all_boxes[idx]
                    paste_xy = (int(box[0]), int(box[1]))
                    crop_id = all_crop_ids[idx]
                    img_id = crop_id.split("_")[0]
                    text_id = crop_id.split("_")[1]
                    if "syn" in crop_id:
                        # paste on gt_image, pc = polygon cropped
                        actual_id = self.gt_img_dict[img_id]["id"].index(int(text_id))
                        pc_alpha = self.gt_img_dict[img_id]["img"][actual_id]
                        newsize = (int(box[2] - box[0]), int(box[3] - box[1]))
                        final_pc_alpha = pc_alpha.resize(newsize)
                        gt_image.paste(final_pc_alpha, paste_xy, final_pc_alpha)

                        # paste on gt_edge_image
                        gt_edge_pc_alpha = self.gt_edge_img_dict[img_id]["img"][
                            actual_id
                        ]
                        gt_edge_final_pc_alpha = gt_edge_pc_alpha.resize(newsize)
                        gt_edge_image.paste(
                            gt_edge_final_pc_alpha, paste_xy, gt_edge_final_pc_alpha
                        )

                        # for low light images
                        # we need to randomly pick on exposure
                        current_low_img_kyes = []
                        for low_img_key in self.all_low_img_keys:
                            temp_img_id = low_img_key.split("_")[0]
                            temp_text_id = low_img_key.split("_")[1]
                            temp_exposure = low_img_key.split("_")[2]
                            if temp_img_id == img_id and temp_text_id == text_id:
                                current_low_img_kyes.append(
                                    f"{temp_img_id}_{temp_exposure}"
                                )
                        sel_low_img_key = np.random.choice(current_low_img_kyes, 1)[0]

                        low_image_pc_alpha = self.low_img_dict[sel_low_img_key]["img"][
                            actual_id
                        ]
                        low_final_pc_alpha = low_image_pc_alpha.resize(newsize)
                        low_image.paste(
                            low_final_pc_alpha, paste_xy, low_final_pc_alpha
                        )

                        low_edge_pc_alpha = self.low_edge_img_dict[sel_low_img_key][
                            "img"
                        ][actual_id]
                        low_edge_final_pc_alpha = low_edge_pc_alpha.resize(newsize)
                        edge_image.paste(
                            low_edge_final_pc_alpha, paste_xy, low_edge_final_pc_alpha
                        )

                # convert pil image back to np array
                low_image = np.array(low_image).astype(np.uint8)
                gt_image = np.array(gt_image).astype(np.uint8)
                edge_image = np.array(edge_image).astype(np.uint8)
                gt_edge_image = np.array(gt_edge_image).astype(np.uint8)

                return low_image, gt_image, edge_image, gt_edge_image

            except FunctionTimedOut:
                # nms timeout, return these images as is
                return low_image, gt_image, edge_image, gt_edge_image
        else:
            # max boxes in patch reached, return these images as is
            return low_image, gt_image, edge_image, gt_edge_image
