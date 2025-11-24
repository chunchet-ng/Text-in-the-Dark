# coding: utf-8
__author__ = "RocketFlash: https://github.com/RocketFlash"

import random

import cv2
import numpy as np


def resize_keep_ar(image, height=500, scale=None):
    if scale is not None:
        image = cv2.resize(image, None, fx=float(scale), fy=float(scale))
    else:
        r = height / float(image.shape[0])
        width = r * image.shape[1]
        image = cv2.resize(image, (int(width), int(height)))
    return image


class SimpleCapAug:
    """
    gt_images_sources - list of gt images paths
    low_images_sources - list of low images paths
    edge_images_sources - list of edge images paths
    gt_edge_images_sources - list of gt edge images paths
    probability_map - mask with brobability values
    mean_h_norm - mean normilized height
    n_objects_range - [min, max] number of objects
    s_range - range of scales of original image size
    h_range - range of objects heights
    x_range - range in camera coordinate system (in meters) [float, float]
    y_range - range in camera coordinate system (in meters) [float, float]
    objects_idxs - objects indexes from dataset to paste [idx1, idx2, ...]
    random_h_flip - source image random horizontal flip
    random_v_flip - source image random vertical flip
    blending_coeff - coefficient of image blending
    image_format - color image format : {bgr, rgb}
    coords_format - output coordinates format: {xyxy, xywh, yolo}
    """

    def __init__(
        self,
        probability_map=None,
        mean_h_norm=None,
        n_objects_range=[1, 6],
        h_range=None,
        s_range=[0.5, 1.5],
        x_range=[200, 500],
        y_range=[100, 300],
        objects_idxs=None,
        random_h_flip=False,
        random_v_flip=False,
        image_format="bgr",
        coords_format="xyxy",
        blending_coeff=0,
    ):
        self.probability_map = probability_map
        self.mean_h_norm = mean_h_norm
        self.n_objects_range = n_objects_range
        self.s_range = s_range
        self.h_range = h_range
        self.x_range = x_range
        self.y_range = y_range
        self.objects_idxs = objects_idxs
        self.random_h_flip = random_h_flip
        self.random_v_flip = random_v_flip
        self.image_format = image_format
        self.coords_format = coords_format
        self.blending_coeff = blending_coeff

    def __call__(
        self,
        gt_images_sources,
        low_images_sources,
        edge_images_sources,
        gt_edge_images_sources,
        gt_image,
        low_image,
        edge_image,
        gt_edge_image,
    ):
        self.gt_images_sources = gt_images_sources
        self.low_images_sources = low_images_sources
        self.edge_images_sources = edge_images_sources
        self.gt_edge_images_sources = gt_edge_images_sources
        return self.generate_objects(gt_image, low_image, edge_image, gt_edge_image)

    def generate_objects(self, gt_image, low_image, edge_image, gt_edge_image):
        n_objects = random.randint(*self.n_objects_range)
        heights = None
        scales = None

        if self.probability_map is not None:
            p_h, p_w = self.probability_map.shape
            prob_map_1d = np.squeeze(self.probability_map.reshape((1, -1)))
            select_indexes = np.random.choice(
                np.arange(prob_map_1d.size), n_objects, p=prob_map_1d
            )
            points = [
                [(select_idx % p_w) / p_w, (select_idx // p_w) / p_h]
                for select_idx in select_indexes
            ]
            points = np.array(points)

            if self.mean_h_norm is not None:
                heights = np.random.uniform(
                    low=self.mean_h_norm * 0.98,
                    high=self.mean_h_norm * 1.02,
                    size=(n_objects, 1),
                )
            else:
                if self.h_range is not None:
                    heights = np.random.uniform(
                        low=self.h_range[0], high=self.h_range[1], size=(n_objects, 1)
                    )
        else:
            points = np.random.randint(
                low=[self.x_range[0], self.y_range[0]],
                high=[self.x_range[1], self.y_range[1]],
                size=(n_objects, 2),
            )
            if self.h_range is not None:
                heights = np.random.randint(
                    low=self.h_range[0], high=self.h_range[1], size=(n_objects, 1)
                )
        if heights is None:
            scales = np.random.uniform(
                low=self.s_range[0], high=self.s_range[1], size=(n_objects, 1)
            )

        return self.generate_objects_coord(
            gt_image, low_image, edge_image, gt_edge_image, points, heights, scales
        )

    def generate_objects_coord(
        self, gt_image, low_image, edge_image, gt_edge_image, points, heights, scales
    ):
        """
        points - numpy array of coordinates in meters with shape [n,2]
        """
        n_objects = points.shape[0]

        if self.objects_idxs is None:
            objects_idxs = [
                random.randint(0, len(self.gt_images_sources) - 1)
                for _ in range(n_objects)
            ]
        else:
            objects_idxs = self.objects_idxs

        assert len(objects_idxs) == points.shape[0]

        gt_image_dst = gt_image.copy()
        dst_h, dst_w, _ = gt_image_dst.shape

        low_image_dst = low_image.copy()
        low_dst_h, low_dst_w, _ = low_image_dst.shape

        edge_image_dst = edge_image.copy()
        edge_dst_h, edge_dst_w, _ = edge_image_dst.shape

        gt_edge_image_dst = gt_edge_image.copy()
        gt_edge_dst_h, gt_edge_dst_w, _ = gt_edge_image_dst.shape

        assert dst_h == low_dst_h and dst_w == low_dst_w, (
            "mismatch image shape between gt and low."
        )
        assert low_dst_h == edge_dst_h and low_dst_w == edge_dst_w, (
            "mismatch image shape between low and edge."
        )

        coords_all = []

        semantic_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)

        for idx, object_idx in enumerate(objects_idxs):
            point = points[idx]
            if heights is not None:
                height = heights[idx]
                scale = None
            else:
                scale = scales[idx]
                height = None

            gt_image_src = self.select_image(self.gt_images_sources, object_idx)
            low_image_src = self.select_image(self.low_images_sources, object_idx)
            edge_image_src = self.select_image(self.edge_images_sources, object_idx)
            gt_edge_image_src = self.select_image(
                self.gt_edge_images_sources, object_idx
            )

            if self.probability_map is not None:
                x_coord, y_coord = int(point[0] * dst_w), int(point[1] * dst_h)
                height *= dst_h
                gt_image_src = resize_keep_ar(gt_image_src, height=height, scale=scale)
                low_image_src = resize_keep_ar(
                    low_image_src, height=height, scale=scale
                )
                edge_image_src = resize_keep_ar(
                    edge_image_src, height=height, scale=scale
                )
                gt_edge_image_src = resize_keep_ar(
                    gt_edge_image_src, height=height, scale=scale
                )
            else:
                x_coord, y_coord = int(point[0]), int(point[1])
                gt_image_src = resize_keep_ar(gt_image_src, height=height, scale=scale)
                low_image_src = resize_keep_ar(
                    low_image_src, height=height, scale=scale
                )
                edge_image_src = resize_keep_ar(
                    edge_image_src, height=height, scale=scale
                )
                gt_edge_image_src = resize_keep_ar(
                    gt_edge_image_src, height=height, scale=scale
                )

            gt_image_dst, coords, mask = self.paste_object(
                gt_image_dst,
                gt_image_src,
                x_coord,
                y_coord,
                self.random_h_flip,
                self.random_v_flip,
            )
            low_image_dst, low_coords, low_mask = self.paste_object(
                low_image_dst,
                low_image_src,
                x_coord,
                y_coord,
                self.random_h_flip,
                self.random_v_flip,
            )
            edge_image_dst, edge_coords, edge_mask = self.paste_object(
                edge_image_dst,
                edge_image_src,
                x_coord,
                y_coord,
                self.random_h_flip,
                self.random_v_flip,
            )
            gt_edge_image_dst, gt_edge_coords, gt_edge_mask = self.paste_object(
                gt_edge_image_dst,
                gt_edge_image_src,
                x_coord,
                y_coord,
                self.random_h_flip,
                self.random_v_flip,
            )

            assert np.array_equal(coords, low_coords) and np.array_equal(
                mask, low_mask
            ), "mismatch coords and mask between gt and low."
            assert np.array_equal(low_coords, edge_coords) and np.array_equal(
                low_mask, edge_mask
            ), "mismatch coords and mask between low and edge."
            assert np.array_equal(edge_coords, gt_edge_coords) and np.array_equal(
                edge_mask, gt_edge_mask
            ), "mismatch coords and mask between low_edge and gt_edge."

            if coords:
                coords_all.append(coords)
                x1, y1, x2, y2 = coords
                curr_mask = mask / 255
                curr_mask = curr_mask.astype(np.uint8)

                roi_mask_sem = semantic_mask[y1:y2, x1:x2]

                mask_inv = cv2.bitwise_not(curr_mask * 255)

                img_sem_bg = cv2.bitwise_and(roi_mask_sem, roi_mask_sem, mask=mask_inv)

                dst_sem = cv2.add(img_sem_bg, curr_mask)

                semantic_mask[y1:y2, x1:x2] = dst_sem

        coords_all = np.array(coords_all)

        if self.coords_format == "yolo":
            x = coords_all.copy()
            x = x.astype(float)
            dw = 1.0 / dst_w
            dh = 1.0 / dst_h
            ws = coords_all[:, 2] - coords_all[:, 0]
            hs = coords_all[:, 3] - coords_all[:, 1]
            x[:, 0] = dw * ((coords_all[:, 0] + ws / 2.0) - 1)
            x[:, 1] = dh * ((coords_all[:, 1] + hs / 2.0) - 1)
            x[:, 2] = dw * ws
            x[:, 3] = dh * hs
            coords_all = x
        elif self.coords_format == "xywh":
            x = coords_all.copy()
            x[:, 2] = coords_all[:, 2] - coords_all[:, 0]
            x[:, 3] = coords_all[:, 3] - coords_all[:, 1]
            coords_all = x

        return (
            gt_image_dst,
            low_image_dst,
            edge_image_dst,
            gt_edge_image_dst,
            coords_all,
            semantic_mask,
        )

    def select_image(self, cropped_images, object_idx):
        source_image_path = cropped_images[object_idx]
        image_src = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
        if self.image_format == "rgb":
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGBA)
        else:
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2BGRA)
        return image_src

    def paste_object(
        self,
        image_dst,
        image_src,
        x_coord,
        y_coord,
        random_h_flip=True,
        random_v_flip=False,
    ):
        src_h, src_w, _ = image_src.shape
        dst_h, dst_w, _ = image_dst.shape
        x_offset, y_offset = x_coord - int(src_w / 2), y_coord - src_h
        y1, y2 = max(y_offset, 0), min(y_offset + src_h, dst_h)
        x1, x2 = max(x_offset, 0), min(x_offset + src_w, dst_w)
        y1_m = 0 if y1 > 0 else -y_offset
        x1_m = 0 if x1 > 0 else -x_offset
        y2_m = src_h if y2 < dst_h - 1 else dst_h - y_offset
        x2_m = src_w if x2 < dst_w - 1 else dst_w - x_offset
        coords = []

        if y1_m >= src_h or x1_m >= src_w or y2_m < 0 or x2_m < 0:
            return image_dst, coords, None

        if random_h_flip:
            if random.uniform(0, 1) > 0.5:
                image_src = cv2.flip(image_src, 1)

        if random_v_flip:
            if random.uniform(0, 1) > 0.5:
                image_src = cv2.flip(image_src, 0)

        # Simple cut and paste without preprocessing
        mask_src = image_src[:, :, 3]
        rgb_img = image_src[:, :, :3]

        if self.blending_coeff > 0:
            beta = 1.0 - self.blending_coeff
            out_img = cv2.addWeighted(
                rgb_img[y1_m:y2_m, x1_m:x2_m],
                self.blending_coeff,
                image_dst[y1:y2, x1:x2],
                beta,
                0.0,
            )
        else:
            mask_inv = cv2.bitwise_not(mask_src)
            img1_bg = cv2.bitwise_and(
                image_dst[y1:y2, x1:x2],
                image_dst[y1:y2, x1:x2],
                mask=mask_inv[y1_m:y2_m, x1_m:x2_m],
            )
            img2_fg = cv2.bitwise_and(
                rgb_img[y1_m:y2_m, x1_m:x2_m],
                rgb_img[y1_m:y2_m, x1_m:x2_m],
                mask=mask_src[y1_m:y2_m, x1_m:x2_m],
            )
            out_img = cv2.add(img1_bg, img2_fg)

        mask_visible = mask_src[y1_m:y2_m, x1_m:x2_m]
        image_dst[y1:y2, x1:x2] = out_img
        coords = [x1, y1, x2, y2]

        return image_dst, coords, mask_visible


class SIMPLE_CAP_AUG_PLAIN:
    """
    gt_images_sources - list of gt images paths
    low_images_sources - list of low images paths
    probability_map - mask with brobability values
    mean_h_norm - mean normilized height
    n_objects_range - [min, max] number of objects
    s_range - range of scales of original image size
    h_range - range of objects heights
    x_range - range in camera coordinate system (in meters) [float, float]
    y_range - range in camera coordinate system (in meters) [float, float]
    objects_idxs - objects indexes from dataset to paste [idx1, idx2, ...]
    random_h_flip - source image random horizontal flip
    random_v_flip - source image random vertical flip
    blending_coeff - coefficient of image blending
    image_format - color image format : {bgr, rgb}
    coords_format - output coordinates format: {xyxy, xywh, yolo}
    """

    def __init__(
        self,
        probability_map=None,
        mean_h_norm=None,
        n_objects_range=[1, 6],
        h_range=None,
        s_range=[0.5, 1.5],
        x_range=[200, 500],
        y_range=[100, 300],
        objects_idxs=None,
        random_h_flip=False,
        random_v_flip=False,
        image_format="bgr",
        coords_format="xyxy",
        blending_coeff=0,
    ):
        self.probability_map = probability_map
        self.mean_h_norm = mean_h_norm
        self.n_objects_range = n_objects_range
        self.s_range = s_range
        self.h_range = h_range
        self.x_range = x_range
        self.y_range = y_range
        self.objects_idxs = objects_idxs
        self.random_h_flip = random_h_flip
        self.random_v_flip = random_v_flip
        self.image_format = image_format
        self.coords_format = coords_format
        self.blending_coeff = blending_coeff

    def __call__(self, gt_images_sources, low_images_sources, gt_image, low_image):
        self.gt_images_sources = gt_images_sources
        self.low_images_sources = low_images_sources
        return self.generate_objects(gt_image, low_image)

    def generate_objects(self, gt_image, low_image):
        n_objects = random.randint(*self.n_objects_range)
        heights = None
        scales = None

        if self.probability_map is not None:
            p_h, p_w = self.probability_map.shape
            prob_map_1d = np.squeeze(self.probability_map.reshape((1, -1)))
            select_indexes = np.random.choice(
                np.arange(prob_map_1d.size), n_objects, p=prob_map_1d
            )
            points = [
                [(select_idx % p_w) / p_w, (select_idx // p_w) / p_h]
                for select_idx in select_indexes
            ]
            points = np.array(points)

            if self.mean_h_norm is not None:
                heights = np.random.uniform(
                    low=self.mean_h_norm * 0.98,
                    high=self.mean_h_norm * 1.02,
                    size=(n_objects, 1),
                )
            else:
                if self.h_range is not None:
                    heights = np.random.uniform(
                        low=self.h_range[0], high=self.h_range[1], size=(n_objects, 1)
                    )
        else:
            points = np.random.randint(
                low=[self.x_range[0], self.y_range[0]],
                high=[self.x_range[1], self.y_range[1]],
                size=(n_objects, 2),
            )
            if self.h_range is not None:
                heights = np.random.randint(
                    low=self.h_range[0], high=self.h_range[1], size=(n_objects, 1)
                )
        if heights is None:
            scales = np.random.uniform(
                low=self.s_range[0], high=self.s_range[1], size=(n_objects, 1)
            )

        return self.generate_objects_coord(gt_image, low_image, points, heights, scales)

    def generate_objects_coord(self, gt_image, low_image, points, heights, scales):
        """
        points - numpy array of coordinates in meters with shape [n,2]
        """
        n_objects = points.shape[0]

        if self.objects_idxs is None:
            objects_idxs = [
                random.randint(0, len(self.gt_images_sources) - 1)
                for _ in range(n_objects)
            ]
        else:
            objects_idxs = self.objects_idxs

        assert len(objects_idxs) == points.shape[0]

        gt_image_dst = gt_image.copy()
        dst_h, dst_w, _ = gt_image_dst.shape

        low_image_dst = low_image.copy()
        low_dst_h, low_dst_w, _ = low_image_dst.shape

        assert dst_h == low_dst_h and dst_w == low_dst_w, (
            "mismatch image shape between gt and low."
        )

        coords_all = []

        semantic_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)

        for idx, object_idx in enumerate(objects_idxs):
            point = points[idx]
            if heights is not None:
                height = heights[idx]
                scale = None
            else:
                scale = scales[idx]
                height = None

            gt_image_src = self.select_image(self.gt_images_sources, object_idx)
            low_image_src = self.select_image(self.low_images_sources, object_idx)

            if self.probability_map is not None:
                x_coord, y_coord = int(point[0] * dst_w), int(point[1] * dst_h)
                height *= dst_h
                gt_image_src = resize_keep_ar(gt_image_src, height=height, scale=scale)
                low_image_src = resize_keep_ar(
                    low_image_src, height=height, scale=scale
                )
            else:
                x_coord, y_coord = int(point[0]), int(point[1])
                gt_image_src = resize_keep_ar(gt_image_src, height=height, scale=scale)
                low_image_src = resize_keep_ar(
                    low_image_src, height=height, scale=scale
                )

            gt_image_dst, coords, mask = self.paste_object(
                gt_image_dst,
                gt_image_src,
                x_coord,
                y_coord,
                self.random_h_flip,
                self.random_v_flip,
            )
            low_image_dst, low_coords, low_mask = self.paste_object(
                low_image_dst,
                low_image_src,
                x_coord,
                y_coord,
                self.random_h_flip,
                self.random_v_flip,
            )

            assert np.array_equal(coords, low_coords) and np.array_equal(
                mask, low_mask
            ), "mismatch coords and mask between gt and low."

            if coords:
                coords_all.append(coords)
                x1, y1, x2, y2 = coords
                curr_mask = mask / 255
                curr_mask = curr_mask.astype(np.uint8)

                roi_mask_sem = semantic_mask[y1:y2, x1:x2]

                mask_inv = cv2.bitwise_not(curr_mask * 255)

                img_sem_bg = cv2.bitwise_and(roi_mask_sem, roi_mask_sem, mask=mask_inv)

                dst_sem = cv2.add(img_sem_bg, curr_mask)

                semantic_mask[y1:y2, x1:x2] = dst_sem

        coords_all = np.array(coords_all)

        if self.coords_format == "yolo":
            x = coords_all.copy()
            x = x.astype(float)
            dw = 1.0 / dst_w
            dh = 1.0 / dst_h
            ws = coords_all[:, 2] - coords_all[:, 0]
            hs = coords_all[:, 3] - coords_all[:, 1]
            x[:, 0] = dw * ((coords_all[:, 0] + ws / 2.0) - 1)
            x[:, 1] = dh * ((coords_all[:, 1] + hs / 2.0) - 1)
            x[:, 2] = dw * ws
            x[:, 3] = dh * hs
            coords_all = x
        elif self.coords_format == "xywh":
            x = coords_all.copy()
            x[:, 2] = coords_all[:, 2] - coords_all[:, 0]
            x[:, 3] = coords_all[:, 3] - coords_all[:, 1]
            coords_all = x

        return gt_image_dst, low_image_dst, coords_all, semantic_mask

    def select_image(self, cropped_images, object_idx):
        source_image_path = cropped_images[object_idx]
        image_src = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
        if self.image_format == "rgb":
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGBA)
        else:
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2BGRA)
        return image_src

    def paste_object(
        self,
        image_dst,
        image_src,
        x_coord,
        y_coord,
        random_h_flip=True,
        random_v_flip=False,
    ):
        src_h, src_w, _ = image_src.shape
        dst_h, dst_w, _ = image_dst.shape
        x_offset, y_offset = x_coord - int(src_w / 2), y_coord - src_h
        y1, y2 = max(y_offset, 0), min(y_offset + src_h, dst_h)
        x1, x2 = max(x_offset, 0), min(x_offset + src_w, dst_w)
        y1_m = 0 if y1 > 0 else -y_offset
        x1_m = 0 if x1 > 0 else -x_offset
        y2_m = src_h if y2 < dst_h - 1 else dst_h - y_offset
        x2_m = src_w if x2 < dst_w - 1 else dst_w - x_offset
        coords = []

        if y1_m >= src_h or x1_m >= src_w or y2_m < 0 or x2_m < 0:
            return image_dst, coords, None

        if random_h_flip:
            if random.uniform(0, 1) > 0.5:
                image_src = cv2.flip(image_src, 1)

        if random_v_flip:
            if random.uniform(0, 1) > 0.5:
                image_src = cv2.flip(image_src, 0)

        # Simple cut and paste without preprocessing
        mask_src = image_src[:, :, 3]
        rgb_img = image_src[:, :, :3]

        if self.blending_coeff > 0:
            beta = 1.0 - self.blending_coeff
            out_img = cv2.addWeighted(
                rgb_img[y1_m:y2_m, x1_m:x2_m],
                self.blending_coeff,
                image_dst[y1:y2, x1:x2],
                beta,
                0.0,
            )
        else:
            mask_inv = cv2.bitwise_not(mask_src)
            img1_bg = cv2.bitwise_and(
                image_dst[y1:y2, x1:x2],
                image_dst[y1:y2, x1:x2],
                mask=mask_inv[y1_m:y2_m, x1_m:x2_m],
            )
            img2_fg = cv2.bitwise_and(
                rgb_img[y1_m:y2_m, x1_m:x2_m],
                rgb_img[y1_m:y2_m, x1_m:x2_m],
                mask=mask_src[y1_m:y2_m, x1_m:x2_m],
            )
            out_img = cv2.add(img1_bg, img2_fg)

        mask_visible = mask_src[y1_m:y2_m, x1_m:x2_m]
        image_dst[y1:y2, x1:x2] = out_img
        coords = [x1, y1, x2, y2]

        return image_dst, coords, mask_visible
