import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image
from skimage import color, feature
from torch.utils.data import Dataset

from dataset.gaussian_cap_aug import GausCapAug
from dataset.simple_cap_aug import SimpleCapAug


# numpy random issue:
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
# https://www.reddit.com/r/MachineLearning/comments/mocpgj/p_using_pytorch_numpy_a_bug_that_plagues/
class LOLTrainSet(Dataset):
    """LOL Train dataset."""

    def __init__(
        self,
        patch_size,
        list_file,
        input_img_dir,
        gt_img_dir,
        edge_dir,
        gt_edge_dir,
        use_cap=False,
        cap_version=-1,
        cap_paths={},
        input_use_canny=False,
        gt_use_canny=False,
        dataset_type=None,
    ):
        self.patch_size = patch_size
        with open(list_file, "r") as f:
            self.list_file_lines = f.readlines()
        self.input_img_dir = input_img_dir
        self.gt_img_dir = gt_img_dir
        self.edge_dir = edge_dir
        self.gt_edge_dir = gt_edge_dir

        self.use_cap = use_cap
        self.cap_version = cap_version
        self.cap_paths = cap_paths

        self.input_use_canny = input_use_canny
        self.gt_use_canny = gt_use_canny

        self.dataset_type = dataset_type

        if self.use_cap:
            if self.cap_version == 1:
                self.gt_cropped_dir = self.cap_paths["gt_cropped_dir"]
                self.input_cropped_dir = self.cap_paths["input_cropped_dir"]
                self.edge_cropped_dir = self.cap_paths["edge_cropped_dir"]
                self.gt_edge_cropped_dir = self.cap_paths["gt_edge_cropped_dir"]

                if (
                    self.gt_cropped_dir is not None
                    and self.input_cropped_dir is not None
                    and self.edge_cropped_dir is not None
                    and self.gt_edge_cropped_dir is not None
                ):
                    self.cap_aug = SimpleCapAug(
                        n_objects_range=[5, 10],
                        h_range=[25, 100],
                        x_range=[0, self.patch_size],
                        y_range=[0, self.patch_size],
                    )
                else:
                    raise ValueError(
                        f"Invalid path found during cap version {self.cap_version} init."
                    )
            elif self.cap_version == 2:
                self.gt_img_root = self.cap_paths["gt_img_root"]
                self.gt_txt_path = self.cap_paths["gt_txt_path"]
                self.low_img_root = self.cap_paths["low_img_root"]
                self.edge_img_root = self.cap_paths["edge_img_root"]
                self.gt_edge_img_root = self.cap_paths["gt_edge_img_root"]

                if (
                    self.gt_img_root is not None
                    and self.gt_txt_path is not None
                    and self.low_img_root is not None
                    and self.edge_img_root is not None
                    and self.gt_edge_img_root is not None
                ):
                    self.cap_aug = GausCapAug(
                        patch_size=self.patch_size,
                        gt_img_root=self.gt_img_root,
                        gt_txt_path=self.gt_txt_path,
                        low_img_root=self.low_img_root,
                        edge_img_root=self.edge_img_root,
                        gt_edge_img_root=self.gt_edge_img_root,
                        dataset_type=self.dataset_type,
                    )
                    logger.info(
                        f"CAP version 2 stats - mean_w: {self.cap_aug.mean_w}, std_w: {self.cap_aug.std_w}, mean_h: {self.cap_aug.mean_h}, std_h: {self.cap_aug.std_h}"
                    )
                else:
                    raise ValueError(
                        f"Invalid path found during cap version {self.cap_version} init."
                    )
            else:
                raise ValueError("Invalid cap_version")

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(" ")
        input_img_name = os.path.basename(img_names[0].strip())
        gt_img_name = os.path.basename(img_names[1].strip())

        filename_no_ext = os.path.splitext(input_img_name)[0]

        gt_img_path = os.path.join(self.gt_img_dir, gt_img_name)
        assert os.path.isfile(gt_img_path)
        gt_img = Image.open(gt_img_path).resize((self.patch_size, self.patch_size))
        gt_img_w, gt_img_h = gt_img.size

        input_img_path = os.path.join(self.input_img_dir, input_img_name)
        assert os.path.isfile(input_img_path)
        input_img = Image.open(input_img_path).resize(
            (self.patch_size, self.patch_size)
        )
        input_img_w, input_img_h = input_img.size

        edge_path = os.path.join(self.edge_dir, input_img_name)
        assert os.path.isfile(edge_path)
        input_edge_img = (
            Image.open(edge_path)
            .resize((self.patch_size, self.patch_size))
            .convert("RGB")
        )
        input_edge_w, input_edge_h = input_edge_img.size

        gt_edge_path = os.path.join(self.gt_edge_dir, gt_img_name)
        assert os.path.isfile(gt_edge_path)
        gt_edge_img = (
            Image.open(gt_edge_path)
            .resize((self.patch_size, self.patch_size))
            .convert("RGB")
        )
        gt_edge_w, gt_edge_h = gt_edge_img.size

        assert input_img_h == gt_img_h == input_edge_h == gt_edge_h
        assert input_img_w == gt_img_w == input_edge_w == gt_edge_w

        # Random crop
        # i, j, h, w = T.RandomCrop.get_params(
        #     input_img, output_size=(self.patch_size, self.patch_size))
        # input_patch = TF.crop(input_img, i, j, h, w)
        # gt_patch = TF.crop(gt_img, i, j, h, w)
        # input_edge_patch = TF.crop(input_edge_img, i, j, h, w)
        # gt_edge_patch = TF.crop(gt_edge_img, i, j, h, w)

        # Stupid fix instead of renaming variables
        input_patch = input_img
        gt_patch = gt_img
        input_edge_patch = input_edge_img
        gt_edge_patch = gt_edge_img

        # Random horizontal flipping
        if torch.randint(0, 2, (1,))[0].item() == 1:
            input_patch = TF.hflip(input_patch)
            input_edge_patch = TF.hflip(input_edge_patch)
            gt_edge_patch = TF.hflip(gt_edge_patch)
            gt_patch = TF.hflip(gt_patch)

        # Random vertical flipping
        if torch.randint(0, 2, (1,))[0].item() == 1:
            input_patch = TF.vflip(input_patch)
            input_edge_patch = TF.vflip(input_edge_patch)
            gt_edge_patch = TF.vflip(gt_edge_patch)
            gt_patch = TF.vflip(gt_patch)

        # Random transpose
        # use numpy here because the object is pil image
        input_patch = np.asarray(input_patch)
        input_edge_patch = np.asarray(input_edge_patch)
        gt_patch = np.asarray(gt_patch)
        gt_edge_patch = np.asarray(gt_edge_patch)

        if torch.randint(0, 2, (1,))[0].item() == 1:
            input_patch = np.transpose(input_patch, (1, 0, 2))
            input_edge_patch = np.transpose(input_edge_patch, (1, 0, 2))
            gt_patch = np.transpose(gt_patch, (1, 0, 2))
            gt_edge_patch = np.transpose(gt_edge_patch, (1, 0, 2))

        if self.use_cap:
            if self.cap_version == 1:
                # to be used for copy and paste aug
                self.gt_source_imgs = sorted(
                    list(
                        Path(os.path.join(self.gt_cropped_dir, gt_img_name)).glob(
                            "*.png"
                        )
                    )
                )
                self.low_source_imgs = sorted(
                    list(
                        Path(os.path.join(self.input_cropped_dir, input_img_name)).glob(
                            "*.png"
                        )
                    )
                )
                self.edge_source_imgs = sorted(
                    list(
                        Path(os.path.join(self.edge_cropped_dir, input_img_name)).glob(
                            "*.png"
                        )
                    )
                )
                self.gt_edge_source_imgs = sorted(
                    list(
                        Path(os.path.join(self.gt_edge_cropped_dir, gt_img_name)).glob(
                            "*.png"
                        )
                    )
                )

                if (
                    len(self.gt_source_imgs) > 0
                    and len(self.low_source_imgs) > 0
                    and len(self.edge_source_imgs) > 0
                    and len(self.gt_edge_source_imgs) > 0
                ):
                    gt_patch, input_patch, input_edge_patch, gt_edge_patch, _, _ = (
                        self.cap_aug(
                            self.gt_source_imgs,
                            self.low_source_imgs,
                            self.edge_source_imgs,
                            self.gt_edge_source_imgs,
                            gt_patch,
                            input_patch,
                            input_edge_patch,
                            gt_edge_patch,
                        )
                    )
            elif self.cap_version == 2:
                txt_name = "gt_img_" + gt_img_name.replace(".png", ".txt")
                txt_path = os.path.join(self.gt_txt_path, txt_name)
                assert os.path.isfile(txt_path), f"Invalid txt_path: {txt_path}"
                input_patch, gt_patch, input_edge_patch, gt_edge_patch = self.cap_aug(
                    input_patch,
                    gt_patch,
                    input_edge_patch,
                    gt_edge_patch,
                    random.randrange(0, self.patch_size // 2),
                    random.randrange(0, self.patch_size // 2),
                    txt_path,
                )
            else:
                raise ValueError("Invalid cap_version")

        # it is stupid to load all the edges and not use it
        # but this is the simplest change to the code
        # so if we want to use canny, we will infer it directly from input or gt
        if self.input_use_canny:
            tmp_input_edge_patch = feature.canny(
                color.rgb2gray(Image.fromarray(input_patch)), sigma=0.5
            )
            input_edge_patch = tmp_input_edge_patch * 255

        if self.gt_use_canny:
            tmp_gt_edge_patch = feature.canny(
                color.rgb2gray(Image.fromarray(gt_patch)), sigma=0.5
            )
            gt_edge_patch = tmp_gt_edge_patch * 255

        assert input_patch.dtype == np.uint8, (
            f"Invalid data type input_patch: {input_patch.dtype}"
        )
        input_patch = T.ToTensor()(input_patch)

        assert gt_patch.dtype == np.uint8, (
            f"Invalid data type gt_patch: {gt_patch.dtype}"
        )
        gt_patch = T.ToTensor()(gt_patch)

        input_edge_patch = Image.fromarray(np.uint8(input_edge_patch)).convert("L")
        gt_edge_patch = Image.fromarray(np.uint8(gt_edge_patch)).convert("L")

        input_edge_patch = T.ToTensor()(input_edge_patch)
        gt_edge_patch = T.ToTensor()(gt_edge_patch)

        r, g, b = input_patch[0, :, :], input_patch[1, :, :], input_patch[2, :, :]
        in_gray_patch = torch.unsqueeze(1.0 - (0.299 * r + 0.587 * g + 0.114 * b), 0)

        sample = {
            "in_img": input_patch,
            "in_gray": in_gray_patch,
            "in_edge": input_edge_patch,
            "gt_img": gt_patch,
            "gt_edge": gt_edge_patch,
            "filename_no_ext": filename_no_ext,
        }

        return sample


class LOLTestSet(Dataset):
    """LOL Test dataset."""

    def __init__(self, target_size, list_file, input_img_dir, gt_img_dir, edge_dir, gt_edge_dir):
        with open(list_file, "r") as f:
            self.list_file_lines = f.readlines()
        self.input_img_dir = input_img_dir
        self.gt_img_dir = gt_img_dir
        self.target_size = target_size
        self.edge_dir = edge_dir
        self.gt_edge_dir = gt_edge_dir

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(" ")
        input_img_name = os.path.basename(img_names[0].strip())
        gt_img_name = os.path.basename(img_names[1].strip())

        filename_no_ext = os.path.splitext(input_img_name)[0]

        input_img_path = os.path.join(self.input_img_dir, input_img_name)
        assert os.path.isfile(input_img_path)

        input_img = Image.open(input_img_path)
        input_img_w, input_img_h = input_img.size

        if max(input_img_h, input_img_w) > self.target_size:
            resize_ratio = self.target_size / max(input_img_h, input_img_w)
        else:
            resize_ratio = 1.0

        target_H, target_W = (
            int(input_img_h * resize_ratio),
            int(input_img_w * resize_ratio),
        )
        dim = (target_W, target_H)

        input_img = input_img.resize(dim, resample=Image.Resampling.LANCZOS)

        W1, H1 = input_img.size

        if (W1 % 32) != 0:
            W1 = W1 + 32 - (W1 % 32)
        if (H1 % 32) != 0:
            H1 = H1 + 32 - (H1 % 32)

        dim = (W1, H1)

        # Because the size of SID image is too big, resizing is unavoidable
        # Alex
        final_ratio_w = W1 / input_img_w
        final_ratio_h = H1 / input_img_h
        input_img = input_img.resize(dim, resample=Image.Resampling.LANCZOS)

        gt_img_path = os.path.join(self.gt_img_dir, gt_img_name)
        assert os.path.isfile(gt_img_path)
        gt_img = Image.open(gt_img_path)
        gt_img_w, gt_img_h = gt_img.size
        gt_img = gt_img.resize(dim, resample=Image.Resampling.LANCZOS)

        edge_path = os.path.join(self.edge_dir, input_img_name)
        assert os.path.isfile(edge_path)
        input_edge_img = Image.open(edge_path).convert("L")
        input_edge_w, input_edge_h = input_edge_img.size
        input_edge_img = input_edge_img.resize(dim, resample=Image.Resampling.LANCZOS)

        gt_edge_path = os.path.join(self.gt_edge_dir, gt_img_name)
        assert os.path.isfile(gt_edge_path)
        gt_edge_img = Image.open(gt_edge_path).convert("L")
        gt_edge_w, gt_edge_h = gt_edge_img.size
        gt_edge_img = gt_edge_img.resize(dim, resample=Image.Resampling.LANCZOS)

        assert input_img_h == gt_img_h == input_edge_h == gt_edge_h
        assert input_img_w == gt_img_w == input_edge_w == gt_edge_w

        input_img = T.ToTensor()(input_img)
        gt_img = T.ToTensor()(gt_img)
        input_edge_img = T.ToTensor()(input_edge_img)
        gt_edge_img = T.ToTensor()(gt_edge_img)

        r, g, b = input_img[0, :, :], input_img[1, :, :], input_img[2, :, :]
        input_gray_img = torch.unsqueeze(1.0 - (0.299 * r + 0.587 * g + 0.114 * b), 0)

        sample = {
            "in_img": input_img,
            "in_edge": input_edge_img,
            "in_gray": input_gray_img,
            "gt_img": gt_img,
            "gt_edge": gt_edge_img,
            "final_ratio_w": final_ratio_w,
            "final_ratio_h": final_ratio_h,
            "filename_no_ext": filename_no_ext,
            "input_img_name": input_img_name,
        }

        return sample


# Make sure each process has different random seed, especially for 'fork' method.
# Check https://github.com/pytorch/pytorch/issues/63311 for more details.
# From YOLOX: https://github.com/Megvii-BaseDetection/YOLOX/blob/7ee17936db849600817d7de05269bfdfb1a0eb48/yolox/exp/yolox_base.py#L189
def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)
