## Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method
## Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, Tong Lu
## https://arxiv.org/pdf/2212.11548.pdf

import argparse
import glob
import os
import time
from collections import OrderedDict

import cv2
import lpips
import numpy as np
import pyiqa
import torch
from loguru import logger
from natsort import natsort
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


class Measure:
    def __init__(self, net="alex", use_gpu=False):
        self.device = "cuda" if use_gpu else "cpu"
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        score = ssim(imgA, imgB, channel_axis=-1)
        return score

    def psnr(self, imgA, imgB):
        psnr_val = psnr(imgA, imgB)
        return psnr_val


class Measure_IQA:
    def __init__(self, use_gpu=False):
        self.device = "cuda" if use_gpu else "cpu"
        self.iqa_lpips_metric = pyiqa.create_metric("lpips", device=self.device)
        self.iqa_psnr_metric = pyiqa.create_metric("psnr", device=self.device)
        self.iqa_ssim_metric = pyiqa.create_metric("ssim", device=self.device)
        self.iqa_niqe_metric = pyiqa.create_metric("niqe", device=self.device)

    def measure(self, imgA, imgB):
        return [
            float(self.iqa_eval(imgA, imgB, metric=f))
            for f in ["psnr", "ssim", "lpips", "niqe"]
        ]

    def iqa_eval(self, imgA, imgB, metric=""):
        tA = t_iqa(imgA).to(self.device)
        tB = t_iqa(imgB).to(self.device)

        if metric == "psnr":
            val = self.iqa_psnr_metric(tA, tB).item()
        elif metric == "ssim":
            val = self.iqa_ssim_metric(tA, tB).item()
        elif metric == "lpips":
            val = self.iqa_lpips_metric(tA, tB).item()
        elif metric == "niqe":
            val = self.iqa_niqe_metric(tB).item()
        else:
            raise NotImplementedError(f"wrong metric name for iqa {metric}.")

        return val


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def t_iqa(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 255


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def format_result(psnr, ssim, lpips):
    return f"PSNR:{psnr:0.5f}, SSIM:{ssim:0.5f}, LPIPS:{lpips:0.5f}"


def resize_img(img, target_size):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    if h != target_h or w != target_w:
        # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
        if h > target_h or w > target_w:
            # shrink
            interpolation = cv2.INTER_AREA
        else:
            # enlarge
            interpolation = cv2.INTER_CUBIC
        return cv2.resize(img, target_size, interpolation=interpolation)
    return img


def measure_dirs(
    dirA, dirB, use_gpu, image_type, per_img_result, target_size, use_iqa=False
):
    t_init = time.time()

    paths_A = fiFindByWildcard(os.path.join(dirA, f"*.{image_type}"))
    paths_B = fiFindByWildcard(os.path.join(dirB, f"*.{image_type}"))

    if per_img_result:
        logger.info("Per image results gt_img, pred_img, PSNR, SSIM, LPIPS, time: ")

    measure = Measure(use_gpu=use_gpu)

    if use_iqa:
        measure_iqa = Measure_IQA(use_gpu=use_gpu)

    results = []
    for pathA, pathB in tqdm(zip(paths_A, paths_B), unit="image"):
        result = OrderedDict()

        t = time.time()

        gt_img = imread(pathA)
        gt_img = resize_img(gt_img, target_size)
        en_img = imread(pathB)
        en_img = resize_img(en_img, target_size)

        result["psnr"], result["ssim"], result["lpips"] = measure.measure(
            gt_img, en_img
        )
        d = time.time() - t

        if per_img_result:
            logger.info(
                f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, {format_result(**result)}, {d:0.1f}"
            )

        results.append(result)

        if use_iqa:
            (
                result["iqa_psnr"],
                result["iqa_ssim"],
                result["iqa_lpips"],
                result["iqa_niqe"],
            ) = measure_iqa.measure(gt_img, en_img)

    psnr = np.mean([result["psnr"] for result in results])
    ssim = np.mean([result["ssim"] for result in results])
    lpips = np.mean([result["lpips"] for result in results])

    if use_iqa:
        iqa_psnr = np.mean([result["iqa_psnr"] for result in results])
        iqa_ssim = np.mean([result["iqa_ssim"] for result in results])
        iqa_lpips = np.mean([result["iqa_lpips"] for result in results])
        iqa_niqe = np.mean([result["iqa_niqe"] for result in results])

    if use_iqa:
        return (
            psnr,
            ssim,
            lpips,
            iqa_psnr,
            iqa_ssim,
            iqa_lpips,
            iqa_niqe,
            time.time() - t_init,
        )
    else:
        return psnr, ssim, lpips, time.time() - t_init


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dirA", default="./datasets/LOL/test/high/", type=str)
    parser.add_argument("-dirB", default="./results/LOL/", type=str)
    parser.add_argument("-image_type", default="png")
    parser.add_argument("--use_gpu", default=True)
    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB
    image_type = args.image_type
    use_gpu = args.use_gpu

    if len(dirA) > 0 and len(dirB) > 0:
        measure_dirs(dirA, dirB, use_gpu, image_type)
