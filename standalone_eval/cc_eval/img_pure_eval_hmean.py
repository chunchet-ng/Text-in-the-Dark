from __future__ import division

import argparse
import os
import platform
import shutil
import time
import zipfile
from datetime import datetime

import cv2
import numpy as np
import skimage
import torch
import torchvision
from loguru import logger
from PIL import Image
from tqdm import tqdm

import cc_eval.cc_utils as utils
import cc_eval.std_eval_utils as std_utils
from cc_eval.CRAFTpytorch.craft import CRAFT


def construct_path_dict(txt_file_path):
    path_dict = {}
    with open(txt_file_path, mode="r") as txt:
        lines = txt.readlines()
        for line in lines:
            line = line.strip().split(" ")
            input_name = os.path.basename(line[0].strip())
            gt_name = os.path.basename(line[1].strip())
            path_dict[input_name] = gt_name
    return path_dict


def main(cfg):
    # define variables
    path_dict = construct_path_dict(cfg.input_path_txt)
    psnr_list, ssim_list = [], []

    final_ratio_w = 1
    final_ratio_h = 1
    target_w = cfg.target_size[0]
    target_h = cfg.target_size[1]
    gt_dir = cfg.gt_img_path
    pred_dir = cfg.img_path

    all_gts = os.listdir(gt_dir)
    all_imgs = os.listdir(pred_dir)

    to_tensor = torchvision.transforms.ToTensor()

    # init CRAFT model
    craft_net = CRAFT()
    craft_net.load_state_dict(
        utils.copyStateDict(False, torch.load(cfg.craft_pretrained_model))
    )
    craft_net.cuda()
    craft_net.eval()

    # init logging
    dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
    logger.remove()
    logger.add(
        f"{cfg.log_out_path}/{dt_string}_console.log",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        backtrace=True,
        diagnose=True,
        colorize=True,
    )

    # log package versions
    logger.info("------Package version:")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"CV2 version: {cv2.__version__}")
    logger.info(f"PIL version: {Image.__version__}")
    logger.info(f"torch version: {torch.__version__}")
    logger.info(f"torchvision version: {torchvision.__version__}")
    logger.info(f"skimage version: {skimage.__version__}")

    if cfg.use_iqa:
        try:
            import pyiqa

            logger.info(f"pyiqa version: {pyiqa.__version__}")
        except ImportError:
            pass

    logger.info(f"------Evaluating Epoch {cfg.epoch}:")
    logger.info("------Configuration Details:")
    for k, v in cfg.items():
        logger.info(f"{k}:{v}")

    with torch.no_grad():
        for img_name in tqdm(all_imgs, unit="image"):
            full_img_path = os.path.join(pred_dir, img_name)

            pil_img = Image.open(full_img_path)
            ori_input_width, ori_input_height = pil_img.size
            assert np.array(pil_img).shape[-1] == 3, (
                f"{full_img_path} - invalid image shape: {np.array(pil_img).shape}"
            )
            if ori_input_width != target_w or ori_input_height != target_h:
                pil_img = pil_img.resize(cfg.target_size, resample=Image.ANTIALIAS)

            pil_img_arr = np.array(pil_img).astype(np.uint8)
            out_img = torch.unsqueeze(to_tensor(pil_img_arr), dim=0).cuda()

            gt_img_name = path_dict[img_name]
            gt_full_img_path = os.path.join(gt_dir, gt_img_name)
            pil_gt_img = Image.open(gt_full_img_path)

            ori_gt_width, ori_gt_height = pil_gt_img.size
            if ori_gt_width != target_w or ori_gt_height != target_h:
                pil_gt_img = pil_gt_img.resize(
                    cfg.target_size, resample=Image.ANTIALIAS
                )

            final_ratio_w = target_w / ori_gt_width
            final_ratio_h = target_h / ori_gt_height

            assert pil_gt_img.size == pil_img.size

            pil_gt_img_arr = np.array(pil_gt_img).astype(np.uint8)
            gt_img = torch.unsqueeze(to_tensor(pil_gt_img_arr), dim=0).cuda()

            if cfg.dataset_type == "sony" or cfg.dataset_type == "fuji":
                filename_no_ext = os.path.splitext(os.path.basename(img_name))[0]
                filename_no_ext = (
                    filename_no_ext.replace("_", "").replace(".", "").replace("s", "")
                )
            else:
                filename_no_ext = os.path.splitext(os.path.basename(img_name))[0]

            if cfg.craft_cfg is not None:
                utils.TextDetection(
                    out_img,
                    [filename_no_ext],
                    cfg.det_txt,
                    "res_img_",
                    craft_net,
                    float(cfg.craft_cfg["text_threshold"]),
                    float(cfg.craft_cfg["low_text"]),
                    float(cfg.craft_cfg["link_threshold"]),
                    final_ratio_w,
                    final_ratio_h,
                )
            else:
                utils.TextDetection(
                    out_img,
                    [filename_no_ext],
                    cfg.det_txt,
                    "res_img_",
                    craft_net,
                    0.7,
                    0.4,
                    0.4,
                    final_ratio_w,
                    final_ratio_h,
                )

            psnr_list.append(utils.PSNR(out_img, gt_img).item())
            ssim_list.append(utils.SSIM(out_img, gt_img).item())

    logger.info(f"ori_input_img_size: {(ori_input_width, ori_input_height)}")
    logger.info(f"ori_gt_img_size: {ori_gt_width, ori_gt_height}")
    # logger.info(f'img_resize: {img_resize}')
    # logger.info(f'gt_img_resize: {gt_img_resize}')
    logger.info(f"resized_input_img_size: {pil_img.size}")
    logger.info(f"resized_gt_img_size: {pil_gt_img.size}")
    logger.info(f"final_ratio_w: {final_ratio_w}")
    logger.info(f"final_ratio_h: {final_ratio_h}")

    # estimating h-mean of IOU.
    # h-mean of TIOU and SIOU are estimated but not shown.
    eval_hmean_time = time.perf_counter()
    if not os.path.exists("/".join(x for x in cfg.zip_path.split("/")[:-1])):
        os.makedirs("/".join(x for x in cfg.zip_path.split("/")[:-1]))
    if os.path.isfile(cfg.zip_path):
        os.remove(cfg.zip_path)

    txts = os.listdir(cfg.det_txt)
    with zipfile.ZipFile(cfg.zip_path, "a") as zip:
        for txt_name in tqdm(txts):
            if ".txt" in txt_name:
                zip.write(
                    filename=os.path.join(cfg.det_txt, txt_name), arcname=txt_name
                )

    # estimating h-mean of IOU.
    # h-mean of TIOU and SIOU are estimated but not shown.
    resDict = utils.eval(
        "icdar15", cfg.gt_path, cfg.zip_path, cfg.out_path, per_sample_result=True
    )
    hmean = resDict["method"]["hmean"]
    precision = resDict["method"]["precision"]
    recall = resDict["method"]["recall"]
    eval_hmean_end_time = time.perf_counter() - eval_hmean_time

    cc_psnr = np.mean(psnr_list)
    cc_ssim = np.mean(ssim_list)

    logger.info(f"------GT_path: {gt_dir}, files: {len(all_gts)}")
    logger.info(f"------Pred_path: {pred_dir}, files: {len(all_imgs)}")
    logger.info("------Measuring using CC's PSNR & SSIM------")
    logger.info(f"------PSNR={cc_psnr}")
    logger.info(f"------SSIM={cc_ssim}")
    logger.info(f"------HMean={hmean}")
    logger.info(f"------HMean_Time={eval_hmean_end_time:.3f}s")

    logger.info(
        "------Measuring using skimage's PSNR & SSIM, and LPIPS (for ref. only)------"
    )

    if cfg.dataset_type.lower() == "icdar15":
        image_type = "jpg"
    else:
        image_type = "png"

    if cfg.use_iqa:
        (
            skim_psnr,
            skim_ssim,
            lpips,
            iqa_psnr,
            iqa_ssim,
            iqa_lpips,
            iqa_niqe,
            extra_time,
        ) = std_utils.measure_dirs(
            gt_dir,
            pred_dir,
            use_gpu=True,
            image_type=image_type,
            per_img_result=cfg.per_img_result,
            target_size=cfg.target_size,
            use_iqa=True,
        )
    else:
        skim_psnr, skim_ssim, lpips, extra_time = std_utils.measure_dirs(
            gt_dir,
            pred_dir,
            use_gpu=True,
            image_type=image_type,
            per_img_result=cfg.per_img_result,
            target_size=cfg.target_size,
            use_iqa=False,
        )

    logger.info(f"------SKIM_PSNR={skim_psnr}")
    logger.info(f"------SKIM_SSIM={skim_ssim}")
    logger.info(f"------LPIPS={lpips}")

    if cfg.use_iqa:
        logger.info(
            "------Measuring using IQA-PyTorch's PSNR & SSIM, LPIPS and NIQE (for ref. only)------"
        )
        logger.info(f"------IQA_PSNR={iqa_psnr}")
        logger.info(f"------IQA_SSIM={iqa_ssim}")
        logger.info(f"------IQA_LPIPS={iqa_lpips}")
        logger.info(f"------IQA_NIQE={iqa_niqe}")

    logger.info(f"------Additional_Eval_Time={extra_time:.3f}s")

    if cfg.del_out:
        logger.info(f"Deleting output file: {cfg.out_path}")
        try:
            shutil.rmtree(cfg.out_path)
        except OSError as e:
            logger.info(f"Error: {e.filename} - {e.strerror}.")

    if cfg.use_iqa:
        out_dict = {
            "ori_psnr": cc_psnr,
            "ori_ssim": cc_ssim,
            "hmean": hmean,
            "precision": precision,
            "recall": recall,
            "skim_psnr": skim_psnr,
            "skim_ssim": skim_ssim,
            "lpips": lpips,
            "iqa_psnr": iqa_psnr,
            "iqa_ssim": iqa_ssim,
            "iqa_lpips": iqa_lpips,
            "iqa_niqe": iqa_niqe,
        }
    else:
        out_dict = {
            "ori_psnr": cc_psnr,
            "ori_ssim": cc_ssim,
            "hmean": hmean,
            "precision": precision,
            "recall": recall,
            "skim_psnr": skim_psnr,
            "skim_ssim": skim_ssim,
            "lpips": lpips,
        }

    return out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--craft_pretrained_model", type=str, default="", help="craft path"
    )
    parser.add_argument("--input_path_txt", type=str, default="", help="img path txt")
    parser.add_argument("--img_path", type=str, default="", help="img path")
    parser.add_argument("--gt_img_path", type=str, default="", help="gt img path")
    parser.add_argument("--zip_path", type=str, default="", help="zip path")
    parser.add_argument(
        "--det_txt", type=str, default="", help="detection text folder path"
    )
    parser.add_argument("--gt_path", type=str, default="", help="gt path")
    parser.add_argument("--out_path", type=str, default="", help="output result path")
    parser.add_argument("--log_out_path", type=str, default="", help="output log path")
    parser.add_argument(
        "--del_out", action="store_true", help="whether to delete eval output"
    )
    parser.add_argument("--dataset_type", type=str, default="", help="dataset type")
    parser.add_argument(
        "--gen_image", action="store_true", help="whether to generate output image"
    )
    parser.add_argument(
        "--img_out_path", type=str, default="", help="output image path"
    )

    cfg = parser.parse_args()

    if cfg.dataset_type == "icdar15":
        cfg.target_size = (1280, 736)
    else:
        cfg.target_size = (4256, 2848)

    args_dict = vars(cfg)

    main(cfg)
