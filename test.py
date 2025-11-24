import argparse
import os
import time
import warnings
import zipfile
from datetime import datetime

import cv2
import lpips
import numpy as np
import PIL
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
warnings.filterwarnings("ignore", category=UserWarning)

from cc_eval.cc_hmean import cc_hmean

from utils import utils
from utils.CRAFTpytorch.craft import CRAFT


def test(cfg):
    logger.info(f"------PIL version: {PIL.__version__}.")

    # define some output folders
    detection_txt_folder = os.path.join(cfg.log_dir, "det_txt")
    zip_path = os.path.join(cfg.log_dir, "det_txt.zip")
    if not os.path.isdir(detection_txt_folder):
        os.makedirs(detection_txt_folder)

    # initialize network
    if cfg.unet_type == "cc_unet":
        from models.unet import CCUNet as UNet

        unet = UNet(
            psa_type=cfg.psa_type,
            use_bias=True,
            use_batch_norm=cfg.use_bn,
            spatial_weight=cfg.spatial_weight,
            channel_weight=cfg.channel_weight,
        )
    elif cfg.unet_type == "cc_unet_nedge":
        from models.unet import CCUNet_NestedEdge as UNet

        unet = UNet(
            psa_type=cfg.psa_type,
            use_bias=True,
            use_batch_norm=cfg.use_bn,
            spatial_weight=cfg.spatial_weight,
            channel_weight=cfg.channel_weight,
        )
    elif cfg.unet_type == "cc_unet_nedge_v2":
        from models.unet import CCUNet_NestedEdge_v2 as UNet

        unet = UNet(
            psa_type=cfg.psa_type,
            use_bias=True,
            use_batch_norm=cfg.use_bn,
            spatial_weight=cfg.spatial_weight,
            channel_weight=cfg.channel_weight,
            use_aux_loss=cfg.aux_loss,
        )
    elif cfg.unet_type == "howard_unet":
        from models.unet import GrayEdgeAttentionUNet as UNet

        unet = UNet()
    elif cfg.unet_type == "plain_unet":
        from models.unet import UNet as UNet

        unet = UNet(use_batch_norm=cfg.use_bn)
    elif cfg.unet_type == "att_plain_unet":
        from models.unet import UNet_Att as UNet

        unet = UNet(use_bias=True, use_batch_norm=cfg.use_bn)
    else:
        raise ValueError(f"Invalid unet_type: {cfg.unet_type}")

    if cfg.use_dp and torch.cuda.device_count() > 1:
        logger.info(f"------Using {torch.cuda.device_count()} GPUs!")
        unet = torch.nn.DataParallel(unet)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet.cuda()

    if os.path.exists(cfg.weights):
        # we need to re-init the optimizer and scheduler when we perform mix training
        load_epoch, unet = utils.load_checkpoint_state(
            cfg.use_dp, "eval", cfg.weights, device, unet
        )

        logger.info(f"------Loaded pretrained model of {load_epoch}th epoch.")
    else:
        logger.info("------No pretrained model.")

    # CRAFT net
    if cfg.use_dp and torch.cuda.device_count() > 1:
        craft_net = torch.nn.DataParallel(CRAFT())
    else:
        craft_net = CRAFT()
    craft_net.load_state_dict(
        utils.copyStateDict(cfg.use_dp, torch.load(cfg.craft_pretrained_model))
    )
    craft_net.cuda()
    craft_net.eval()

    # Load LPIPS model
    lpips_model = lpips.LPIPS(net="alex")
    lpips_model.cuda()

    if cfg.test_dataset_type == "sony":
        from dataset.sony import SonyTestSet as TestSet
    elif cfg.test_dataset_type == "fuji":
        from dataset.fuji import FujiTestSet as TestSet
    elif cfg.test_dataset_type == "icdar15":
        from dataset.icdar15 import IC15TestSet as TestSet
    elif cfg.test_dataset_type == "lol":
        from dataset.lol import LOLTestSet as TestSet
    else:
        raise ValueError(f"Invalid test_dataset_type: {cfg.test_dataset_type}")

    logger.info(f"------Evaluated using image size: {cfg.target_size}.")
    if cfg.test_dataset_type == "lol":
        test_dataset = TestSet(
            cfg.target_size,
            list_file=cfg.test_list_file,
            input_img_dir=cfg.test_input_dir,
            gt_img_dir=cfg.test_gt_dir,
            edge_dir=cfg.test_edge_dir,
        )
    else:
        test_dataset = TestSet(
            cfg.target_size,
            list_file=cfg.test_list_file,
            input_img_dir=cfg.test_input_dir,
            gt_img_dir=cfg.test_gt_dir,
            edge_dir=cfg.test_edge_dir,
            gt_edge_dir=cfg.test_gt_edge_dir,
            ratio_multiplier=cfg.test_ratio_multiplier,
            input_use_canny=cfg.input_use_canny,
            gt_use_canny=cfg.gt_use_canny,
        )

    if cfg.test_ratio_multiplier > 0:
        logger.info(
            f"Multiplying low light image with exposure ratio of {cfg.test_ratio_multiplier} for test set"
        )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=True,
    )

    highest_hmean = 0.0

    # Always define testing_img_folder for cc_hmean call
    testing_img_folder = os.path.join(cfg.log_dir, cfg.out_img_folder)
    if cfg.gen_image:
        if not os.path.isdir(testing_img_folder):
            os.makedirs(testing_img_folder)

    with torch.no_grad():
        eval_time = time.perf_counter()

        unet.eval()
        psnr_list, ssim_list, lpips_list = [], [], []

        with tqdm(test_dataloader, unit="batch") as tqdm_loader:
            for sample in tqdm_loader:
                dt_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                tqdm_loader.set_description(f"[{dt_string}] Eval Epoch [{load_epoch}]")

                eval_sample_time = time.perf_counter()

                input_img_name = sample["input_img_name"][0]  # input filename
                in_img = sample["in_img"].cuda()
                gt_img = sample["gt_img"].cuda()
                in_edge_img = sample["in_edge"].cuda()

                eval_unet_time = time.perf_counter()
                if cfg.unet_type == "cc_unet":
                    out_img = unet(in_img, in_edge_img)
                elif cfg.unet_type == "howard_unet":
                    in_gray_img = sample["in_gray"].cuda()
                    out_img = unet(in_img, in_gray_img, in_edge_img)
                elif cfg.unet_type == "plain_unet" or cfg.unet_type == "att_plain_unet":
                    out_img = unet(in_img)
                elif (
                    cfg.unet_type == "cc_unet_nedge"
                    or cfg.unet_type == "cc_unet_nedge_v2"
                ):
                    out_img, _ = unet(in_img, in_edge_img)
                else:
                    raise ValueError(f"Invalid unet_type: {cfg.unet_type}")
                eval_unet_end_time = time.perf_counter() - eval_unet_time

                psnr_list.append(utils.PSNR(out_img, gt_img).item())
                ssim_list.append(utils.SSIM(out_img, gt_img).item())
                lpips_list.append(utils.LPIPS(out_img, gt_img, lpips_model).item())

                # The bboxes will be rescaled back to the original size
                # for the h-mean estimation
                eval_craft_time = time.perf_counter()
                utils.TextDetection(
                    out_img,
                    sample["filename_no_ext"],
                    detection_txt_folder,
                    cfg.file_prefix,
                    craft_net,
                    cfg.text_threshold,
                    cfg.link_threshold,
                    cfg.low_text,
                    sample["final_ratio_w"],
                    sample["final_ratio_h"],
                )
                eval_craft_end_time = time.perf_counter() - eval_craft_time

                if cfg.gen_image:
                    # output_img = to_pil(torch.squeeze(out_img, 0))
                    # filename = os.path.join(testing_img_folder, input_img_name)
                    # output_img.save(filename)

                    output_img = utils.Tensor2OpenCV(out_img)
                    filename = os.path.join(testing_img_folder, input_img_name)
                    output_img = np.uint8(output_img)
                    cv2.imwrite(filename, output_img)

                eval_end_sample_time = time.perf_counter() - eval_sample_time
                log_str = "UNET_Time=%.3f, CRAFT_Time=%.3f, Total_Time=%.3f" % (
                    eval_unet_end_time,
                    eval_craft_end_time,
                    eval_end_sample_time,
                )
                tqdm_loader.postfix = log_str

        # manually zip all detection files and 1st delete the zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)

        eval_hmean_time = time.perf_counter()
        txts = os.listdir(detection_txt_folder)
        with zipfile.ZipFile(zip_path, "a") as zip:
            for txt_name in txts:
                if ".txt" in txt_name:
                    zip.write(
                        filename=os.path.join(detection_txt_folder, txt_name),
                        arcname=txt_name,
                    )

        # estimating h-mean of IOU.
        # h-mean of TIOU and SIOU are estimated but not shown.
        resDict = utils.eval(cfg.eval_dataset_type, cfg.gt_path, zip_path, cfg.log_dir)
        hmean = round(resDict["method"]["hmean"], 3)
        eval_hmean_end_time = time.perf_counter() - eval_hmean_time
        per_eval_time = time.perf_counter() - eval_time

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list)

        logger.info(
            "------final_ratio_w=%.3f final_ratio_h=%.3f"
            % (sample["final_ratio_w"].item(), sample["final_ratio_h"].item())
        )
        logger.info(
            "------HMEAN_Time=%.3f per_eval_time=%.3f"
            % (eval_hmean_end_time, per_eval_time)
        )
        logger.info("------hmean={}, the highest_hmean={}".format(hmean, highest_hmean))
        logger.info(
            "------PSNR={}, SSIM={}, LPIPS={}".format(avg_psnr, avg_ssim, avg_lpips)
        )

        logger.remove()
        cc_hmean(
            cfg.test_dataset_type, testing_img_folder, cfg.log_dir, load_epoch, False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="configs/cfg_sony.yaml", help="cfgl path"
    )
    parser.add_argument(
        "--unet-type", type=str, default="cc_unet", help="select unet type"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Total batch size for all gpus."
    )
    parser.add_argument("--weights", type=str, default="", help="initial weights path")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="maximum number of dataloader workers",
    )
    parser.add_argument("--gen-image", action="store_true", help="to output image")
    parser.add_argument("--use-dp", action="store_true", help="to use DataParallel")
    parser.add_argument("--use-bn", action="store_true", help="to use BatchNorm")
    parser.add_argument(
        "--project", default="results", help="save to project/name/eval_name"
    )
    parser.add_argument("--name", default="exp", help="save to project/name/eval_name")
    parser.add_argument(
        "--eval-name", default="eval", help="save to project/name/eval_name"
    )
    parser.add_argument(
        "--spatial-weight", type=float, default=0.8, help="weight for spatial attention"
    )
    parser.add_argument(
        "--channel-weight", type=float, default=0.2, help="weight for channel attention"
    )
    parser.add_argument("--psa-type", default="normal", help="which psa to use")
    parser.add_argument(
        "--aux-loss",
        action="store_true",
        help="whether to use aux loss for cc_unet_nedge_v2",
    )
    parser.add_argument(
        "--test-ratio-multiplier",
        type=float,
        default=0.0,
        help="the exposure ratio multiplier to be used.",
    )
    parser.add_argument(
        "--input-use-canny",
        action="store_true",
        default=False,
        help="whether to use canny edge for input image",
    )
    parser.add_argument(
        "--gt-use-canny",
        action="store_true",
        default=False,
        help="whether to use canny edge for gt image",
    )
    parser.add_argument(
        "--out-img-folder", type=str, default="out_img", help="output image folder"
    )
    parser.add_argument(
        "--concat-input",
        action="store_true",
        default=False,
        help="whether to concat input for testing",
    )
    opt = parser.parse_args()
    args_dict = vars(opt)

    # do remember that parse_args will override values in cfg
    cfg = utils.load_config(opt.cfg)
    cfg.update(args_dict)
    cfg.log_dir = f"{opt.project}/{opt.name}/{opt.eval_name}"

    dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
    logger.remove()
    logger.add(
        f"{cfg.log_dir}/{dt_string}_console.log",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        backtrace=True,
        diagnose=True,
        colorize=True,
    )
    logger.info("------Configuration Details:")
    for k, v in cfg.items():
        logger.info(f"{k}:{v}")

    test(cfg)
