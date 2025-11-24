import argparse
import copy
import os
import time
import warnings
import zipfile
from datetime import datetime

import cv2
import lpips
import numpy as np
import torch
import torchvision.transforms as T
import wandb
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
warnings.filterwarnings("ignore", category=UserWarning)

from utils import utils
from utils.CRAFTpytorch.craft import CRAFT
from utils.RCFpytorch.models import RCF


def train(cfg, writer):
    logger.info(f"------Init training for {cfg.epochs} epochs.")
    cfg.epochs += 1

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

    # set up solver and scheduler
    cfg.learning_rate = cfg.learning_rate * cfg.batch_size

    if cfg.use_sgd:
        optimizer = optim.SGD(unet.parameters(), lr=cfg.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=100)
    else:
        optimizer = optim.Adam(unet.parameters(), lr=cfg.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[2000], gamma=cfg.scheduler_gamma
        )

    if os.path.exists(cfg.weights):
        # we need to re-init the optimizer and scheduler when we perform mix training
        if cfg.reinit_opt:
            logger.info("Reinit optimizer and scheduler")
            load_epoch, unet, _, _ = utils.load_checkpoint_state(
                cfg.use_dp, "train", cfg.weights, device, unet, optimizer, scheduler
            )
            optimizer = optim.Adam(unet.parameters(), lr=cfg.learning_rate)
            # dont have to use this because we already lower the lr, training will never reach 9999
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[9999], gamma=cfg.scheduler_gamma
            )
        else:
            load_epoch, unet, optimizer, scheduler = utils.load_checkpoint_state(
                cfg.use_dp, "train", cfg.weights, device, unet, optimizer, scheduler
            )

        # by default the ckpt will be saved before eval.
        # this is to make sure that when eval faces OOM, the ckpt is saved before crashing
        # if cfg.eval_when_resume:
        # when this option is used, theoretically the total_epoch will +1
        # because we are starting from the previously saved ckpt without +1
        # so that the eval will be carried out
        # use this option when the code crashed at eval phase
        #     start_epoch = load_epoch
        # else:
        #     start_epoch = load_epoch + 1
        start_epoch = load_epoch + 1
        logger.info(f"------Loaded pretrained model of {load_epoch}th epoch.")
    else:
        logger.info("------No pretrained model.")
        start_epoch = 1

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

    # RCF
    if cfg.use_rcf:
        if cfg.unet_type != "cc_unet_nedge" and cfg.unet_type != "cc_unet_nedge_v2":
            if cfg.use_dp and torch.cuda.device_count() > 1:
                rcf_net = torch.nn.DataParallel(RCF())
            else:
                rcf_net = RCF()
            logger.info("Using RCF for edge prediction.")
            rcf_net.load_state_dict(
                utils.copyStateDict(cfg.use_dp, torch.load(cfg.rcf_pretrained_model))
            )
            rcf_net.cuda()
            rcf_net.eval()

    # Load LPIPS model
    lpips_model = lpips.LPIPS(net="alex")
    lpips_model.cuda()

    # preparing dataloader for train/test/visualization
    from utils.utils import worker_init_reset_seed as worker_init_fn

    if cfg.mix_train:
        train_dataset_list = cfg.train_dataset_type.split("_")
        torch_dataset_list = []
        for x in train_dataset_list:
            temp_cfg = copy.deepcopy(cfg)

            # only prepare for cap version 2
            if x == "sony":
                cap_paths = {
                    "gt_img_root": cfg.sony_train_gt_dir,
                    "gt_txt_path": cfg.sony_gt_txt_path,
                    "low_img_root": cfg.sony_train_input_dir,
                    "edge_img_root": cfg.sony_train_edge_dir,
                    "gt_edge_img_root": cfg.sony_train_gt_edge_dir,
                }
                temp_cfg.train_list_file = cfg.sony_train_list_file
                temp_cfg.train_input_dir = cfg.sony_train_input_dir
                temp_cfg.train_gt_dir = cfg.sony_train_gt_dir
                temp_cfg.train_edge_dir = cfg.sony_train_edge_dir
                temp_cfg.train_gt_edge_dir = cfg.sony_train_gt_edge_dir
                temp_cfg.train_ratio_multiplier = cfg.sony_train_ratio_multiplier
            elif x == "fuji":
                cap_paths = {
                    "gt_img_root": cfg.fuji_train_gt_dir,
                    "gt_txt_path": cfg.fuji_gt_txt_path,
                    "low_img_root": cfg.fuji_train_input_dir,
                    "edge_img_root": cfg.fuji_train_edge_dir,
                    "gt_edge_img_root": cfg.fuji_train_gt_edge_dir,
                }
                temp_cfg.train_list_file = cfg.fuji_train_list_file
                temp_cfg.train_input_dir = cfg.fuji_train_input_dir
                temp_cfg.train_gt_dir = cfg.fuji_train_gt_dir
                temp_cfg.train_edge_dir = cfg.fuji_train_edge_dir
                temp_cfg.train_gt_edge_dir = cfg.fuji_train_gt_edge_dir
                temp_cfg.train_ratio_multiplier = cfg.fuji_train_ratio_multiplier
            elif x == "icdar15":
                cap_paths = {
                    "gt_img_root": cfg.icdar15_train_gt_dir,
                    "gt_txt_path": cfg.icdar15_gt_txt_path,
                    "low_img_root": cfg.icdar15_train_input_dir,
                    "edge_img_root": cfg.icdar15_train_edge_dir,
                    "gt_edge_img_root": cfg.icdar15_train_gt_edge_dir,
                }
                temp_cfg.train_list_file = cfg.icdar15_train_list_file
                temp_cfg.train_input_dir = cfg.icdar15_train_input_dir
                temp_cfg.train_gt_dir = cfg.icdar15_train_gt_dir
                temp_cfg.train_edge_dir = cfg.icdar15_train_edge_dir
                temp_cfg.train_gt_edge_dir = cfg.icdar15_train_gt_edge_dir
                temp_cfg.train_ratio_multiplier = cfg.icdar15_train_ratio_multiplier

            temp_cfg.train_dataset_type = x
            temp_train_dataset = get_train_set(temp_cfg, cap_paths=cap_paths)
            torch_dataset_list.append(temp_train_dataset)

        logger.info(
            f"Combining {len(train_dataset_list)} datasets, {train_dataset_list}"
        )
        train_dataset = torch.utils.data.ConcatDataset(torch_dataset_list)
    else:
        train_dataset = get_train_set(cfg)

    if cfg.test_dataset_type == "sony":
        from dataset.sony import SonyTestSet as TestSet
    elif cfg.test_dataset_type == "fuji":
        from dataset.fuji import FujiTestSet as TestSet
    elif cfg.test_dataset_type == "lol":
        from dataset.lol import LOLTestSet as TestSet
    elif cfg.test_dataset_type == "icdar15":
        from dataset.icdar15 import IC15TestSet as TestSet
    else:
        raise ValueError(f"Invalid test_dataset_type: {cfg.test_dataset_type}")

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

    if cfg.train_ratio_multiplier > 0:
        logger.info(
            f"Multiplying low light image with exposure ratio of {cfg.train_ratio_multiplier} for train set"
        )

    if cfg.test_ratio_multiplier > 0:
        logger.info(
            f"Multiplying low light image with exposure ratio of {cfg.test_ratio_multiplier} for test set"
        )

    if cfg.input_use_canny or cfg.gt_use_canny:
        logger.info(
            f"Use canny edge for input: {cfg.input_use_canny} and gt: {cfg.gt_use_canny}"
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
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

    # init wandb for logging
    if cfg.use_wandb:
        if cfg.wandb_id == "-1":
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=dict(cfg))
        else:
            wandb.init(
                id=cfg.wandb_id,
                project=cfg.wandb_project,
                name=cfg.wandb_name,
                config=dict(cfg),
                resume="must",
            )
            logger.info(f"Resuming wandb run id: {cfg.wandb_id}")

    if cfg.eval_when_resume and os.path.exists(cfg.weights):
        tmp_highest_hmean = test(
            cfg,
            load_epoch,
            unet,
            test_dataloader,
            detection_txt_folder,
            craft_net,
            zip_path,
            optimizer,
            scheduler,
            highest_hmean,
            lpips_model,
            writer,
        )
        if tmp_highest_hmean != -1:
            highest_hmean = tmp_highest_hmean

    with tqdm(
        range(start_epoch, cfg.epochs), desc="All Epochs", unit="epoch"
    ) as tqdm_all:
        for epoch in tqdm_all:
            avg_edge_loss = 0.0
            avg_mae_loss = 0.0
            avg_ms_ssim_loss = 0.0
            avg_text_loss = 0.0
            avg_all_loss = 0.0

            epoch_time = time.perf_counter()

            if os.path.isdir(cfg.log_dir + "%04d" % epoch):
                continue

            edge_loss_list = []
            mae_loss_list = []
            ms_ssim_loss_list = []
            text_loss_list = []
            all_loss_list = []

            # Training
            unet.train()

            with tqdm(train_dataloader, unit="batch") as tqdm_loader:
                for sample in tqdm_loader:
                    dt_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    tqdm_loader.set_description(
                        f"[{dt_string}] Epoch [{epoch}/{cfg.epochs}]"
                    )

                    sample_time = time.perf_counter()
                    in_imgs = sample["in_img"].cuda()
                    gt_imgs = sample["gt_img"].cuda()
                    in_edge_imgs = sample["in_edge"].cuda()
                    gt_edge_imgs = sample["gt_edge"].cuda()

                    if cfg.multi_scale_patch:
                        new_patch_size = cfg.patch_size + (
                            32 * torch.randint(0, 4, (1,))[0].item()
                        )
                        resize_transform = T.Resize(
                            size=(new_patch_size, new_patch_size)
                        )
                        in_imgs = resize_transform(in_imgs)
                        gt_imgs = resize_transform(gt_imgs)
                        in_edge_imgs = resize_transform(in_edge_imgs)
                        gt_edge_imgs = resize_transform(gt_edge_imgs)

                    optimizer.zero_grad()

                    unet_time = time.perf_counter()
                    if cfg.unet_type == "cc_unet":
                        out_imgs = unet(in_imgs, in_edge_imgs)
                    elif cfg.unet_type == "howard_unet":
                        in_gray_imgs = sample["in_gray"].cuda()
                        out_imgs = unet(in_imgs, in_gray_imgs, in_edge_imgs)
                    elif (
                        cfg.unet_type == "plain_unet"
                        or cfg.unet_type == "att_plain_unet"
                    ):
                        out_imgs = unet(in_imgs)
                    elif (
                        cfg.unet_type == "cc_unet_nedge"
                        or cfg.unet_type == "cc_unet_nedge_v2"
                    ):
                        out_imgs, out_pms = unet(in_imgs, in_edge_imgs)
                    else:
                        raise ValueError(f"Invalid unet_type: {cfg.unet_type}")
                    unet_end_time = time.perf_counter() - unet_time

                    loss_time = time.perf_counter()
                    if (
                        cfg.unet_type == "cc_unet_nedge"
                        or cfg.unet_type == "cc_unet_nedge_v2"
                    ):
                        edge_loss = utils.EdgeBCELoss(out_pms, gt_edge_imgs)
                    else:
                        if cfg.use_rcf:
                            if cfg.batch_size > 1:
                                edge_loss = utils.RCFEdgeLoss_loop(
                                    out_imgs, gt_edge_imgs, rcf_net, device
                                )
                            else:
                                edge_loss = utils.RCFEdgeLoss(
                                    out_imgs, gt_edge_imgs, rcf_net, device
                                )
                        else:
                            edge_loss = torch.Tensor([0]).cuda()

                    if cfg.use_smooth_l1:
                        mae_loss = utils.Smooth_L1_Loss(
                            out_imgs, gt_imgs, device, smooth_l1_beta=cfg.smooth_l1_beta
                        )
                    else:
                        mae_loss = utils.L1_Loss(out_imgs, gt_imgs, device)

                    ms_ssim_loss = utils.MS_SSIMLoss(out_imgs, gt_imgs)
                    text_loss = utils.TextDetectionLoss(
                        out_imgs, gt_imgs, craft_net, device
                    )
                    loss = (
                        cfg.mae_loss_w * mae_loss
                        + cfg.ms_ssim_loss_w * ms_ssim_loss
                        + cfg.text_loss_w * text_loss
                        + cfg.edge_loss_w * edge_loss
                    )
                    loss_end_time = time.perf_counter() - loss_time

                    bp_time = time.perf_counter()
                    loss.backward()
                    optimizer.step()
                    bp_end_time = time.perf_counter() - bp_time

                    # loss for the entire epoch
                    edge_loss_list.append(edge_loss.item())
                    mae_loss_list.append(mae_loss.item())
                    ms_ssim_loss_list.append(ms_ssim_loss.item())
                    text_loss_list.append(text_loss.item())
                    all_loss_list.append(loss.item())

                    sample_end_time = time.perf_counter() - sample_time
                    log_str = (
                        "All_Loss=%.3f, UNET_Time=%.3f, LOSS_Time=%.3f, BP_Time=%.3f, Total_Time=%.3f"
                        % (
                            np.mean(all_loss_list),
                            unet_end_time,
                            loss_end_time,
                            bp_end_time,
                            sample_end_time,
                        )
                    )
                    if cfg.multi_scale_patch:
                        log_str = f"P_Size={new_patch_size}, {log_str}"

                    tqdm_loader.postfix = log_str

            per_epoch_time = time.perf_counter() - epoch_time
            # logger.info("------per_epoch_time=%.3f" % (per_epoch_time))

            avg_edge_loss = np.mean(edge_loss_list)
            avg_mae_loss = np.mean(mae_loss_list)
            avg_ms_ssim_loss = np.mean(ms_ssim_loss_list)
            avg_text_loss = np.mean(text_loss_list)
            avg_all_loss = np.mean(all_loss_list)

            writer.add_scalar("Train/Edge_Loss", avg_edge_loss, epoch)
            writer.add_scalar("Train/MAE_Loss", avg_mae_loss, epoch)
            writer.add_scalar("Train/MS_SSIM_Loss", avg_ms_ssim_loss, epoch)
            writer.add_scalar("Train/Text_Loss", avg_text_loss, epoch)
            writer.add_scalar("Train/All_Loss", avg_all_loss, epoch)

            if cfg.use_sgd:
                writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
            else:
                writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], epoch)

            if cfg.use_wandb:
                wandb.log({"Train/Edge_Loss": avg_edge_loss}, step=epoch)
                wandb.log({"Train/MAE_Loss": avg_mae_loss}, step=epoch)
                wandb.log({"Train/MS_SSIM_Loss": avg_ms_ssim_loss}, step=epoch)
                wandb.log({"Train/Text_Loss": avg_text_loss}, step=epoch)
                wandb.log({"Train/All_Loss": avg_all_loss}, step=epoch)

                if cfg.use_sgd:
                    wandb.log({"Train/LR": optimizer.param_groups[0]["lr"]}, step=epoch)
                else:
                    wandb.log({"Train/LR": scheduler.get_last_lr()[0]}, step=epoch)

            epoch_log_str = (
                "Edge_Loss=%.3f, MAE_Loss=%.3f, MS_SSIM_Loss=%.3f, Text_Loss=%.3f, All_Loss=%.3f, Epoch_Time=%.3f"
                % (
                    avg_edge_loss,
                    avg_mae_loss,
                    avg_ms_ssim_loss,
                    avg_text_loss,
                    avg_all_loss,
                    per_epoch_time,
                )
            )
            tqdm_all.postfix = epoch_log_str

            # save the current model using
            # different name for reproducing the best results
            if epoch % cfg.model_save_freq == 0:
                utils.save_checkpoint_state(
                    cfg.use_dp,
                    cfg.log_dir + "{}.pt".format(epoch),
                    epoch,
                    unet,
                    optimizer,
                    scheduler,
                )

            if epoch % cfg.test_freq == 0:
                tmp_highest_hmean = test(
                    cfg,
                    epoch,
                    unet,
                    test_dataloader,
                    detection_txt_folder,
                    craft_net,
                    zip_path,
                    optimizer,
                    scheduler,
                    highest_hmean,
                    lpips_model,
                    writer,
                )
                if tmp_highest_hmean != -1:
                    highest_hmean = tmp_highest_hmean

            if cfg.use_sgd:
                scheduler.step(avg_all_loss)
            else:
                scheduler.step()

    writer.close()
    if cfg.use_wandb:
        wandb.finish()


def get_train_set(cfg, cap_paths=None):
    if cfg.train_dataset_type == "sony":
        from dataset.sony import SonyTrainSet as TrainSet
    elif cfg.train_dataset_type == "fuji":
        from dataset.fuji import FujiTrainSet as TrainSet
    elif cfg.train_dataset_type == "lol":
        from dataset.lol import LOLTrainSet as TrainSet
    elif cfg.train_dataset_type == "icdar15":
        from dataset.icdar15 import IC15TrainSet as TrainSet
    else:
        raise ValueError(f"Invalid train_dataset_type: {cfg.train_dataset_type}")

    if cfg.use_cap:
        if cfg.cap_version == 1:
            cap_paths = {
                "gt_cropped_dir": cfg.gt_cropped_dir,
                "input_cropped_dir": cfg.input_cropped_dir,
                "edge_cropped_dir": cfg.edge_cropped_dir,
                "gt_edge_cropped_dir": cfg.gt_edge_cropped_dir,
            }
        elif cfg.cap_version == 2:
            if cap_paths is None:
                cap_paths = {
                    "gt_img_root": cfg.train_gt_dir,
                    "gt_txt_path": cfg.gt_txt_path,
                    "low_img_root": cfg.train_input_dir,
                    "edge_img_root": cfg.train_edge_dir,
                    "gt_edge_img_root": cfg.train_gt_edge_dir,
                }

        if "use_naive" not in cfg.keys():
            cfg["use_naive"] = False

        if cfg.use_naive:
            logger.info("------Please use naive cap version 2 for ablation study only!")

        if cfg.train_dataset_type == "sony":
            train_dataset = TrainSet(
                patch_size=cfg.patch_size,
                list_file=cfg.train_list_file,
                input_img_dir=cfg.train_input_dir,
                gt_img_dir=cfg.train_gt_dir,
                edge_dir=cfg.train_edge_dir,
                gt_edge_dir=cfg.train_gt_edge_dir,
                use_cap=cfg.use_cap,
                cap_version=cfg.cap_version,
                cap_paths=cap_paths,
                ratio_multiplier=cfg.train_ratio_multiplier,
                input_use_canny=cfg.input_use_canny,
                gt_use_canny=cfg.gt_use_canny,
                dataset_type=cfg.train_dataset_type,
                use_naive=cfg.use_naive,
            )
        elif cfg.train_dataset_type == "lol":
            train_dataset = TrainSet(
                patch_size=cfg.patch_size,
                list_file=cfg.train_list_file,
                input_img_dir=cfg.train_input_dir,
                gt_img_dir=cfg.train_gt_dir,
                edge_dir=cfg.train_edge_dir,
                gt_edge_dir=cfg.train_gt_edge_dir,
                use_cap=cfg.use_cap,
                cap_version=cfg.cap_version,
                cap_paths=cap_paths,
                input_use_canny=cfg.input_use_canny,
                gt_use_canny=cfg.gt_use_canny,
                dataset_type=cfg.train_dataset_type,
            )
        elif cfg.train_dataset_type == "fuji":
            train_dataset = TrainSet(
                patch_size=cfg.patch_size,
                list_file=cfg.train_list_file,
                input_img_dir=cfg.train_input_dir,
                gt_img_dir=cfg.train_gt_dir,
                edge_dir=cfg.train_edge_dir,
                gt_edge_dir=cfg.train_gt_edge_dir,
                use_cap=cfg.use_cap,
                cap_version=cfg.cap_version,
                cap_paths=cap_paths,
                ratio_multiplier=cfg.train_ratio_multiplier,
                input_use_canny=cfg.input_use_canny,
                gt_use_canny=cfg.gt_use_canny,
                dataset_type=cfg.train_dataset_type,
                use_naive=cfg.use_naive,
            )

        logger.info(
            f"Using copy and paste augmentation version {cfg.cap_version} for {cfg.train_dataset_type} dataset"
        )
    else:
        if cfg.train_dataset_type == "lol":
            train_dataset = TrainSet(
                patch_size=cfg.patch_size,
                list_file=cfg.train_list_file,
                input_img_dir=cfg.train_input_dir,
                gt_img_dir=cfg.train_gt_dir,
                edge_dir=cfg.train_edge_dir,
                gt_edge_dir=cfg.train_gt_edge_dir,
                input_use_canny=cfg.input_use_canny,
                gt_use_canny=cfg.gt_use_canny,
                dataset_type=cfg.train_dataset_type,
            )
        elif cfg.train_dataset_type == "sony" or cfg.train_dataset_type == "icdar15":
            train_dataset = TrainSet(
                patch_size=cfg.patch_size,
                list_file=cfg.train_list_file,
                input_img_dir=cfg.train_input_dir,
                gt_img_dir=cfg.train_gt_dir,
                edge_dir=cfg.train_edge_dir,
                gt_edge_dir=cfg.train_gt_edge_dir,
                ratio_multiplier=cfg.train_ratio_multiplier,
                input_use_canny=cfg.input_use_canny,
                gt_use_canny=cfg.gt_use_canny,
                dataset_type=cfg.train_dataset_type,
            )

    return train_dataset


def test(
    cfg,
    epoch,
    unet,
    test_dataloader,
    detection_txt_folder,
    craft_net,
    zip_path,
    optimizer,
    scheduler,
    highest_hmean,
    lpips_model,
    writer,
):
    if cfg.gen_image:
        testing_img_folder = os.path.join(cfg.log_dir, "out_img")
        if not os.path.isdir(testing_img_folder):
            os.makedirs(testing_img_folder)

    with torch.no_grad():
        logger.info(f"\n------Evaluating epoch {epoch}.")
        eval_time = time.perf_counter()
        return_highest_hmean = False

        unet.eval()
        psnr_list, ssim_list, lpips_list = [], [], []

        with tqdm(test_dataloader, unit="batch") as tqdm_loader:
            for sample in tqdm_loader:
                dt_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                tqdm_loader.set_description(f"[{dt_string}] Eval Epoch [{epoch}]")

                eval_sample_time = time.perf_counter()

                input_img_name = sample["input_img_name"]  # input filename
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

        if hmean > highest_hmean:
            highest_hmean = hmean
            return_highest_hmean = True
            # save the best model using the same name for resuming
            utils.save_checkpoint_state(
                cfg.use_dp,
                os.path.join(cfg.log_dir, "best_hmean.pt"),
                epoch,
                unet,
                optimizer,
                scheduler,
            )
            logger.info(
                f"------Saved CKPT with best hmean of {hmean} at epoch {epoch}."
            )

        per_eval_time = time.perf_counter() - eval_time

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list)
        logger.info(
            "------HMEAN_Time=%.3f per_eval_time=%.3f"
            % (eval_hmean_end_time, per_eval_time)
        )
        logger.info("------hmean={}, the highest_hmean={}".format(hmean, highest_hmean))
        logger.info(
            "------PSNR={}, SSIM={}, LPIPS={}".format(avg_psnr, avg_ssim, avg_lpips)
        )

        writer.add_scalar("Test/psnr", avg_psnr, epoch)
        writer.add_scalar("Test/ssim", avg_ssim, epoch)
        writer.add_scalar("Test/lpips", avg_lpips, epoch)
        writer.add_scalar("Test/hmean", hmean, epoch)

        if cfg.use_wandb:
            wandb.log({"Test/psnr": avg_psnr}, step=epoch)
            wandb.log({"Test/ssim": avg_ssim}, step=epoch)
            wandb.log({"Test/lpips": avg_lpips}, step=epoch)
            wandb.log({"Test/hmean": hmean}, step=epoch)

        if return_highest_hmean:
            return highest_hmean
        else:
            return -1


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
        "--workers", type=int, default=8, help="maximum number of dataloader workers"
    )
    parser.add_argument("--epochs", type=int, default=500, help="total training epochs")
    parser.add_argument(
        "--model-save-freq", type=int, default=100, help="frequency of model saving"
    )
    parser.add_argument(
        "--test-freq", type=int, default=500, help="frequency of model testing"
    )
    parser.add_argument("--gen-image", action="store_true", help="to output image")
    parser.add_argument("--use-dp", action="store_true", help="to use DataParallel")
    parser.add_argument("--use-bn", action="store_true", help="to use BatchNorm")
    parser.add_argument(
        "--use-wandb", action="store_true", help="to use wandb for logging"
    )
    parser.add_argument(
        "--wandb-project", default="results", help="project name for wandb"
    )
    parser.add_argument("--wandb-name", default="exp", help="run name for wandb")
    parser.add_argument(
        "--wandb-id", type=str, default="-1", help="unique wandb id for resume"
    )
    parser.add_argument("--project", default="results", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
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
        "--use-last-edge",
        action="store_true",
        help="whether to use edge decoder at last stage for cc_unet_nedge_v2",
    )
    parser.add_argument(
        "--last-edge-att",
        action="store_true",
        help="whether to use multiple the edge map back at last stage for cc_unet_nedge_v2",
    )
    parser.add_argument(
        "--use-sgd", action="store_true", help="whether to use sgd or adam"
    )
    parser.add_argument(
        "--multi-scale-patch",
        action="store_true",
        help="whether to use multi scale patch",
    )
    parser.add_argument(
        "--use-cap",
        action="store_true",
        help="whether to use copy and paste augmentation",
    )
    parser.add_argument(
        "--cap-version",
        type=int,
        default=1,
        help="version of copy and paste augmentation",
    )
    parser.add_argument(
        "--eval-when-resume",
        action="store_true",
        help="whether to eval at the resume epoch",
    )
    parser.add_argument(
        "--train-ratio-multiplier",
        type=float,
        default=0.0,
        help="the exposure ratio multiplier to be used.",
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
        "--use-smooth-l1",
        action="store_true",
        default=False,
        help="whether to use smooth_l1_loss",
    )
    parser.add_argument(
        "--smooth-l1-beta", type=float, default=1.0, help="beta for smooth_l1_loss"
    )
    parser.add_argument(
        "--use-rcf",
        action="store_true",
        default=False,
        help="whether to use rcf for edge_loss",
    )
    parser.add_argument(
        "--reinit-opt", action="store_true", default=False, help="whether to reinit opt"
    )
    parser.add_argument(
        "--concat-input",
        action="store_true",
        default=False,
        help="whether to concat input for training",
    )
    opt = parser.parse_args()
    args_dict = vars(opt)

    # do remember that parse_args will override values in cfg
    cfg = utils.load_config(opt.cfg)
    cfg.update(args_dict)
    cfg.log_dir = f"{opt.project}/{opt.name}/"

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

    # tensorboard log
    writer = SummaryWriter(log_dir=cfg.log_dir + "logs")

    train(cfg, writer)
