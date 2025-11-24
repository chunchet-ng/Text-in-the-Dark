import math
import os
import random
import uuid
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from easydict import EasyDict as edict
from pytorch_msssim import ms_ssim, ssim

from cc_eval.tiou import rrc_evaluation_funcs as tiou
from cc_eval.tiou import script as tiou_script


def copyStateDict(load_parallel, state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        if load_parallel:
            name = "module." + name
        new_state_dict[name] = v
    return new_state_dict


def save_checkpoint_state(ddp, path, epoch, model, optimizer, scheduler):
    if ddp:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
    else:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
    torch.save(checkpoint, path)
    del checkpoint


def load_checkpoint_state(
    load_parallel, mode, path, device, model, optimizer=None, scheduler=None
):
    checkpoint = torch.load(path, map_location=device)
    epoch = checkpoint["epoch"]
    model_state_dict = copyStateDict(load_parallel, checkpoint["model_state_dict"])
    model.load_state_dict(model_state_dict)
    if mode == "train":
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return epoch, model, optimizer, scheduler
    else:
        return epoch, model


def load_config(fileName):
    config = None
    with open(fileName, "r") as f:
        config = edict(yaml.safe_load(f))
        for key, val in config.items():
            if isinstance(val, str):
                if "project_path" in val:
                    config[key] = val.replace("$(project_path)", config.project_path)
                if "data_path" in val:
                    config[key] = val.replace("$(data_path)", config.data_path)
    return config


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def SSIM(out_image, gt_image):
    out_image = torch.clamp(out_image, min=0.0, max=1.0)
    return ssim(out_image, gt_image, data_range=1, size_average=True)


def PSNR(out_image, gt_image):
    out_image = torch.clamp(out_image, min=0.0, max=1.0)
    mse = torch.mean((out_image - gt_image) ** 2)
    return 10 * torch.log10(1.0 / mse)


def L1_Loss(out_image, gt_image, device):
    assert out_image.shape == gt_image.shape, (
        f"out shape: {out_image.shape}, gt shape: {gt_image.shape}"
    )
    return torch.nn.functional.l1_loss(out_image.to(device), gt_image.to(device))


def Smooth_L1_Loss(out_image, gt_image, device, smooth_l1_beta=1.0):
    assert out_image.shape == gt_image.shape, (
        f"out shape: {out_image.shape}, gt shape: {gt_image.shape}"
    )
    return torch.nn.functional.smooth_l1_loss(
        out_image.to(device), gt_image.to(device), beta=smooth_l1_beta
    )


def MS_SSIMLoss(out_image, gt_image):
    return 1 - ms_ssim(out_image, gt_image, data_range=1, size_average=True)


def EdgeBCELoss(out_images, gt_images):
    costs = torch.zeros([gt_images.shape[0]]).cuda()
    gt_masks = []
    for gt_image in gt_images:
        mask = gt_image.clone()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        gt_masks.append(mask)

    for out_image_split in out_images:
        count = 0
        for out, gt_mask in zip(out_image_split, gt_masks):
            costs[count] += torch.nn.functional.binary_cross_entropy(
                out.cuda(), gt_images[count].cuda(), weight=gt_mask
            )
            count += 1
    return torch.mean(costs)


def Tensor2OpenCV(img):
    # BxCxHxW (B=1) to HxWxC
    if img.size(dim=0) == 1:
        img = img.squeeze(0)
    img = img.permute(1, 2, 0).cpu().data.numpy()
    img = np.minimum(np.maximum(img, 0), 1)
    img = img * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def BinaryTensor2OpenCV(img):
    # BxCxHxW (B=1) to HxWxC
    if img.size(dim=0) == 1:
        img = img.squeeze(0)
    img = img.cpu().data.numpy()
    img = np.minimum(np.maximum(img, 0), 1)
    img = img * 255

    return img


def nparray2OpenCV(img):
    # CxHxW to HxWxC
    img = img.transpose(1, 2, 0)
    img = np.minimum(np.maximum(img, 0), 1)
    img = np.uint8(img * 255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


"""
def Tensor2OpenCV2Tensor(img):
    img = img.squeeze(0)
    img = img.permute(1, 2, 0).cpu().data.numpy()
    img = np.minimum(np.maximum(img, 0), 1)
    img = np.uint8(img * 255)
    img = np.expand_dims(np.float32(img) / 255.0), axis=0)
    img_tensor = torch.from_numpy(img).permute(0,3,1,2)
    return img_tensor
"""


def Tensor2CPU(img):
    # BxCxHxW (B=1) to CxHxW
    img = img.squeeze(0)
    img = img.cpu().data.numpy()
    img = np.minimum(np.maximum(img, 0), 1)

    return img


def Tensors2OpenCV(imgs):
    # BxCxHxW to HxWxC
    imgs_list = []
    for img in imgs:
        img = img.permute(1, 2, 0).cpu().data.numpy()
        img = np.minimum(np.maximum(img, 0), 1)
        img = img * 255

        imgs_list.append(img)

    return imgs_list


def OpenCV2fourDTensor(img):
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = np.expand_dims(np.float32(input_img / 255.0), axis=0)
    img_tensor = torch.from_numpy(input_img).permute(0, 3, 1, 2)

    return img_tensor


def OpenCV2ThreeTensor(img):
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = np.float32(input_img / 255.0)
    img_tensor = torch.from_numpy(input_img).permute(2, 0, 1)

    return img_tensor


def Tensor2OpenCVnWrite(img, filename, final_ratio_w, final_ratio_h):
    # BxCxHxW (B=1) to HxWxC
    img = img.squeeze(0)
    img = img.permute(1, 2, 0).cpu().data.numpy()
    img = np.minimum(np.maximum(img, 0), 1)
    img = img * 255

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = np.uint8(img)

    # resize to the original size
    H = img.shape[0]
    W = img.shape[1]

    dim = (int(W / final_ratio_w), int(H / final_ratio_h))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite(filename, img)


def normalizeMeanVarianceTensor_01(in_img):
    img = TF.normalize(in_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return img


def normalizeMeanVarianceTensor(in_img):
    img = TF.normalize(
        in_img,
        mean=(0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0),
        std=(0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0),
    )
    return img


def edge_normalizeMeanVarianceTensor(in_img):
    img = TF.normalize(
        in_img, mean=(0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0), std=(1, 1, 1)
    )
    return img


def OpenCV2CHW(img):
    # HxWxC to CxHxW
    img = np.transpose(img, (2, 0, 1)).astype(np.uint8)

    return img


def DrawOpenCVImage(img, boxes):
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))

        poly = poly.reshape(-1, 2)
        cv2.polylines(
            img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2
        )

    return img


def warpCoord(Minv, pt):
    # unwarp corodinates
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape
    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex = x - niter, x + w + niter + 1
        sy, ey = y - niter, y + h + niter + 1
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = (
            np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
            .transpose()
            .reshape(-1, 2)
        )
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    boxes, labels, mapper = getDetBoxes_core(
        textmap, linkmap, text_threshold, link_threshold, low_text
    )

    return boxes


def TextDetectionLoss(out_image, gt_image, net, device):
    # out_image:  BxCxHxW
    out_image = torch.clamp(out_image, min=0, max=1)
    out_image = out_image * 255
    out_image = normalizeMeanVarianceTensor(out_image)

    gt_image = torch.clamp(gt_image, min=0, max=1)
    gt_image = gt_image * 255
    gt_image = normalizeMeanVarianceTensor(gt_image)

    out_pred, _ = net(out_image)
    gt_pred, _ = net(gt_image)

    out_text = out_pred[:, :, :, 0]
    gt_text = gt_pred[:, :, :, 0]

    return L1_Loss(out_text, gt_text, device)


def RCFEdgeLoss(out_image, gt_image, rcf_net, device):
    # out_image:  BxCxHxW
    out_image = torch.clamp(out_image, min=0, max=1)
    out_image = out_image * 255
    out_image = edge_normalizeMeanVarianceTensor(out_image)
    out_image = torch.squeeze(out_image, 0).detach().cpu().numpy()
    scale = [0.5, 1, 1.5]
    in_ = out_image.transpose((1, 2, 0))
    _, H, W = out_image.shape
    ms_fuse = np.zeros((H, W), np.float32)
    for k in range(len(scale)):
        im_ = cv2.resize(
            in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR
        )
        im_ = im_.transpose((2, 0, 1))
        im_cuda = torch.unsqueeze(torch.from_numpy(im_).to(device), 0)
        results = rcf_net(im_cuda)
        fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
        fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
        ms_fuse += fuse_res
        del results, im_cuda
    ms_fuse = ms_fuse / len(scale)

    gt_image = torch.clamp(gt_image, min=0, max=1).squeeze(0).squeeze(0)
    input_image = torch.from_numpy(ms_fuse).to(device)
    edge_loss = L1_Loss(input_image, gt_image, device)

    return edge_loss


def RCFEdgeLoss_loop(out_images, gt_images, rcf_net, device):
    # out_image:  BxCxHxW
    out_images = torch.clamp(out_images, min=0, max=1)
    out_images = out_images * 255
    out_images = edge_normalizeMeanVarianceTensor(out_images)

    edge_loss = 0
    for gt_image, out_image in zip(gt_images, out_images):
        out_image = torch.squeeze(out_image, 0).detach().cpu().numpy()
        scale = [0.5, 1, 1.5]
        in_ = out_image.transpose((1, 2, 0))
        _, H, W = out_image.shape
        ms_fuse = np.zeros((H, W), np.float32)
        for k in range(len(scale)):
            im_ = cv2.resize(
                in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR
            )
            im_ = im_.transpose((2, 0, 1))
            im_cuda = torch.unsqueeze(torch.from_numpy(im_).to(device), 0)
            results = rcf_net(im_cuda)
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse_res
            del results, im_cuda
        ms_fuse = ms_fuse / len(scale)

        gt_image = torch.clamp(gt_image, min=0, max=1).squeeze(0).squeeze(0).to(device)
        input_image = torch.from_numpy(ms_fuse).to(device)
        edge_loss += L1_Loss(input_image, gt_image, device)

    return edge_loss / out_images.shape[0]


def RCFEdgeLoss_multi(out_image, gt_image, rcf_net):
    # out_image:  BxCxHxW
    out_image = torch.clamp(out_image, min=0, max=1)
    out_image = out_image * 255
    out_image = edge_normalizeMeanVarianceTensor(out_image)
    _, _, H, W = (
        out_image.shape[0],
        out_image.shape[1],
        out_image.shape[2],
        out_image.shape[3],
    )

    # repeat for 3 scales [0.5, 1, 1.5]
    repeated = out_image.repeat_interleave(3, dim=1)

    scale_h_1, scale_w_1 = int(H * 0.5), int(W * 0.5)
    scale_h_2, scale_w_2 = int(H * 1.5), int(W * 1.5)
    resize_1 = TF.resize(repeated[:, 0::3, :, :], (scale_h_1, scale_w_1))
    resize_2 = TF.resize(repeated[:, 2::3, :, :], (scale_h_2, scale_w_2))
    ori = repeated[:, 1::3, :, :]

    out_resize_1 = TF.resize(rcf_net(resize_1)[-1], (H, W))
    out_resize_2 = TF.resize(rcf_net(resize_2)[-1], (H, W))
    out_ori = rcf_net(ori)[-1]

    assert out_resize_1.shape == out_resize_2.shape == out_ori.shape, (
        "Invalid output resized shapes at RCF."
    )

    final_out = torch.div(out_resize_1 + out_resize_2 + out_ori, 3)

    assert final_out.shape == gt_image.shape, "Invalid output and gt shapes at RCF."

    gt_image = torch.clamp(gt_image, min=0, max=1)
    edge_loss = L1_Loss(final_out, gt_image)

    del repeated
    del resize_1
    del resize_2
    del ori
    del out_resize_1
    del out_resize_2
    del out_ori
    del final_out

    return edge_loss


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)

        if isinstance(ratio_w, torch.Tensor):
            ratio_w = ratio_w.item()

        if isinstance(ratio_h, torch.Tensor):
            ratio_h = ratio_h.item()

        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)

    return polys


def TextDetectionResult(input_image, net, text_threshold, link_threshold, low_text):
    out_image = torch.clamp(input_image, min=0, max=1)
    out_image = out_image * 255.0
    out_image = normalizeMeanVarianceTensor(out_image)

    with torch.no_grad():
        out_pred, _ = net(out_image)

    score_text = out_pred[0, :, :, 0].cpu().data.numpy()
    score_link = out_pred[0, :, :, 1].cpu().data.numpy()

    boxes = getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text
    )

    return boxes, score_text


def TextDetectionVisualize(
    out_imgs,
    craft_net,
    text_threshold,
    link_threshold,
    low_text,
    final_ratio_w=None,
    final_ratio_h=None,
):
    # craft text detection: out_imgs should be BxCxHxW
    boxes, score_text = TextDetectionResult(
        out_imgs, craft_net, text_threshold, link_threshold, low_text
    )

    final_ratio_w = 1
    final_ratio_h = 1

    boxes = adjustResultCoordinates(boxes, final_ratio_w, final_ratio_h)

    # from BxCxHxW (0-1) tensor to HxWxC (0-255) numpy array
    out_imgs = Tensor2OpenCV(out_imgs)

    # RGB2BGR
    # out_imgs = cv2.cvtColor(out_imgs, cv2.COLOR_RGB2BGR)

    # rescale according to final_ratio_w*W, final_ratio_h*H
    H = out_imgs.shape[0]
    W = out_imgs.shape[1]
    dim = (int(final_ratio_w * W), int(final_ratio_h * H))
    out_imgs = cv2.resize(out_imgs, dim, interpolation=cv2.INTER_AREA)

    # overlapping the boxes on out_imgs
    out_imgs = DrawOpenCVImage(out_imgs, boxes)

    # BGR2RGB for out_imgs
    out_imgs = cv2.cvtColor(out_imgs, cv2.COLOR_BGR2RGB)

    # numpy array to CHW numpy array
    out_imgs = OpenCV2CHW(out_imgs)

    # score map and affinity map generation
    C, H, W = out_imgs.shape
    dim = (int(W), int(H))

    # the shape of score_text is HxW
    score_text = cv2.resize(score_text, dim, interpolation=cv2.INTER_AREA)

    score_text_heatmap = cvt2HeatmapImg(score_text)

    # BGR2RGB for score_text
    score_text_heatmap = cv2.cvtColor(score_text_heatmap, cv2.COLOR_BGR2RGB)

    # back to CxHxW
    score_text_heatmap = np.transpose(score_text_heatmap, (2, 0, 1))

    return out_imgs, score_text_heatmap


def TextDetection(
    out_imgs,
    input_filename,
    output_folder,
    file_prefix,
    craft_net,
    text_threshold,
    link_threshold,
    low_text,
    final_ratio_w=None,
    final_ratio_h=None,
):
    # craft text detection: out_imgs should be BxCxHxW
    boxes, score_text = TextDetectionResult(
        out_imgs, craft_net, text_threshold, link_threshold, low_text
    )

    # boxes will be multiplied by 2
    if final_ratio_w is None:
        final_ratio_w = 1
    if final_ratio_h is None:
        final_ratio_h = 1

    boxes = adjustResultCoordinates(boxes, 1 / final_ratio_w, 1 / final_ratio_h)
    txt_filename = os.path.join(output_folder, file_prefix + input_filename[0] + ".txt")

    with open(txt_filename, "w") as f:
        for i, box in enumerate(boxes):
            # outF = open(txt_filename, "w")
            concatenated_string = ""
            box = np.array(box).astype(np.int32).reshape((-1))
            concatenated_string = ",".join([str(p) for p in box]) + "\n"
            f.write(concatenated_string)


def eval(eval_dataset_type, gt_path, res_path, output_path, per_sample_result=True):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    params = {
        "flag": eval_dataset_type,
        "o": output_path,  # result in json format
        "s": res_path,
        "g": gt_path,
    }

    tiou_default_params = tiou_script.default_evaluation_params()
    if per_sample_result:
        tiou_default_params["PER_SAMPLE_RESULTS"] = True

    if eval_dataset_type == "icdar15":
        resDict = tiou.main_evaluation(
            params,
            tiou_default_params,
            tiou_script.validate_data,
            tiou_script.evaluate_method,
        )

    return resDict


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)
