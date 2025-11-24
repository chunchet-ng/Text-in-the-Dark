import os
import os.path as osp

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models import RCF


def multi_scale_test(model, image):
    scale = [0.5, 1, 1.5]
    in_ = image.transpose((1, 2, 0))
    _, H, W = image.shape
    ms_fuse = np.zeros((H, W), np.float32)
    for k in range(len(scale)):
        im_ = cv2.resize(
            in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR
        )
        im_ = im_.transpose((2, 0, 1))
        im_cuda = torch.unsqueeze(torch.from_numpy(im_).cuda(), 0)
        results = model(im_cuda)
        fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
        fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
        ms_fuse += fuse_res
        del results, im_cuda
    ms_fuse = ms_fuse / len(scale)
    ms_fuse = ((ms_fuse) * 255).astype(np.uint8)
    return ms_fuse


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ckpt_path = "./rcf/bsds500_pascal_model.pth"
    splits = ["train", "test"]

    with torch.no_grad():
        model = RCF()
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)
        model = model.cuda()
        model.eval()
        for split in splits:
            in_dirs = [
                f"../dataset/Sony/{split}/short",
                f"../dataset/Sony/{split}/long",
            ]
            save_dirs = [
                f"../dataset/Sony/{split}/cc_edge",
                f"../dataset/Sony/{split}/cc_edge_gt",
            ]
            for in_dir, save_dir in zip(in_dirs, save_dirs):
                if not osp.isdir(save_dir):
                    os.makedirs(save_dir)

                img_list = os.listdir(in_dir)
                mean = np.array(
                    [104.00698793, 116.66876762, 122.67891434], dtype=np.float32
                )

                for idx, filename in enumerate(tqdm(img_list)):
                    print(f"[{idx + 1}/{len(img_list)}] Processing {filename}.")
                    full_path = osp.join(in_dir, filename)
                    image = cv2.imread(full_path)
                    im_w, im_h = image.shape[1], image.shape[0]
                    image = cv2.resize(
                        image, (3200, 2144), interpolation=cv2.INTER_LINEAR
                    )
                    image = np.array(image, dtype=np.float32)
                    image = (image - mean).transpose((2, 0, 1))
                    fuse_res = multi_scale_test(model, image)
                    fuse_res = cv2.resize(
                        fuse_res, (im_w, im_h), interpolation=cv2.INTER_LINEAR
                    )
                    cv2.imwrite(osp.join(save_dir, filename), fuse_res)
