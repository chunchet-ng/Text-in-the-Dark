import os

from easydict import EasyDict as edict

from cc_eval.img_pure_eval_hmean import main as hmean_eval


def cc_hmean(
    dataset_type,
    img_path,
    experiments_folder,
    epoch,
    per_img_result,
    manual_craft_sel=None,
    use_iqa=False,
):
    """_summary_

    Args:
        dataset_type (str): type of dataset. Must be one of [icdar15, lol, sony, fuji].
        img_path (str): Path to input/predicted images folder.
        experiments_folder (str): Path to experiments folder to store craft output.
        epoch (str): Current epoch number, for logging purposes only.
        per_img_result (bool): Whether to print per image result for scikit-image evaluation.
        manual_craft_sel (str): Manually override pretrained craft model type.
        use_iqa (bool): Whether to report IQA-PyTorch's metrics.

    Log folder structure:
    {experiments_folder}/cc_eval/epoch_{epoch}/craft
        |- hmean_output/
        |- log
        |- txts
        |_ zips

    Returns:
        out_dict (dict): {
            'ori_psnr': cc_psnr,
            'ori_ssim': cc_ssim,
            'hmean': hmean,
            'skim_psnr': skim_psnr,
            'skim_ssim': skim_ssim,
            'lpips': lpips,
            }
    """

    # define variables
    cfg = edict()
    cfg.img_path = img_path
    cfg.dataset_type = dataset_type.lower()
    cfg.gen_image = False
    cfg.del_out = False
    cfg.epoch = epoch
    cfg.per_img_result = per_img_result
    cfg.use_iqa = use_iqa

    assert dataset_type.lower() in ["icdar15", "lol", "sony", "fuji"], (
        f"dataset_type must be one of \
        [icdar15, lol, sony, fuji], but recevied {cfg.dataset_type}"
    )
    assert not os.path.isfile(experiments_folder), (
        f"experiments_folder must be a folder. \
        Please recheck at path {experiments_folder}"
    )

    cfg.manual_craft_sel = manual_craft_sel
    if cfg.manual_craft_sel is not None:
        assert cfg.manual_craft_sel.lower() in ["icdar15", "mlt"], (
            f"manual_craft_sel must be one of \
        [icdar15, mlt], but recevied {cfg.manual_craft_sel}"
        )

    experiments_folder = os.path.join(experiments_folder, f"cc_eval/epoch_{epoch}")
    if not os.path.exists(os.path.join(experiments_folder, "craft/")):
        os.makedirs(os.path.join(experiments_folder, "craft/"))

    cfg.zip_path = os.path.join(experiments_folder, "craft/zips/test.zip")
    cfg.det_txt = os.path.join(experiments_folder, "craft/txts")
    cfg.out_path = os.path.join(experiments_folder, "craft/hmean_output")
    cfg.log_out_path = os.path.join(experiments_folder, "craft/log")

    if not os.path.exists(cfg.det_txt):
        os.makedirs(cfg.det_txt)

    if not os.path.exists(cfg.out_path):
        os.makedirs(cfg.out_path)

    if not os.path.exists(cfg.log_out_path):
        os.makedirs(cfg.log_out_path)

    if cfg.dataset_type == "lol":
        cfg.input_path_txt = "./dataset/LOL/LOL_test_list.txt"
        cfg.gt_path = "./sid_text/LOL/version_2/detection/lol_det_test.zip"
        cfg.gt_img_path = "./dataset/LOL/eval15/high"
    elif cfg.dataset_type == "sony":
        cfg.input_path_txt = "./dataset/Sony/Sony_test_list.txt"
        cfg.gt_path = "./sid_text/version_1/sony/test/sony_test_short.zip"
        cfg.gt_img_path = "./dataset/Sony/test/long"
    elif cfg.dataset_type == "fuji":
        cfg.input_path_txt = "./dataset/Fuji/Fuji_test_list.txt"
        cfg.gt_path = "./sid_text/version_2/fuji/test/fuji_test_short.zip"
        cfg.gt_img_path = "./dataset/Fuji/test/long"
    elif cfg.dataset_type == "icdar15":
        cfg.input_path_txt = "./dataset/IC15_004/input_image_test.txt"
        cfg.gt_path = "./sid_text/version_1/icdar15/test.zip"
        cfg.gt_img_path = "./dataset/IC15_004/test/gt_image"
    else:
        raise ValueError(f"Invalid value for dataset_type {cfg.dataset_type}")

    if cfg.manual_craft_sel == "icdar15":
        cfg.craft_pretrained_model = (
            "./Text-in-the-Dark/utils/CRAFTpytorch/craft_ic15_20k.pth"
        )
    elif cfg.manual_craft_sel == "mlt":
        cfg.craft_pretrained_model = (
            "./Text-in-the-Dark/utils/CRAFTpytorch/craft_mlt_25k.pth"
        )
    elif cfg.manual_craft_sel is None:
        if cfg.dataset_type == "icdar15" or cfg.dataset_type == "lol":
            cfg.craft_pretrained_model = (
                "./Text-in-the-Dark/utils/CRAFTpytorch/craft_ic15_20k.pth"
            )
        else:
            cfg.craft_pretrained_model = (
                "./Text-in-the-Dark/utils/CRAFTpytorch/craft_mlt_25k.pth"
            )
    else:
        raise ValueError(
            f"Invalid value for craft_pretrained_model {cfg.craft_pretrained_model}"
        )

    if cfg.dataset_type == "icdar15":
        cfg.target_size = (1280, 736)
    elif "lol" in cfg.dataset_type:
        cfg.target_size = (608, 416)
    elif cfg.dataset_type == "sony" or cfg.dataset_type == "fuji":
        cfg.target_size = (4256, 2848)
    else:
        raise ValueError

    out_dict = hmean_eval(cfg)

    return out_dict


if __name__ == "__main__":
    # set dataset_type as sony, fuji, icdar15, lol
    # icdar15 can be used for both sony's and fuji's icdar15_v1 and v3
    dataset_type = "sony"

    # path to input/predicted images
    img_path = ""

    # path to experiments folder to store craft output
    experiments_folder = "test/"

    # current epoch number, for logging purposes only
    epoch = 10

    # whether to print per image result for scikit-image evaluation
    per_img_result = False

    # structure of out_dict, skim stands for scikit-image
    """
        out_dict = {
            'ori_psnr': cc_psnr,
            'ori_ssim': cc_ssim,
            'hmean': hmean,
            'precision': precision,
            'recall': recall,
            'skim_psnr': skim_psnr,
            'skim_ssim': skim_ssim,
            'lpips': lpips,
        }
    """

    out_dict = cc_hmean(
        dataset_type, img_path, experiments_folder, epoch, per_img_result
    )

    # [optional] manual_craft_sel is provided to override craft model selection
    # must be either icdar15 or mlt. Use it as follows:
    # out_dict = cc_hmean(dataset_type, img_path, experiments_folder, epoch, per_img_result, manual_craft_sel='mlt')

    # [optional] use_iqa is provided to decide whether to use IQA pytorch for evaluation
    # structure of out_dict, skim stands for scikit-image, iqa stands for IQA pytorch
    """
        out_dict = {
            'ori_psnr': cc_psnr,
            'ori_ssim': cc_ssim,
            'hmean': hmean,
            'precision': precision,
            'recall': recall,
            'skim_psnr': skim_psnr,
            'skim_ssim': skim_ssim,
            'lpips': lpips,
            'iqa_psnr': iqa_psnr,
            'iqa_ssim': iqa_ssim,
            'iqa_lpips': iqa_lpips,
            'iqa_niqe': iqa_niqe,
        }
    """
    # out_dict = cc_hmean(dataset_type, img_path, experiments_folder, epoch, per_img_result, use_iqa=True)

    # [optional] specify craft_cfg for brute forcing with different CRAFT params
    """
        craft_cfg = {
            'text_threshold': 0.7,
            'low_text': 0.4,
            'link_threshold': 0.4,
        }
    """
    # out_dict = cc_hmean(dataset_type, img_path, experiments_folder, epoch, per_img_result, craft_cfg=craft_cfg)

    # [optional] specify brute_force to bypass dataset_type checking during brute forcing
    # out_dict = cc_hmean(dataset_type, img_path, experiments_folder, epoch, per_img_result, craft_cfg=craft_cfg, brute_force=True)
