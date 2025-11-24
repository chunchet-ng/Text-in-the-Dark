# Standalone evaluation code

### Package installation
```bash
pip install ./Text-in-the-Dark/standalone_eval
```

### Example Usage

```python
from cc_eval.cc_hmean import cc_hmean


if __name__ == '__main__':
    # set dataset_type as sony, fuji, icdar15, lol
    # icdar15 can be used for both sony's and fuji's icdar15_v1 and v3
    dataset_type = 'sony'
    
    # path to input/predicted images
    img_path = ''
    
    # path to experiments folder to store craft output 
    experiments_folder = 'test/'
    
    # current epoch number, for logging purposes only
    epoch = 10

    # whether to print per image result for scikit-image evaluation
    per_img_result = False
    
    # structure of out_dict, skim stands for scikit-image
    '''
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
    '''
    
    out_dict = cc_hmean(dataset_type, img_path, experiments_folder, epoch, per_img_result)
    
    # [optional] manual_craft_sel is provided to override craft model selection
    # must be either icdar15 or mlt. Use it as follows:
    # out_dict = cc_hmean(dataset_type, img_path, experiments_folder, epoch, per_img_result, manual_craft_sel='mlt')
    
    # [optional] use_iqa is provided to decide whether to use IQA pytorch for evaluation
    # structure of out_dict, skim stands for scikit-image, iqa stands for IQA pytorch
    '''
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
    '''
    # out_dict = cc_hmean(dataset_type, img_path, experiments_folder, epoch, per_img_result, use_iqa=True)
```