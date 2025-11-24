# Text in the Dark

###  Official repository of the paper: "Text in the Dark: Extremely Low-Light Text Image Enhancement"

Authored By:
Chun Chet Ng*, Che-Tsung Lin*, Zhi Qin Tan, Wan Jun Nah, Xinyu Wang, Jie Long Kew, Pohao Hsu, Shang Hong Lai, Chee Seng Chan, Christopher Zach

*Equal Contribution

Released On: December 20, 2024

***

[Project Page](https://chunchet-ng.github.io/Text-in-the-Dark/) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0923596524001231) | [Text in the Dark Dataset](#text-in-the-dark-dataset) | [Installation](#installation) | [Training](#training) | [Evaluation](#evaluation) | [Bibtex](#citation)

## Abstract:
Extremely low-light text images pose significant challenges for scene text detection. Existing methods enhance these images using low-light image enhancement techniques before text detection. However, they fail to address the importance of low-level features, which are essential for optimal performance in downstream scene text tasks. Further research is also limited by the scarcity of extremely low-light text datasets. To address these limitations, we propose a novel, text-aware extremely low-light image enhancement framework. Our approach first integrates a Text-Aware Copy-Paste (Text-CP) augmentation method as a preprocessing step, followed by a dual-encoder–decoder architecture enhanced with Edge-Aware attention modules. We also introduce text detection and edge reconstruction losses to train the model to generate images with higher text visibility. Additionally, we propose a Supervised Deep Curve Estimation (Supervised-DCE) model for synthesizing extremely low-light images, allowing training on publicly available scene text datasets such as IC15. To further advance this domain, we annotated texts in the extremely low-light See In the Dark (SID) and ordinary LOw-Light (LOL) datasets. The proposed framework is rigorously tested against various traditional and deep learning-based methods on the newly labeled SID-Sony-Text, SID-Fuji-Text, LOL-Text, and synthetic extremely low-light IC15 datasets. Our extensive experiments demonstrate notable improvements in both image enhancement and scene text tasks, showcasing the model’s efficacy in text detection under extremely low-light conditions.

## Text in the Dark Dataset
> ***The Text in the Dark dataset is released [here](https://bit.ly/text-in-the-dark).***

The Text in the Dark dataset is created based on the combination of the following low light datasets:

1. See in the Dark (SID) dataset - Sony Set
2. See in the Dark (SID) dataset - Fuji Set
3. LOw Light (LOL) dataset

The annotation format for this dataset is following the ICDAR's annotations format:

``<x1,y1,x2,y2,x3,y3,x4,y4,text_class> or <L,T,R,B,text_class>, text_class can be "Text" or "###"``

Please note that texts of "###" class are ignored during evaluation.

### Statistics - Long Exposure Images
#### SID-Sony Set:
| Subset | Images | Legible Text | Illegible Text | Total Text |
| :--- | :----: | :----: | :----: | :----: |
| Train | 161 | 5,937 | 2,128 | 8,065 |
| Test | 50 | 611 | 359 | 970 |

#### SID-Fuji Set:
| Subset | Images | Legible Text | Illegible Text | Total Text |
| :--- | :----: | :----: | :----: | :----: |
| Train | 135 | 6,213 | 4,534 | 10,747 |
| Test | 41 | 1,018 | 1,083 | 2,101 |

#### LOL:
| Subset | Images | Legible Text | Illegible Text | Total Text |
| :--- | :----: | :----: | :----: | :----: |
| Train | 485 | 613 | 1,423 | 2,036 |
| Test | 15 | 28 | 45 | 73 |

### Statistics - Short Exposure Images
#### SID-Sony Set:
| Subset | Images | Legible Text | Illegible Text | Total Text |
| :--- | :----: | :----: | :----: | :----: |
| Train | 280 | 10,396 | 3,866 | 14,262 |
| Test | 598 | 8,210 | 4,976 | 13,186 |

#### SID-Fuji Set:
| Subset | Images | Legible Text | Illegible Text | Total Text |
| :--- | :----: | :----: | :----: | :----: |
| Train | 286 | 13,540 | 10,316 | 23,856 |
| Test | 524 | 12,768 | 14,036 | 26,804 |

#### LOL:
| Subset | Images | Legible Text | Illegible Text | Total Text |
| :--- | :----: | :----: | :----: | :----: |
| Train | 485 | 613 | 1,423 | 2,036 |
| Test | 15 | 28 | 45 | 73 |

## Installation

### Environment Setup

This project requires Python 3.9+ and PyTorch. We recommend using Conda to manage the environment.

#### Using Conda Environment File

```bash
# Clone the repository
git clone https://github.com/your-username/Text-in-the-Dark.git
cd Text-in-the-Dark

# Create and activate the conda environment
conda env create -f low_light_text.yml
conda activate tid
```

### Pretrained Models

Download the pretrained CRAFT and RCF models and place them in the appropriate directories:
- CRAFT model: Place in `utils/CRAFTpytorch/`
- RCF model: Place in `utils/RCFpytorch/`

### Dataset Preparation

Download the Text in the Dark dataset from [here](https://bit.ly/text-in-the-dark) and organize the data according to the structure expected by the training and testing scripts.

### Evaluation Package Installation

For evaluation, install the `cc_eval` package:

```bash
pip install ./standalone_eval
```

This package provides the evaluation metrics including H-mean, PSNR, SSIM, and LPIPS for text detection and image quality assessment.

## Training

To train the model, use the `train.py` script with a configuration file:

```bash
python train.py --config configs/your_config.yml
```

Key training arguments:
- `--config`: Path to the YAML configuration file
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--workers`: Number of data loading workers
- `--use-wandb`: Enable Weights & Biases logging
- `--use-dp`: Use DataParallel for multi-GPU training

Example:
```bash
python train.py --config configs/sony_config.yml --batch-size 4 --epochs 100 --use-wandb
```

## Evaluation

To evaluate a trained model, use the `test.py` script:

```bash
python test.py --config configs/your_config.yml --weights path/to/checkpoint.pth --gen-image
```

Key evaluation arguments:
- `--config`: Path to the YAML configuration file
- `--weights`: Path to the trained model checkpoint
- `--gen-image`: Generate and save enhanced images
- `--batch-size`: Batch size for evaluation
- `--workers`: Number of data loading workers

Example:
```bash
python test.py --config configs/sony_config.yml --weights checkpoints/best_model.pth --gen-image
```

The evaluation will output:
- H-mean (text detection performance)
- PSNR and SSIM (image quality metrics)
- LPIPS (perceptual similarity)
- Enhanced images (if `--gen-image` is specified)

***

## Citation
If you wish to cite the paper published at ICPR 2022, [Extremely Low-Light Image Enhancement with Scene Text Restoration](https://ieeexplore.ieee.org/document/9956716):

```bibtex
@inproceedings{icpr2022_ellie,
  author={Hsu, Po-Hao
  and Lin, Che-Tsung
  and Ng, Chun Chet
  and Long Kew, Jie
  and Tan, Mei Yih
  and Lai, Shang-Hong
  and Chan, Chee Seng
  and Zach, Christopher},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
  title={Extremely Low-Light Image Enhancement with Scene Text Restoration}, 
  year={2022},
  pages={317-323}}
```

If you wish to cite the latest version of Text in the Dark dataset published at Signal Processing: Image Communication, [Text in the Dark: Extremely Low-Light Text Image Enhancement](https://www.sciencedirect.com/science/article/abs/pii/S0923596524001231):

```bibtex
@article{spic_text_in_the_dark,
  title = {Text in the dark: Extremely low-light text image enhancement},
  journal = {Signal Processing: Image Communication},
  volume = {130},
  pages = {117222},
  year = {2025},
  issn = {0923-5965},
  doi = {https://doi.org/10.1016/j.image.2024.117222},
  url = {https://www.sciencedirect.com/science/article/pii/S0923596524001231},
  author = {Che-Tsung Lin and Chun Chet Ng and Zhi Qin Tan and Wan Jun Nah and Xinyu Wang and 
  Jie Long Kew and Pohao Hsu and Shang Hong Lai and Chee Seng Chan and Christopher Zach},
}
```

## Feedback
We welcome all suggestions and opinions (both positive and negative) on this work. Please contact us by sending an email to `ngchunchet95 at gmail.com` or `cs.chan at um.edu.my`.

## Acknowledgement
We would like to express our gratitude for the contributions of computing resources and annotation platforms by [ViTrox Corporation Berhad](https://www.vitrox.com/). Their generous support has made this work possible.

## License and Copyright
This project is open source under the BSD-3 license (refer to the [LICENSE](LICENSE.txt) file for more information).

&#169; 2024 Universiti Malaya.