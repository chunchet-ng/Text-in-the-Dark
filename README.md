# Text in the Dark

Official repository of the paper: "Text in the Dark: Extremely Low-Light Text Image
Enhancement"

Authored By:
Chun Chet Ng*, Che-Tsung Lin*, Zhi Qin Tan, Wan Jun Nah, Xinyu Wang, Jie Long Kew, Pohao Hsu, Shang Hong Lai, Chee Seng Chan, Christopher Zach

*Equal Contribution

***The manuscript is currently under review. This work is an extension of our ICPR 2022 paper, Extremely Low-Light Image Enhancement with Scene Text Restoration.***

[Project Page]() | [Paper]() | [Text in the Dark Dataset](#text-in-the-dark-dataset) | [Extremely Low-Light Text Image Enhancement Model](#extremely-low-light-text-image-enhancement-model-elite) | [Bibtex](#citation)

## Abstract:
Text extraction in extremely low-light images is challenging. Although existing low-light image enhancement methods can enhance images as preprocessing before text extraction, they do not focus on scene text. Further research is also hindered by the lack of extremely low-light text datasets. Thus, we propose a novel extremely low-light image enhancement framework with an edge-aware attention module to focus on scene text regions. Our method is trained with text detection and edge reconstruction losses to
emphasize low-level scene text features. Additionally, we present a Supervised Deep Curve Estimation model to synthesize extremely low-light images based on the public ICDAR15 (IC15) dataset. We also labeled texts in the extremely low-light See In the Dark (SID) and ordinary LOw-Light (LOL) datasets to benchmark extremely low-light scene text tasks. Extensive experiments prove our model outperforms state-of-the-art methods on all datasets.

## Text in the Dark Dataset
***The Text in the Dark dataset will be released upon the acceptance of the manuscript. Please stay tuned!***

## Extremely Low-Light Text Image Enhancement Model (ELITE)
***The training and testing code for the proposed method will be released upon the acceptance of the manuscript. Please stay tuned!***

## Citation
If you wish to cite the paper published at ICPR 2022, Extremely Low-Light Image Enhancement with Scene Text Restoration:

```bibtex
@inproceedings{icdar2021_ictext,
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

If you wish to cite the lastest version of Text in the Dark dataset and our proposed method:

***Our paper is currently under review. We will update this section when it is published.***

## Feedback
We welcome all suggestions and opinions (both positive and negative) on this work. Please contact us by sending an email to `ngchunchet95 at gmail.com` or `cs.chan at um.edu.my`.

## Acknowledgement
We would like to express our gratitude for the contributions of computing resources and annotation platforms by [ViTrox Corporation Berhad](https://www.vitrox.com/). Their generous support has made this work possible.

## License and Copyright
This project is open source under the BSD-3 license (refer to the [LICENSE](LICENSE.txt) file for more information).

&#169; 2023 Universiti Malaya.
