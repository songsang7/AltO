# Unsupervised Homography Estimation on Multimodal Image Pair via Alternating Optimization (AltO)

This is an official PyTorch implementation of "[Unsupervised Homography Estimation on Multimodal Image Pair via Alternating Optimization](https://arxiv.org/abs/2411.13036)" to be presented at the Thirty-Eighth Annual Conference on Neural Information Processing Systems ([NeurIPS 2024](https://neurips.cc/virtual/2024/poster/92937)).

![](https://github.com/songsang7/AltO/blob/main/paper_figures/overall_archi_whitebackgrounded.svg)

## 1. Requirements
- Python
- PyTorch
- Albumentations
- Kornia[x]
- OpenCV-python

## 2. Datasets
- Google Map & Google Earth : [DLKFM](https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography)
- Deep NIR
   - [Overview](https://inkyusa.github.io/deepNIR_dataset/overview/synth/)
   - [Download](https://www.kaggle.com/datasets/enddl22/deepnir-nir-rgb-nirscene1-dataset)
   - In our paper, 'nirscene_img_aug_100_oversample' is used.

## 3. How to Run

Run ***main.py*** without any options. By default, it performs both training and testing, but if you want to execute only specific tasks, comment out unnecessary parts in main.py. If you want to change hyperparameters or paths, check ***image_matching/ui/hardcoded.py***. For more details, see the ***Static View (AltO).drawio.pdf***.
