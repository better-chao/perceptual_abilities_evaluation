# Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model

[Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model](https://arxiv.org/abs/2203.14940)

## Abstract

DetPro introduces a novel method for open-vocabulary object detection by leveraging continuous prompt representation learning based on pre-trained vision-language models.  It addresses key challenges in prompt engineering for detection with the following innovations: (i) a background interpretation scheme that incorporates image background proposals into prompt training to improve representation;  (ii) a context grading scheme that separates image foreground proposals for tailored and precise prompt training, enabling better alignment with detection tasks.  Unlike previous classification-oriented prompt learning methods, DetPro is specifically designed for detection scenarios.
![detpro-overview](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/images/detpro-overview.png)

## Installation

```
Please see ![get_started.md](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation and the basic usage of MMDetection.

pip install -r requirements/build.txt
pip install -e .
pip install git+https://github.com/openai/CLIP.git
pip uninstall pycocotools -y
pip uninstall mmpycocotools -y
pip install mmpycocotools
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

## Open Vocabulary Generalization Evaluation
### 1.Dataset Preparation

In this task, we use COCO and LVIS. we use the weights trained on LVIS-base to evaluate on COCO to assess the model's generalization capability. Similarly, we use the weights trained on LVIS-base to evaluate on COCO.

### 2.Config Preparation

[LVIS-base Config](https://github.com/dyabel/detpro/blob/main/configs/lvis/detpro_ens_20e.py)

### 3.Weights Preparation

Download the [LVIS-base checkpoint](https://drive.google.com/file/d/1ktTMZWFjUAGjzjlOdzxGfKQR8u9x_OmX/view?usp=sharing) to the weights directory.
Download the [prompt weight](https://drive.google.com/file/d/1T-Ydo0YgneDbZYU2hu3wWm9MZ2plxGT_/view?usp=sharing) to the weights directory.

### 4.Evaluation

./tools/dist_test.sh  configs/transfer/transfer_coco.py workdirs/vild_ens_20e_fg_bg_5_10_end/epoch_20.pth 8 --eval bbox --cfg-options model.roi_head.load_feature=False model.roi_head.prompt_path=checkpoints/coco/fg_bg_5_10_voc_ens.pth model.roi_head.fixed_lambda=0.6
