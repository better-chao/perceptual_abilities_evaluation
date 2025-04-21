# PromptDet: Towards Open-vocabulary Detection using Uncurated Images Detection using Uncurated Images

[PromptDet: Towards Open-vocabulary Detection using Uncurated Images Detection using Uncurated Images](https://arxiv.org/abs/2203.16513)

## Abstract

PromptDet introduces a scalable approach to expanding an object detector's vocabulary towards novel and unseen categories with zero manual annotations.  It achieves this through the following contributions: (i) a two-stage open-vocabulary object detector that classifies class-agnostic object proposals using a text encoder from pre-trained visual-language models, enhancing generalization;  (ii) regional prompt learning that aligns the textual embedding space with regional visual object features to better pair the visual latent space (from RPN box proposals) with the pre-trained text encoder;  (iii) a self-training framework that leverages uncurated and noisy web images to scale the learning procedure, enabling detection across a broader spectrum of objects.

![promptdet-overview](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/images/promptdet-overview.png)

## Installation

```
MMDetection version 2.16.0.

Please see ![get_started.md](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation and the basic usage of MMDetection.
```

## Open Vocabulary Generalization Evaluation
### 1.Dataset Preparation

In this task, we use COCO and LVIS. we use the weights trained on LVIS-base to evaluate on COCO to assess the model's generalization capability. Similarly, we use the weights trained on LVIS-base to evaluate on COCO.

### 2.Config Preparation

[LVIS-base Config]([https://github.com/facebookresearch/Detic/blob/main/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml](https://github.com/fcjian/PromptDet/blob/master/configs/promptdet/promptdet_r50_fpn_sample1e-3_mstrain_6x_lvis_v1_self_train.py))

### 3.Weights Preparation

Download the [LVIS-base checkpoint](https://drive.google.com/file/d/1hxVx2eI220_9legRozZTTQONswaLETYd/view?usp=sharing) to the weights directory.

### 4.Evaluation

./tools/dist_test.sh configs/promptdet/promptdet_r50_fpn_sample1e-3_mstrain_6x_lvis_v1.py weights/promptdet_r50_fpn_sample1e-3_mstrain_6x_lvis_v1_self_train.pth 2 --eval bbox segm
