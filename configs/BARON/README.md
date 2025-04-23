# BARON

[Aligning Bag of Regions for Open-Vocabulary Object Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Aligning_Bag_of_Regions_for_Open-Vocabulary_Object_Detection_CVPR_2023_paper.pdf)

## Abstract

Pre-trained vision-language models (VLMs) learn to align vision and language representations on large-scale datasets, where each image-text pair usually contains a bag of semantic concepts. However, existing open-vocabulary object detectors only align region embeddings individually with the corresponding features extracted from the VLMs. Such a design leaves the compositional structure of semantic concepts in a scene under-exploited, although the structure may be implicitly learned by the VLMs. In this work, we propose to align the embedding of bag of regions beyond individual regions. The proposed method groups contextually interrelated regions as a bag. The embeddings of regions in a bag are treated as embeddings of words in a sentence, and they are sent to the text encoder of a VLM to obtain the bag-of-regions embedding, which is learned to be aligned to the corresponding features extracted by a frozen VLM. Applied to the commonly used Faster R-CNN, our approach surpasses the previous best results by 4.6 box AP50 and 2.8 mask AP on novel categories of open-vocabulary COCO and LVIS benchmarks, respectively.
<img src="..\..\images\baron-overview.png" >

## Installation

This project is based on [MMDetection 3.x](https://github.com/open-mmlab/mmdetection/tree/3.x)

It requires the following OpenMMLab packages:

- MMEngine >= 0.6.0
- MMCV-full >= v2.0.0rc4
- MMDetection >= v3.0.0rc6
- lvisapi

```bash
pip install openmim mmengine
mim install "mmcv>=2.0.0rc4"
pip install git+https://github.com/lvis-dataset/lvis-api.git
mim install mmdet>=3.0.0rc6
```

## Open Vocabulary Generalization Evaluation
### 1.Dataset Preparation

In this task, we use COCO and LVIS. we use the weights trained on LVIS-base to evaluate on COCO to assess the model's generalization capability. Similarly, we use the weights trained on LVIS-base to evaluate on COCO.

### 2.Config Preparation

[LVIS-base Config](https://github.com/wusize/ovdet/blob/main/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py)

### 3.Weights Preparation

Download the [LVIS-base checkpoint](https://drive.google.com/drive/folders/1JTM0uoPQZtq7lnhZxCBwjxBUca9omYR9?usp=sharing) to the weights directory.


### 4.Evaluation

```bash
GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=12 bash tools/slurm_test.sh PARTITION test \ 
path/to/the/cfg/file path/to/the/checkpoint
```

