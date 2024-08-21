# CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching

[CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching](https://arxiv.org/abs/2303.13076)

## Abstract

CORA introduces a DETR-style framework that effectively adapts the CLIP model for open-vocabulary detection by employing two key strategies: Region Prompting and Anchor Pre-Matching.   Region Prompting tackles the distribution mismatch by enhancing the region features of the CLIP-based region classifier, ensuring that the model can accurately classify objects within image regions rather than relying solely on whole-image features.   Anchor Pre-Matching, on the other hand, facilitates the learning of generalizable object localization.  It does so by using a class-aware matching mechanism that associates object queries with dynamic anchor boxes.  This pre-matching process allows for efficient and class-specific localization of objects, which is crucial for detecting novel classes during inference.

![cora-overview](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/images/cora-overview.png)

## Installation

```
# cora
git clone git@github.com:tgxs002/CORA.git
cd CORA

# environment
conda create -n cora python=3.9.12
conda activate cora
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch

# install detectron2
Please install detectron2 as instructed in the official tutorial (https://detectron2.readthedocs.io/en/latest/tutorials/install.html). We use version==0.6 in our experiments.

# dependencies
pip install -r requirements.txt

# cuda operators
cd ./models/ops
sh ./make.sh
```

## Fine-grained Understanding Evaluation

### 1.Dataset Preparation

In this task, we use FG-OVD benchmark, see[FG-OVD]()
