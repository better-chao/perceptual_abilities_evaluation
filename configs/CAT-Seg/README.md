# CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation

[CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2303.11797)

## Abstract

Open-vocabulary semantic segmentation presents the challenge of labeling each pixel within an image based on a wide range of text descriptions. In this work, CAT-Seg introduce a novel cost-based approach to adapt vision-language foundation models, notably CLIP, for the intricate task of semantic segmentation. Through aggregating the cosine similarity score, i.e., the cost volume between image and text embeddings, this method potently adapts CLIP for segmenting seen and unseen classes by fine-tuning its encoders, addressing the challenges faced by existing methods in handling unseen classes. Building upon this, CAT-Seg explore methods to effectively aggregate the cost volume considering its multi-modal nature of being established between image and text embeddings.



![img](D:\data\科研\多模态大模型评估\perceptual_abilities_evaluation\images\catseg-overview.png)

## Installation

### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.13 is recommended and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

An example of installation is shown below:

```
git clone https://github.com/KU-CVLAB/CAT-Seg.git
cd CAT-Seg
conda create -n catseg python=3.8
conda activate catseg
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Zero-Shot Evaluation

### 1. Dataset Preparation

In the zero-shot setting, we use ADE20K-150, ADE20k-847, PAS-20, PAS-20g, PC-59 and PC-459 datasets, please follow [dataset preperation](https://github.com/KU-CVLAB/CAT-Seg/blob/main/datasets/README.md)

### 2. Config Preparation

For CAT-Seg-B, we use this [config](https://github.com/KU-CVLAB/CAT-Seg/blob/main/configs/vitb_384.yaml), and for CAT-Seg-L, we use this [config](https://github.com/KU-CVLAB/CAT-Seg/blob/main/configs/vitl_336.yaml)

### 3. Weights Preparation

Please download the checkpoints of [CAT-Seg-B](https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_base.pth) and [CAT-Seg-L](https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_large.pth) to the weights directory in CAT-Seg.

### 4. Evaluation

**Evaluation script**

```bash
sh eval.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] MODEL.WEIGHTS path/to/weights.pth

# for CAT-Seg-B
sh eval.sh configs/vitb_384.yaml 4 output/ MODEL.WEIGHTS path/to/CAT-Seg-B/weights.pth
# for CAT-Seg-L
sh eval.sh configs/vitl_336.yaml 4 output/ MODEL.WEIGHTS path/to/CAT-Seg-L/weights.pth
```
