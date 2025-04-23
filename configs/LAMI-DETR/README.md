# LAMI-DETR

[LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction](https://arxiv.org/pdf/2407.11335)

## Abstract

Existing methods enhance open-vocabulary object detection by leveraging the robust open-vocabulary recognition capabilities of Vision Language Models (VLMs), such as CLIP. However, two main challenges emerge: (1) A deficiency in concept representation, where the category names in CLIP’s text space lack textual and visual knowledge. (2) An overfitting tendency towards base categories, with the open vocabulary knowledge biased towards base categories during the transfer from VLMs to detectors. To address these challenges, we propose the Language Model Instruction (LaMI) strategy, which leverages the relationships between visual concepts and applies them within a simple yet effective DETR-like detector, termed LaMI-DETR. LaMI utilizes GPT to construct visual concepts and employs T5 to investigate visual similarities across categories. These inter-category relationships refine concept representation and avoid overfitting to base categories. Comprehensive experiments validate our approach’s superior performance over existing methods in the same rigorous setting without reliance on external training resources. LaMI-DETR achieves a rare box AP of 43.4 on OV-LVIS,
 surpassing the previous best by 7.8 rare box AP.
<img src="..\..\images\LAMI-DETR-overview.png" >

## Installation

The code is tested under python=3.9 torch=1.10.0 [cuda=11.7](https://drive.google.com/file/d/1A57019pFuRRjaQAVAv_lfeWWBWadfcgE/view?usp=sharing). Please [download](https://drive.google.com/file/d/1nIq4gAHvNYSaC_dnozVtHeTH0Rsw-DHY/view?usp=sharing) and unzip this environment under your conda envs dir.
```shell
cd your_conda_envs_path
unzip tar -xvf lami.tar
vim your_conda_envs_path/lami/bin/pip
change '#!~/.conda/envs/lami/bin/python' to '#!your_conda_envs_path/lami/bin/python'
export CUDA_HOME=/usr/local/cuda-11.7
```

or you can create a conda environment and activate it. Install `PyTorch` following the [official documentation](https://pytorch.org/).
For example,
```shell
conda create -n lami python=3.9
conda activate lami
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
export CUDA_HOME=/usr/local/cuda-11.7
```

Check the torch installation.
```shell
python
>>> import torch
>>> torch.cuda.is_available()
True
>>> from torch.utils.cpp_extension import CUDA_HOME
>>> CUDA_HOME
'/usr/local/cuda-11.7'
>>> exit()
```

Install the [detectron2](https://github.com/facebookresearch/detectron2) and [detrex](https://github.com/IDEA-Research/detrex).
```shell
cd LaMI-DETR
pip install -e detectron2
pip install -e .
```

## Open Vocabulary Generalization Evaluation
### 1.Dataset Preparation

In this task, we use COCO and LVIS. we use the weights trained on LVIS-base to evaluate on COCO to assess the model's generalization capability. Similarly, we use the weights trained on LVIS-base to evaluate on COCO.

### 2.Config Preparation

[LVIS-base Config](https://github.com/eternaldolphin/LaMI-DETR/blob/main/lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py)

### 3.Weights Preparation

Download the [LVIS-base checkpoint](https://drive.google.com/file/d/1DRIYuaW4oV_ghFLRX2VG-cALWsF0fxyk/view?usp=sharing) to the weights directory.


### 4.Evaluation

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --config-file lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py --num-gpus 4 --eval-only train.init_checkpoint=pretrained_models/lami_convnext_large_12ep_lvis/model_final.pth


