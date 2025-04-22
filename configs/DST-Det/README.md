# Exploiting unlabeled data with vision and language models for object detection
Paper is this [URL](https://arxiv.org/abs/2310.01393).The official code is the [DST-Det](https://github.com/xushilin1/dst-det).  
  
## Abstract
This paper presents a novel method for open-vocabulary object detection (OVOD) that aims to detect objects \textit{beyond} the set of categories observed during training. Our approach proposes a dynamic self-training strategy that leverages the zero-shot classification capabilities of pre-trained vision-language models, such as CLIP, to classify proposals as novel classes directly. Unlike previous works that ignore novel classes during training and rely solely on the region proposal network (RPN) for novel object detection, our method selectively filters proposals based on specific design criteria. The resulting set of identified proposals serves as pseudo labels for novel classes during the training phase, enabling our self-training strategy to improve the recall and accuracy of novel classes in a self-training manner without requiring additional annotations or datasets. Empirical evaluations on the LVIS and COCO datasets demonstrate significant improvements over the baseline performance without incurring additional parameters or computational costs during inference. Notably, our method achieves a 1.7% improvement over the previous F-VLM method on the LVIS validation set. Moreover, combined with offline pseudo label generation, our method improves the strong baselines over 10 % mAP on COCO.
![image](https://github.com/xushilin1/dst-det/blob/main/assets/figs/teaser.png)
## Installation
### Requirements
Install pytorch
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch  
```  
Install the MMDetection library.  
```
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.7.0
MMCV_WITH_OPS=1 pip install -e . -v
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.28.1
pip install -e . -v
```    
Install the EVA-CLIP  
```
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
pip install -e . -v
```  
### Dataset Prepare
See [Data Preparation](https://github.com/xushilin1/dst-det?tab=readme-ov-file).  
### Inference
python ./test.py
### Configs
See ./configs/fvit/coco/fvit_vitl14_coco.py
Download the pretrained weights [here](https://huggingface.co/shilinxu/dst-det/tree/main) .  
