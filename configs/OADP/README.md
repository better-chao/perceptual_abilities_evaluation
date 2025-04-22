# Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection
Paper is this [URL](https://arxiv.org/pdf/2303.05892).The official code is the [OADP](https://github.com/LutingWang/OADP).  
  
## Abstract
Open-vocabulary object detection aims to provide object detectors trained on a fixed set of object categories with the generalizability to detect objects described by arbitrary text queries. Previous methods adopt knowledge distillation to extract knowledge from Pretrained Vision-andLanguage Models (PVLMs) and transfer it to detectors. However, due to the non-adaptive proposal cropping and single-level feature mimicking processes, they suffer from information destruction during knowledge extraction and inefficient knowledge transfer. To remedy these limitations, we propose an Object-Aware Distillation Pyramid (OADP) framework, including an Object-Aware Knowledge Extraction (OAKE) module and a Distillation Pyramid (DP) mechanism. When extracting object knowledge from PVLMs, the former adaptively transforms object proposals and adopts object-aware mask attention to obtain precise and complete knowledge of objects. The latter introduces global and block distillation for more comprehensive knowledge transfer to compensate for the missing relation information in object distillation. Extensive experiments show that our method achieves significant improvement compared to current methods.
[image](../../images/oadp.png)
## Installation
### Requirements
Install pytorch
```
conda create -n oadp python=3.10
conda activate oadp

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```  
Install the MMDetection library.  
```
pip install openmim
mim install mmcv_full==1.7.0
pip install mmdet==2.25.2
```    
Install other dependencies,
```
pip install todd_ai==0.3.0
pip install git+https://github.com/LutingWang/CLIP.git
pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
pip install nni scikit-learn==1.1.3
```  
### Dataset Prepare
See [Datasets](https://github.com/LutingWang/OADP).  
### Inference
You could follow the under command: 
```
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.dp.test configs/dp/oadp_ov_coco_eval.py work_dirs/oadp_ov_coco/iter_32000.pth
```  
### Configs
See ./configs/dp/oadp_ov_coco_eval.py

### Checkpoints
See [Results](https://github.com/LutingWang/OADP). 
