# perceptual_abilities_evaluation of ZegFormer
This is the official code for the [ZegFormer](https://arxiv.org/abs/2112.07910) (CVPR 2022).  

ZegFormer is the first framework that decouple the zero-shot semantic segmentation into:   1) class-agnostic segmentation and 2) segment-level zero-shot classification
## Abstract
Zero-shot semantic segmentation (ZS3) aims to segment the novel categories that have not been seen in the training. Existing works formulate ZS3 as a pixel-level zeroshot classification problem, and transfer semantic knowledge from seen classes to unseen ones with the help of language models pre-trained only with texts. While simple, the pixel-level ZS3 formulation shows the limited capability to integrate vision-language models that are often pre-trained with image-text pairs and currently demonstrate great potential for vision tasks. Inspired by the observation that humans often perform segment-level semantic labeling, we propose to decouple the ZS3 into two sub-tasks: 1) a classagnostic grouping task to group the pixels into segments. 2) a zero-shot classification task on segments. The former task does not involve category information and can be directly transferred to group pixels for unseen classes. The latter task performs at segment-level and provides a natural way to leverage large-scale vision-language models pre-trained with image-text pairs (e.g. CLIP) for ZS3. Based on the decoupling formulation, we propose a simple and effective zero-shot semantic segmentation model, called ZegFormer, which outperforms the previous methods on ZS3 standard benchmarks by large margins, e.g., 22 points on the PASCAL VOC and 3 points on the COCO-Stuff in terms of mIoU for unseen classes. Code will be released at https://github.com/dingjiansw101/ZegFormer.

![image](https://github.com/dingjiansw101/ZegFormer/raw/main/figures/adeinferenceCOCO.png)
## Installation
### Requirements
Linux or macOS with Python ≥ 3.6  
PyTorch ≥ 1.7 and torchvision that matches the PyTorch installation. Install them together at pytorch.org to make sure of this. Note, please check PyTorch version matches that is required by Detectron2.  
Detectron2: follow Detectron2 installation instructions.  
OpenCV is optional but needed by demo and visualization  
pip install -r requirements.txt  
An example of installation is shown below:  
```
conda create -n zegformer python==3.7  
conda activate zegformer  
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html  
python -m pip install detectron2 -f \  
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html  

git clone https://github.com/dingjiansw101/ZegFormer.git  
cd ZegFormer  
pip install -r requirements.txt  
```  
## Small target segmentation
Here we evaluate the performance of ZegFormer on small target segmentation, taking the dataset as an example.  
### Dataset Prepare
See [data preparation](https://github.com/dingjiansw101/ZegFormer/blob/main/datasets/README.md)  
### Inference Demo with Pre-trained Models
Download the checkpoints of ZegFormer from [weights](https://drive.google.com/drive/u/0/folders/1qcIe2mE1VRU1apihsao4XvANJgU5lYgm)  
For example  
```  
python demo/demo.py --config-file configs/coco-stuff/zegformer_R101_bs32_60k_vit16_coco-stuff_gzss_eval.yaml \  
  --input /ZegFormer/input/COCO_train2014_000000000077.jpg /gpfsdata/home/wanqiao/program/ZegFormer/input/COCO_train2014_000000000113.jpg \  
  --output /ZegFormer/result \  
  --opts MODEL.WEIGHTS /ZegFormer/weights/zegformer_R101_bs32_60k_vit16_coco-stuff.pth  
```  
### Training & Evaluation in Command Line  
To train models with R-101 backbone, download the [pre-trained model](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl)   
Setting the dataset as this [URL](https://github.com/dingjiansw101/ZegFormer/blob/main/datasets/README.md)  
For example  
```  
python ./train_net.py \  
  --config-file configs/coco-stuff/zegformer_R101_bs32_60k_vit16_coco-stuff.yaml \  
  --num-gpus 1 \  
  --eval-only MODEL.WEIGHTS /ZegFormer/pretrained_model/R-101.pkl  
```  
