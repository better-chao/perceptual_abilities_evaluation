# perceptual_abilities_evaluation of zegclip
This is the official code for the [ZegClip](https://github.com/ZiqinZhou66/ZegCLIP).  
  
## Abstract
Recently, CLIP has been applied to pixel-level zero-shot learning tasks via a two-stage scheme. The general idea is to first generate class-agnostic region proposals and then feed the cropped proposal regions to CLIP to utilize its image-level zero-shot classification capability. While effective, such a scheme requires two image encoders, one for proposal generation and one for CLIP, leading to a complicated pipeline and high computational cost. In this work, we pursue a simpler-and-efficient one-stage solution that directly extends CLIP's zero-shot prediction capability from image to pixel level. Our investigation starts with a straightforward extension as our baseline that generates semantic masks by comparing the similarity between text and patch embeddings extracted from CLIP. However, such a paradigm could heavily overfit the seen classes and fail to generalize to unseen classes. To handle this issue, we propose three simple-but-effective designs and figure out that they can significantly retain the inherent zero-shot capacity of CLIP and improve pixel-level generalization ability. Incorporating those modifications leads to an efficient zero-shot semantic segmentation system called ZegCLIP. Through extensive experiments on three public benchmarks, ZegCLIP demonstrates superior performance, outperforming the state-of-the-art methods by a large margin under both ''inductive'' and ''transductive'' zero-shot settings. In addition, compared with the two-stage method, our one-stage ZegCLIP achieves a speedup of about 5 times faster during inference.

![image](https://github.com/ZiqinZhou66/ZegCLIP/blob/main/figs/overview.png)
## Installation
### Requirements
Install pytorch
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch  
```  
Install the mmsegmentation library and some required packages.  
```
pip install mmcv-full==1.4.4 mmsegmentation==0.24.0 pip install scipy timm==0.3.2
```    
Then:  
```
git clone https://github.com/ZiqinZhou66/ZegCLIP.git  
cd ZegCLIP  
pip install -r requirements.txt  
```
## Small target segmentation
Here we evaluate the performance of ZegCLIP on small target segmentation, taking the dataset as an example.  
### Dataset Prepare
See [data preparation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md).  
### Inference
python test.py ./path/to/config ./path/to/model.pth --eval=mIoU  
For example  
```  
CUDA_VISIBLE_DEVICES="0" python test.py configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py /ZegCLIP/models_weights/coco_inductive_512_vit_base.pth --eval=mIoU  
```
Download the pretrained weights [here](https://drive.google.com/file/d/12M6T97o9wyxbJKrR7zLfFMDsGVTiq4WY/view?usp=share_link) .  
