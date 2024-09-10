# Cascade-CLIP: Cascaded Vision-Language Embeddings Alignment for Zero-Shot Semantic Segmentation
Paper is this [URL](https://arxiv.org/abs/2406.00670).The official code is the [Cascade-CLIP](https://github.com/HVision-NKU/Cascade-CLIP).   
 
## Abstract
Pre-trained vision-language models, e.g., CLIP, have been successfully applied to zero-shot semantic segmentation. Existing CLIP-based approaches primarily utilize visual features from the last layer to align with text embeddings, while they neglect the crucial information in intermediate layers that contain rich object details. However, we find that directly aggregating the multi-level visual features weakens the zero-shot ability for novel classes. The large differences between the visual features from different layers make these features hard to align well with the text embeddings. We resolve this problem by introducing a series of independent decoders to align the multi-level visual features with the text embeddings in a cascaded way, forming a novel but simple framework named Cascade-CLIP. Our Cascade-CLIP is flexible and can be easily applied to existing zero-shot semantic segmentation methods. Experimental results show that our simple Cascade-CLIP achieves superior zero-shot performance on segmentation benchmarks, like COCO-Stuff, Pascal-VOC, and Pascal-Context.  

![image](https://github.com/HVision-NKU/Cascade-CLIP/blob/main/figs/overview.png)
## Installation
### Requirements
See [installation instructions](https://github.com/HVision-NKU/Cascade-CLIP/blob/main/INSTALL.md).  
An example of installation is shown below:  
```
conda create --name cascade_clip python=3.7 -y  
conda activate cascade_clip  
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch  
pip install openmim  
mim install mmcv-full==1.5.0  
pip install mmsegmentation==0.24.0  
pip install -r requirements.txt    
```   
### Dataset Prepare
See [data preparation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md).  
### Inference
```
python test.py config/xxx/xxx.py ./output/model.pth --eval=mIoU
```
For example  
```  
python test.py /Cascade-CLIP/configs/coco_171/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py /Cascade-CLIP/weights/coco_model.pth  --eval=mIoU  
```
Download the pretrained weights [here](https://pan.baidu.com/s/1OwBKOKr0-GkTmmv_K6OLqw?pwd=9mkw).  
