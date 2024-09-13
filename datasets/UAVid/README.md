# UAVid: A Semantic Segmentation Dataset for UAV Imagery
[UAVid: A Semantic Segmentation Dataset for UAV Imagery](https://arxiv.org/abs/1810.10438) .  
 
## Abstract
UAVid dataset is a new high-resolution UAV semantic segmentation dataset as a complement, which brings new challenges, including large scale variation, moving object recognition and temporal consistency preservation. Our UAV dataset consists of 30 video sequences capturing 4K high-resolution images in slanted views. In total, 300 images have been densely labeled with 8 classes for the semantic labeling task. We have provided several deep learning baseline methods with pre-training, among which the proposed Multi-Scale-Dilation net performs the best via multi-scale feature extraction.  

## Dataset Path and Annotation Explanation 
```
/gpfsdata/home/wanqiao/dataset/UAVid
|---train  
|   |---Images  
|   |   |---xxx.png
|   |   |---xxx.png
|   |   |...   
|   |---Labels
|   |   |---xxx.png
|   |   |---xxx.png  
|   |   |...  
|---val  
|   |---Images  
|   |   |---xxx.png
|   |   |---xxx.png
|   |   |...   
|   |---Labels
|   |   |---xxx.png
|   |   |---xxx.png  
|   |   |...   
```    
