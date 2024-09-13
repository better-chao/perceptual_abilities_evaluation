# UAVid: A Semantic Segmentation Dataset for UAV Imagery
[UAVid: A Semantic Segmentation Dataset for UAV Imagery](https://arxiv.org/abs/1810.10438) .  
 
## Abstract
There already exist several semantic segmentation datasets for comparison among semantic segmentation methods in complex urban scenes, such as the Cityscapes and CamVid datasets, where the side views of the objects are captured with a camera mounted on the driving car. There also exist semantic labeling datasets for the airborne images and the satellite images, where the top views of the objects are captured. However, only a few datasets capture urban scenes from an oblique Unmanned Aerial Vehicle (UAV) perspective, where both of the top view and the side view of the objects can be observed, providing more information for object recognition. UAVid dataset is a new high-resolution UAV semantic segmentation dataset as a complement, which brings new challenges, including large scale variation, moving object recognition and temporal consistency preservation. Our UAV dataset consists of 30 video sequences capturing 4K high-resolution images in slanted views. In total, 300 images have been densely labeled with 8 classes for the semantic labeling task. We have provided several deep learning baseline methods with pre-training, among which the proposed Multi-Scale-Dilation net performs the best via multi-scale feature extraction.  

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
