# CamVid:The Cambridge-driving Labeled Video Database
[Segmentation and Recognition Using Structure from Motion Point Clouds](https://link.springer.com/chapter/10.1007/978-3-540-88682-2_5) .  
The github link is https://github.com/lih627/CamVid/tree/main .
 
## Abstract
The CamVid dataset is a road scene dataset used for semantic segmentation tasks, particularly in urban environments. It contains video sequences captured from a vehicle driving through streets, with pixel-level annotations for 11 object classes, including roads, buildings, pedestrians, and vehicles. CamVid provides essential ground truth for the development and evaluation of computer vision models focused on understanding and segmenting real-world urban scenes.  

## Dataset Path and Annotation Explanation 
```
/gpfsdata/home/wanqiao/dataset/CamVid
|---CamVid_Label   
|   |---xxx.png  
|   |...  
|---CamVid_RGB    
|   |---xxx.png  
|   |...  
|---CamVidColor11  
|   |---xxx.png
|   |...
|---CamVidGray
|   |---xxx.png
|   |...
|---splits
|   |---camvid_test.txt
|   |---camvid_train.txt
|   |---camvid_trainval.txt
|   |---camvid_val.txt
# python camvid_data_split.py
|---sortes_camvid
|   |---test
|   |   |---images
|   |   |   |---xxx.png
|   |   |---labels
|   |   |   |---xxx.png
|   |---train
|   |---val  
```    
