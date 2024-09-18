# LVIS: A Dataset for Large Vocabulary Instance Segmentation

[LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://github.com/lvis-dataset/lvis-api)

## Abstract

COCO-C, VOC-C, Cityscapes-C are the robustness evaluation datasets, 
consisting of 15 types of corruptions and 5 levels of severities. 
The original robustness evaluation was performed by incorporating data augmentation 
into image preprocessing, which was quite slow. 
To speed up the evaluation process, we stored the augmented images on disk, 
creating 15*5 sub-datasets for further evaluation. 


## Dataset Path and Annotation Explanation
```shell
COCO: 
/gpfsdata/home/huangziyue/data/coco_new
|---val2017_corruptions
|   |---{corruption}_Svr{severity}
|   |   |---image1
|   |   |...
|   |...

VOC
|---VOC_test_corruptions
|...

cityscapes
|---val_corruptions
|...


```
The corruption value for {corruption}_Svr{severity} is in [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter',
        'saturate'
    ], and the severity value is [1,2,3,4,5].

The greater the severity value, the more significant the change in the image.








