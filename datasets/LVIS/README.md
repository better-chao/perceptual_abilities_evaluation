# LVIS: A Dataset for Large Vocabulary Instance Segmentation

[LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://github.com/lvis-dataset/lvis-api)

## Abstract

LVIS (pronounced ‘el-vis’): is a dataset for Large Vocabulary Instance Segmentation. When complete, it will feature more than 2 million high-quality instance segmentation masks for over 1200 entry-level object categories in 164k images. The LVIS API enables reading and interacting with annotation files, visualizing annotations, and evaluating results.

LVIS uses the COCO 2017 train, validation, and test image sets. If you have already downloaded the COCO images, you only need to use the LVIS annotations. LVIS val set contains images from COCO 2017 train in addition to the COCO 2017 val split.

LVIS has annotations for instance segmentations in a format similar to COCO. The annotations are stored using JSON. The LVIS API can be used to access and manipulate annotations. The JSON file has the following format:


## Dataset Path and Annotation Explanation
```shell
/gpfsdata/home/huangziyue/data/coco_new
|---train2017
|   |---xxx
|   |---xxx
|   |---xxx
|   |...
|---val2017
|   |---xxx
|   |---xxx
|   |---xxx
|   |...
|---annotations
|   |---lvis_od_train.json
|   |---lvis_od_val.json
|   |---lvis_v1_minival_inserted_image_name.json
|   |---ovd_ins_train2017_all.json
|   |---ovd_ins_train2017_b.json
|   |---ovd_ins_train2017_t.json
|   |---ovd_ins_val2017_all.json
|   |---ovd_ins_val2017_b.json
|   |---ovd_ins_val2017_t.json
```
train2017: The training images of COCO 2017.

val2017: The validation images of COCO 2017.

lvis_od_train.json: The training annotations of LVIS, in COCO format.

lvis_od_val.json: The validation annotations of LVIS, in COCO format.

## YAML Config Explanation
```shell
LVIS/
|---minival.yaml
|---val.yaml
```
minival.yaml: Small validation set.
val.yaml:     Full validation set.








