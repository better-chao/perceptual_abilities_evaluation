# Caltech-UCSD Birds-200-2011 (CUB-200-2011)

Download Caltech-UCSD Birds-200-2011 (CUB-200-2011) [here](https://www.vision.caltech.edu/datasets/cub_200_2011/).

## Abstract
Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations.
- Number of categories: 200
- Number of images: 11,788
- Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box

## Dataset Folder Structure
Uncompress dataset to get the following folder structure
```shell
datasets/cub200_2011
|---attributes
|   |---xxx
|   |---xxx
|   |---xxx
|   |...
|---images
|   |---xxx
|   |---xxx
|   |---xxx
|   |...
|---parts
|   |---xxx
|   |---xxx
|   |---xxx
|   |...
|---bounding_boxes.txt
|---classes.txt
|---image_class_labels.txt
|---images.txt
|---train_test_split.txt
```

## Convert into COCO annotation format
Running the script to convert the annotataion of Stanford Dogs Dataset into COCO format.
```shell
python cub200_2011_to_coco_dataset.py --root_path datasets/cub200_2011 --output_path datasets/cub200_2011_train.json
```
