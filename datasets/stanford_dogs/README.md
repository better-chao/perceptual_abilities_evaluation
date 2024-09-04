# Stanford Dogs Dataset

Download Stanford Dogs Dataset [here](http://vision.stanford.edu/aditya86/ImageNetDogs/).

## Abstract
The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. Contents of this dataset:
- Number of categories: 120
- Number of images: 20,580
- Annotations: Class labels, Bounding boxes

## Dataset Folder Structure
Uncompress dataset to get the following folder structure
```shell
datasets/stanford_dogs
|---Annotation
|   |---xxx
|   |---xxx
|   |---xxx
|   |...
|---Images
|   |---xxx
|   |---xxx
|   |---xxx
|   |...
|---file_list.mat
|---train_list.mat
|---test_list.mat
```

## Convert into COCO annotation format
Running the script to convert the annotataion of Stanford Dogs Dataset into COCO format.
```shell
python stanford_dogs_to_coco_dataset.py --root_path datasets/stanford_dogs --output_path datasets/stanford_dogs_train.json
```

