# Open Vocabulary Object Detection with Pseudo Bounding-Box Labels

[Open Vocabulary Object Detection with Pseudo Bounding-Box Labels](https://arxiv.org/abs/2111.09452)

## Abstract

Despite great progress in object detection, most existing methods work only on a limited set of object categories, due to the tremendous human effort needed for bounding-box annotations of training data. To alleviate the problem, recent open vocabulary and zero-shot detection methods attempt to detect novel object categories beyond those seen during training. They achieve this goal by training on a pre-defined base categories to induce generalization to novel objects. However, their potential is still constrained by the small set of base categories available for training. To enlarge the set of base classes, we propose a method to automatically generate pseudo bounding-box annotations of diverse objects from large-scale image-caption pairs. Our method leverages the localization ability of pre-trained vision-language models to generate pseudo bounding-box labels and then directly uses them for training object detectors. Experimental results show that our method outperforms the state-of-the-art open vocabulary detector by 8% AP on COCO novel categories, by 6.3% AP on PASCAL VOC, by 2.3% AP on Objects365 and by 2.8% AP on LVIS. 

<img src="..\..\images\pb-ovd-overview.jpg" >

## Installation
Following the below steps to install PB-OVD

```angular2
conda create --name ovd
conda activate ovd
cd $INSTALL_DIR

bash ovd_install.sh
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

cd ../
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
python setup.py build develop
```

## Fine-grained Evalutation

### 1. Dataset Preparation

In the zero-shot setting, we use Stanford Dogs and CUB200-2011 datasets, please follow [Stanford Dogs](datasets/stanford_dogs/README.md) and [CUB200-2011](datasets/cub200_2011/README.md) to place the datasets correctly and convert the annotations into COCO format.

### 2. Checkpoint And Config Preparation

Download the checkpoint of PB-OVD finetuned on COCO dataset from [here](https://storage.cloud.google.com/sfr-pb-ovd-research/models/finetune.pth). 

Put finetune_cub200_2011.yaml and finetune_stanford_dogs.yaml under the folder of PB-OVD/configs

Add the below values to the DATASETS in maskrcnn_benchmark/config/paths_catalog.py
```shell
"stanford_dogs_coco_train":{
    "img_dir": "stanford_dogs/Images",
    "ann_file": "stanford_dogs_train_clipemb.json"
},
"stanford_dogs_coco_val":{
    "img_dir": "stanford_dogs/Images",
    "ann_file": "stanford_dogs_val_clipemb.json"
},
"cub200_2011_coco_train":{
    "img_dir": "cub200_2011/images",
    "ann_file": "cub200_2011_train_clipemb.json"
},
"cub200_2011_coco_val":{
    "img_dir": "cub200_2011/images",
    "ann_file": "cub200_2011_val_clipemb.json"
}
```

### 3. Finetune And Evaluation

Finetune PB-OVD on Stanford Dogs datasets following
```shell
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py  --distributed \
--config-file configs/finetune_stanford_dogs.yaml \
MODEL.WEIGHT models_finetune.pth \
OUTPUT_DIR work_dirs/stanford_dogs_finetune
```

Finetune PB-OVD on CUB200-2011 datasets following
```shell
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py  --distributed \
--config-file configs/finetune_cub200_2011.yaml \
MODEL.WEIGHT models_finetune.pth \
OUTPUT_DIR work_dirs/cub200_2011_finetune
```
The evaluation will be made after training and the result can be found in the training log.
