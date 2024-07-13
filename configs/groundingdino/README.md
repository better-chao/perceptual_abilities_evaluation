
# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

<!-- [ALGORITHM] -->

## Abstract

Grounding DINO is an open-set object detector by marrying Transformer-based detector DINO with grounded pre-training, which can detect arbitrary objects with human inputs such as category names or referring expressions. The key solution of open-set object detection is introducing language to a closed-set detector for open-set concept generalization. Grounding DINO can be divided a closed-set detector into three phases and propose a tight fusion solution, which includes a feature enhancer, a language-guided query selection, and a cross-modality decoder for cross-modality fusion. 

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/42299757/0ed51aeb-3d53-42d8-8563-f6d21364ac95"/>
</div>

## Installation
Follow the [instructions](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/grounding_dino/README.md) to configure the environment.


## Domain-related Evaluation
Here we evaluate the performance of GroundingDino on cross-domain data, taking the [Diverse Weather](https://github.com/AmingWu/Single-DGOD) dataset as an example.
### 1. Dataset Preparation
Download [Diverse Weather](https://github.com/AmingWu/Single-DGOD) Datasets and convert the dataset into coco format. Diverse Weather consists of images from five different weather conditions in urban scenes: daytime sunny, night sunny, dusk rainy, night rainy, and daytime foggy. Examples from the five scenes are illustrated as following. Here, we view daytime sunny scene as source domain and test on the other four challenging weather conditions (unseen target domains).

<img src="./images/overview.png" width="96%" height="96%">

<img src="./images/weather_diverse.png" width="96%" height="96%">

### 2. Config Preparation

Due to the simplicity and small number of cat datasets, we use 8 cards to train 20 epochs, scale the learning rate accordingly, and do not train the language model, only the visual model.

The Details of the configuration can be found in [grounding_dino_swin-t_finetune_8xb2_20e_cat](grounding_dino_swin-t_finetune_8xb2_20e_cat.py)

### 3. Visualization and Evaluation

Due to the Grounding DINO is an open detection model, so it can be detected and evaluated even if it is not trained on the cat dataset.

The single image visualization is as follows:

```shell
cd mmdetection
python demo/image_demo.py data/cat/images/IMG_20211205_120756.jpg configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py --weights https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth --texts cat.
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/89173261-16f1-4fd9-ac63-8dc2dcda6616" alt="cat dataset"/>
</div>

The test dataset evaluation on single card is as follows:

```shell
python tools/test.py configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth
```

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.867
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.931
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.867
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.903
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.907
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.907
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.907
```

### 4. Model Training and Visualization

```shell
./tools/dist_train.sh configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py 8 --work-dir cat_work_dir
```

The model will be saved based on the best performance on the test set. The performance of the best model (at epoch 16) is as follows:

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.905
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.923
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.905
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.927
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.937
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.937
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.937
```

We can find that after fine-tuning training, the training of the cat dataset is increased from 86.7 to 90.5.

If we do single image inference visualization again, the result is as follows:

```shell
cd mmdetection
python demo/image_demo.py data/cat/images/IMG_20211205_120756.jpg configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py --weights cat_work_dir/best_coco_bbox_mAP_epoch_16.pth --texts cat.
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/5a027b00-8adb-4283-a47b-2f7a0a2c96d4" alt="cat dataset"/>
</div>
