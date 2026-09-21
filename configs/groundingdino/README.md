
# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

<!-- [ALGORITHM] -->

## Abstract

Grounding DINO is an open-set object detector by marrying Transformer-based detector DINO with grounded pre-training, which can detect arbitrary objects with human inputs such as category names or referring expressions. The key solution of open-set object detection is introducing language to a closed-set detector for open-set concept generalization. Grounding DINO can be divided a closed-set detector into three phases and propose a tight fusion solution, which includes a feature enhancer, a language-guided query selection, and a cross-modality decoder for cross-modality fusion. 

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/42299757/0ed51aeb-3d53-42d8-8563-f6d21364ac95"/>
</div>
 
## Installation
Here we use groundingdino's mmdetection implementation, clone the [repo](https://github.com/open-mmlab/mmdetection/tree/dev-3.x) and follow the [instructions](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/grounding_dino/README.md) to configure the environment.


## Domain-related Evaluation
Here we evaluate the performance of GroundingDino on cross-domain data, taking the [Diverse Weather](https://github.com/AmingWu/Single-DGOD) dataset as an example.
### 1. Dataset Preparation
Download [Diverse Weather](https://github.com/AmingWu/Single-DGOD) Datasets and convert the dataset into coco format. Diverse Weather consists of images from five different weather conditions in urban scenes: daytime sunny, night sunny, dusk rainy, night rainy, and daytime foggy. Examples from the five scenes are illustrated as following. Here, we view daytime sunny scene as source domain and test on the other four challenging weather conditions (unseen target domains).

<img src="../../images/weather_diverse.png" width="96%" height="96%">

### 2. Config Preparation

Here we consider two evaluation methods: 1. Zero prediction：Directly use the weights to evaluate on 5 scenarios; 2. Full finetune: Use the weights to perform full finetuning on the source domain, and then evaluate the performance on the unseen target domain.

The Details of the configuration can be found in [grounding_dino_swin-t_finetune_8xb2_20e_weather](grounding_dino_swin-t_finetune_8xb2_20e_weather.py), and we put this configuration file into ./configs/grounding_dino in [repo](https://github.com/open-mmlab/mmdetection/tree/dev-3.x).

### 3. Zero prediction

```shell
# $MMDETROOT represents the main directory of the [repo](https://github.com/open-mmlab/mmdetection/tree/dev-3.x).

cd $MMDETROOT

# download the pretrained weights
wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth

# perform the zero prediction
python tools/test.py configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_weather.py groundingdino_swint_ogc_mmdet-822d7e9d.pth
```

### 4. Source Finetune and evaluation

```shell
# finetune on the source domain dataset, here, we use 4 cards to train 20 epochs, scale the learning rate accordingly, and do not train the language model, only the visual model.

./tools/dist_train.sh configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_weather.py 4 --work-dir work_dir/weather_work_dir

```
No best-checkpoint selection is performed: the checkpoint of the final epoch is the one reported.

```shell
# Then we test the final-epoch checkpoint on the different unseen target datasets (20 epochs, so epoch_20.pth).

python tools/test.py configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_weather.py work_dir/weather_work_dir/epoch_20.pth

```


## Visual Prompt Fine-tuning (unified feature-space prompt)

All backbone, neck, encoder, decoder and text-encoder parameters are frozen; the
only trainable parameters are one 50x50 tensor per neck level, added to the
multi-scale image feature map after bilinear resizing to each level.

- module: [grounding_dino_visual_prompt_module.py](grounding_dino_visual_prompt_module.py)
- config: [grounding_dino_visual_prompt_tuning.py](grounding_dino_visual_prompt_tuning.py)

```shell
# put both files under $MMDETROOT/configs/grounding_dino/ and run
./tools/dist_train.sh configs/grounding_dino/grounding_dino_visual_prompt_tuning.py 4 \
    --work-dir work_dirs/grounding_dino_visual_prompt
```

Set `model.prompt_channels` to `[256] * model.neck.num_outs`.
