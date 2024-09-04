# Side Adapter Network for Open-Vocabulary Semantic Segmentation

[Side Adapter Network for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2302.12242)

## Abstract

This paper presents a new framework for open-vocabulary semantic segmentation with the pre-trained vision-language model, named Side Adapter Network (SAN). Our approach models the semantic segmentation task as a region recognition problem. A side network is attached to a frozen CLIP model with two branches: one for predicting mask proposals, and the other for predicting attention bias which is applied in the CLIP model to recognize the class of masks. This decoupled design has the benefit CLIP in recognizing the class of mask proposals. Since the attached side network can reuse CLIP features, it can be very light. In addition, the entire network can be trained end-to-end, allowing the side network to be adapted to the frozen CLIP model, which makes the predicted mask proposals CLIP-aware. Our approach is fast, accurate, and only adds a few additional trainable parameters. We evaluate our approach on multiple semantic segmentation benchmarks. Our method significantly outperforms other counterparts, with up to 18 times fewer trainable parameters and 19 times faster inference speed.

<img src="..\..\images\san-overview.png" >

## Installation

```bash
# Clone the repository
git clone https://github.com/MendelXu/SAN.git
# Navigate to the project directory
cd SAN
# Install the dependencies
bash install.sh
```

## Zero-Shot Evaluation

### 1. Dataset Preparation

The data should be organized like:

```
datasets/
    VOC2012/
        ...
        images_detectron2/
        annotations_detectron2/
        annotations_detectron2_bg/
    pcontext/
        ...
        val/
    pcontext_full/
        ...
        val/
    ADEChallengeData2016/
        ...
        images/
        annotations_detectron2/
    ADE20K_2021_17_01/
        ...
        images/
        annotations_detectron2/        
```

For ADE20k and Pascal VOC, follow the tutorial in [CAT-Seg](https://github.com/KU-CVLAB/CAT-Seg/blob/main/datasets/README.md)

For Pascal Context, follow the tutorial in [SimSeg](https://github.com/MendelXu/zsseg.baseline)

### 2. Config Preparation

For SAN-ViT-B, we use this [config](https://github.com/MendelXu/SAN/blob/main/configs/san_clip_vit_res4_coco.yaml), and for SAN-ViT-L, we use this [config](https://github.com/MendelXu/SAN/blob/main/configs/san_clip_vit_large_res4_coco.yaml)

### 3. Weights Preparation

Please download the checkpoints of [SAN-ViT-B](https://huggingface.co/Mendel192/san/resolve/main/san_vit_b_16.pth) and [SAN-ViT-L](https://huggingface.co/Mendel192/san/resolve/main/san_vit_large_14.pth) to the weights directory in SAN.

### 4. Evaluation

Before the evaluation, please download the [register_voc.py](./register_voc.py)  and put it into `./san/data/datasets/`. And replace the `train_net.py` file with this [file](./train_net.py).

```bash
python train_net.py --eval-only --config-file [path/to/config] --num-gpus 4 OUTPUT_DIR [OUTPUT_DIR] MODEL.WEIGHTS [path/to/checkpoint] DATASETS.TEST "('pcontext_sem_seg_val','ade20k_sem_seg_val','pcontext_full_sem_seg_val','ade20k_full_sem_seg_val', 'voc_sem_seg_val', 'voc_sem_seg_test_background')"
```

