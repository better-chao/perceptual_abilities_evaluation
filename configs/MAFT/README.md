# Learning Mask-aware CLIP Representations for Zero-Shot Segmentation

Recently, pre-trained vision-language models have been increasingly used to tackle the challenging zero-shot segmentation task. To maintain the CLIP's zero-shot transferability, previous practices favour to freeze CLIP during training. However, in the paper, we reveal that CLIP is insensitive to different mask proposals and tends to produce similar predictions for various mask proposals of the same image. This issue mainly relates to the fact that CLIP is trained with image-level supervision. To alleviate this issue, we propose a simple yet effective method, named Mask-aware Fine-tuning (MAFT). Specifically, Image-Proposals CLIP Encoder (IP-CLIP Encoder) is proposed to handle arbitrary numbers of image and mask proposals simultaneously. Then, mask-aware loss and self-distillation loss are designed to fine-tune IP-CLIP Encoder, ensuring CLIP is responsive to different mask proposals while not sacrificing transferability. In this way, mask-aware representations can be easily learned to make the true positives stand out. Notably, our solution can seamlessly plug into most existing methods without introducing any new parameters during the fine-tuning process.

<div>
    <image src="../../images/MAFT-overview.png" />
</div>

## Installation
1. Clone the repository
    ```
    git clone https://github.com/jiaosiyu1999/MAFT.git
    ```
2. Navigate to the project directory
    ```bash
    cd MAFT
    ```
3. Install the dependencies
    ```bash
    bash install.sh
    cd freeseg/modeling/heads/ops
    sh make.sh
    ```
## Zero-Shot Evaluation

### 1. Dataset Preparation

**Cross-datasets setting**

In the cross-datasets zero-shot setting, we use ADE20K-150, ADE20k-847, PAS-20, PAS-20g, PC-59 and PC-459 datasets.

The data should be organized like:

```
datasets/
 ADEChallengeData2016/
    images/
    annotations_detectron2/
 ADE20K_2021_17_01/
    images/
    annotations_detectron2/
 VOCdevkit/
   VOC2012/
    images_detectron2/
    annotations_ovs/
    annotations_detectron2_bg/
  VOC2010/
    images/
    annotations_detectron2_ovs/
      pc59_val/
      pc459_val/    
```

For ADE20K-150, ADE20k-847, PC-59, PC-459, PAS-20, please follow the tutorial in [MAFT](https://github.com/jiaosiyu1999/MAFT/blob/master/datasets/README.md)

For PAS-20g, please follow the tutorial in [CAT-Seg](https://github.com/KU-CVLAB/CAT-Seg/blob/main/datasets/README.md)

Download the [register_voc](./zero-shot-config/register_voc.py) and put it into `./freeseg/data/datasets`. 

### 2. Config Preparation

**Cross-datasets setting**

Download the [config](./zero-shot-config/eval.yaml) and replace `configs/coco-stuff-164k-156/eval.yaml`

### 3. Weights Preparation

**Cross-datasets setting**

Please download the checkpoints of [MAFT_Vitb](https://drive.google.com/file/d/1J3QBMrU65pa8750q5hHiU7bTSsQ5gIAB/view?usp=sharing) to the weights directory.

### 4. Evaluation

**Cross-datasets Evaluation setting**

Replace the `train_net.py` file with this [file](./zero-shot-config/train_net.py).

```bash
# 1. Download MAFT-ViT-B.
# 2. put it at `out/model.pt`.
# 3. evaluation
python train_net.py --config-file configs/coco-stuff-164k-156/eval.yaml --num-gpus [gpu_nums] --eval-only 
```

