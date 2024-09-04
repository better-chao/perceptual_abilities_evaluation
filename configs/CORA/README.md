# CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching

[CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching](https://arxiv.org/abs/2303.13076)

## Abstract

CORA introduces a DETR-style framework that effectively adapts the CLIP model for open-vocabulary detection by employing two key strategies: Region Prompting and Anchor Pre-Matching.   Region Prompting tackles the distribution mismatch by enhancing the region features of the CLIP-based region classifier, ensuring that the model can accurately classify objects within image regions rather than relying solely on whole-image features.   Anchor Pre-Matching, on the other hand, facilitates the learning of generalizable object localization.  It does so by using a class-aware matching mechanism that associates object queries with dynamic anchor boxes.  This pre-matching process allows for efficient and class-specific localization of objects, which is crucial for detecting novel classes during inference.

![cora-overview](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/images/cora-overview.png)

## Installation

```
# cora
git clone git@github.com:tgxs002/CORA.git
cd CORA

# environment
conda create -n cora python=3.9.12
conda activate cora
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch

# install detectron2
Please install detectron2 as instructed in the official tutorial (https://detectron2.readthedocs.io/en/latest/tutorials/install.html). We use version==0.6 in our experiments.

# dependencies
pip install -r requirements.txt

# cuda operators
cd ./models/ops
sh ./make.sh
```

## Fine-grained Understanding Evaluation

### 1.Dataset Preparation

In this task, we use FG-OVD benchmark, see[FG-OVD](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/datasets/FG-OVD/README.md)

### 2.Config Preparation

We utilize the model with Resnet50x4 as the backbone. [Config](https://github.com/tgxs002/CORA/blob/master/configs/COCO/R50x4_dab_ovd_3enc_apm128_splcls0.2_relabel_noinit.sh)

### 3.Weights Preparation

Download the [checkpoint](https://drive.google.com/file/d/115osjVyv86vjG_b0W83vPQryXxdIDsv_/view?usp=share_link) to the weights directory in CORA.

### 4.Evaluation

First we use the following command for inference.

```
python cora_inferences.py \
--dataset ../FG-OVD/benchmarks/1_attributes.json \
--out ./work_dir/1_attributes.pkl \
--n_hardnegatives 5 \
--resume ./weights/COCO_RN50x4.pth \
--eval \
--batch_size 2 \
--backbone clip_RN50x4 \
--ovd \
--region_prompt_path logs/region_prompt_R50x4.pth \
--dim_feedforward 1024 \
--use_nms \
--num_queries 1000 \
--anchor_pre_matching \
--remove_misclassified \
--condition_on_text \
--enc_layers 3 \
--text_dim 640 \
--condition_bottleneck 128 \
--label_version RN50x4base_prev \
--disable_init
```
'--dataset' refers to the data to be evaluated in json format
'--out' refers to where the output results are stored
'--n_hardnegatives' refers to negative sample number, which is default to 5 for difficulty-based evaluation and 2 for attribute-based evaluation.
The other parameters are the cora configuration.

Then we use the following command to evaluate.

```
python evaluate_map.py \
--predictions work_dir/1_attributes.pkl \
--ground_truth ../FG-OVD/benchmarks/1_attributes.json \
--out work_dir/1_attributes_result.json
```
'--predictions' refer to the results of model inference
'--ground_truth' refers to the data to be evaluated in json format
'--out' refers to where the output evaluation results are stored.

### 5.Results

| benchmark | Hard | Medium | Easy | Trivial | Color | Material | Pattern | Transp |
 |---|---|---|---|---|---|---|---|---|
| Origin | 13.8 | 20.0 | 20.4 | 35.1 | 25.0 | 19.3 | 22.0 | 27.9 |
| Ours | 14.7 | 22.1 | 24.3 | 35.2 | 24.7 | 18.7 | 20.1 | 27.0 |
