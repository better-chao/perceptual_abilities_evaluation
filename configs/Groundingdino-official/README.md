# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

## Abstract

Detic introduces an innovative approach to object detection by leveraging image-level supervision to expand the detector's vocabulary to an unprecedented scale.  It achieves this through a simple yet effective strategy that trains the classifier of a detector on expansive image classification datasets, thereby transcending the limitations imposed by traditional detection datasets.  This approach not only simplifies the implementation process but also enhances compatibility across various detection architectures and backbones, leading to significant improvements in detecting a vast array of object categories, including those not previously annotated with bounding boxes.

![detic-overview](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/images/detic-overview.png)

## Installation

Here we are using the official implementation of Grounding DINO.

```
git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO/

pip install -e .
```

## Fine-grained Understanding Evaluation

### 1.Dataset Preparation

In this task, we use FG-OVD benchmark, see[FG-OVD](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/datasets/FG-OVD/README.md)

### 2.Config Preparation

We utilize Swin-T as the backbone. [Config](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py)

### 3.Weights Preparation

Download the [checkpoint](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) to the weights directory in GroundingDINO.

### 4.Evaluation

First we use the following command for inference.

```
python groundingdino_inference.py \
    --checkpoint_path "weights/groundingdino_swint_ogc.pth" \
    --dataset_path "../FG-OVD/benchmarks/1_attributes.json" \
    --imgs_path "/gpfsdata/home/yangshuai/data/coco" \
    --out "eval_fgovd/1_attributes.pkl" \
    --n_hardnegatives 5
```
'--dataset' refers to the data to be evaluated in json format
'--out' refers to where the output results are stored
'--n_hardnegatives' refers to negative sample number, which is default to 5 for difficulty-based evaluation and 2 for attribute-based evaluation.

Then we use the following command to evaluate.

```
python evaluate_map.py \
--predictions eval_fgovd/1_attributes.pkl \
--ground_truth ../FG-OVD/benchmarks/1_attributes.json \
--out eval_fgovd/1_attributes_result.json
```
'--predictions' refer to the results of model inference
'--ground_truth' refers to the data to be evaluated in json format
'--out' refers to where the output evaluation results are stored.

### 5.Results

| benchmark | Hard | Medium | Easy | Trivial | Color | Material | Pattern | Transp |
 |---|---|---|---|---|---|---|---|---|
| Origin | 16.6 | 27.9 | 30.1 | 62.7 | 41.0 | 30.2 | 31.2 | 25.4 |
| Ours | 17.2 | 28.3 | 30.9 | 62.9 | 41.6 | 30.4 | 31.3 | 26.9 |
