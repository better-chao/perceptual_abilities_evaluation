# Detecting Twenty-thousand Classes using Image-level Supervision

[Detecting Twenty-thousand Classes using Image-level Supervision](https://arxiv.org/abs/2201.02605)

## Abstract

Detic introduces an innovative approach to object detection by leveraging image-level supervision to expand the detector's vocabulary to an unprecedented scale.  It achieves this through a simple yet effective strategy that trains the classifier of a detector on expansive image classification datasets, thereby transcending the limitations imposed by traditional detection datasets.  This approach not only simplifies the implementation process but also enhances compatibility across various detection architectures and backbones, leading to significant improvements in detecting a vast array of object categories, including those not previously annotated with bounding boxes.

![detic-overview](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/images/detic-overview.png)

## Installation

```
conda create -n detic python=3.9 -y
conda activate detic
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd Detic
pip install -r requirements.txt
```

## Fine-grained Understanding Evaluation

### 1.Dataset Preparation

In this task, we use FG-OVD benchmark, see[FG-OVD](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/datasets/FG-OVD/README.md)

### 2.Config Preparation

We utilize Swin-B as the backbone and ImageNet-21K pre-training. [Config](https://github.com/facebookresearch/Detic/blob/main/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml)

### 3.Weights Preparation

Download the [checkpoint](https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth) to the weights directory in detic.

### 4.Evaluation

First we use the following command for inference.

```
python detic_inference.py\
 --dataset ../FG-OVD/benchmarks/1_attributes.json\
 --out eval_fgovd/1_attributes.pkl
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
| Origin | 11.5 | 18.6 | 18.6 | 69.7 | 21.5 | 38.8 | 30.1 | 28.0 |
| Ours | 11.5 | 18.6 | 18.8 | 69.7 | 21.6 | 38.8 | 30.3 | 24.8 |
