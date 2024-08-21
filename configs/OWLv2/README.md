# Scaling Open-Vocabulary Object Detection

[Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)

## Abstract

OWLv2 is an advanced open-vocabulary object detection model that scales up detection data through self-training, leveraging weak supervision from abundant Web image-text pairs.   It addresses key challenges in self-training, such as label space selection, pseudo-annotation filtering, and training efficiency.   With an optimized architecture and a self-training recipe called OWL-ST, OWLv2 surpasses previous state-of-the-art detectors at comparable training scales and achieves significant improvements when scaled to over 1 billion examples.

![owlv2-overview](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/images/owlv2-overview.png)

## Installation

Here we use the owlvit model available in the transformers open source library.

```
# environment
conda create -n owlvit python=3.10 -y
conda activate owlvit
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# transformers install
pip install transformers==4.38.0

```

## Fine-grained Understanding Evaluation

### 1.Dataset Preparation

In this task, we use FG-OVD benchmark, see[FG-OVD](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/datasets/FG-OVD/README.md)

### 2.Config Preparation

We evaluated configurations with ViT B/16 and ViT L/14 backbones.

### 3.Weights Preparation

Download the [owlvit-large-patch14](https://huggingface.co/google/owlv2-large-patch14) and [owlvit-base-patch16](https://huggingface.co/google/owlv2-base-patch16) to the weights directory.

### 4.Evaluation

First we use the following command for inference.

```
python owl2_inference.py\
 --dataset ../FG-OVD/benchmarks/1_attributes.json\
 --out ./work_dir/1_attributes.pkl\
 --n_hardnegatives 5
```

'--dataset' refers to the data to be evaluated in json format
'--out' refers to where the output results are stored
'--n_hardnegatives' refers to negative sample number, which is default to 5 for difficulty-based evaluation and 2 for attribute-based evaluation.
The other parameters are the cora configuration.

Then we use the following command to evaluate.

```
python evaluate_map.py\
 --predictions ./work_dir/1_attributes.pkl\
 --ground_truth ../FG-OVD/benchmarks/1_attributes.json\
 --out ./work_dir/1_attributes_result.json \
 --n_hardnegatives 5
```

'--predictions' refer to the results of model inference
'--ground_truth' refers to the data to be evaluated in json format
'--out' refers to where the output evaluation results are stored.
'--n_hardnegatives' refers to negative sample number, which is default to 5 for difficulty-based evaluation and 2 for attribute-based evaluation.
The other parameters are the cora configuration.

### 5.Results

| benchmark | Hard | Medium | Easy | Trivial | Color | Material | Pattern | Transp |
 |---|---|---|---|---|---|---|---|---|
| Origin-base | 25.3 | 38.5 | 40.0 | 52.9 | 45.1 | 33.5 | 19.2 | 28.5 |
| Ours-base | 25.4 | 39.0 | 40.5 | 54.4 | 45.2 | 33.6 | 19.3 | 28.5 |
| Origin-large | 25.4 | 41.2 | 42.8 | 63.2 | 53.3 | 36.9 | 23.3 | 12.2 |
| Ours-large | 25.6 | 41.8 | 43.3 | 65.0 | 53.4 | 37.0 | 23.4 | 12.2 |
