# Simple Open-Vocabulary Object Detection with Vision Transformers

[Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

## Abstract

OWL-ViT is a method for open-vocabulary object detection that leverages the power of Vision Transformers (ViTs). It employs a standard ViT architecture, pre-trained contrastively on a large image-text dataset, and fine-tuned end-to-end for detection tasks. With minimal modifications to the original model, OWL-ViT achieves strong performance on zero-shot and one-shot image-conditioned detection, demonstrating the effectiveness of simple architectures combined with large-scale pre-training for open-vocabulary settings. The method simplifies the detection process by removing the final token pooling layer and attaching lightweight classification and box heads directly to the transformer output tokens, enabling open-vocabulary classification with class-name embeddings. OWL-ViT showcases consistent improvements with increased image-level pre-training and model size, setting a new benchmark for open-world detection tasks.

![owlvit-overview](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/images/owlvit-overview.png)

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

Download the [owlvit-large-patch14](https://huggingface.co/google/owlvit-large-patch14) and [owlvit-base-patch16](https://huggingface.co/google/owlvit-base-patch16) to the weights directory.

### 4.Evaluation

First we use the following command for inference.

```
python owl_inference.py\
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
| Origin-base | 26.2 | 39.8 | 38.4 | 53.9 | 45.3 | 37.3 | 26.6 | 34.1 |
| Ours-base | 26.4 | 40.4 | 38.9 | 55.4 | 45.5 | 37.4 | 26.8 | 34.1 |
| Origin-large | 26.5 | 39.3 | 44.0 | 65.1 | 43.8 | 44.9 | 36.0 | 29.2 |
| Ours-large | 26.6 | 39.8 | 44.5 | 67.0 | 44.0 | 45.0 | 36.2 | 29.2 |
