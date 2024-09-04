# OPEN-VOCABULARY OBJECT DETECTION VIA VISION AND LANGUAGE KNOWLEDGE DISTILLATION

[OPEN-VOCABULARY OBJECT DETECTION VIA VISION AND LANGUAGE KNOWLEDGE DISTILLATION](https://arxiv.org/abs/2104.13921)

## Abstract

ViLD introduces an innovative approach to open-vocabulary object detection by distilling knowledge from a pretrained image classifier —— CLIP into a two-stage detector.   It employs the teacher model to generate embeddings for category texts and image regions from object proposals, then trains a student detector to align its region embeddings with those provided by the teacher.   This method excels at detecting objects described by arbitrary text inputs, even for categories not seen during training, and demonstrates strong performance on benchmarks like LVIS, outperforming supervised models on novel categories.   ViLD's effectiveness is further highlighted by its ability to generalize to other datasets without finetuning, showcasing its potential for scalable detection of diverse categories.

![vild-overview](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/images/vild-overview.png)

## Installation
Here we follow the official implementation of FG-OVD, using the vild implemented in google colab.

```
# environment
conda create -n vild python=3.9.12
conda activate vild

# install tensorflow
conda install cudnn==8.9.2.26
conda install cudatoolkit==11.8.0

pip install tensorflow==2.9.0

# install clip
pip install git+https://github.com/openai/CLIP.git

# install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# dependencies
pip install opencv-python
pip install pycocotools
pip install lvis

```

## Fine-grained Understanding Evaluation

### 1.Dataset Preparation

In this task, we use FG-OVD benchmark, see[FG-OVD](https://github.com/better-chao/perceptual_abilities_evaluation/blob/main/datasets/FG-OVD/README.md)

### 2.Config Preparation

We utilize the model with Resnet-152 as the backbone.

### 3.Weights Preparation

gsutil cp -r gs://cloud-tpu-checkpoints/detection/projects/vild/colab/image_path_v2 ./

### 4.Evaluation

First we use the following command for inference.

```
python detectors_inferences/vild_inference.py\
 --dataset benchmarks/1_attributes.json\
 --out work_dir/1_attributes.pkl\
 --n_hardnegatives 5
```
'--dataset' refers to the data to be evaluated in json format
'--out' refers to where the output results are stored
'--n_hardnegatives' refers to negative sample number, which is default to 5 for difficulty-based evaluation and 2 for attribute-based evaluation.

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
| Origin | 22.1 | 36.1 | 39.9 | 56.6 | 43.2 | 34.9 | 24.5 | 30.1 |
| Ours | 22.1 | 36.1 | 40.0 | 56.6 | 43.1 | 34.8 | 24.9 | 30.6 |
