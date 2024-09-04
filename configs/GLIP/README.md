# GLIP: Grounded Language-Image Pre-training

[GLIP: Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857)

<!-- [ALGORITHM] -->

## Abstract

This paper presents a grounded language-image pre-training (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making the learned representation semantic-rich. In our experiments, we pre-train GLIP on 27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs. The learned representations demonstrate strong zero-shot and few-shot transferability to various object-level recognition tasks. 1) When directly evaluated on COCO and LVIS (without seeing any images in COCO during pre-training), GLIP achieves 49.8 AP and 26.9 AP, respectively, surpassing many supervised baselines. 2) After fine-tuned on COCO, GLIP achieves 60.8 AP on val and 61.5 AP on test-dev, surpassing prior SoTA. 3) When transferred to 13 downstream object detection tasks, a 1-shot GLIP rivals with a fully-supervised Dynamic Head.

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/b87228d7-f000-4a5d-b103-fe535984417a"/>
</div>

## Installation
Here we use mmdetection toolbox, clone the [repo](https://github.com/open-mmlab/mmdetection/tree/v3.3.0) and follow the [instructions](https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs/glip/README.md) to configure the environment.


```shell
cd $MMDETROOT

# source installation
pip install -r requirements/multimodal.txt

# or mim installation
mim install mmdet[multimodal]
```

## Dense-object-related Evaluation
### 1. Zero prediction
```shell
python tools/test.py configs/GLIP/glip.py 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_c_mmdet-2fc427dd.pth' 
```

### 2. Full finetune
```shell
./dist_train.sh configs/GLIP/glip.py 4
```

### 3. Linear prediction
```shell
./dist_train.sh configs/GLIP/glip_linear.py 4
```

### 4. Prompt tuning
First copy glip_head_prompt_tuning.py in your current work directory, and import it in train.py
```shell
# Add this line in train.py
# from glip_head_prompt_tuning import ATSSVLFusionHead_PromptTuning

./dist_train.sh configs/GLIP/glip_prompt_tuning.py 4
```

### Results

#### CrowdHuman

| **Model** | **Tune**  | **AP** | **AP50** | **AP75** | **APs** | **APm** | **APl** |
| --------- | --------- | ------ | -------- | -------- | ------- | ------- | ------- |
| GLIP      | Zero-shot | 0.197  | 0.429    | 0.159    | 0.079   | 0.159   | 0.248   |
| GLIP      | Prompt    | 0.206  | 0.448    | 0.167    | 0.083   | 0.168   | 0.257   |
| GLIP      | Linear    | 0.503  | 0.827    | 0.520    | 0.374   | 0.505   | 0.535   |
| GLIP      | Full      | 0.505  | 0.829    | 0.523    | 0.380   | 0.508   | 0.535   |
| G-DINO    | Zero-shot | 0.259  | 0.556    | 0.225    | 0.187   | 0.242   | 0.304   |
| G-DINO    | Linear    | 0.557  | 0.880    | 0.595    | 0.463   | 0.558   | 0.579   |
| G-DINO    | Full      | 0.563  | 0.881    | 0.605    | 0.465   | 0.563   | 0.586   |

#### Ochuman (eval only)

| **Model** | **Tune**  | **AP** | **AP50** | **AP75** | **APs** | **APm** | **APl** |
| --------- | --------- | ------ | -------- | -------- | ------- | ------- | ------- |
| GLIP      | Zero-shot | 0.371  | 0.574    | 0.380    | nan     | 0.023   | 0.383   |

