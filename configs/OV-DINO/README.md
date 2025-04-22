# OV-DINO: Unified Open-Vocabulary Detection with Language-Aware Selective Fusion

[OV-DINO: Unified Open-Vocabulary Detection with Language-Aware Selective Fusion](https://arxiv.org/abs/2407.07844)

<!-- [ALGORITHM] -->

## Abstract

Open-vocabulary detection is a challenging task due to the requirement of detecting objects based on class names, including those not encountered during training. Existing methods have shown strong zero-shot detection capabilities through pre-training and pseudo-labeling on diverse large-scale datasets. However, these approaches encounter two main challenges: (i) how to effectively eliminate data noise from pseudo-labeling, and (ii) how to efficiently leverage the language-aware capability for region-level cross-modality fusion and alignment. To address these challenges, we propose a novel unified open-vocabulary detection method called OV-DINO, which is pre-trained on diverse large-scale datasets with language-aware selective fusion in a unified framework. Specifically, we introduce a Unified Data Integration (UniDI) pipeline to enable end-to-end training and eliminate noise from pseudo-label generation by unifying different data sources into detection-centric data format. In addition, we propose a Language-Aware Selective Fusion (LASF) module to enhance the cross-modality alignment through a language-aware query selection and fusion process. We evaluate the performance of the proposed OV-DINO on popular open-vocabulary detection benchmarks, achieving state-of-the-art results with an AP of 50.6% on the COCO benchmark and 40.1% on the LVIS benchmark in a zero-shot manner, demonstrating its strong generalization ability. Furthermore, the fine-tuned OV-DINO on COCO achieves 58.4% AP, outperforming many existing methods with the same backbone. 

<div align=center>
<img src="https://github.com/wanghao9610/OV-DINO/raw/main/docs/ovdino_framework.png"/>
</div>

## Installation
Please clone the [OVDIN](https://github.com/wanghao9610/OV-DINO/) repo and follow the [instructions](https://github.com/wanghao9610/OV-DINO/blob/main/README.md) to configure the environment.

```bash
# clone this project
git clone https://github.com/wanghao9610/OV-DINO.git
cd OV-DINO
export root_dir=$(realpath ./)
cd $root_dir/ovdino

# Optional: set CUDA_HOME for cuda11.6.
# OV-DINO utilizes the cuda11.6 default, if your cuda is not cuda11.6, you need first export CUDA_HOME env manually.
export CUDA_HOME="your_cuda11.6_path"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo -e "$log_format cuda version:\n$(nvcc -V)"

# create conda env for ov-dino
conda create -n ovdino -y
conda activate ovdino
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install gcc=9 gxx=9 -c conda-forge -y # Optional: install gcc9
python -m pip install -e detectron2-717ab9
pip install -e ./

# Optional: create conda env for ov-sam, it may not compatible with ov-dino, so we create a new env.
# ov-sam = ov-dino + sam2
conda create -n ovsam -y
conda activate ovsam
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# install the sam2 following the sam2 project.
# please refer to https://github.com/facebookresearch/segment-anything-2.git
# download sam2 checkpoints and put them to inits/sam2
python -m pip install -e detectron2-717ab9
pip install -e ./
```

After configuring the environment, copy the [ovdino](./ovdino) folder to your local OV-DINO repository.


## Evaluation
### 1. Zero prediction
```shell
sh eval_custom.sh
```

### 2. Visual Finetune
```shell
sh ft_custom.sh
```

### 3. Text Prompt tuning
```shell
sh text_custom.sh
```