# MM-OVOD

[Multi-Modal Classifiers for Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.05493)

## Abstract

The goal of this paper is open-vocabulary object detection (OVOD) – building a model that can detect objects beyond the set of categories seen at training, thus enabling the user to specify categories of interest at inference without the need for model retraining. We adopt a standard two-stage object detector architecture, and explore three ways for specifying novel categories: via language descriptions, via image exemplars, or via a combination of the two. We make three contributions: first, we prompt a large language model (LLM) to generate informative language descriptions for object classes, and construct powerful text-based classifiers; second, we employ a visual aggregator on image exemplars that can ingest any number of images as input, forming vision-based classifiers; and third, we provide a simple method to fuse information from language descriptions and image exemplars, yielding a multi-modal classifier. When evaluating on the challenging LVIS open-vocabulary benchmark we demonstrate that: (i) our text-based classifiers outperform all previous OVOD works; (ii) our vision-based classifiers perform as well as text-based classifiers in prior work; (iii) using multi-modal classifiers perform better than either modality alone; and finally, (iv) our text-based and multi-modal classifiers yield better performance than a fully-supervised detector.
<img src="..\..\images\mm-ovod-overview.jpg" >

## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.8.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).


### Author conda environment setup
```bash
conda create --name mm-ovod python=3.8 -y
conda activate mm-ovod
conda install pytorch torchvision=0.9.2 torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
git checkout 2b98c273b240b54d2d0ee6853dc331c4f2ca87b9
pip install -e .

cd ..
git clone https://github.com/prannaykaul/mm-ovod.git --recurse-submodules
cd mm-ovod
pip install -r requirements.txt
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

Our project (like Detic) use a submodule: [CenterNet2](https://github.com/xingyizhou/CenterNet2.git). If you forget to add `--recurse-submodules`, do `git submodule init` and then `git submodule update`.


### Downloading pre-trained ResNet-50 backbone
We use the ResNet-50 backbone pre-trained on ImageNet-21k-P from [here](
https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth). Please download it from the previous link, place it in the `${mm-ovod_ROOT}/checkpoints` folder and use the following command to convert it for use with detectron2:
```bash
cd ${mm-ovod_ROOT}
mkdir checkpoints
cd checkpoints
wget https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth
python ../tools/convert-thirdparty-pretrained-model-to-d2.py --path resnet50_miil_21k.pth
```

### Downloading pre-trained visual aggregator
The pretrained model for the visual aggregator is required if one wants to use their own image exemplars to produce a vison-based
classifier. The model can be downloaded from [here](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/visual_aggregator_ckpt_4_transformer.pth.tar) and should be placed in the `${mm-ovod_ROOT}/checkpoints` folder.
```bash
cd ${mm-ovod_ROOT}
mkdir checkpoints
cd checkpoints
wget https://robots.ox.ac.uk/~prannay/public_models/mm-ovod/visual_aggregator_ckpt_4_transformer.pth.tar
tar -xf visual_aggregator_ckpt_4_transformer.pth.tar
rm visual_aggregator_ckpt_4_transformer.pth.tar
```

## Open Vocabulary Generalization Evaluation
### 1.Dataset Preparation

In this task, we use COCO and LVIS. we use the weights trained on LVIS-base to evaluate on COCO to assess the model's generalization capability. Similarly, we use the weights trained on LVIS-base to evaluate on COCO.

### 2.Config Preparation

[LVIS-base Config](https://github.com/prannaykaul/mm-ovod/blob/main/configs/lvis-base_in-l_r50_4x_4x_clip_multi_modal_agg.yaml)

### 3.Weights Preparation

Download the [LVIS-base checkpoint](https://www.robots.ox.ac.uk/~prannay/public_models/mm-ovod/lvis-base_in-l_r50_4x_4x_clip_multi_modal_agg.pth.tar) to the weights directory.


### 4.Evaluation

python train_net_auto.py --num-gpus 4 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth

