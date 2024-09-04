# perceptual_abilities_evaluation of PADing
This is the official code for the [PADing](https://openaccess.thecvf.com/content/CVPR2023/papers/He_Primitive_Generation_and_Semantic-Related_Alignment_for_Universal_Zero-Shot_Segmentation_CVPR_2023_paper.pdf) (CVPR 2023).  

Primitive Generation and Semantic-related Alignment for Universal Zero-Shot Segmentation study universal zero-shot segmentation to achieve 1)panoptic, 2)instance, 3)and semantic segmentation for novel categories without any training samples.  
## Abstract
We study universal zero-shot segmentation in this work to achieve panoptic, instance, and semantic segmentation for novel categories without any training samples. Such zero-shot segmentation ability relies on inter-class relationships in semantic space to transfer the visual knowledge learned from seen categories to unseen ones. Thus, it is desired to well bridge semantic-visual spaces and apply the semantic relationships to visual feature learning. We introduce a generative model to synthesize features for unseen categories, which links semantic and visual spaces as well as address the issue of lack of unseen training data. Furthermore, to mitigate the domain gap between semantic and visual spaces, firstly, we enhance the vanilla generator with learned primitives, each of which contains fine-grained attributes related to categories, and synthesize unseen features by selectively assembling these primitives. Secondly, we propose to disentangle the visual feature into the semantic-related part and the semantic-unrelated part that contains useful visual classification clues but is less relevant to semantic representation. The inter-class relationships of semantic-related visual features are then required to be aligned with those in semantic space, thereby transferring semantic knowledge to visual feature learning. The proposed approach achieves impressively state-of-theart performance on zero-shot panoptic segmentation, instance segmentation, and semantic segmentation.Code will be released at https://github.com/heshuting555/PADing.

![image](https://github.com/heshuting555/PADing/blob/main/imgs/framework.png)
## Installation
### Requirements
The code is tested under CUDA 11.1, Pytorch 1.9.0 and Detectron2 0.6.   
1. Install [Detectron2](https://github.com/facebookresearch/detectron2) following the [manual](https://detectron2.readthedocs.io/en/latest/)  
2. Run `sh make.sh` under `PADing/modeling/pixel_decoder/ops` (Note: 1-2 steps you can also follow the installation process of [Mask2Former](https://github.com/facebookresearch/Mask2Former))  
3. Install other required packages: pip install -r requirements.txt  
4. Prepare the dataset following [datasets/README.md](https://github.com/heshuting555/PADing/blob/main/datasets/README.md)  
An example of installation is shown below:  
```
conda create -n pading python==3.7  
conda activate pading  
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html  
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html  

git clone https://github.com/heshuting555/PADing.git 
cd PADing  
pip install -r requirements.txt  
```  
## Small target segmentation
Here we evaluate the performance of PADing on small target segmentation, taking the dataset as an example.  
### Dataset Prepare
See [data preparation](https://github.com/heshuting555/PADing/blob/main/datasets/README.md).  
### Inference
For example  
```  
CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --config-file configs/panoptic-segmentation/PADing.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [output_dir]  
```  
### Training  
Firstly, download the pretrained weights [here](https://drive.google.com/drive/folders/1ynhW1vc_KpLQC_O1MrSuRt4dn8ZYTwa4?usp=sharing) or you can train vanilla mask2former backbone using seen classes and convert it using the following command:   
For example  
```  
python train_net_pretrain.py --config-file configs/panoptic-segmentation/pretrain.yaml --num-gpus 8

python tools/preprocess_pretrained_weight.py --task_name panoptic --input_file panoptic_pretrain/model_final.pth  
```
### Trained Models and logs
Download pretrained weights [here](https://drive.google.com/drive/folders/1ynhW1vc_KpLQC_O1MrSuRt4dn8ZYTwa4?usp=sharing).

Download final trained PADing weights for inference [here](https://drive.google.com/drive/folders/1ynhW1vc_KpLQC_O1MrSuRt4dn8ZYTwa4?usp=sharing).
