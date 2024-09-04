# CLIPTrase

## [ECCV24] Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation

## 1. Introduction
> CLIP, as a vision-language model, has significantly advanced Open-Vocabulary Semantic Segmentation (OVSS) with its zero-shot capabilities. Despite its success, its application to OVSS faces challenges due to its initial image-level alignment training, which affects its performance in tasks requiring detailed local context. Our study delves into the impact of CLIP's [CLS] token on patch feature correlations, revealing a dominance of "global" patches that hinders local feature discrimination. To overcome this, we propose CLIPtrase, a novel training-free semantic segmentation strategy that enhances local feature awareness through recalibrated self-correlation among patches. This approach demonstrates notable improvements in segmentation accuracy and the ability to maintain semantic coherence across objects.
Experiments show that we are 22.3\% ahead of CLIP on average on 9 segmentation benchmarks, outperforming existing state-of-the-art training-free methods.

Full paper and supplementary materials: arxiv



## 2. Code

### 2.1. Environments

+ base environment: pytorch==1.12.1, torchvision==0.13.1 (CUDA11.3)
```
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
+ Detectron2 version: install detectron2==0.6 additionally
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

+ LOCAL

/home/yajie/anaconda2023/envs/cliptrase
+ Source code 
```
git clone https://github.com/leaves162/CLIPtrase.git
```
### 2.2. Training-free OVSS
+ Running with single GPU
```
python clip_self_correlation.py
```
+ Running with multiple GPUs in the detectron2 version
  
  Update: We provide detectron2 framework version, the clip state keys are modified and can be found [here](https://drive.google.com/file/d/1mZtNhYCJzL1jDfc4oO6e7rqbKiKSBGz9/view?usp=drive_link), you can download and put it in `outputs` folder.
  
  Note: The results of the d2 version are slightly different from those in the paper due to differences in preprocessing and resolution.
```
python -W ignore train_net.py --eval-only --config-file configs/clip_self_correlation.yaml --num-gpus 4 OUTPUT_DIR your_output_path MODEL.WEIGHTS your_model_path
```


## Citation 
+ If you find this project useful, please consider citing:
```
@InProceedings{shao2024explore,
    title={Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation},
    author={Tong Shao and Zhuotao Tian and Hang Zhao and Jingyong Su},
    booktitle={European Conference on Computer Vision},
    organization={Springer},
    year={2024}
}
```


