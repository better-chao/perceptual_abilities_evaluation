# Extract Free Dense Labels from CLIP [[Project Page](https://www.mmlab-ntu.com/project/maskclip/)]
```
        ███╗   ███╗ █████╗ ███████╗██╗  ██╗ ██████╗██╗     ██╗██████╗
        ████╗ ████║██╔══██╗██╔════╝██║ ██╔╝██╔════╝██║     ██║██╔══██╗
        ██╔████╔██║███████║███████╗█████╔╝ ██║     ██║     ██║██████╔╝
        ██║╚██╔╝██║██╔══██║╚════██║██╔═██╗ ██║     ██║     ██║██╔═══╝
        ██║ ╚═╝ ██║██║  ██║███████║██║  ██╗╚██████╗███████╗██║██║
        ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝╚═╝
```
This is the code for our paper: [Extract Free Dense Labels from CLIP](https://arxiv.org/abs/2112.01071).

This repo is a fork of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). So the installation and data preparation is pretty similar.

# Installation
**Step 0.** Install PyTorch and Torchvision following [official instructions](https://pytorch.org/get-started/locally/), e.g.,

```shell
pip install torch torchvision
# FYI, we're using torch==1.9.1 and torchvision==0.10.1
```

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).
```shell
pip install -U openmim
mim install mmcv-full
```

**Step 2.** Install [CLIP](https://github.com/openai/CLIP).
```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

**Step 3.** Install MaskCLIP.
```shell
git clone https://github.com/chongzhou96/MaskCLIP.git
cd MaskCLIP
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## Environment
+ LOCAL

    /home/yajie/anaconda2023/envs/tcl

# Dataset Preparation
Please refer to [dataset_prepare.md](docs/en/dataset_prepare.md#prepare-datasets). In our paper, we experiment with [Pascal VOC](docs/en/dataset_prepare.md#pascal-voc), [Pascal Context](docs/en/dataset_prepare.md#pascal-context), and [COCO Stuff 164k](docs/en/dataset_prepare.md#coco-stuff-164k).

# MaskCLIP
MaskCLIP doesn't require any training. We only need to (1) download and convert the CLIP model and (2) prepare the text embeddings of the objects of interest.

**Step 0.** Download and convert the CLIP models, e.g.,
```shell
mkdir -p pretrain
python tools/maskclip_utils/convert_clip_weights.py --model ViT16 --backbone
# Other options for model: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT32, ViT16, ViT14
```

**Step 1.** Prepare the text embeddings of the objects of interest, e.g.,
```shell
python tools/maskclip_utils/prompt_engineering.py --model ViT16 --class-set context
# Other options for model: RN50, RN101, RN50x4, RN50x16, ViT32, ViT16
# Other options for class-set: voc, context, stuff
# Actually, we've played around with many more interesting target classes. (See prompt_engineering.py)
```

**Step 2.** Get quantitative results (mIoU):
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU
# e.g., python tools/test.py configs/maskclip/maskclip_vit16_520x520_pascal_context_59.py pretrain/ViT16_clip_backbone.pth --eval mIoU
```

**Step 3. (optional)** Get qualitative results:
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${OUTPUT_DIR}
# e.g., python tools/test.py configs/maskclip/maskclip_vit16_520x520_pascal_context_59.py pretrain/ViT16_clip_backbone.pth --show-dir output/
```

# MaskCLIP+
MaskCLIP+ trains another segmentation model with pseudo labels extracted from MaskCLIP.

**Step 0.** Download and convert the CLIP models, e.g.,
```shell
mkdir -p pretrain
python tools/maskclip_utils/convert_clip_weights.py --model ViT16
# Other options for model: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT32, ViT16, ViT14
```

**Step 1.** Prepare the text embeddings of the target dataset, e.g.,
```shell
python tools/maskclip_utils/prompt_engineering.py --model ViT16 --class-set context
# Other options for model: RN50, RN101, RN50x4, RN50x16, ViT32, ViT16
# Other options for class-set: voc, context, stuff
```

**Train.** Depending on your setup (single/mutiple GPU(s), multiple machines), the training script can be different. Here, we give an example of multiple GPUs on a single machine. For more infomation, please refer to [train.md](docs/en/train.md).
```shell
sh tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
# e.g., sh tools/dist_train.sh configs/maskclip_plus/zero_shot/maskclip_plus_r50_deeplabv3plus_r101-d8_480x480_40k_pascal_context.py 4
```

**Inference.** See step 2 and step 3 under the MaskCLIP section. (We will release the trained models soon.)


# Citation
If you use MaskCLIP or this code base in your work, please cite
```
@InProceedings{zhou2022maskclip,
    author = {Zhou, Chong and Loy, Chen Change and Dai, Bo},
    title = {Extract Free Dense Labels from CLIP},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2022}
}
```

# Contact
For questions about our paper or code, please contact [Chong Zhou](mailto:chong033@ntu.edu.sg).