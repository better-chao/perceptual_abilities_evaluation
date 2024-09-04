# Transferable and Principled Efficiency for Open-Vocabulary Segmentation



[Transferable and Principled Efficiency for Open-Vocabulary Segmentation](https://arxiv.org/abs/2404.07448)

<!-- [ALGORITHM] -->

## Abstract

<!-- Open-Vocabulary Segmentation approach often introduces heavy computational overheads for two challenges: 1) large model sizes of the backbone; 2) expensive costs during the fine-tuning. Traditional efficiency methods such as model compression and efficient fine-tuning can address these challenges, but their solutions cannot be easily transferred and necessitate re-training on different models.  -->
OpenTrans propose a transferable open-vocabulary segmentation technic, to establish a principled and seamlessly transferable efficiency across various OVS frameworks. First, to address the large model size of the CLIP image encoder, OpenTrans explore small subnetwork by iterative magnitude pruning. Second, to address the heavy OVS fine-tuning computation problem, OpenTrans proposes to select only some layers at the lighter tail of the pre-trained weight spectrum for updating.

![](../../images/opentrans-overview.png)
 
## Installation
We install OpenTrans using the official [github repository](https://github.com/Xujxyang/OpenTrans) and follow the [instructions](https://github.com/bytedance/fc-clip/blob/main/INSTALL.md) to configure the environment.

### Prepare Datasets
A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

FC-CLIP has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  # panoptic datasets
  ADEChallengeData2016/
  coco/
  cityscapes/
  mapillary_vistas/
  # semantic datasets
  VOCdevkit/
  ADE20K_2021_17_01/
  pascal_ctx_d2/
  pascal_voc_d2/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.


### Getting Start
When you have prepared the environment and dataset according to the description of fc-clip, you also need to configure the method for running the selective layer based on the requirements we provide. In addition, you would need to download the pre-trained weights for CLIP-RT50.\
Let's get started!\
Please use the following command to load the data path:
```bash
export DETECTRON2_DATASETS= ******
```
Then, you can run our pruning method using the following statement:
```bash
python3 plain_train_net_imp.py --resume --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKD OUTPUT_DIR rebuttal_test MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 2.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED True
```
If it is the first step of training and you need to reference mask2former/maskformer_model_voca_dense_kd.py or use the command line to disable other unnecessary losses and only use kd_loss.\
You can use the following command to perform layer-wise training:
```bash
python3 plain_train_net.py --resume --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegMaskFormerDenseTextKD OUTPUT_DIR rebuttla_clip MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.PER_REGION.PLAIN_LOSS_WEIGHT 2.0 MODEL.KD.KD_WEIGHT 2.0 MODEL.GROUNDING.LOSS_WEIGHT 2.0 MODEL.KD.ENABLED True MODEL.WEIGHTS path/to/pruned_weight
```
If you want to activate both strategies simultaneously, you can enable layer-wise in plain_train_net_imp.py:\
The specific algorithms for these two strategies can be found in the pruning.py and utils.py files. You can adjust them according to your specific requirements.\
To perform the transfer experiment to DeepLab, you can use the following command:
```bash
python3 plain_train_net_deeplab.py --resume --num-gpus 4 --config-file configs/coco/panoptic-segmentation/Deeplabv3_R50_bs16_50ep_voca_ground_perRegion.yaml MODEL.META_ARCHITECTURE OpenSegDeeplabv3Text OUTPUT_DIR test MODEL.KD.STU_SOURCE prior MODEL.KD.TEC_SOURCE attn_g INPUT.DATASET_MAPPER_NAME detr_panoptic_caption_kd MODEL.PER_REGION.DENSE_SUPERVISE True MODEL.WEIGHTS path/to/pruned_weight
```
After training, you can evaluate all the model using the following command:
```bash
python3 train_net.py --num-gpus 4 --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep_voca_ground_perRegion.yaml --eval-only DATASETS.TEST '("coco_2017_val_panoptic_caption",)' MODEL.META_ARCHITECTURE OpenSegMaskFormer MODEL.WEIGHTS path/to/your_weight
```
To adjust the code for the pruning process and transfer, please refer to the *do_train* function in the *plain_train_net* script.
If you want to load our subnetwork into any other detectron2-based model(such as fc-clip), we provide the following hook to assist you:
```python
class ExtraOperationsHook(HookBase):
    def __init__(self, trainer, first_iter, second_iter):
        super().__init__()
        self.trainer = trainer
        self.first_iter = first_iter
        self.second_iter = second_iter
        self.a = 0

    def after_step(self):
        current_iter = self.trainer.iter
        if current_iter == self.first_iter:
            # load mask
            print("load mask", current_iter)
            pruning.see_zero_rate(self.trainer.model.module.backbone.clip_model.visual)
            load_name_a = './pruned_mask/seg_backbone_imp{}.pth'.format(21)
            mask_ckpt_a = torch.load(load_name_a, map_location="cuda")
            mask_dict_a = pruning.extract_mask(mask_ckpt_a['backbone'])
            pruning.imagenet_pruning_model_custom_res50v1(self.trainer.model.module.backbone.clip_model.visual, mask_dict_a)
            pruning.see_zero_rate(self.trainer.model.module.backbone.clip_model.visual)
        if current_iter == self.second_iter:
            # remove mask
            pruning.see_zero_rate(self.trainer.model.module.backbone.clip_model.visual)
            print("remove prune and save", current_iter)
            pruning.imagenet_remove_model_custom_res50v1(self.trainer.model.module.backbone.clip_model.visual, False)
            pruning.see_zero_rate(self.trainer.model.module.backbone.clip_model.visual)
            # save the model
            self.trainer.checkpointer.save("model_{:07d}_needed".format(current_iter))
```