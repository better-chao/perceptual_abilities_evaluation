save_interval = -1
val_interval = 4
train_batch_size_per_gpu = 2
val_batch_size_per_gpu = 2
base_lr = 0.01
max_epochs = 40
max_keep_ckpts = 0
close_mosaic_epochs = 20

data_root = 'data/VOC'
train_ann_file = 'coco_annotations/trainval_coco_ann.json'
train_data_prefix = 'VOC_trainval'

val_ann_file = 'coco_annotations/test_coco_ann.json'
val_data_prefix = 'VOC_test'
class_text_path = 'data/texts/voc_class_texts.json'

class_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

num_classes = len(class_name)
num_training_classes = num_classes

metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)]
)

img_scale = (1024, 1024)  # width, height

########################################################
# ======================default_runtime======================
custom_imports = dict(
    imports=['yolo_world',
             'M_yolow.datasets.transforms.transforms'],
    allow_failed_imports=False)
default_scope = 'mmyolo'
randomness = dict(seed=2024)
seed = 2024
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = './checkpoints/YOLOW/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth'
resume = False
backend_args = None
# ========================Frequently modified parameters======================
train_num_workers = 2

filter_empty_gt = True
persistent_workers = True

# -----data related-----
dataset_type = 'YOLOv5CocoDataset'

val_num_workers = 2
batch_shapes_cfg = None

# -----train val related-----
affine_scale = 0.5
max_aspect_ratio = 100
lr_factor = 0.01
weight_decay = 0.0005


# ===============================Unmodified in most cases====================
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotationsWithoutMask', with_bbox=True),
]
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]


train_pipeline = [
    *pre_transform,
    dict(
        type='MultiModalMosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotationsWithoutMask', with_bbox=True),
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='RandomLoadText',
         num_neg_samples=(num_training_classes, num_training_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='RandomLoadText',
         num_neg_samples=(num_training_classes, num_training_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotationsWithoutMask', with_bbox=True),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='yolow_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=data_root,
            metainfo=metainfo,
            ann_file=train_ann_file,
            data_prefix=dict(img=train_data_prefix),
            filter_cfg=dict(filter_empty_gt=True, min_size=32)),
        class_text_path=class_text_path,
        pipeline=train_pipeline
    ))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=data_root,
            metainfo=metainfo,
            ann_file=val_ann_file,
            data_prefix=dict(img=val_data_prefix),
            filter_cfg=dict(filter_empty_gt=filter_empty_gt, min_size=32)),
        class_text_path=class_text_path,
        pipeline=test_pipeline)
)

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 100),
    ann_file=data_root + '/' + val_ann_file,
    classwise=True,
    metric='bbox')

# Inference on val dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu,
    ),
    constructor='YOLOv5OptimizerConstructor',
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'backbone.text_model': dict(lr_mult=0.01),
    #         'logit_scale': dict(weight_decay=0.0)
    #     }),
)

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_interval,
        save_best=None,
        max_keep_ckpts=max_keep_ckpts))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        4)])


val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

########################################################

model = dict(
    backbone=dict(
        image_model=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            arch='P5',
            deepen_factor=1.0,
            last_stage_out_channels=512,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            type='YOLOv8CSPDarknet',
            widen_factor=1.0),
        text_model=dict(
            frozen_modules=[
                'all',
            ],
            model_name='./checkpoints/clip-vit-base-patch32',
            type='HuggingCLIPLanguageBackbone'),
        type='MultiModalYOLOBackbone'),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            embed_dims=512,
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                512,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=num_classes,
            reg_max=16,
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            widen_factor=1.0),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOWorldHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOWDetDataPreprocessor'),
    mm_neck=True,
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        deepen_factor=1.0,
        embed_channels=[
            128,
            256,
            256,
        ],
        guide_channels=512,
        in_channels=[
            256,
            512,
            512,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        num_heads=[
            4,
            8,
            8,
        ],
        out_channels=[
            256,
            512,
            512,
        ],
        type='YOLOWorldPAFPN',
        widen_factor=1.0),
    num_test_classes=num_classes,
    num_train_classes=num_classes,
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=num_classes,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLOWorldDetector')
