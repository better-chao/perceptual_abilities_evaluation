_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/default_runtime.py'
]
# find_unused_parameters=True
auto_scale_lr = dict(enable=True, base_batch_size=256)
norm_cfg = dict(type='SyncBN', requires_grad=True)
head_norm_cfg = dict(type='MMSyncBN', requires_grad=True)

num_classes = 1203
# model settings
model = dict(
    type='Fvlm',
    backbone_name='RN50x64',
    use_res_feature=False,
    backbone=dict(
        _delete_=True,
        type='ModifiedResNet',
        layers=(3,15,36,10),
        output_dim=1024,
        heads=64,
        input_resolution=448,
        width=128
    ),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048, 4096],
        out_channels=256, # 256 or 512
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        type='FvlmRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        vlm_roi_extractor = dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=4096,
            featmap_strides=[32]),
        bbox_head=dict(
            type='FvlmBBoxHead',
            norm_cfg=head_norm_cfg,
            fixed_temperature=0,
            learned_temperature=50.0,
            vlm_temperature=100.0,
            alpha=0.35,
            beta=0.65,
            attnpool=dict(
                type='AttentionPool2d',
                spacial_dim=14,
                embed_dim=4096,
                num_heads=64,
                output_dim=1024),
            class_embed="datasets/embeddings/lvis_v1_with_background_clip_rn50x64_multi_emb.pt",
            seen_classes='datasets/lvis_v1_seen_classes.json',
            unseen_classes='datasets/lvis_v1_unseen_classes.json',
            all_classes='datasets/lvis_v1_all_classes.json',
            num_classes=num_classes,
            with_cls=False,
            num_shared_convs=4,
            num_shared_fcs=2,
            num_cls_fcs=1,
            num_reg_fcs=1,
            reg_class_agnostic=True,
            fc_out_channels=1024,
            loss_cls=dict(
                type='CrossEntropyLoss',
                class_weight=[1.0] * num_classes + [0.9],
                loss_weight=1.0
            )
        ),
        mask_head = dict(
            type='FCNMaskHead',
            num_classes=num_classes,
            class_agnostic=True,
            norm_cfg=head_norm_cfg,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5)))


optimizer = dict(type='SGD', lr=0.36, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.009,
    step=[2304*2, 2592*2, 2736*2])
evaluation = dict(interval=2880*2, metric=['segm'])
checkpoint_config = dict(interval=1000)
runner = dict(type='IterBasedRunner', max_iters=2880*2)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])



dataset_type = 'LVISV1DatasetOV'
data_root = 'data/lvis_v1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024, 1024)

file_client_args = dict(backend='disk')

# Standard Scale Jittering (SSJ) resizes and crops an image
# with a resize range of 0.8 to 1.25 of the original image size.
load_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size=image_size),
]
train_pipeline = [
    # dict(type='CopyPaste', max_num_pasted=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v1_train_seen_1203_cat.json',
            img_prefix=data_root,
            pipeline=load_pipeline),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline))



custom_imports = dict(
    imports=[
        'datasets',
        'models.dst_det',
    ],
    allow_failed_imports=False)