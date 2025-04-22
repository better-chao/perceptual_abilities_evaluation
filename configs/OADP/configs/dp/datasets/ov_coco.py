_base_ = [
    'coco_detection.py',
]

categories = 'coco'
dataset_type = 'OV_COCO'
data_root = 'data/coco/'
oake_root = data_root + 'oake/'
ann_file_prefix = data_root + 'annotations/instances_'
norm = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
trainer = dict(
    dataloader=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_prefix + 'train2017.48.json',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='LoadCLIPFeatures',
                    default=dict(
                        task_name='train2017',
                        type='PthAccessLayer',
                    ),
                    globals_=dict(data_root=oake_root + 'globals'),
                    blocks=dict(data_root=oake_root + 'blocks'),
                    objects=dict(data_root=oake_root + 'objects'),
                ),
                dict(
                    type='Resize',
                    img_scale=[(1330, 640), (1333, 800)],
                    multiscale_mode='range',
                    keep_ratio=True,
                ),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Normalize', **norm),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='ToTensor',
                    keys=['block_bboxes', 'block_labels', 'object_bboxes'],
                ),
                dict(
                    type='ToDataContainer',
                    fields=[
                        dict(key='clip_blocks'),
                        dict(key='block_bboxes'),
                        dict(key='block_labels'),
                        dict(key='clip_objects'),
                        dict(key='object_bboxes'),
                    ],
                ),
                dict(
                    type='Collect',
                    keys=[
                        'img', 'gt_bboxes', 'gt_labels', 'clip_global',
                        'clip_blocks', 'block_bboxes', 'block_labels',
                        'clip_objects', 'object_bboxes'
                    ],
                ),
            ],
        ),
    ),
)
validator = dict(
    dataloader=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_prefix + 'val2017.65.min.json',
        ),
    ),
)
