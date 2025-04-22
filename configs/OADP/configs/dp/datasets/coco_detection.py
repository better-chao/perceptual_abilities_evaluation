dataset_type = 'CocoDataset'
data_root = 'data/coco/'
ann_file_prefix = data_root + 'annotations/instances_'
norm = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
trainer = dict(
    dataloader=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_prefix + 'train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Normalize', **norm),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ],
        ),
    ),
)
validator = dict(
    dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_prefix + 'val2017.json',
            img_prefix=data_root + 'val2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(1333, 800),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(type='Normalize', **norm),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img']),
                    ],
                ),
            ],
        ),
    ),
)
