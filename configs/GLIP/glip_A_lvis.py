_base_ = 'mmdet::glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'

lang_model_name = '/gpfsdata/home/buaa_liuchenguang/TTA_workspace/GroundingDINO/bert-no-uncased'

dataset_type = 'LVISV1Dataset'
data_root = '/gpfsdata/home/buaa_liuchenguang/github_eval/OpenDataLab___CrowdHuman/raw/CrowdHuman/crowdhuman/'

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth'  # noqa

backend_args = None

class_name = ('pedestrian')

palette = [(220, 20, 60)]

metainfo = dict(classes=class_name, palette=palette)

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='pillow',
        backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='GTBoxSubOne_GLIP'),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 480), (1333, 560), (1333, 640), (1333, 720),
                (1333, 800)],
        keep_ratio=True,
        resize_type='FixScaleResize',
        backend='pillow'),
    dict(type='RandomFlip_GLIP', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

'''
dict(
        type='RandomSamplingNegPos',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
'''

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/crowdhuman_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
        return_classes=True,
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/crowdhuman_val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=test_pipeline,
        backend_args=backend_args,
        return_classes=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root + 'annotations/lvis_od_val.json')
test_evaluator = val_evaluator

# ----------------------------------

model = dict(
    language_model=dict(
        name=lang_model_name,
    ),
    bbox_head=dict(
        lang_model_name=lang_model_name,
    ),
    test_cfg=dict(
        max_per_img=300,
        chunked_size=40)
)


work_dir = 'work_dirs/glip_A'
