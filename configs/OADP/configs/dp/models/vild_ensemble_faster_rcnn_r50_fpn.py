_base_ = [
    'faster_rcnn_r50_fpn.py',
]

model = dict(
    type='ViLD',
    backbone=dict(style='caffe', init_cfg=None),
    neck=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
    roi_head=dict(
        type='ViLDEnsembleRoIHead',
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            num_classes=None,
            reg_class_agnostic=True,
        ),
        object_head=dict(type='Shared4Conv1FCObjectBBoxHead'),
    ),
    distiller=dict(
        type='SelfDistiller',
        student_hooks=dict(
            objects=dict(
                inputs=tuple(),
                action=dict(
                    type='StandardHook',
                    path='.roi_head._object_head.fc_cls._linear',
                ),
            ),
        ),
        adapts=dict(),
        losses=dict(
            loss_clip_objects=dict(
                inputs=('objects', 'clip_objects'),
                action=dict(
                    type='L1Loss',
                    weight=dict(type='WarmupScheduler', gain=256, end=200),
                ),
            ),
        ),
    ),
    test_cfg=dict(rcnn=dict(
        score_thr=0.0,
        max_per_img=300,
    )),
)
