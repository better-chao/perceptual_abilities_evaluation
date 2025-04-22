_base_ = [
    'datasets/ov_lvis.py',
    # 'models/vild_ensemble_faster_rcnn_r50_fpn.py',
    # 'models/block.py',
    'models/oadp_faster_rcnn_r50_fpn.py',
    'models/mask.py',
    'schedules/2x.py',
    'base.py',
]

cls_predictor_cfg = dict(
    type='ViLDClassifier',
    prompts='data/prompts/detpro_lvis.pth',
    scaler=dict(
        train=0.01,
        val=0.007,
    ),
)
model = dict(
    # type='OADP',
    global_head=dict(
        classifier=dict(
            type='ViLDClassifier',
            prompts='data/prompts/detpro_lvis.pth',
            out_features=1203,
        ),
    ),
    roi_head=dict(
        # type='OADPRoIHead',
        bbox_head=dict(cls_predictor_cfg=cls_predictor_cfg),
        object_head=dict(cls_predictor_cfg=cls_predictor_cfg),
        block_head=dict(cls_predictor_cfg=cls_predictor_cfg),
    ),
)
