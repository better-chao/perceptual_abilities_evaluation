_base_ = [
    'datasets/ov_coco.py',
    'models/vild_ensemble_faster_rcnn_r50_fpn.py',
    'schedules/40k.py',
    'base.py',
]

cls_predictor_cfg = dict(
    type='ViLDClassifier',
    prompts='data/prompts/vild.pth',
    scaler=dict(
        train=0.01,
        val=0.007,
    ),
)
model = dict(
    roi_head=dict(
        bbox_head=dict(cls_predictor_cfg=cls_predictor_cfg),
        object_head=dict(cls_predictor_cfg=cls_predictor_cfg),
    ),
)
