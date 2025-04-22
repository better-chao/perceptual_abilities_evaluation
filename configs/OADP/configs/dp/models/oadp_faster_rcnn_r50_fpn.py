_base_ = [
    'vild_ensemble_faster_rcnn_r50_fpn.py',
    'global_.py',
    'block.py',
]

model = dict(
    type='OADP',
    roi_head=dict(type='OADPRoIHead'),
)
