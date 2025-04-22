_base_ = [
    'objects_coco.py',
]

train = dict(
    dataloader=dict(
        dataset=dict(
            type="LVISDataset",
            root='data/coco',
            annFile='data/lvis_v1/annotations/lvis_v1_train.json',
            output_dir='data/lvis_v1/oake/objects/train2017',
            proposal_file='data/lvis_v1/proposals/oln_r50_fpn_lvis_train.pkl',
        ),
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(
            type="LVISDataset",
            root='data/coco',
            annFile='data/lvis_v1/annotations/lvis_v1_val.json',
            output_dir='data/lvis_v1/oake/objects/val2017',
            proposal_file='data/lvis_v1/proposals/oln_r50_fpn_lvis_val.pkl',
        ),
    ),
)
