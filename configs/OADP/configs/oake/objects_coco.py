_base_ = [
    'base.py',
]

train = dict(
    dataloader=dict(
        dataset=dict(
            type="COCODataset",
            output_dir='data/coco/oake/objects/train2017',
            proposal_file='data/coco/proposals/oln_r50_fpn_coco_train.pkl',
            proposal_sorted=True,
        ),
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(
            type="COCODataset",
            output_dir='data/coco/oake/objects/val2017',
            proposal_file='data/coco/proposals/oln_r50_fpn_coco_val.pkl',
            proposal_sorted=True,
        ),
    ),
)
log = dict(interval=5)
mini_batch_size = 512
