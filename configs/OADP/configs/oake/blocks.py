_base_ = [
    'base.py',
]

train = dict(
    dataloader=dict(
        dataset=dict(output_dir='data/coco/oake/blocks/train2017'),
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(output_dir='data/coco/oake/blocks/val2017'),
    ),
)
log = dict(interval=10)
