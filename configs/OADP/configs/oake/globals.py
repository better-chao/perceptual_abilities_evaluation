_base_ = [
    'base.py',
]

train = dict(
    dataloader=dict(
        dataset=dict(output_dir='data/coco/oake/globals/train2017'),
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(output_dir='data/coco/oake/globals/val2017'),
    ),
)
log = dict(interval=50)
