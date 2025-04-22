train = dict(
    dataloader=dict(
        dataset=dict(
            root='data/coco/train2017',
            annFile='data/coco/annotations/instances_train2017.json',
        ),
        num_workers=2,
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(
            root='data/coco/val2017',
            annFile='data/coco/annotations/instances_val2017.json',
        ),
        num_workers=2,
    ),
)
