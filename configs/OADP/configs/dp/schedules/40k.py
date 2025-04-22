_base_ = [
    'base.py',
]

interval = 2000
trainer = dict(
    lr_config=dict(
        by_epoch=False,
        step=[30000],
    ),
    runner=dict(type='IterBasedRunner', max_iters=4e4),
    workflow=[('train', interval)],
    checkpoint_config=dict(
        by_epoch=False,
        interval=interval,
        save_last=True,
    ),
    evaluation=dict(interval=interval),
)
