_base_ = [
    'base.py',
]

trainer = dict(
    lr_config=dict(step=[16, 19]),
    runner=dict(type='EpochBasedRunner', max_epochs=24),
    workflow=[('train', 1)],
    checkpoint_config=dict(interval=1),
    evaluation=dict(interval=4),
)
