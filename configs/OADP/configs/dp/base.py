trainer = dict(
    log_config=dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False),
        ],
    ),
    custom_hooks=[dict(type='NumClassCheckHook')],
    fp16=dict(loss_scale=dict(init_scale=64.0)),
    log_level='INFO',
    resume_from=None,
    load_from='pretrained/soco/soco_star_mask_rcnn_r50_fpn_400e.pth',
    seed=3407,
    optimizer=dict(weight_decay=2.5e-5),
)
validator = dict(
    # fp16=True,
    fp16=False,
)
