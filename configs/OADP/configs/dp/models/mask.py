model = dict(
    roi_head=dict(
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            class_agnostic=True,
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_mask=True,
                loss_weight=1.0,
            ),
            num_classes=None,
        ),
    ),
    train_cfg=dict(rcnn=dict(mask_size=28)),
    test_cfg=dict(rcnn=dict(mask_thr_binary=0.5)),
)
