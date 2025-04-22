model = dict(
    roi_head=dict(
        block_head=dict(
            type='Shared2FCBlockBBoxHead',
            topk=5,
            loss=dict(
                type='AsymmetricLoss',
                weight=dict(type='WarmupScheduler', gain=16, end=1000),
                gamma_neg=4,
                gamma_pos=0,
            ),
        ),
    ),
    distiller=dict(
        student_hooks=dict(
            blocks=dict(
                inputs=tuple(),
                action=dict(
                    type='StandardHook',
                    path='.roi_head._block_head.fc_cls._linear',
                ),
            ),
        ),
        losses=dict(
            loss_clip_blocks=dict(
                inputs=('blocks', 'clip_blocks'),
                action=dict(
                    type='L1Loss',
                    weight=dict(type='WarmupScheduler', gain=128, end=200),
                ),
            ),
            loss_clip_block_relations=dict(
                inputs=('blocks', 'clip_blocks'),
                action=dict(
                    type='RKDLoss',
                    weight=dict(type='WarmupScheduler', gain=8, end=200),
                ),
            ),
        ),
    ),
)
