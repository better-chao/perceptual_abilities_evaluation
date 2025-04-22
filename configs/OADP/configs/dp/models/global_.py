model = dict(
    global_head=dict(
        topk=20,
        classifier=dict(in_features=256),
        loss=dict(
            type='AsymmetricLoss',
            weight=dict(type='WarmupScheduler', gain=4, end=2000),
            gamma_neg=4,
            gamma_pos=0,
        ),
    ),
    distiller=dict(
        student_hooks=dict(
            global_=dict(
                inputs=tuple(),
                action=dict(
                    type='StandardHook',
                    path='._global_head._classifier._linear',
                ),
            ),
        ),
        losses=dict(
            loss_clip_global=dict(
                inputs=('global_', 'clip_global'),
                action=dict(
                    type='MSELoss',
                    weight=dict(type='WarmupScheduler', gain=0.5, end=200),
                    # FIXME: in todd, sum should multiply by number of GPUs
                    reduction='sum',
                ),
            ),
        ),
    ),
)
