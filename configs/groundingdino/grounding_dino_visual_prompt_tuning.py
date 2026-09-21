# -*- coding: utf-8 -*-
# Visual Prompt Fine-tuning for GroundingDINO: the unified feature-space prompt.
# Built on the closed-set COCO schedule (12 epochs, step 11); swap the dataset
# section to run it on any other benchmark of this study.
_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

custom_imports = dict(
    imports=['configs.groundingdino.grounding_dino_visual_prompt_module'],
    allow_failed_imports=False)

model = dict(
    type='GroundingDINO_VisualPrompt',
    prompt_size=50,
    # one 50x50 tensor per neck level; must match model.neck.num_outs/out_channels
    prompt_channels=[256, 256, 256, 256])

# every module is frozen by a zero learning-rate multiplier; only the prompt trains
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.),
            'neck': dict(lr_mult=0.),
            'bbox_head': dict(lr_mult=0.),
            'language_model': dict(lr_mult=0.),
            'encoder': dict(lr_mult=0.),
            'decoder': dict(lr_mult=0.),
            'query_embedding': dict(lr_mult=0.),
            'memory_trans_fc': dict(lr_mult=0.),
            'visual_prompt': dict(lr_mult=1.),
        }),
    clip_grad=None)

work_dir = 'work_dirs/grounding_dino_coco_visual_prompt'

# no best-checkpoint selection: every result in the paper is read from the
# final-epoch checkpoint
default_hooks = dict(checkpoint=dict(interval=1, save_best=None))
