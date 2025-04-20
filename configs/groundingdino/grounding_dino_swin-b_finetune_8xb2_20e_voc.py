_base_ = [
    '../grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
]

# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'  # noqa

data_root = '/gpfsdata/home/huangziyue/data/VOC/'
base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                                        'scale_factor', 'text', 'custom_entities', 'caption_prompt')
base_train_pipeline = _base_.train_pipeline
base_train_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                                        'scale_factor', 'flip', 'flip_direction', 'text',
                                        'custom_entities', 'caption_prompt')
class_name = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', )
num_classes = len(class_name)
metainfo = dict(classes=class_name)
caption_prompt = None

model = dict(bbox_head=dict(num_classes=num_classes),
            type='GroundingDINOTextprompt',
            )

train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='coco_annotations/trainval_coco_ann.json',
        data_prefix=dict(img='VOC_trainval/'),
        pipeline=base_train_pipeline,
        caption_prompt=caption_prompt,
        test_mode=False,
        return_classes=True))

val_dataloader = dict(
    batch_size=10,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='coco_annotations/test_coco_ann.json',
        data_prefix=dict(img='VOC_test/'),
        pipeline=base_test_pipeline,
        caption_prompt=caption_prompt,
        test_mode=False,
        return_classes=True))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'coco_annotations/test_coco_ann.json', classwise=True)
test_evaluator = val_evaluator

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=30, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=max_epoch)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(lr_mult=0.),
            'backbone': dict(lr_mult=0.),
            'language_model': dict(lr_mult=0),
            'neck': dict(lr_mult=0.),
            'bbox_head': dict(lr_mult=0.),
            'encoder': dict(lr_mult=0.),
            'decoder': dict(lr_mult=0.),
            'query_embedding': dict(lr_mult=0.),
            'text_feat_map': dict(lr_mult=0.),
            'level_embed': dict(lr_mult=0.),
            'memory_trans_fc': dict(lr_mult=0.),
            'memory_trans_norm': dict(lr_mult=0.),
            'tunable_linear':dict(lr_mult=1.),
        }))

auto_scale_lr = dict(base_batch_size=16)
