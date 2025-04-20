_base_ = [
    '../grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
]

# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'  # noqa

data_root = '/gpfsdata/home/huangziyue/data/coco_new/'
base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                                        'scale_factor', 'text', 'custom_entities', 'caption_prompt')
base_train_pipeline = _base_.train_pipeline
base_train_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                                        'scale_factor', 'flip', 'flip_direction', 'text',
                                        'custom_entities', 'caption_prompt')
class_name = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', )
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
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=base_train_pipeline,
        caption_prompt=caption_prompt,
        test_mode=False,
        return_classes=True))

val_dataloader = dict(
    batch_size=10,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        pipeline=base_test_pipeline,
        caption_prompt=caption_prompt,
        test_mode=False,
        return_classes=True))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json', classwise=True)
test_evaluator = val_evaluator

max_epoch = 10

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

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
# ./tools/dist_train.sh my_groundingdino/configs/text_prompt/grounding_dino_swin-b_finetune_8xb2_20e_coco.py 3 --work-dir work_dir/coco-tp