_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

data_root = 'data/weather/daytime_clear/'
# test_data_root = 'data/weather/daytime_clear/'
# test_data_root = 'data/weather/daytime_foggy/'
# test_data_root = 'data/weather/dusk_rainy/'
# test_data_root = 'data/weather/night_clear/'
test_data_root = 'data/weather/night_rainy/'
base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                                        'scale_factor', 'text', 'custom_entities')
base_train_pipeline = _base_.train_pipeline
base_train_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                                        'scale_factor', 'flip', 'flip_direction', 'text',
                                        'custom_entities')
class_name = ("bus", "bike", "car", "motor", "person", "rider", "truck", )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[
    (220, 20, 60),   # 深红色
    (0, 255, 0),     # 绿色
    (0, 0, 255),     # 蓝色
    (255, 255, 0),   # 黄色
    (255, 165, 0),   # 橙色
    (128, 0, 128),   # 紫色
    (0, 255, 255)    # 青色
    ])
model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='voc07_train.json',
        data_prefix=dict(img=''),
        pipeline=base_train_pipeline, 
        test_mode=False,
        return_classes=True))

val_dataloader = dict(
    batch_size=12,
    dataset=dict(
        metainfo=metainfo,
        data_root=test_data_root,
        ann_file='voc07_test.json',
        data_prefix=dict(img=''),
        pipeline=base_test_pipeline, 
        test_mode=False,
        return_classes=True))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=test_data_root + 'voc07_test.json')
test_evaluator = val_evaluator

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
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
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)
