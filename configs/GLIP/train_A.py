# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo
from glip_head_prompt_tuning import ATSSVLFusionHead_PromptTuning


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='configs/glip/glip_A_long.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(custom_args):
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif custom_args['work_dir'] is not None:
        cfg.work_dir = custom_args['work_dir']
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.data_root = custom_args['data_root']
    cfg.train_dataloader.dataset.data_root = custom_args['data_root']
    cfg.train_dataloader.dataset.data_prefix.img = custom_args['train_data_prefix']
    cfg.train_dataloader.dataset.ann_file = custom_args['train_ann_file']
    cfg.test_dataloader.dataset.data_root = custom_args['data_root']
    cfg.test_dataloader.dataset.data_prefix.img = custom_args['val_data_prefix']
    cfg.test_dataloader.dataset.ann_file = custom_args['val_ann_file']
    cfg.val_dataloader.dataset.data_root = custom_args['data_root']
    cfg.val_dataloader.dataset.data_prefix.img = custom_args['val_data_prefix']
    cfg.val_dataloader.dataset.ann_file = custom_args['val_ann_file']
    cfg.val_evaluator.ann_file = custom_args['data_root'] + custom_args['val_ann_file']
    cfg.test_evaluator = cfg.val_evaluator
    cfg.class_name = custom_args['class_name']
    cfg.metainfo.classes = custom_args['class_name']
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo
    cfg.test_dataloader.dataset.metainfo = cfg.metainfo
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo
    cfg.train_cfg.max_epochs = custom_args['epochs']
    cfg.train_cfg.val_interval = 1
    cfg.train_dataloader.batch_size = 6
    cfg.val_dataloader.batch_size = 1
    cfg.test_dataloader.batch_size = 1

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    # /home/jinqing/workshop/PB-OVD-master/datasets/stanford_dogs_train_clipemb.json /home/DATASET_PUBLIC/Stanford_Dogs
    '''
    data_root = ''
    train_ann_file = '/home/jinqing/workshop/PB-OVD-master/datasets/stanford_dogs_train_clipemb.json'
    train_data_prefix = '/home/DATASET_PUBLIC/Stanford_Dogs/Images/'  # Stanford_Dogs
    val_ann_file = '/home/DATASET_PUBLIC/Stanford_Dogs/stanford_dogs_val_clipemb.json'
    val_data_prefix = '/home/DATASET_PUBLIC/Stanford_Dogs/Images'
    dataset_name = 'Stanford_Dogs'
    '''

    data_root = ''      # CUB-200-2011
    train_ann_file = '/home/jinqing/workshop/PB-OVD-master/datasets/cub200_2011_train_clipemb.json'
    train_data_prefix = '/home/DATASET_PUBLIC/CUB_200_2011/CUB_200_2011/images'
    val_ann_file = '/home/DATASET_PUBLIC/CUB_200_2011/CUB_200_2011/cub200_2011_val_clipemb.json'
    val_data_prefix = '/home/DATASET_PUBLIC/CUB_200_2011/CUB_200_2011/images'
    dataset_name = 'CUB-200-2011'

    '''
    data_root = ''
    train_ann_file = '/home/DATASET_PUBLIC/domain_generalization/weather/daytime_clear/voc07_train.json'
    train_data_prefix = '/home/DATASET_PUBLIC/domain_generalization/weather/daytime_clear/'  # Day-Clear
    val_ann_file = '/home/DATASET_PUBLIC/domain_generalization/weather/daytime_clear/voc07_test.json'
    val_data_prefix = '/home/DATASET_PUBLIC/domain_generalization/weather/daytime_clear/'
    dataset_name = 'Day-Clear'
    '''
    '''
    data_root = ''
    train_ann_file = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/VOC2012/VOC2012_6_trainval.json'
    train_data_prefix = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/VOC2012/'  # Pascal-VOC
    val_ann_file = train_ann_file
    val_data_prefix = train_data_prefix
    dataset_name = 'Pascal-VOC'
    '''
    '''
    data_root = ''
    train_ann_file = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/cityscapes_in_voc/cityscapes_car_trainval.json'
    train_data_prefix = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/cityscapes_in_voc/'  # Cityscapes-Car
    val_ann_file = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/kitti/kitti_trainval.json'
    val_data_prefix = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/kitti'
    dataset_name = 'Cityscapes-Car'
    '''
    '''
    data_root = ''
    train_data_prefix = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/kitti'
    train_ann_file = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/kitti/kitti_trainval.json'    # Kitti
    val_data_prefix = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/cityscapes_in_voc'
    val_ann_file = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/cityscapes_in_voc/cityscapes_car_test.json'
    dataset_name = 'Kitti'
    '''
    '''
    data_root = ''
    train_data_prefix = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/sim10k/'
    train_ann_file = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/sim10k/sim10k_trainval10k.json'  # Sim10k
    val_data_prefix = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/cityscapes_in_voc/'
    val_ann_file = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/cityscapes_in_voc/cityscapes_car_test.json'
    dataset_name = 'Sim10k'
    '''
    '''
    data_root = ''
    train_data_prefix = '/gpfsdata/home/huangziyue/data/VOC/VOC_trainval'
    train_ann_file = '/gpfsdata/home/huangziyue/data/VOC/coco_annotations/trainval_coco_ann.json'
    val_data_prefix = '/gpfsdata/home/huangziyue/data/VOC/VOC_test'
    val_ann_file = '/gpfsdata/home/huangziyue/data/VOC/coco_annotations/test_coco_ann.json'
    dataset_name = 'VOC'
    '''
    '''
    data_root = ''
    train_data_prefix = '/home/DATASET_PUBLIC/coco/train2017/'
    train_ann_file = '/home/jinqing/workshop/PB-OVD-master/datasets/instances_train2017_seen_2_clipemb.json'
    val_data_prefix = '/home/DATASET_PUBLIC/coco/val2017/'
    val_ann_file = '/home/jinqing/workshop/PB-OVD-master/datasets/instances_val2017_all_2_clipemb.json'
    dataset_name = 'COCO-ov'
    '''
    '''
    data_root = ''
    train_data_prefix = '/gpfsdata/home/huangziyue/data/coco_new/'
    train_ann_file = '/gpfsdata/home/yangshuai/data/lvis_v1/annotations/lvis_v1_train.json'
    val_data_prefix = '/gpfsdata/home/yangshuai/data/lvis_v1/'
    val_ann_file = '/gpfsdata/home/yangshuai/data/lvis_v1/annotations/lvis_v1_val.json'
    dataset_name = 'LVIS'
    '''

    epochs = 12

    import json

    with open(data_root + train_ann_file, 'r') as f:
        data = json.load(f)

    categories = data['categories']
    category_list = [category['name'] for category in categories]
    class_name = tuple(category_list)
    print(class_name)
    args = {
        'data_root': data_root,
        'train_ann_file': train_ann_file,
        'train_data_prefix': train_data_prefix,
        'val_ann_file': val_ann_file,
        'val_data_prefix': val_data_prefix,
        'work_dir': 'work_dirs/glip-A_' + dataset_name + '_ft_' + str(epochs),
        'class_name': class_name,
        'epochs': epochs
    }
    main(args)
