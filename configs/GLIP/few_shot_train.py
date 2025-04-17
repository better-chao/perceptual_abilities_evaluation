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
    parser.add_argument('--config', default='configs/glip/glip_prompt_tuning_A.py')
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
    cfg.test_evaluator.format_only = True
    cfg.test_evaluator.outfile_prefix = custom_args['outfile_prefix']
    cfg.train_cfg.max_epochs = custom_args['epochs']
    cfg.default_hooks.checkpoint.interval = 40
    cfg.train_dataloader.batch_size = 1
    cfg.val_dataloader.batch_size = 1

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
    import json
    with open('docs/glip-A_few-shot_record.json', 'r') as rc:
        record = json.load(rc)
        prefix_list = list(record)
    with open('docs/generate_few_shot_infos.json', 'r') as rf:
        all_dict = json.load(rf)
        for dataset_name in all_dict:
            data_info = all_dict[dataset_name]
            data_root = data_info['data_root']
            train_ann_files = data_info['train_ann_files']
            train_data_prefix = data_info['train_data_prefix']
            val_ann_file = data_info['val_ann_file']
            val_data_prefix = data_info['val_data_prefix']
            class_name = tuple(data_info['class_name'])
            epochs = 40
            type_num = 12
            for i in range(type_num):
                train_ann_file = train_ann_files[i]
                shot = 10
                if i < 3:
                    shot = 1
                elif i < 6:
                    shot = 3
                elif i < 9:
                    shot = 5
                seed = 300
                if (i % 3) == 0:
                    seed = 3
                elif (i % 3) == 1:
                    seed = 30
                result_prefix = dataset_name + '_shot' + str(shot) + '_seed' + str(seed)
                if result_prefix in prefix_list:
                    continue
                args = {
                    'data_root': data_root + '/',
                    'train_ann_file': train_ann_file,
                    'train_data_prefix': train_data_prefix,
                    'val_ann_file': val_ann_file,
                    'val_data_prefix': val_data_prefix,
                    'work_dir': 'work_dirs/few_shot_tp/glip-A_' + result_prefix,
                    'class_name': class_name,
                    'epochs': epochs,
                    'outfile_prefix': 'results/few_shot_tp/glip-A_' + result_prefix + '.json'
                }
                main(args)
                prefix_list.append(result_prefix)
                with open('docs/glip-A_few-shot_record.json', 'w') as rc:
                    json.dump(prefix_list, rc)
