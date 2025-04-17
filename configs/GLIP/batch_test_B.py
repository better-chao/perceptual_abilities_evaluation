# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo
from glip_head_prompt_tuning import ATSSVLFusionHead_PromptTuning


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    # parser.add_argument('--config', default='configs/glip/lvis/glip_atss_swin-t_bc_fpn_dyhead_pretrain_zeroshot_mini-lvis.py')
    parser.add_argument('--config', default='configs/glip/glip_B_lvis_text.py')
    parser.add_argument('--checkpoint', default='https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_b_mmdet-6dfbd102.pth')
    parser.add_argument(
        '--work-dir',
        default="tmp_eval_dir")
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument('--tta', action='store_true')
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
    # testing speed.
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
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # cfg.load_from = args.checkpoint
    cfg.load_from = custom_args['checkpoint']
    # import pdb; pdb.set_trace()
    # cfg.data_root = custom_args['data_root']
    cfg.test_dataloader.dataset.data_root = custom_args['data_root']
    cfg.test_dataloader.dataset.data_prefix.img = custom_args['val_data_prefix']
    cfg.test_dataloader.dataset.ann_file = custom_args['val_ann_file']
    cfg.test_dataloader.dataset.data_root = custom_args['data_root']
    cfg.val_dataloader.dataset.data_prefix.img = custom_args['val_data_prefix']
    cfg.val_dataloader.dataset.ann_file = custom_args['val_ann_file']
    cfg.val_evaluator.ann_file = custom_args['data_root'] + custom_args['val_ann_file']
    cfg.test_evaluator = cfg.val_evaluator
    cfg.class_name = custom_args['class_name']
    cfg.metainfo.classes = custom_args['class_name']
    cfg.test_dataloader.dataset.metainfo = cfg.metainfo
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo
    cfg.val_dataloader.batch_size = 1
    cfg.test_dataloader.batch_size = 1

    # import pdb; pdb.set_trace()

    # if args.show or args.show_dir:
    #     cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    # if args.out is not None:
    #     assert args.out.endswith(('.pkl', '.pickle')), \
    #         'The dump file must be a pkl file.'
    #     runner.test_evaluator.metrics.append(
    #         DumpDetResults(out_file_path=args.out))

    # start testing
    # import pdb; pdb.set_trace()
    metric = runner.test()
    return metric


if __name__ == '__main__':
    '''
    data_root = '/gpfsdata/home/huangziyue/data/VOC/'
    val_ann_file = 'coco_annotations/test_coco_ann.json'
    val_data_prefix = 'VOC_test_corruptions/'    # VOC_test_corruptions/brightness_Svr1
    '''
    glip_type = 'B'

    data_root = ''
    val_ann_file = '/gpfsdata/home/yangshuai/data/lvis_v1/annotations/lvis_v1_val.json'
    val_data_prefix = '/gpfsdata/home/huangziyue/data/coco_new/'  # LVIS
    result_inf = 'LVIS--LVIS_tp'

    checkpoint = '/gpfsdata/home/wangjunzhe/VLM_Eval/mmdetection/work_dirs/glip-B_LVIS_tp_10/epoch_10.pth'

    val_alldir = [val_data_prefix]  # COCO
    '''
    val_alldir = os.listdir(data_root+val_data_prefix)      # COCO_C
    val_alldir = [os.path.join(val_data_prefix, p) for p in val_alldir]
    '''
    import json
    with open(data_root+val_ann_file, 'r') as f:
        data = json.load(f)

    categories = data['categories']
    category_list = [category['name'] for category in categories]

    class_name = tuple(category_list)
    print(class_name)

    results_list = []

    # import pdb; pdb.set_trace()
    cnt = 0
    for val_img_path in val_alldir:
        # data_root = '/gpfsdata/home/buaa_liuchenguang/github_eval/OpenDataLab___OCHuman/raw/OCHuman/OCHuman/'
        # val_ann_file = 'annotations/ochuman_coco_format_val_range_0.00_1.00.json'
        # val_img_path = 'images/'    # VOC_test_corruptions/brightness_Svr1
        # class_name = ('person')
        args = {
            'data_root': data_root,
            'val_ann_file': val_ann_file,
            'val_data_prefix': val_img_path,
            'class_name': class_name,
            'checkpoint': checkpoint
        }
        metric = main(args)
        results_list.append([val_img_path, metric])
        # break
        cnt += 1
        # if cnt >= 100:
        #     break
    
    import pickle
    write_file = 'results/visual_finetuning/result_list_' + glip_type + '_' + result_inf + '.pkl'
    with open(write_file, 'wb') as f:
        pickle.dump(results_list, f)
