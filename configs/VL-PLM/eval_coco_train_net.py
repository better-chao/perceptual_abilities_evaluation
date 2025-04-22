import sys
import os

# 获取项目的根路径
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将项目根路径添加到 sys.path
if project_path not in sys.path:
    sys.path.append(project_path)
print(sys.path)

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    LVISEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.data.build import build_detection_train_loader

from VL_PLM.config import add_detector_config
from VL_PLM.data.dataset_mapper import DatasetMapperVLPLM as DatasetMapper
from VL_PLM.data.augumentation import build_lsj_aug
from VL_PLM.evaluation.coco_evaluation import COCO_evaluator

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    results = OrderedDict()

    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))

        metadata = MetadataCatalog.get(dataset_name)
        distributed = comm.get_world_size() > 1
        if distributed:
            model.module.roi_heads.box_predictor.set_class_embeddings(metadata.class_emb_mtx)
        else:
            model.roi_heads.box_predictor.set_class_embeddings(metadata.class_emb_mtx)

        evaluator_type = metadata.evaluator_type

        if evaluator_type == "lvis":
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco' in evaluator_type:
            evaluator = COCO_evaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
        results[dataset_name] = inference_on_dataset(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False, use_lsj=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
            checkpointer.resume_or_load(
                cfg.MODEL.WEIGHTS, resume=resume,
            ).get("iteration", -1) + 1
    )
    if cfg.SOLVER.RESET_ITER:
        logger.info('Reset loaded iteration. Start training from iteration 0.')
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    mapper = DatasetMapper(cfg, True)
    if use_lsj:
        mapper.augmentations = build_lsj_aug(image_size=1024)
        mapper.recompute_boxes = True

    data_loader = build_detection_train_loader(cfg, mapper=mapper)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    distributed = comm.get_world_size() > 1
    if distributed:
        model.module.roi_heads.box_predictor.set_class_embeddings(metadata.class_emb_mtx)
    else:
        model.roi_heads.box_predictor.set_class_embeddings(metadata.class_emb_mtx)

    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):

            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                                 for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and iteration % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter
            ):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                    (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_detector_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.TEST.AUG.ENABLED:
            logger.info("Running inference with test-time augmentation ...")
            model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True
        )

    do_train(cfg, model, resume=args.resume, use_lsj=args.use_lsj)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument('--manual_device', default='')
    args.add_argument('--use_lsj', action='store_true', default=False)
    """
    eval_coco_2_coco: 
    
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.286
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.438
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.307
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.182
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.306
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.372
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.241
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.421
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.449
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.295
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570
    [12/07 22:43:07 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
    |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
    |:------:|:------:|:------:|:------:|:------:|:------:|
    | 28.556 | 43.761 | 30.680 | 18.242 | 30.569 | 37.232 |
    [12/07 22:43:07 d2.evaluation.coco_evaluation]: Evaluation results for AP50 
    |  results_base  |  results_novel  |
    |:--------------:|:---------------:|
    |     60.054     |     34.423      |
    [12/07 22:43:07 d2.evaluation.coco_evaluation]: Per-category bbox AP50: 
    | category      | AP50   | category     | AP50   | category       | AP50   |
    |:--------------|:-------|:-------------|:-------|:---------------|:-------|
    | person        | 83.843 | bicycle      | 60.553 | car            | 73.764 |
    | motorcycle    | 76.585 | airplane     | 63.456 | bus            | 76.969 |
    | train         | 85.333 | truck        | 56.247 | boat           | 55.491 |
    | traffic light | 0.000  | fire hydrant | 0.000  | stop sign      | 0.000  |
    | parking meter | 0.000  | bench        | 39.019 | bird           | 59.064 |
    | cat           | 42.060 | dog          | 59.797 | horse          | 80.694 |
    | sheep         | 62.933 | cow          | 62.293 | elephant       | 76.946 |
    | bear          | 85.399 | zebra        | 89.742 | giraffe        | 87.928 |
    | backpack      | 37.042 | umbrella     | 12.871 | handbag        | 34.386 |
    | tie           | 3.502  | suitcase     | 67.506 | frisbee        | 85.264 |
    | skis          | 43.591 | snowboard    | 10.795 | sports ball    | 0.000  |
    | kite          | 66.642 | baseball bat | 0.000  | baseball glove | 0.000  |
    | skateboard    | 5.587  | surfboard    | 63.749 | tennis racket  | 0.000  |
    | bottle        | 59.027 | wine glass   | 1.666  | cup            | 31.607 |
    | fork          | 55.889 | knife        | 5.091  | spoon          | 29.531 |
    | bowl          | 55.800 | banana       | 45.991 | apple          | 28.284 |
    | sandwich      | 48.297 | orange       | 43.970 | broccoli       | 46.538 |
    | carrot        | 38.756 | hot dog      | 31.217 | pizza          | 72.736 |
    | donut         | 54.986 | cake         | 41.559 | chair          | 49.621 |
    | couch         | 40.561 | potted plant | 0.000  | bed            | 57.208 |
    | dining table  | 0.248  | toilet       | 70.389 | tv             | 74.739 |
    | laptop        | 67.767 | mouse        | 76.811 | remote         | 50.933 |
    | keyboard      | 20.433 | cell phone   | 0.000  | microwave      | 76.871 |
    | oven          | 50.430 | toaster      | 60.891 | sink           | 7.727  |
    | refrigerator  | 73.967 | book         | 33.669 | clock          | 73.927 |
    | vase          | 51.562 | scissors     | 23.931 | teddy bear     | 0.000  |
    | hair drier    | 0.000  | toothbrush   | 39.228 |                |        |
    Loading and preparing results...
    DONE (t=4.89s)
    creating index...
    index created!
    
    eval_coco_2_voc:
    
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.463
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.697
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.503
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.197
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.408
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.539
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.364
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.592
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.611
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.360
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.574
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.668
    [12/07 23:06:42 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
    |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
    |:------:|:------:|:------:|:------:|:------:|:------:|
    | 46.302 | 69.677 | 50.263 | 19.700 | 40.834 | 53.858 |
    [12/07 23:06:42 d2.evaluation.coco_evaluation]: Evaluation results for AP50 
    |  results_base  |  results_novel  |
    |:--------------:|:---------------:|
    |     82.130     |     69.268      |
    [12/07 23:06:42 d2.evaluation.coco_evaluation]: Per-category bbox AP50: 
    | category    | AP50   | category    | AP50   | category   | AP50   |
    |:------------|:-------|:------------|:-------|:-----------|:-------|
    | aeroplane   | 69.649 | bicycle     | 88.283 | bird       | 84.493 |
    | boat        | 65.242 | bottle      | 75.218 | bus        | 81.329 |
    | car         | 86.888 | cat         | 50.112 | chair      | 66.699 |
    | cow         | 80.441 | diningtable | 0.046  | dog        | 65.191 |
    | horse       | 92.177 | motorbike   | 91.254 | person     | 89.480 |
    | pottedplant | 1.889  | sheep       | 80.530 | sofa       | 54.932 |
    | train       | 92.290 | tvmonitor   | 77.387 |            |        |
    
    
    
    eval_coco_2_lvis:
    [12/09 22:38:46 detectron2]: Evaluation results for lvis_eval in csv format:
    [12/09 22:38:46 d2.evaluation.testing]: copypaste: Task: bbox
    [12/09 22:38:46 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
    [12/09 22:38:46 d2.evaluation.testing]: copypaste: 1.5652,2.2844,1.6594,1.2020,2.1289,3.4428
    [12/09 22:38:46 d2.evaluation.testing]: copypaste: Task: segm
    [12/09 22:38:46 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
    [12/09 22:38:46 d2.evaluation.testing]: copypaste: 0.2153,0.5734,0.1205,0.1145,0.2670,0.6605
    
    eval_coco_2_lvis_mini:
    [12/09 23:46:20 d2.evaluation.testing]: copypaste: Task: bbox
    [12/09 23:46:20 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
    [12/09 23:46:20 d2.evaluation.testing]: copypaste: 2.1311,3.1362,2.2580,1.8764,3.1848,5.6057
    [12/09 23:46:20 d2.evaluation.testing]: copypaste: Task: segm
    [12/09 23:46:20 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
    [12/09 23:46:20 d2.evaluation.testing]: copypaste: 0.2836,0.6944,0.2021,0.2447,0.4579,1.1216

    eval_coco_2_obj365: 
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.050
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.075
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.054
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.024
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.050
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.066
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.115
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.069
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.125
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.181
    [12/10 01:14:02 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
    |  AP   |  AP50  |  AP75  |  APs  |  APm  |  APl  |
    |:-----:|:------:|:------:|:-----:|:-----:|:-----:|
    | 5.015 | 7.524  | 5.376  | 2.354 | 4.990 | 8.738 |

    
    """
    #################
    input_args = [
        '--config', 'configs/eval_coco_2_obj365.yaml',
        '--num-gpus', '1',
        '--eval-only',
        '--resume'
    ]
    #################
    args = args.parse_args(input_args)


    if args.manual_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.manual_device
    args.dist_url = 'tcp://127.0.0.1:{}'.format(
        torch.randint(11111, 60000, (1,))[0].item())
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
