import itertools

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detrex.data import DetrDatasetMapper
# from detrex.data.datasets import register_custom_ovd_instances
from omegaconf import OmegaConf

dataloader = OmegaConf.create()

# Case 0: If you want to define it by yourself, you can change it on ovdino/detrex/data/datasets/custom_ovd.py, you need to uncomment the code (ovdino/detrex/data/datasets/__init__.py L21) first.
# Case 1 (Recommend): If you follow the coco format, you need uncomment and change the following code.
# # 1. Define custom_meta_info, just a example, you need to change it to your own.
# meta_info = {
#     "thing_dataset_id_to_contiguous_id": {
#         0: 0,
#         1: 0,
#     },  # key: dataset_id, value: contiguous_id
#     "thing_classes": ["category_0", "category_1"],  # category names
# }
# # 2. Register custom train dataset.
# register_custom_ovd_instances(
#     "custom_train_ovd_unipro",  # dataset_name
#     meta_info,
#     "/path/to/train.json",  # annotations_json_file
#     "/path/to/train/images",  # image_root
#     2,  # number_of_classes, default: 2. You also need to change model.num_classes in the ovdino/projects/ovdino/configs/ovdino_swin_tiny224_bert_base_ft_custom_24ep.py#L37.
#     "full",  # template, default: full
# )
# # 3. Register custom val dataset.
# register_custom_ovd_instances(
#     "custom_val_ovd_unipro",
#     meta_info,
#     "/path/to/val.json",
#     "/path/to/val/images",
#     2,
#     "full",
# )
# # 4. Optional, register custom test dataset.
# register_custom_ovd_instances(
#     "custom_test_ovd",
#     meta_info,
#     "/path/to/test.json",
#     "/path/to/test/images",
#     2,
#     "full",  # choices: ["identity", "simple", "full"]
# )

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="custom_train_ovd_unipro"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(
                    480,
                    512,
                    544,
                    576,
                    608,
                    640,
                    672,
                    704,
                    736,
                    768,
                    800,
                ),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(
                    480,
                    512,
                    544,
                    576,
                    608,
                    640,
                    672,
                    704,
                    736,
                    768,
                    800,
                ),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="custom_val_ovd_unipro", filter_empty=False
    ),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
