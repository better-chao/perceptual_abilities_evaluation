# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_migc": ("coco/coco_images_results_rv60B1", "coco/annotations/coco_images_results_rv60B1_coco_annos.json"),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}
_PREDEFINED_SPLITS_COCO["coco_ovd"] = {
    "coco_2017_ovd_all_train": ("coco/train2017", "coco/regionclip_ov_annotations/ovd_ins_train2017_all.json"),
    "coco_2017_ovd_b_train": ("coco/train2017", "coco/regionclip_ov_annotations/ovd_ins_train2017_b.json"),
    "coco_2017_ovd_t_train": ("coco/train2017", "coco/regionclip_ov_annotations/ovd_ins_train2017_t.json"),
    "coco_2017_ovd_all_test": ("coco/val2017", "coco/regionclip_ov_annotations/ovd_ins_val2017_all.json"),
    "coco_2017_ovd_b_test": ("coco/val2017", "coco/regionclip_ov_annotations/ovd_ins_val2017_b.json"),
    "coco_2017_ovd_t_test": ("coco/val2017", "coco/regionclip_ov_annotations/ovd_ins_val2017_t.json"),
}

# 这里是使用coco_c数据集测试鲁棒性
_PREDEFINED_SPLITS_COCO["coco_c"] = {
    "coco_c_speckle_noise_Svr3": ("coco_new/val2017_corruptions/speckle_noise_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_saturate_Svr4": ("coco_new/val2017_corruptions/saturate_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_glass_blur_Svr5": ("coco_new/val2017_corruptions/glass_blur_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_fog_Svr4": ("coco_new/val2017_corruptions/fog_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_contrast_Svr2": ("coco_new/val2017_corruptions/contrast_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_snow_Svr2": ("coco_new/val2017_corruptions/snow_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_contrast_Svr1": ("coco_new/val2017_corruptions/contrast_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_saturate_Svr1": ("coco_new/val2017_corruptions/saturate_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_fog_Svr2": ("coco_new/val2017_corruptions/fog_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_gaussian_blur_Svr2": ("coco_new/val2017_corruptions/gaussian_blur_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_brightness_Svr3": ("coco_new/val2017_corruptions/brightness_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_glass_blur_Svr2": ("coco_new/val2017_corruptions/glass_blur_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_glass_blur_Svr4": ("coco_new/val2017_corruptions/glass_blur_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_gaussian_noise_Svr3": ("coco_new/val2017_corruptions/gaussian_noise_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_saturate_Svr2": ("coco_new/val2017_corruptions/saturate_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_motion_blur_Svr4": ("coco_new/val2017_corruptions/motion_blur_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_defocus_blur_Svr3": ("coco_new/val2017_corruptions/defocus_blur_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_impulse_noise_Svr2": ("coco_new/val2017_corruptions/impulse_noise_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_frost_Svr4": ("coco_new/val2017_corruptions/frost_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_zoom_blur_Svr2": ("coco_new/val2017_corruptions/zoom_blur_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_pixelate_Svr2": ("coco_new/val2017_corruptions/pixelate_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_motion_blur_Svr2": ("coco_new/val2017_corruptions/motion_blur_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_gaussian_noise_Svr2": ("coco_new/val2017_corruptions/gaussian_noise_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_brightness_Svr5": ("coco_new/val2017_corruptions/brightness_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_shot_noise_Svr5": ("coco_new/val2017_corruptions/shot_noise_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_elastic_transform_Svr5": ("coco_new/val2017_corruptions/elastic_transform_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_zoom_blur_Svr3": ("coco_new/val2017_corruptions/zoom_blur_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_fog_Svr5": ("coco_new/val2017_corruptions/fog_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_contrast_Svr4": ("coco_new/val2017_corruptions/contrast_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_defocus_blur_Svr4": ("coco_new/val2017_corruptions/defocus_blur_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_speckle_noise_Svr5": ("coco_new/val2017_corruptions/speckle_noise_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_frost_Svr5": ("coco_new/val2017_corruptions/frost_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_fog_Svr3": ("coco_new/val2017_corruptions/fog_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_motion_blur_Svr1": ("coco_new/val2017_corruptions/motion_blur_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_snow_Svr4": ("coco_new/val2017_corruptions/snow_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_gaussian_blur_Svr1": ("coco_new/val2017_corruptions/gaussian_blur_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_jpeg_compression_Svr3": ("coco_new/val2017_corruptions/jpeg_compression_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_impulse_noise_Svr1": ("coco_new/val2017_corruptions/impulse_noise_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_impulse_noise_Svr5": ("coco_new/val2017_corruptions/impulse_noise_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_brightness_Svr2": ("coco_new/val2017_corruptions/brightness_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_spatter_Svr1": ("coco_new/val2017_corruptions/spatter_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_elastic_transform_Svr2": ("coco_new/val2017_corruptions/elastic_transform_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_jpeg_compression_Svr5": ("coco_new/val2017_corruptions/jpeg_compression_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_glass_blur_Svr3": ("coco_new/val2017_corruptions/glass_blur_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_glass_blur_Svr1": ("coco_new/val2017_corruptions/glass_blur_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_defocus_blur_Svr2": ("coco_new/val2017_corruptions/defocus_blur_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_spatter_Svr5": ("coco_new/val2017_corruptions/spatter_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_contrast_Svr3": ("coco_new/val2017_corruptions/contrast_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_pixelate_Svr3": ("coco_new/val2017_corruptions/pixelate_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_zoom_blur_Svr4": ("coco_new/val2017_corruptions/zoom_blur_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_brightness_Svr4": ("coco_new/val2017_corruptions/brightness_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_elastic_transform_Svr1": ("coco_new/val2017_corruptions/elastic_transform_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_defocus_blur_Svr5": ("coco_new/val2017_corruptions/defocus_blur_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_spatter_Svr2": ("coco_new/val2017_corruptions/spatter_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_motion_blur_Svr5": ("coco_new/val2017_corruptions/motion_blur_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_impulse_noise_Svr4": ("coco_new/val2017_corruptions/impulse_noise_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_frost_Svr3": ("coco_new/val2017_corruptions/frost_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_elastic_transform_Svr4": ("coco_new/val2017_corruptions/elastic_transform_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_saturate_Svr5": ("coco_new/val2017_corruptions/saturate_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_speckle_noise_Svr2": ("coco_new/val2017_corruptions/speckle_noise_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_snow_Svr5": ("coco_new/val2017_corruptions/snow_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_speckle_noise_Svr1": ("coco_new/val2017_corruptions/speckle_noise_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_frost_Svr2": ("coco_new/val2017_corruptions/frost_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_jpeg_compression_Svr4": ("coco_new/val2017_corruptions/jpeg_compression_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_pixelate_Svr1": ("coco_new/val2017_corruptions/pixelate_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_brightness_Svr1": ("coco_new/val2017_corruptions/brightness_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_contrast_Svr5": ("coco_new/val2017_corruptions/contrast_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_jpeg_compression_Svr2": ("coco_new/val2017_corruptions/jpeg_compression_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_gaussian_blur_Svr3": ("coco_new/val2017_corruptions/gaussian_blur_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_defocus_blur_Svr1": ("coco_new/val2017_corruptions/defocus_blur_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_spatter_Svr3": ("coco_new/val2017_corruptions/spatter_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_shot_noise_Svr1": ("coco_new/val2017_corruptions/shot_noise_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_spatter_Svr4": ("coco_new/val2017_corruptions/spatter_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_frost_Svr1": ("coco_new/val2017_corruptions/frost_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_fog_Svr1": ("coco_new/val2017_corruptions/fog_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_pixelate_Svr5": ("coco_new/val2017_corruptions/pixelate_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_speckle_noise_Svr4": ("coco_new/val2017_corruptions/speckle_noise_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_snow_Svr3": ("coco_new/val2017_corruptions/snow_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_saturate_Svr3": ("coco_new/val2017_corruptions/saturate_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_gaussian_noise_Svr1": ("coco_new/val2017_corruptions/gaussian_noise_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_shot_noise_Svr2": ("coco_new/val2017_corruptions/shot_noise_Svr2", "coco_new/annotations/instances_val2017.json"),
    "coco_c_zoom_blur_Svr1": ("coco_new/val2017_corruptions/zoom_blur_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_pixelate_Svr4": ("coco_new/val2017_corruptions/pixelate_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_gaussian_blur_Svr5": ("coco_new/val2017_corruptions/gaussian_blur_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_shot_noise_Svr3": ("coco_new/val2017_corruptions/shot_noise_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_gaussian_noise_Svr4": ("coco_new/val2017_corruptions/gaussian_noise_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_gaussian_blur_Svr4": ("coco_new/val2017_corruptions/gaussian_blur_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_gaussian_noise_Svr5": ("coco_new/val2017_corruptions/gaussian_noise_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_shot_noise_Svr4": ("coco_new/val2017_corruptions/shot_noise_Svr4", "coco_new/annotations/instances_val2017.json"),
    "coco_c_elastic_transform_Svr3": ("coco_new/val2017_corruptions/elastic_transform_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_motion_blur_Svr3": ("coco_new/val2017_corruptions/motion_blur_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_zoom_blur_Svr5": ("coco_new/val2017_corruptions/zoom_blur_Svr5", "coco_new/annotations/instances_val2017.json"),
    "coco_c_jpeg_compression_Svr1": ("coco_new/val2017_corruptions/jpeg_compression_Svr1", "coco_new/annotations/instances_val2017.json"),
    "coco_c_impulse_noise_Svr3": ("coco_new/val2017_corruptions/impulse_noise_Svr3", "coco_new/annotations/instances_val2017.json"),
    "coco_c_snow_Svr1": ("coco_new/val2017_corruptions/snow_Svr1", "coco_new/annotations/instances_val2017.json"),
}

# 这里是使用voc_c数据集测试鲁棒性
_PREDEFINED_SPLITS_COCO["voc_c"] = {
    "speckle_noise_Svr3": ("VOC/VOC_test_corruptions/speckle_noise_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "saturate_Svr4": ("VOC/VOC_test_corruptions/saturate_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "glass_blur_Svr5": ("VOC/VOC_test_corruptions/glass_blur_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "fog_Svr4": ("VOC/VOC_test_corruptions/fog_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "contrast_Svr2": ("VOC/VOC_test_corruptions/contrast_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "snow_Svr2": ("VOC/VOC_test_corruptions/snow_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "contrast_Svr1": ("VOC/VOC_test_corruptions/contrast_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "saturate_Svr1": ("VOC/VOC_test_corruptions/saturate_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "fog_Svr2": ("VOC/VOC_test_corruptions/fog_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "gaussian_blur_Svr2": ("VOC/VOC_test_corruptions/gaussian_blur_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "brightness_Svr3": ("VOC/VOC_test_corruptions/brightness_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "glass_blur_Svr2": ("VOC/VOC_test_corruptions/glass_blur_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "glass_blur_Svr4": ("VOC/VOC_test_corruptions/glass_blur_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "gaussian_noise_Svr3": ("VOC/VOC_test_corruptions/gaussian_noise_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "saturate_Svr2": ("VOC/VOC_test_corruptions/saturate_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "motion_blur_Svr4": ("VOC/VOC_test_corruptions/motion_blur_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "defocus_blur_Svr3": ("VOC/VOC_test_corruptions/defocus_blur_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "impulse_noise_Svr2": ("VOC/VOC_test_corruptions/impulse_noise_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "frost_Svr4": ("VOC/VOC_test_corruptions/frost_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "zoom_blur_Svr2": ("VOC/VOC_test_corruptions/zoom_blur_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "pixelate_Svr2": ("VOC/VOC_test_corruptions/pixelate_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "motion_blur_Svr2": ("VOC/VOC_test_corruptions/motion_blur_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "gaussian_noise_Svr2": ("VOC/VOC_test_corruptions/gaussian_noise_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "brightness_Svr5": ("VOC/VOC_test_corruptions/brightness_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "shot_noise_Svr5": ("VOC/VOC_test_corruptions/shot_noise_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "elastic_transform_Svr5": ("VOC/VOC_test_corruptions/elastic_transform_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "zoom_blur_Svr3": ("VOC/VOC_test_corruptions/zoom_blur_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "fog_Svr5": ("VOC/VOC_test_corruptions/fog_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "contrast_Svr4": ("VOC/VOC_test_corruptions/contrast_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "defocus_blur_Svr4": ("VOC/VOC_test_corruptions/defocus_blur_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "speckle_noise_Svr5": ("VOC/VOC_test_corruptions/speckle_noise_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "frost_Svr5": ("VOC/VOC_test_corruptions/frost_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "fog_Svr3": ("VOC/VOC_test_corruptions/fog_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "motion_blur_Svr1": ("VOC/VOC_test_corruptions/motion_blur_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "snow_Svr4": ("VOC/VOC_test_corruptions/snow_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "gaussian_blur_Svr1": ("VOC/VOC_test_corruptions/gaussian_blur_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "jpeg_compression_Svr3": ("VOC/VOC_test_corruptions/jpeg_compression_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "impulse_noise_Svr1": ("VOC/VOC_test_corruptions/impulse_noise_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "impulse_noise_Svr5": ("VOC/VOC_test_corruptions/impulse_noise_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "brightness_Svr2": ("VOC/VOC_test_corruptions/brightness_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "spatter_Svr1": ("VOC/VOC_test_corruptions/spatter_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "elastic_transform_Svr2": ("VOC/VOC_test_corruptions/elastic_transform_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "jpeg_compression_Svr5": ("VOC/VOC_test_corruptions/jpeg_compression_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "glass_blur_Svr3": ("VOC/VOC_test_corruptions/glass_blur_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "glass_blur_Svr1": ("VOC/VOC_test_corruptions/glass_blur_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "defocus_blur_Svr2": ("VOC/VOC_test_corruptions/defocus_blur_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "spatter_Svr5": ("VOC/VOC_test_corruptions/spatter_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "contrast_Svr3": ("VOC/VOC_test_corruptions/contrast_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "pixelate_Svr3": ("VOC/VOC_test_corruptions/pixelate_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "zoom_blur_Svr4": ("VOC/VOC_test_corruptions/zoom_blur_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "brightness_Svr4": ("VOC/VOC_test_corruptions/brightness_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "elastic_transform_Svr1": ("VOC/VOC_test_corruptions/elastic_transform_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "defocus_blur_Svr5": ("VOC/VOC_test_corruptions/defocus_blur_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "spatter_Svr2": ("VOC/VOC_test_corruptions/spatter_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "motion_blur_Svr5": ("VOC/VOC_test_corruptions/motion_blur_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "impulse_noise_Svr4": ("VOC/VOC_test_corruptions/impulse_noise_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "frost_Svr3": ("VOC/VOC_test_corruptions/frost_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "elastic_transform_Svr4": ("VOC/VOC_test_corruptions/elastic_transform_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "saturate_Svr5": ("VOC/VOC_test_corruptions/saturate_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "speckle_noise_Svr2": ("VOC/VOC_test_corruptions/speckle_noise_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "snow_Svr5": ("VOC/VOC_test_corruptions/snow_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "speckle_noise_Svr1": ("VOC/VOC_test_corruptions/speckle_noise_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "frost_Svr2": ("VOC/VOC_test_corruptions/frost_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "jpeg_compression_Svr4": ("VOC/VOC_test_corruptions/jpeg_compression_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "pixelate_Svr1": ("VOC/VOC_test_corruptions/pixelate_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "brightness_Svr1": ("VOC/VOC_test_corruptions/brightness_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "contrast_Svr5": ("VOC/VOC_test_corruptions/contrast_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "jpeg_compression_Svr2": ("VOC/VOC_test_corruptions/jpeg_compression_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "gaussian_blur_Svr3": ("VOC/VOC_test_corruptions/gaussian_blur_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "defocus_blur_Svr1": ("VOC/VOC_test_corruptions/defocus_blur_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "spatter_Svr3": ("VOC/VOC_test_corruptions/spatter_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "shot_noise_Svr1": ("VOC/VOC_test_corruptions/shot_noise_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "spatter_Svr4": ("VOC/VOC_test_corruptions/spatter_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "frost_Svr1": ("VOC/VOC_test_corruptions/frost_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "fog_Svr1": ("VOC/VOC_test_corruptions/fog_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "pixelate_Svr5": ("VOC/VOC_test_corruptions/pixelate_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "speckle_noise_Svr4": ("VOC/VOC_test_corruptions/speckle_noise_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "snow_Svr3": ("VOC/VOC_test_corruptions/snow_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "saturate_Svr3": ("VOC/VOC_test_corruptions/saturate_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "gaussian_noise_Svr1": ("VOC/VOC_test_corruptions/gaussian_noise_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "shot_noise_Svr2": ("VOC/VOC_test_corruptions/shot_noise_Svr2", "VOC/coco_annotations/test_coco_ann.json"),
    "zoom_blur_Svr1": ("VOC/VOC_test_corruptions/zoom_blur_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "pixelate_Svr4": ("VOC/VOC_test_corruptions/pixelate_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "gaussian_blur_Svr5": ("VOC/VOC_test_corruptions/gaussian_blur_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "shot_noise_Svr3": ("VOC/VOC_test_corruptions/shot_noise_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "gaussian_noise_Svr4": ("VOC/VOC_test_corruptions/gaussian_noise_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "gaussian_blur_Svr4": ("VOC/VOC_test_corruptions/gaussian_blur_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "gaussian_noise_Svr5": ("VOC/VOC_test_corruptions/gaussian_noise_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "shot_noise_Svr4": ("VOC/VOC_test_corruptions/shot_noise_Svr4", "VOC/coco_annotations/test_coco_ann.json"),
    "elastic_transform_Svr3": ("VOC/VOC_test_corruptions/elastic_transform_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "motion_blur_Svr3": ("VOC/VOC_test_corruptions/motion_blur_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "zoom_blur_Svr5": ("VOC/VOC_test_corruptions/zoom_blur_Svr5", "VOC/coco_annotations/test_coco_ann.json"),
    "jpeg_compression_Svr1": ("VOC/VOC_test_corruptions/jpeg_compression_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
    "impulse_noise_Svr3": ("VOC/VOC_test_corruptions/impulse_noise_Svr3", "VOC/coco_annotations/test_coco_ann.json"),
    "snow_Svr1": ("VOC/VOC_test_corruptions/snow_Svr1", "VOC/coco_annotations/test_coco_ann.json"),
}

# 这里是使用cityscapes_c数据集测试鲁棒性
_PREDEFINED_SPLITS_COCO["cityscapes_c"] = {
    "cityscapes_c_speckle_noise_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/speckle_noise_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_saturate_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/saturate_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_glass_blur_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/glass_blur_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_fog_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/fog_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_contrast_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/contrast_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_snow_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/snow_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_contrast_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/contrast_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_saturate_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/saturate_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_fog_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/fog_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_gaussian_blur_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/gaussian_blur_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_brightness_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/brightness_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_glass_blur_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/glass_blur_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_glass_blur_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/glass_blur_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_gaussian_noise_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/gaussian_noise_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_saturate_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/saturate_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_motion_blur_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/motion_blur_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_defocus_blur_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/defocus_blur_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_impulse_noise_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/impulse_noise_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_frost_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/frost_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_zoom_blur_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/zoom_blur_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_pixelate_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/pixelate_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_motion_blur_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/motion_blur_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_gaussian_noise_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/gaussian_noise_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_brightness_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/brightness_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_shot_noise_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/shot_noise_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_elastic_transform_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/elastic_transform_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_zoom_blur_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/zoom_blur_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_fog_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/fog_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_contrast_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/contrast_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_defocus_blur_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/defocus_blur_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_speckle_noise_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/speckle_noise_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_frost_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/frost_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_fog_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/fog_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_motion_blur_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/motion_blur_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_snow_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/snow_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_gaussian_blur_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/gaussian_blur_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_jpeg_compression_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/jpeg_compression_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_impulse_noise_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/impulse_noise_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_impulse_noise_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/impulse_noise_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_brightness_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/brightness_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_spatter_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/spatter_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_elastic_transform_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/elastic_transform_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_jpeg_compression_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/jpeg_compression_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_glass_blur_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/glass_blur_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_glass_blur_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/glass_blur_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_defocus_blur_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/defocus_blur_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_spatter_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/spatter_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_contrast_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/contrast_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_pixelate_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/pixelate_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_zoom_blur_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/zoom_blur_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_brightness_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/brightness_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_elastic_transform_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/elastic_transform_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_defocus_blur_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/defocus_blur_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_spatter_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/spatter_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_motion_blur_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/motion_blur_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_impulse_noise_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/impulse_noise_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_frost_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/frost_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_elastic_transform_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/elastic_transform_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_saturate_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/saturate_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_speckle_noise_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/speckle_noise_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_snow_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/snow_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_speckle_noise_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/speckle_noise_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_frost_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/frost_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_jpeg_compression_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/jpeg_compression_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_pixelate_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/pixelate_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_brightness_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/brightness_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_contrast_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/contrast_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_jpeg_compression_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/jpeg_compression_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_gaussian_blur_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/gaussian_blur_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_defocus_blur_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/defocus_blur_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_spatter_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/spatter_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_shot_noise_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/shot_noise_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_spatter_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/spatter_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_frost_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/frost_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_fog_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/fog_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_pixelate_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/pixelate_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_speckle_noise_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/speckle_noise_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_snow_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/snow_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_saturate_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/saturate_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_gaussian_noise_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/gaussian_noise_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_shot_noise_Svr2": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/shot_noise_Svr2", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_zoom_blur_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/zoom_blur_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_pixelate_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/pixelate_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_gaussian_blur_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/gaussian_blur_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_shot_noise_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/shot_noise_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_gaussian_noise_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/gaussian_noise_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_gaussian_blur_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/gaussian_blur_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_gaussian_noise_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/gaussian_noise_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_shot_noise_Svr4": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/shot_noise_Svr4", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_elastic_transform_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/elastic_transform_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_motion_blur_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/motion_blur_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_zoom_blur_Svr5": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/zoom_blur_Svr5", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_jpeg_compression_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/jpeg_compression_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_impulse_noise_Svr3": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/impulse_noise_Svr3", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "cityscapes_c_snow_Svr1": ("gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/snow_Svr1", "home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
}

# 这里测试cityscapes
_PREDEFINED_SPLITS_COCO["cityscapes"] = {
    "cityscapes_train":("fasterrcnnda/cityscapes/train", "cityscapes/coco_format/instancesonly_filtered_gtFine_train.json"),
    "cityscapes_val": ("fasterrcnnda/cityscapes/val", "cityscapes/coco_format/instancesonly_filtered_gtFine_val.json"),
    "foggy_cityscapes_val": ("fasterrcnnda/foggy_cityscapes/val", "foggy_cityscapes/coco_format/foggy_instancesonly_filtered_gtFine_val.json"),
    "cityscapes_in_voc_val": ("cityscapes_in_voc", "cityscapes_in_voc/cityscapes_car_test.json"),
    "cityscapes_in_voc_train": ("cityscapes_in_voc", "cityscapes_in_voc/cityscapes_car_trainval.json"),
}

# 这里测试域泛化
_PREDEFINED_SPLITS_COCO["domain_generalization"] = {
    "daytime_clear_train": ("weather/daytime_clear", "weather/daytime_clear/voc07_train.json"),
    "dusk_rainy_val": ("weather/dusk_rainy", "weather/dusk_rainy/voc07_test.json"),
    "night_rainy_val": ("weather/night_rainy", "weather/night_rainy/voc07_test.json"),
    "daytime_foggy_val": ("weather/daytime_foggy", "weather/daytime_foggy/voc07_test.json"),
    "daytime_clear_val": ("weather/daytime_clear", "weather/daytime_clear/voc07_test.json"),
    "night_clear_val": ("weather/night_clear", "weather/night_clear/voc07_test.json"),
}

# 这里测试域自适应
_PREDEFINED_SPLITS_COCO["domain_adaptation"] = {
    "sim10k": ("sim10k", "sim10k/sim10k_trainval10k.json"),
    "kitti": ("kitti", "kitti/kitti_trainval.json"),
    "watercolor": ("watercolor", "watercolor/watercolor_test.json"),
    "comic": ("comic", "comic/comic_test.json"),
}

# 这里测试voc test
_PREDEFINED_SPLITS_COCO["voc"] = {
    "voc_train": ("VOC/VOC_trainval", "VOC/coco_annotations/trainval_coco_ann.json"),
    "voc_test": ("VOC/VOC_test", "VOC/coco_annotations/test_coco_ann.json"),
}

# 这里测试 object365 v1
_PREDEFINED_SPLITS_COCO["object365"] = {
    "obj365_train": ("Objects365_v1/train", "Objects365_v1/objects365_train.json"),
    "obj365_val": ("Objects365_v1/val", "Objects365_v1/objects365_val.json"),
}

# 这里测试细粒度
_PREDEFINED_SPLITS_COCO["fine_grained"] = {
    "dogs_train": ("DATASET_PUBLIC/Stanford_Dogs/Images", "jinqing/workshop/PB-OVD-master/datasets/stanford_dogs_train_clipemb.json"),
    "dogs_val": ("DATASET_PUBLIC/Stanford_Dogs/Images", "DATASET_PUBLIC/Stanford_Dogs/stanford_dogs_val_clipemb.json"),
    "birds_train": ("DATASET_PUBLIC/CUB_200_2011/CUB_200_2011/images", "jinqing/workshop/PB-OVD-master/datasets/cub200_2011_train_clipemb.json"),
    "birds_val": ("DATASET_PUBLIC/CUB_200_2011/CUB_200_2011/images", "DATASET_PUBLIC/CUB_200_2011/CUB_200_2011/cub200_2011_val_clipemb.json"),
}

# 这里测试 dense crowdhuman ochuman widerperson
_PREDEFINED_SPLITS_COCO["dense"] = {
    "crowdhuman_val": ("OpenDataLab___CrowdHuman/raw/CrowdHuman/crowdhuman/val", "OpenDataLab___CrowdHuman/raw/CrowdHuman/crowdhuman/annotations/crowdhuman_val.json"),
    "crowdhuman_train": ("OpenDataLab___CrowdHuman/raw/CrowdHuman/crowdhuman/train", "OpenDataLab___CrowdHuman/raw/CrowdHuman/crowdhuman/annotations/crowdhuman_train.json"),
    "ochuman_val": ("OpenDataLab___OCHuman/raw/OCHuman/OCHuman/images", "OpenDataLab___OCHuman/raw/OCHuman/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json"),
    "widerperson_val": ("WiderPerson/Images", "WiderPerson/val.json"),
    "widerperson_train": ("WiderPerson/Images", "WiderPerson/train.json"),
}

# 这里测试odinw
_PREDEFINED_SPLITS_COCO["odinw"] = {
    "AerialMaritimeDrone_large_shot1_seed3": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot1_seed3.json"),
    "AerialMaritimeDrone_large_shot1_seed30": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot1_seed30.json"),
    "AerialMaritimeDrone_large_shot1_seed300": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot1_seed300.json"),
    "AerialMaritimeDrone_large_shot3_seed3": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot3_seed3.json"),
    "AerialMaritimeDrone_large_shot3_seed30": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot3_seed30.json"),
    "AerialMaritimeDrone_large_shot3_seed300": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot3_seed300.json"),
    "AerialMaritimeDrone_large_shot5_seed3": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot5_seed3.json"),
    "AerialMaritimeDrone_large_shot5_seed30": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot5_seed30.json"),
    "AerialMaritimeDrone_large_shot5_seed300": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot5_seed300.json"),
    "AerialMaritimeDrone_large_shot10_seed3": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot10_seed3.json"),
    "AerialMaritimeDrone_large_shot10_seed30": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot10_seed30.json"),
    "AerialMaritimeDrone_large_shot10_seed300": ("odinw/AerialMaritimeDrone/large/train", "odinw/AerialMaritimeDrone/large/train/fewshot_train_shot10_seed300.json"),
    "AerialMaritimeDrone_tiled_shot1_seed3": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot1_seed3.json"),
    "AerialMaritimeDrone_tiled_shot1_seed30": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot1_seed30.json"),
    "AerialMaritimeDrone_tiled_shot1_seed300": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot1_seed300.json"),
    "AerialMaritimeDrone_tiled_shot3_seed3": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot3_seed3.json"),
    "AerialMaritimeDrone_tiled_shot3_seed30": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot3_seed30.json"),
    "AerialMaritimeDrone_tiled_shot3_seed300": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot3_seed300.json"),
    "AerialMaritimeDrone_tiled_shot5_seed3": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot5_seed3.json"),
    "AerialMaritimeDrone_tiled_shot5_seed30": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot5_seed30.json"),
    "AerialMaritimeDrone_tiled_shot5_seed300": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot5_seed300.json"),
    "AerialMaritimeDrone_tiled_shot10_seed3": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot10_seed3.json"),
    "AerialMaritimeDrone_tiled_shot10_seed30": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot10_seed30.json"),
    "AerialMaritimeDrone_tiled_shot10_seed300": ("odinw/AerialMaritimeDrone/tiled/train", "odinw/AerialMaritimeDrone/tiled/train/fewshot_train_shot10_seed300.json"),
    "AmericanSignLanguageLetters_shot1_seed3": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot1_seed3.json"),
    "AmericanSignLanguageLetters_shot1_seed30": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot1_seed30.json"),
    "AmericanSignLanguageLetters_shot1_seed300": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot1_seed300.json"),
    "AmericanSignLanguageLetters_shot3_seed3": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot3_seed3.json"),
    "AmericanSignLanguageLetters_shot3_seed30": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot3_seed30.json"),
    "AmericanSignLanguageLetters_shot3_seed300": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot3_seed300.json"),
    "AmericanSignLanguageLetters_shot5_seed3": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot5_seed3.json"),
    "AmericanSignLanguageLetters_shot5_seed30": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot5_seed30.json"),
    "AmericanSignLanguageLetters_shot5_seed300": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot5_seed300.json"),
    "AmericanSignLanguageLetters_shot10_seed3": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot10_seed3.json"),
    "AmericanSignLanguageLetters_shot10_seed30": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot10_seed30.json"),
    "AmericanSignLanguageLetters_shot10_seed300": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/fewshot_train_shot10_seed300.json"),
    "Aquarium_shot1_seed3": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot1_seed3.json"),
    "Aquarium_shot1_seed30": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot1_seed30.json"),
    "Aquarium_shot1_seed300": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot1_seed300.json"),
    "Aquarium_shot3_seed3": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot3_seed3.json"),
    "Aquarium_shot3_seed30": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot3_seed30.json"),
    "Aquarium_shot3_seed300": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot3_seed300.json"),
    "Aquarium_shot5_seed3": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot5_seed3.json"),
    "Aquarium_shot5_seed30": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot5_seed30.json"),
    "Aquarium_shot5_seed300": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot5_seed300.json"),
    "Aquarium_shot10_seed3": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot10_seed3.json"),
    "Aquarium_shot10_seed30": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot10_seed30.json"),
    "Aquarium_shot10_seed300": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/fewshot_train_shot10_seed300.json"),
    "BCCD_shot1_seed3": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot1_seed3.json"),
    "BCCD_shot1_seed30": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot1_seed30.json"),
    "BCCD_shot1_seed300": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot1_seed300.json"),
    "BCCD_shot3_seed3": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot3_seed3.json"),
    "BCCD_shot3_seed30": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot3_seed30.json"),
    "BCCD_shot3_seed300": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot3_seed300.json"),
    "BCCD_shot5_seed3": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot5_seed3.json"),
    "BCCD_shot5_seed30": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot5_seed30.json"),
    "BCCD_shot5_seed300": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot5_seed300.json"),
    "BCCD_shot10_seed3": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot10_seed3.json"),
    "BCCD_shot10_seed30": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot10_seed30.json"),
    "BCCD_shot10_seed300": ("odinw/BCCD/BCCD.v3-raw.coco/train", "odinw/BCCD/BCCD.v3-raw.coco/train/fewshot_train_shot10_seed300.json"),
    "boggleBoards_shot1_seed3": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot1_seed3.json"),
    "boggleBoards_shot1_seed30": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot1_seed30.json"),
    "boggleBoards_shot1_seed300": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot1_seed300.json"),
    "boggleBoards_shot3_seed3": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot3_seed3.json"),
    "boggleBoards_shot3_seed30": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot3_seed30.json"),
    "boggleBoards_shot3_seed300": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot3_seed300.json"),
    "boggleBoards_shot5_seed3": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot5_seed3.json"),
    "boggleBoards_shot5_seed30": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot5_seed30.json"),
    "boggleBoards_shot5_seed300": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot5_seed300.json"),
    "boggleBoards_shot10_seed3": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot10_seed3.json"),
    "boggleBoards_shot10_seed30": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot10_seed30.json"),
    "boggleBoards_shot10_seed300": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/fewshot_train_shot10_seed300.json"),
    "brackishUnderwater_shot1_seed3": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot1_seed3.json"),
    "brackishUnderwater_shot1_seed30": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot1_seed30.json"),
    "brackishUnderwater_shot1_seed300": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot1_seed300.json"),
    "brackishUnderwater_shot3_seed3": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot3_seed3.json"),
    "brackishUnderwater_shot3_seed30": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot3_seed30.json"),
    "brackishUnderwater_shot3_seed300": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot3_seed300.json"),
    "brackishUnderwater_shot5_seed3": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot5_seed3.json"),
    "brackishUnderwater_shot5_seed30": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot5_seed30.json"),
    "brackishUnderwater_shot5_seed300": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot5_seed300.json"),
    "brackishUnderwater_shot10_seed3": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot10_seed3.json"),
    "brackishUnderwater_shot10_seed30": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot10_seed30.json"),
    "brackishUnderwater_shot10_seed300": ("odinw/brackishUnderwater/960x540/train", "odinw/brackishUnderwater/960x540/train/fewshot_train_shot10_seed300.json"),
    "ChessPieces_shot1_seed3": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot1_seed3.json"),
    "ChessPieces_shot1_seed30": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot1_seed30.json"),
    "ChessPieces_shot1_seed300": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot1_seed300.json"),
    "ChessPieces_shot3_seed3": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot3_seed3.json"),
    "ChessPieces_shot3_seed30": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot3_seed30.json"),
    "ChessPieces_shot3_seed300": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot3_seed300.json"),
    "ChessPieces_shot5_seed3": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot5_seed3.json"),
    "ChessPieces_shot5_seed30": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot5_seed30.json"),
    "ChessPieces_shot5_seed300": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot5_seed300.json"),
    "ChessPieces_shot10_seed3": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot10_seed3.json"),
    "ChessPieces_shot10_seed30": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot10_seed30.json"),
    "ChessPieces_shot10_seed300": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/train", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/fewshot_train_shot10_seed300.json"),
    "CottontailRabbits_shot1_seed3": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot1_seed3.json"),
    "CottontailRabbits_shot1_seed30": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot1_seed30.json"),
    "CottontailRabbits_shot1_seed300": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot1_seed300.json"),
    "CottontailRabbits_shot3_seed3": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot3_seed3.json"),
    "CottontailRabbits_shot3_seed30": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot3_seed30.json"),
    "CottontailRabbits_shot3_seed300": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot3_seed300.json"),
    "CottontailRabbits_shot5_seed3": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot5_seed3.json"),
    "CottontailRabbits_shot5_seed30": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot5_seed30.json"),
    "CottontailRabbits_shot5_seed300": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot5_seed300.json"),
    "CottontailRabbits_shot10_seed3": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot10_seed3.json"),
    "CottontailRabbits_shot10_seed30": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot10_seed30.json"),
    "CottontailRabbits_shot10_seed300": ("odinw/CottontailRabbits/train", "odinw/CottontailRabbits/train/fewshot_train_shot10_seed300.json"),
    "dice_mediumColor_shot1_seed3": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot1_seed3.json"),
    "dice_mediumColor_shot1_seed30": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot1_seed30.json"),
    "dice_mediumColor_shot1_seed300": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot1_seed300.json"),
    "dice_mediumColor_shot3_seed3": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot3_seed3.json"),
    "dice_mediumColor_shot3_seed30": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot3_seed30.json"),
    "dice_mediumColor_shot3_seed300": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot3_seed300.json"),
    "dice_mediumColor_shot5_seed3": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot5_seed3.json"),
    "dice_mediumColor_shot5_seed30": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot5_seed30.json"),
    "dice_mediumColor_shot5_seed300": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot5_seed300.json"),
    "dice_mediumColor_shot10_seed3": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot10_seed3.json"),
    "dice_mediumColor_shot10_seed30": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot10_seed30.json"),
    "dice_mediumColor_shot10_seed300": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/fewshot_train_shot10_seed300.json"),
    "DroneControl_shot1_seed3": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot1_seed3.json"),
    "DroneControl_shot1_seed30": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot1_seed30.json"),
    "DroneControl_shot1_seed300": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot1_seed300.json"),
    "DroneControl_shot3_seed3": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot3_seed3.json"),
    "DroneControl_shot3_seed30": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot3_seed30.json"),
    "DroneControl_shot3_seed300": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot3_seed300.json"),
    "DroneControl_shot5_seed3": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot5_seed3.json"),
    "DroneControl_shot5_seed30": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot5_seed30.json"),
    "DroneControl_shot5_seed300": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot5_seed300.json"),
    "DroneControl_shot10_seed3": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot10_seed3.json"),
    "DroneControl_shot10_seed30": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot10_seed30.json"),
    "DroneControl_shot10_seed300": ("odinw/DroneControl/Drone Control.v3-raw.coco/train", "odinw/DroneControl/Drone Control.v3-raw.coco/train/fewshot_train_shot10_seed300.json"),
    "EgoHands_generic_shot1_seed3": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot1_seed3.json"),
    "EgoHands_generic_shot1_seed30": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot1_seed30.json"),
    "EgoHands_generic_shot1_seed300": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot1_seed300.json"),
    "EgoHands_generic_shot3_seed3": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot3_seed3.json"),
    "EgoHands_generic_shot3_seed30": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot3_seed30.json"),
    "EgoHands_generic_shot3_seed300": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot3_seed300.json"),
    "EgoHands_generic_shot5_seed3": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot5_seed3.json"),
    "EgoHands_generic_shot5_seed30": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot5_seed30.json"),
    "EgoHands_generic_shot5_seed300": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot5_seed300.json"),
    "EgoHands_generic_shot10_seed3": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot10_seed3.json"),
    "EgoHands_generic_shot10_seed30": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot10_seed30.json"),
    "EgoHands_generic_shot10_seed300": ("odinw/EgoHands/generic/train", "odinw/EgoHands/generic/train/fewshot_train_shot10_seed300.json"),
    "EgoHands_specific_shot1_seed3": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot1_seed3.json"),
    "EgoHands_specific_shot1_seed30": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot1_seed30.json"),
    "EgoHands_specific_shot1_seed300": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot1_seed300.json"),
    "EgoHands_specific_shot3_seed3": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot3_seed3.json"),
    "EgoHands_specific_shot3_seed30": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot3_seed30.json"),
    "EgoHands_specific_shot3_seed300": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot3_seed300.json"),
    "EgoHands_specific_shot5_seed3": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot5_seed3.json"),
    "EgoHands_specific_shot5_seed30": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot5_seed30.json"),
    "EgoHands_specific_shot5_seed300": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot5_seed300.json"),
    "EgoHands_specific_shot10_seed3": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot10_seed3.json"),
    "EgoHands_specific_shot10_seed30": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot10_seed30.json"),
    "EgoHands_specific_shot10_seed300": ("odinw/EgoHands/specific/train", "odinw/EgoHands/specific/train/fewshot_train_shot10_seed300.json"),
    "HardHatWorkers_shot1_seed3": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot1_seed3.json"),
    "HardHatWorkers_shot1_seed30": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot1_seed30.json"),
    "HardHatWorkers_shot1_seed300": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot1_seed300.json"),
    "HardHatWorkers_shot3_seed3": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot3_seed3.json"),
    "HardHatWorkers_shot3_seed30": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot3_seed30.json"),
    "HardHatWorkers_shot3_seed300": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot3_seed300.json"),
    "HardHatWorkers_shot5_seed3": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot5_seed3.json"),
    "HardHatWorkers_shot5_seed30": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot5_seed30.json"),
    "HardHatWorkers_shot5_seed300": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot5_seed300.json"),
    "HardHatWorkers_shot10_seed3": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot10_seed3.json"),
    "HardHatWorkers_shot10_seed30": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot10_seed30.json"),
    "HardHatWorkers_shot10_seed300": ("odinw/HardHatWorkers/raw/train", "odinw/HardHatWorkers/raw/train/fewshot_train_shot10_seed300.json"),
    "MaskWearing_shot1_seed3": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot1_seed3.json"),
    "MaskWearing_shot1_seed30": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot1_seed30.json"),
    "MaskWearing_shot1_seed300": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot1_seed300.json"),
    "MaskWearing_shot3_seed3": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot3_seed3.json"),
    "MaskWearing_shot3_seed30": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot3_seed30.json"),
    "MaskWearing_shot3_seed300": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot3_seed300.json"),
    "MaskWearing_shot5_seed3": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot5_seed3.json"),
    "MaskWearing_shot5_seed30": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot5_seed30.json"),
    "MaskWearing_shot5_seed300": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot5_seed300.json"),
    "MaskWearing_shot10_seed3": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot10_seed3.json"),
    "MaskWearing_shot10_seed30": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot10_seed30.json"),
    "MaskWearing_shot10_seed300": ("odinw/MaskWearing/raw/train", "odinw/MaskWearing/raw/train/fewshot_train_shot10_seed300.json"),
    "MountainDewCommercial_shot1_seed3": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot1_seed3.json"),
    "MountainDewCommercial_shot1_seed30": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot1_seed30.json"),
    "MountainDewCommercial_shot1_seed300": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot1_seed300.json"),
    "MountainDewCommercial_shot3_seed3": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot3_seed3.json"),
    "MountainDewCommercial_shot3_seed30": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot3_seed30.json"),
    "MountainDewCommercial_shot3_seed300": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot3_seed300.json"),
    "MountainDewCommercial_shot5_seed3": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot5_seed3.json"),
    "MountainDewCommercial_shot5_seed30": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot5_seed30.json"),
    "MountainDewCommercial_shot5_seed300": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot5_seed300.json"),
    "MountainDewCommercial_shot10_seed3": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot10_seed3.json"),
    "MountainDewCommercial_shot10_seed30": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot10_seed30.json"),
    "MountainDewCommercial_shot10_seed300": ("odinw/MountainDewCommercial/train", "odinw/MountainDewCommercial/train/fewshot_train_shot10_seed300.json"),
    "NorthAmericaMushrooms_shot1_seed3": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot1_seed3.json"),
    "NorthAmericaMushrooms_shot1_seed30": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot1_seed30.json"),
    "NorthAmericaMushrooms_shot1_seed300": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot1_seed300.json"),
    "NorthAmericaMushrooms_shot3_seed3": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot3_seed3.json"),
    "NorthAmericaMushrooms_shot3_seed30": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot3_seed30.json"),
    "NorthAmericaMushrooms_shot3_seed300": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot3_seed300.json"),
    "NorthAmericaMushrooms_shot5_seed3": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot5_seed3.json"),
    "NorthAmericaMushrooms_shot5_seed30": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot5_seed30.json"),
    "NorthAmericaMushrooms_shot5_seed300": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot5_seed300.json"),
    "NorthAmericaMushrooms_shot10_seed3": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot10_seed3.json"),
    "NorthAmericaMushrooms_shot10_seed30": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot10_seed30.json"),
    "NorthAmericaMushrooms_shot10_seed300": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/fewshot_train_shot10_seed300.json"),
    "openPoetryVision_shot1_seed3": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot1_seed3.json"),
    "openPoetryVision_shot1_seed30": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot1_seed30.json"),
    "openPoetryVision_shot1_seed300": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot1_seed300.json"),
    "openPoetryVision_shot3_seed3": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot3_seed3.json"),
    "openPoetryVision_shot3_seed30": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot3_seed30.json"),
    "openPoetryVision_shot3_seed300": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot3_seed300.json"),
    "openPoetryVision_shot5_seed3": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot5_seed3.json"),
    "openPoetryVision_shot5_seed30": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot5_seed30.json"),
    "openPoetryVision_shot5_seed300": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot5_seed300.json"),
    "openPoetryVision_shot10_seed3": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot10_seed3.json"),
    "openPoetryVision_shot10_seed30": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot10_seed30.json"),
    "openPoetryVision_shot10_seed300": ("odinw/openPoetryVision/512x512/train", "odinw/openPoetryVision/512x512/train/fewshot_train_shot10_seed300.json"),
    "OxfordPets_by-breed_shot1_seed3": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot1_seed3.json"),
    "OxfordPets_by-breed_shot1_seed30": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot1_seed30.json"),
    "OxfordPets_by-breed_shot1_seed300": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot1_seed300.json"),
    "OxfordPets_by-breed_shot3_seed3": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot3_seed3.json"),
    "OxfordPets_by-breed_shot3_seed30": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot3_seed30.json"),
    "OxfordPets_by-breed_shot3_seed300": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot3_seed300.json"),
    "OxfordPets_by-breed_shot5_seed3": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot5_seed3.json"),
    "OxfordPets_by-breed_shot5_seed30": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot5_seed30.json"),
    "OxfordPets_by-breed_shot5_seed300": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot5_seed300.json"),
    "OxfordPets_by-breed_shot10_seed3": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot10_seed3.json"),
    "OxfordPets_by-breed_shot10_seed30": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot10_seed30.json"),
    "OxfordPets_by-breed_shot10_seed300": ("odinw/OxfordPets/by-breed/train", "odinw/OxfordPets/by-breed/train/fewshot_train_shot10_seed300.json"),
    "OxfordPets_by-species_shot1_seed3": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot1_seed3.json"),
    "OxfordPets_by-species_shot1_seed30": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot1_seed30.json"),
    "OxfordPets_by-species_shot1_seed300": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot1_seed300.json"),
    "OxfordPets_by-species_shot3_seed3": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot3_seed3.json"),
    "OxfordPets_by-species_shot3_seed30": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot3_seed30.json"),
    "OxfordPets_by-species_shot3_seed300": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot3_seed300.json"),
    "OxfordPets_by-species_shot5_seed3": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot5_seed3.json"),
    "OxfordPets_by-species_shot5_seed30": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot5_seed30.json"),
    "OxfordPets_by-species_shot5_seed300": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot5_seed300.json"),
    "OxfordPets_by-species_shot10_seed3": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot10_seed3.json"),
    "OxfordPets_by-species_shot10_seed30": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot10_seed30.json"),
    "OxfordPets_by-species_shot10_seed300": ("odinw/OxfordPets/by-species/train", "odinw/OxfordPets/by-species/train/fewshot_train_shot10_seed300.json"),
    "Packages_shot1_seed3": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot1_seed3.json"),
    "Packages_shot1_seed30": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot1_seed30.json"),
    "Packages_shot1_seed300": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot1_seed300.json"),
    "Packages_shot3_seed3": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot3_seed3.json"),
    "Packages_shot3_seed30": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot3_seed30.json"),
    "Packages_shot3_seed300": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot3_seed300.json"),
    "Packages_shot5_seed3": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot5_seed3.json"),
    "Packages_shot5_seed30": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot5_seed30.json"),
    "Packages_shot5_seed300": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot5_seed300.json"),
    "Packages_shot10_seed3": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot10_seed3.json"),
    "Packages_shot10_seed30": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot10_seed30.json"),
    "Packages_shot10_seed300": ("odinw/Packages/Raw/train", "odinw/Packages/Raw/train/fewshot_train_shot10_seed300.json"),
    "PascalVOC_shot1_seed3": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot1_seed3.json"),
    "PascalVOC_shot1_seed30": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot1_seed30.json"),
    "PascalVOC_shot1_seed300": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot1_seed300.json"),
    "PascalVOC_shot3_seed3": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot3_seed3.json"),
    "PascalVOC_shot3_seed30": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot3_seed30.json"),
    "PascalVOC_shot3_seed300": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot3_seed300.json"),
    "PascalVOC_shot5_seed3": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot5_seed3.json"),
    "PascalVOC_shot5_seed30": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot5_seed30.json"),
    "PascalVOC_shot5_seed300": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot5_seed300.json"),
    "PascalVOC_shot10_seed3": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot10_seed3.json"),
    "PascalVOC_shot10_seed30": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot10_seed30.json"),
    "PascalVOC_shot10_seed300": ("odinw/PascalVOC/train", "odinw/PascalVOC/train/fewshot_train_shot10_seed300.json"),
    "pistols_shot1_seed3": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot1_seed3.json"),
    "pistols_shot1_seed30": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot1_seed30.json"),
    "pistols_shot1_seed300": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot1_seed300.json"),
    "pistols_shot3_seed3": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot3_seed3.json"),
    "pistols_shot3_seed30": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot3_seed30.json"),
    "pistols_shot3_seed300": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot3_seed300.json"),
    "pistols_shot5_seed3": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot5_seed3.json"),
    "pistols_shot5_seed30": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot5_seed30.json"),
    "pistols_shot5_seed300": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot5_seed300.json"),
    "pistols_shot10_seed3": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot10_seed3.json"),
    "pistols_shot10_seed30": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot10_seed30.json"),
    "pistols_shot10_seed300": ("odinw/pistols/export", "odinw/pistols/export/fewshot_train_shot10_seed300.json"),
    "PKLot_shot1_seed3": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot1_seed3.json"),
    "PKLot_shot1_seed30": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot1_seed30.json"),
    "PKLot_shot1_seed300": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot1_seed300.json"),
    "PKLot_shot3_seed3": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot3_seed3.json"),
    "PKLot_shot3_seed30": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot3_seed30.json"),
    "PKLot_shot3_seed300": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot3_seed300.json"),
    "PKLot_shot5_seed3": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot5_seed3.json"),
    "PKLot_shot5_seed30": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot5_seed30.json"),
    "PKLot_shot5_seed300": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot5_seed300.json"),
    "PKLot_shot10_seed3": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot10_seed3.json"),
    "PKLot_shot10_seed30": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot10_seed30.json"),
    "PKLot_shot10_seed300": ("odinw/PKLot/640/train", "odinw/PKLot/640/train/fewshot_train_shot10_seed300.json"),
    "plantdoc_shot1_seed3": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot1_seed3.json"),
    "plantdoc_shot1_seed30": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot1_seed30.json"),
    "plantdoc_shot1_seed300": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot1_seed300.json"),
    "plantdoc_shot3_seed3": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot3_seed3.json"),
    "plantdoc_shot3_seed30": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot3_seed30.json"),
    "plantdoc_shot3_seed300": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot3_seed300.json"),
    "plantdoc_shot5_seed3": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot5_seed3.json"),
    "plantdoc_shot5_seed30": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot5_seed30.json"),
    "plantdoc_shot5_seed300": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot5_seed300.json"),
    "plantdoc_shot10_seed3": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot10_seed3.json"),
    "plantdoc_shot10_seed30": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot10_seed30.json"),
    "plantdoc_shot10_seed300": ("odinw/plantdoc/416x416/train", "odinw/plantdoc/416x416/train/fewshot_train_shot10_seed300.json"),
    "pothole_shot1_seed3": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot1_seed3.json"),
    "pothole_shot1_seed30": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot1_seed30.json"),
    "pothole_shot1_seed300": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot1_seed300.json"),
    "pothole_shot3_seed3": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot3_seed3.json"),
    "pothole_shot3_seed30": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot3_seed30.json"),
    "pothole_shot3_seed300": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot3_seed300.json"),
    "pothole_shot5_seed3": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot5_seed3.json"),
    "pothole_shot5_seed30": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot5_seed30.json"),
    "pothole_shot5_seed300": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot5_seed300.json"),
    "pothole_shot10_seed3": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot10_seed3.json"),
    "pothole_shot10_seed30": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot10_seed30.json"),
    "pothole_shot10_seed300": ("odinw/pothole/train", "odinw/pothole/train/fewshot_train_shot10_seed300.json"),
    "Raccoon_shot1_seed3": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot1_seed3.json"),
    "Raccoon_shot1_seed30": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot1_seed30.json"),
    "Raccoon_shot1_seed300": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot1_seed300.json"),
    "Raccoon_shot3_seed3": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot3_seed3.json"),
    "Raccoon_shot3_seed30": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot3_seed30.json"),
    "Raccoon_shot3_seed300": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot3_seed300.json"),
    "Raccoon_shot5_seed3": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot5_seed3.json"),
    "Raccoon_shot5_seed30": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot5_seed30.json"),
    "Raccoon_shot5_seed300": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot5_seed300.json"),
    "Raccoon_shot10_seed3": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot10_seed3.json"),
    "Raccoon_shot10_seed30": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot10_seed30.json"),
    "Raccoon_shot10_seed300": ("odinw/Raccoon/Raccoon.v2-raw.coco/train", "odinw/Raccoon/Raccoon.v2-raw.coco/train/fewshot_train_shot10_seed300.json"),
    "selfdrivingCar_shot1_seed3": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot1_seed3.json"),
    "selfdrivingCar_shot1_seed30": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot1_seed30.json"),
    "selfdrivingCar_shot1_seed300": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot1_seed300.json"),
    "selfdrivingCar_shot3_seed3": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot3_seed3.json"),
    "selfdrivingCar_shot3_seed30": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot3_seed30.json"),
    "selfdrivingCar_shot3_seed300": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot3_seed300.json"),
    "selfdrivingCar_shot5_seed3": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot5_seed3.json"),
    "selfdrivingCar_shot5_seed30": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot5_seed30.json"),
    "selfdrivingCar_shot5_seed300": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot5_seed300.json"),
    "selfdrivingCar_shot10_seed3": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot10_seed3.json"),
    "selfdrivingCar_shot10_seed30": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot10_seed30.json"),
    "selfdrivingCar_shot10_seed300": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/fewshot_train_shot10_seed300.json"),
    "ShellfishOpenImages_shot1_seed3": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot1_seed3.json"),
    "ShellfishOpenImages_shot1_seed30": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot1_seed30.json"),
    "ShellfishOpenImages_shot1_seed300": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot1_seed300.json"),
    "ShellfishOpenImages_shot3_seed3": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot3_seed3.json"),
    "ShellfishOpenImages_shot3_seed30": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot3_seed30.json"),
    "ShellfishOpenImages_shot3_seed300": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot3_seed300.json"),
    "ShellfishOpenImages_shot5_seed3": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot5_seed3.json"),
    "ShellfishOpenImages_shot5_seed30": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot5_seed30.json"),
    "ShellfishOpenImages_shot5_seed300": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot5_seed300.json"),
    "ShellfishOpenImages_shot10_seed3": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot10_seed3.json"),
    "ShellfishOpenImages_shot10_seed30": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot10_seed30.json"),
    "ShellfishOpenImages_shot10_seed300": ("odinw/ShellfishOpenImages/raw/train", "odinw/ShellfishOpenImages/raw/train/fewshot_train_shot10_seed300.json"),
    "ThermalCheetah_shot1_seed3": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot1_seed3.json"),
    "ThermalCheetah_shot1_seed30": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot1_seed30.json"),
    "ThermalCheetah_shot1_seed300": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot1_seed300.json"),
    "ThermalCheetah_shot3_seed3": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot3_seed3.json"),
    "ThermalCheetah_shot3_seed30": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot3_seed30.json"),
    "ThermalCheetah_shot3_seed300": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot3_seed300.json"),
    "ThermalCheetah_shot5_seed3": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot5_seed3.json"),
    "ThermalCheetah_shot5_seed30": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot5_seed30.json"),
    "ThermalCheetah_shot5_seed300": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot5_seed300.json"),
    "ThermalCheetah_shot10_seed3": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot10_seed3.json"),
    "ThermalCheetah_shot10_seed30": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot10_seed30.json"),
    "ThermalCheetah_shot10_seed300": ("odinw/ThermalCheetah/train", "odinw/ThermalCheetah/train/fewshot_train_shot10_seed300.json"),
    "thermalDogsAndPeople_shot1_seed3": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot1_seed3.json"),
    "thermalDogsAndPeople_shot1_seed30": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot1_seed30.json"),
    "thermalDogsAndPeople_shot1_seed300": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot1_seed300.json"),
    "thermalDogsAndPeople_shot3_seed3": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot3_seed3.json"),
    "thermalDogsAndPeople_shot3_seed30": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot3_seed30.json"),
    "thermalDogsAndPeople_shot3_seed300": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot3_seed300.json"),
    "thermalDogsAndPeople_shot5_seed3": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot5_seed3.json"),
    "thermalDogsAndPeople_shot5_seed30": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot5_seed30.json"),
    "thermalDogsAndPeople_shot5_seed300": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot5_seed300.json"),
    "thermalDogsAndPeople_shot10_seed3": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot10_seed3.json"),
    "thermalDogsAndPeople_shot10_seed30": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot10_seed30.json"),
    "thermalDogsAndPeople_shot10_seed300": ("odinw/thermalDogsAndPeople/train", "odinw/thermalDogsAndPeople/train/fewshot_train_shot10_seed300.json"),
    "UnoCards_shot1_seed3": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot1_seed3.json"),
    "UnoCards_shot1_seed30": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot1_seed30.json"),
    "UnoCards_shot1_seed300": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot1_seed300.json"),
    "UnoCards_shot3_seed3": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot3_seed3.json"),
    "UnoCards_shot3_seed30": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot3_seed30.json"),
    "UnoCards_shot3_seed300": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot3_seed300.json"),
    "UnoCards_shot5_seed3": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot5_seed3.json"),
    "UnoCards_shot5_seed30": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot5_seed30.json"),
    "UnoCards_shot5_seed300": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot5_seed300.json"),
    "UnoCards_shot10_seed3": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot10_seed3.json"),
    "UnoCards_shot10_seed30": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot10_seed30.json"),
    "UnoCards_shot10_seed300": ("odinw/UnoCards/raw/train", "odinw/UnoCards/raw/train/fewshot_train_shot10_seed300.json"),
    "VehiclesOpenImages_shot1_seed3": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot1_seed3.json"),
    "VehiclesOpenImages_shot1_seed30": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot1_seed30.json"),
    "VehiclesOpenImages_shot1_seed300": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot1_seed300.json"),
    "VehiclesOpenImages_shot3_seed3": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot3_seed3.json"),
    "VehiclesOpenImages_shot3_seed30": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot3_seed30.json"),
    "VehiclesOpenImages_shot3_seed300": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot3_seed300.json"),
    "VehiclesOpenImages_shot5_seed3": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot5_seed3.json"),
    "VehiclesOpenImages_shot5_seed30": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot5_seed30.json"),
    "VehiclesOpenImages_shot5_seed300": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot5_seed300.json"),
    "VehiclesOpenImages_shot10_seed3": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot10_seed3.json"),
    "VehiclesOpenImages_shot10_seed30": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot10_seed30.json"),
    "VehiclesOpenImages_shot10_seed300": ("odinw/VehiclesOpenImages/416x416/train", "odinw/VehiclesOpenImages/416x416/train/fewshot_train_shot10_seed300.json"),
    "websiteScreenshots_shot1_seed3": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot1_seed3.json"),
    "websiteScreenshots_shot1_seed30": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot1_seed30.json"),
    "websiteScreenshots_shot1_seed300": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot1_seed300.json"),
    "websiteScreenshots_shot3_seed3": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot3_seed3.json"),
    "websiteScreenshots_shot3_seed30": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot3_seed30.json"),
    "websiteScreenshots_shot3_seed300": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot3_seed300.json"),
    "websiteScreenshots_shot5_seed3": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot5_seed3.json"),
    "websiteScreenshots_shot5_seed30": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot5_seed30.json"),
    "websiteScreenshots_shot5_seed300": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot5_seed300.json"),
    "websiteScreenshots_shot10_seed3": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot10_seed3.json"),
    "websiteScreenshots_shot10_seed30": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot10_seed30.json"),
    "websiteScreenshots_shot10_seed300": ("odinw/websiteScreenshots/train", "odinw/websiteScreenshots/train/fewshot_train_shot10_seed300.json"),
    "WildfireSmoke_shot1_seed3": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot1_seed3.json"),
    "WildfireSmoke_shot1_seed30": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot1_seed30.json"),
    "WildfireSmoke_shot1_seed300": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot1_seed300.json"),
    "WildfireSmoke_shot3_seed3": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot3_seed3.json"),
    "WildfireSmoke_shot3_seed30": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot3_seed30.json"),
    "WildfireSmoke_shot3_seed300": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot3_seed300.json"),
    "WildfireSmoke_shot5_seed3": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot5_seed3.json"),
    "WildfireSmoke_shot5_seed30": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot5_seed30.json"),
    "WildfireSmoke_shot5_seed300": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot5_seed300.json"),
    "WildfireSmoke_shot10_seed3": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot10_seed3.json"),
    "WildfireSmoke_shot10_seed30": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot10_seed30.json"),
    "WildfireSmoke_shot10_seed300": ("odinw/WildfireSmoke/train", "odinw/WildfireSmoke/train/fewshot_train_shot10_seed300.json"),
    "AerialMaritimeDrone_large_test": ("odinw/AerialMaritimeDrone/large/test", "odinw/AerialMaritimeDrone/large/test/annotations_without_background.json"),
    "AerialMaritimeDrone_tiled_test": ("odinw/AerialMaritimeDrone/tiled/test", "odinw/AerialMaritimeDrone/tiled/test/annotations_without_background.json"),
    "AmericanSignLanguageLetters_test": ("odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/test", "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/test/annotations_without_background.json"),
    "Aquarium_test": ("odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/test", "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/test/annotations_without_background.json"),
    "BCCD_test": ("odinw/BCCD/BCCD.v3-raw.coco/test", "odinw/BCCD/BCCD.v3-raw.coco/test/annotations_without_background.json"),
    "boggleBoards_test": ("odinw/boggleBoards/416x416AutoOrient/export", "odinw/boggleBoards/416x416AutoOrient/export/test_annotations_without_background.json"),
    "brackishUnderwater_test": ("odinw/brackishUnderwater/960x540/test", "odinw/brackishUnderwater/960x540/test/annotations_without_background.json"),
    "ChessPieces_test": ("odinw/ChessPieces/Chess Pieces.v23-raw.coco/test", "odinw/ChessPieces/Chess Pieces.v23-raw.coco/test/annotations_without_background.json"),
    "CottontailRabbits_test": ("odinw/CottontailRabbits/test", "odinw/CottontailRabbits/test/annotations_without_background.json"),
    "dice_mediumColor_test": ("odinw/dice/mediumColor/export", "odinw/dice/mediumColor/export/test_annotations_without_background.json"),
    "DroneControl_test": ("odinw/DroneControl/Drone Control.v3-raw.coco/test", "odinw/DroneControl/Drone Control.v3-raw.coco/test/annotations_without_background.json"),
    "EgoHands_generic_test": ("odinw/EgoHands/generic/test", "odinw/EgoHands/generic/test/annotations_without_background.json"),
    "EgoHands_specific_test": ("odinw/EgoHands/specific/test", "odinw/EgoHands/specific/test/annotations_without_background.json"),
    "HardHatWorkers_test": ("odinw/HardHatWorkers/raw/test", "odinw/HardHatWorkers/raw/test/annotations_without_background.json"),
    "MaskWearing_test": ("odinw/MaskWearing/raw/test", "odinw/MaskWearing/raw/test/annotations_without_background.json"),
    "MountainDewCommercial_test": ("odinw/MountainDewCommercial/test", "odinw/MountainDewCommercial/test/annotations_without_background.json"),
    "NorthAmericaMushrooms_test": ("odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/test", "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/test/annotations_without_background.json"),
    "openPoetryVision_test": ("odinw/openPoetryVision/512x512/test", "odinw/openPoetryVision/512x512/test/annotations_without_background.json"),
    "OxfordPets_by-breed_test": ("odinw/OxfordPets/by-breed/test", "odinw/OxfordPets/by-breed/test/annotations_without_background.json"),
    "OxfordPets_by-species_test": ("odinw/OxfordPets/by-species/test", "odinw/OxfordPets/by-species/test/annotations_without_background.json"),
    "Packages_test": ("odinw/Packages/Raw/test", "odinw/Packages/Raw/test/annotations_without_background.json"),
    "PascalVOC_test": ("odinw/PascalVOC/valid", "odinw/PascalVOC/valid/annotations_without_background.json"),
    "pistols_test": ("odinw/pistols/export", "odinw/pistols/export/test_annotations_without_background.json"),
    "PKLot_test": ("odinw/PKLot/640/test", "odinw/PKLot/640/test/annotations_without_background.json"),
    "plantdoc_test": ("odinw/plantdoc/416x416/test", "odinw/plantdoc/416x416/test/annotations_without_background.json"),
    "pothole_test": ("odinw/pothole/test", "odinw/pothole/test/annotations_without_background.json"),
    "Raccoon_test": ("odinw/Raccoon/Raccoon.v2-raw.coco/test", "odinw/Raccoon/Raccoon.v2-raw.coco/test/annotations_without_background.json"),
    "selfdrivingCar_test": ("odinw/selfdrivingCar/fixedLarge/export", "odinw/selfdrivingCar/fixedLarge/export/test_annotations_without_background.json"),
    "ShellfishOpenImages_test": ("odinw/ShellfishOpenImages/raw/test", "odinw/ShellfishOpenImages/raw/test/annotations_without_background.json"),
    "ThermalCheetah_test": ("odinw/ThermalCheetah/test", "odinw/ThermalCheetah/test/annotations_without_background.json"),
    "thermalDogsAndPeople_test": ("odinw/thermalDogsAndPeople/test", "odinw/thermalDogsAndPeople/test/annotations_without_background.json"),
    "UnoCards_test": ("odinw/UnoCards/raw/test", "odinw/UnoCards/raw/test/annotations_without_background.json"),
    "VehiclesOpenImages_test": ("odinw/VehiclesOpenImages/416x416/test", "odinw/VehiclesOpenImages/416x416/test/annotations_without_background.json"),
    "websiteScreenshots_test": ("odinw/websiteScreenshots/test", "odinw/websiteScreenshots/test/annotations_without_background.json"),
    "WildfireSmoke_test": ("odinw/WildfireSmoke/test", "odinw/WildfireSmoke/test/annotations_without_background.json"),
}

# 这里训练 voc


_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        if dataset_name == 'coco_ovd' or dataset_name == 'coco_c' or dataset_name == 'voc_c' or dataset_name == 'voc' \
            or dataset_name == 'cityscapes' or dataset_name == 'cityscapes_c' or dataset_name == 'odinw' \
                or dataset_name == 'domain_generalization' or dataset_name == 'domain_adaptation' \
                    or dataset_name == 'fine_grained' or dataset_name == 'dense' \
                        or dataset_name == 'object365':  # for zero-shot split
            for key, (image_root, json_file) in splits_per_dataset.items():
                # Assume pre-defined datasets live in `./datasets`.
                register_coco_instances(
                    key,
                    {}, # empty metadata, it will be overwritten in load_coco_json() function
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )
        else: # default splits
            for key, (image_root, json_file) in splits_per_dataset.items():
                # Assume pre-defined datasets live in `./datasets`.
                register_coco_instances(
                    key,
                    _get_builtin_metadata(dataset_name),
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )

# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    # openset setting
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis_v1/annotations/lvis_v1_train.json"),
        "lvis_v1_val": ("lvis_v1/", "lvis_v1/annotations/lvis_v1_val.json"),
        "lvis_v1_minival": ("lvis_v1/", "lvis_v1/annotations/lvis_v1_minival.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    # custom image setting
    "lvis_v1_custom_img": {
        "lvis_v1_train_custom_img": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val_custom_img": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev_custom_img": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge_custom_img": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    # regular fully supervised setting
    "lvis_v1_fullysup": {
        "lvis_v1_train_fullysup": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val_fullysup": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev_fullysup": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge_fullysup": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            if dataset_name == "lvis_v1":
                args = {'filter_open_cls': True, 'run_custom_img': False}
            elif dataset_name == 'lvis_v1_custom_img':
                args = {'filter_open_cls': False, 'run_custom_img': True}
            elif dataset_name == 'lvis_v1_fullysup':
                args = {'filter_open_cls': False, 'run_custom_img': False}
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                args,
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)