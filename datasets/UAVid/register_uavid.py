import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CLASS_NAMES = (
        "Background",    # 背景
    "Building",      # 建筑物
    "Tree",          # 树木
    "Moving Car",    # 移动车辆
    "Road",          # 道路
    "Low Vegetation",# 低矮植被
    "Static Car",    # 静止车辆
    "Human"
    )

PALETTE = [[0, 0, 0],
           [128, 0, 0],
           [0, 128, 0],
           [192, 0, 192],
           [128, 64, 128],
           [64, 64, 0],
           [64, 0, 128],
           [128, 128, 0]]  



def _get_uavid_meta():
    return {
        "stuff_classes": CLASS_NAMES,
       # "stuff_colors": CAMVID_COLORS,
    }


def register_all_uavid(root):
    root = os.path.join(root, "uavid")
    meta = _get_uavid_meta()

    for name, image_dirname, sem_seg_dirname in [
        #("train", "JPEGImages", "annotations_detectron2/train"),
        ("val", "val/Images", "annotations_detectron2/val/Labels"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"uavid_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="png"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_uavid(_root)
