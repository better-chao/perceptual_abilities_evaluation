import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CLASS_NAMES = (
        "Other", 
    "Facade", 
    "Road", 
    "Vegetation", 
    "Vehicle", 
    "Roof"
    )

PALETTE = [[0, 0, 0],
    [102, 102, 156],
    [128, 64, 128],
    [107, 142, 35],

    [0, 0, 142],
    [70, 0, 70]]  



def _get_udd6_meta():
    return {
        "stuff_classes": CLASS_NAMES,
       # "stuff_colors": CAMVID_COLORS,
    }


def register_all_udd6(root):
    root = os.path.join(root, "udd6")
    meta = _get_udd6_meta()

    for name, image_dirname, sem_seg_dirname in [
        #("train", "JPEGImages", "annotations_detectron2/train"),
        ("val", "val/src", "annotations_detectron2/val/gt"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"udd6_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="JPG"
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
register_all_udd6(_root)
