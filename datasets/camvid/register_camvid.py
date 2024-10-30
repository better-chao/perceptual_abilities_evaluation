import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CLASS_NAMES = (
        'Sky', 'Building', 'Pole', 'Road', 'Sidewalk',
        'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Void'
    )

PALETTE = [[128, 128, 128],
           [128, 0, 0],
           [192, 192, 128],
           [128, 64, 128],
           [0, 0, 192],
           [128, 128, 0],
           [192, 128, 128],
           [64, 64, 128],
           [64, 0, 128],
           [64, 64, 0],
           [0, 128, 192],
           [0,0,0]]  



def _get_camvid_meta():
    return {
        "stuff_classes": CLASS_NAMES,
       # "stuff_colors": CAMVID_COLORS,
    }


def register_all_camvid(root):
    root = os.path.join(root, "camvid")
    meta = _get_camvid_meta()

    for name, image_dirname, sem_seg_dirname in [
        #("train", "JPEGImages", "annotations_detectron2/train"),
        ("test", "camvid_test", "annotations_detectron2/camvid_test_labels"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"camvid_sem_seg_{name}"
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
register_all_camvid(_root)
