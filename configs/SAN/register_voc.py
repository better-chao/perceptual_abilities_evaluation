import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tv",
]


def _get_voc_meta(cat_list):
    voc_colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                  [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
                  [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                  [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
                  [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    
    ret = {
        "stuff_classes": cat_list,
        "stuff_colors" : voc_colors,
    }
    return ret


def register_all_voc_11k(root):
    root = os.path.join(root, "VOC2012")
    meta = _get_voc_meta(CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "JPEGImages", "annotations_detectron2/train"),
        ("val", "JPEGImages", "annotations_detectron2/val"),
        ("test_background", "JPEGImages", "annotations_detectron2_bg/val")
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"voc_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        if "background" in name:
            MetadataCatalog.get(all_name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg_background", ignore_label=255,
                                          stuff_classes=meta["stuff_classes"] + ["background"], stuff_colors=meta["stuff_colors"])
        else:
            MetadataCatalog.get(all_name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255,
                **meta,
            )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_voc_11k(_root)
