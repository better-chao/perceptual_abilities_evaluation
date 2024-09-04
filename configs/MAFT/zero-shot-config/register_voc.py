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

color = [{"color": [220, 20, 60],},
    {"color": [119, 11, 32], },
    {"color": [0, 0, 142],},
    {"color": [0, 0, 230], },
    {"color": [106, 0, 228], },
    {"color": [0, 60, 100], },
    {"color": [0, 80, 100], },
    {"color": [0, 0, 70], },
    {"color": [0, 0, 192], },
    {"color": [250, 170, 30], },
    {"color": [100, 170, 30],},
    {"color": [220, 220, 0], },
    {"color": [175, 116, 175], },
    {"color": [250, 0, 30], },
    {"color": [165, 42, 42], },
    {"color": [255, 77, 255], },
    {"color": [0, 226, 252], },
    {"color": [182, 182, 255], },
    {"color": [0, 82, 0],},
    {"color": [120, 166, 157], },]

def _get_voc_meta(cat_list):
    colorlist = [i['color'] for i in color]
    ret = {
        "stuff_classes": cat_list,
        "stuff_colors": colorlist,
    }
    return ret


def register_all_voc_11k(root):
    # root = os.path.join(root, "VOC2012")
    meta = _get_voc_meta(CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "JPEGImages", "annotations_ovs/train"),
        ("val", "JPEGImages", "annotations_ovs/val"),
        ("test_background", "JPEGImages", "annotations_detectron2_bg/val"),
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


_root = os.getenv("DETECTRON2_DATASETS", "datasets/VOCdevkit/VOC2012")
register_all_voc_11k(_root)
