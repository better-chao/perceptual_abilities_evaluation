import os

from .custom_ovd import register_custom_ovd_instances


CUSTOM_CATEGORIES = []

NUM_CATEGORY = len(CUSTOM_CATEGORIES)


def _get_custom_instances_meta():
    thing_ids = [k["id"] for k in CUSTOM_CATEGORIES]
    assert len(thing_ids) == NUM_CATEGORY, len(thing_ids)
    # Mapping from the incontiguous category id to contiguous id.
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CUSTOM_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

_PREDEFINED_SPLITS = {}

def register_all_custom_instances(root):
    for key, (
        image_root,
        json_file,
        num_sampled_classes,
        template,
    ) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datas`.
        register_custom_ovd_instances(
            key,
            _get_custom_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            num_sampled_classes,
            template=template,
            test_mode=True if "val" in key else False,
        )

def update(val):
    global NUM_CATEGORY
    NUM_CATEGORY = val

_root = ''
register_all_custom_instances(_root)
