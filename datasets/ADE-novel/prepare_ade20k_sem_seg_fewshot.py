# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import os.path as osp
from pathlib import Path

import numpy as np
import random
import json
import tqdm
from PIL import Image

from detectron2.data.datasets.builtin_meta import ADE20K_SEM_SEG_CATEGORIES

def convert(input, output, index=None):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    if index is not None:
        mapping = {i: k for k, i in enumerate(index)}
        img = np.vectorize(lambda x: mapping[x] if x in mapping else 255)(
            img.astype(np.float)
        ).astype(np.uint8)
    img = np.where(np.isin(img, ignore_cls_ids), 255, img)
    Image.fromarray(img).save(output)
    return img


if __name__ == "__main__":
    dataset_dir = (
        Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "ADEChallengeData2016"
    )
    print('Caution: we only generate the validation set!')

    support_file_names = {}
    with open("datasets/cls_json/ade150_novel.json", "r") as file:
        unseen_cls_names = json.load(file)

    unseen_cls_ids = []
    ignore_cls_ids = []
    cls_map = {}
    for name in ["validation"]:
        support_file_names[name] = []
        cls_dict = {} 
        cls_unseen = {}
        cls_ls_X = []
        annotation_dir = dataset_dir / "annotations" / name
        output_dir = dataset_dir / "support_annotations_detectron2" / name
        cls_dir = dataset_dir / "support_cls_annotations_detectron2" / name
        fewshot_cls_name_dir = dataset_dir / "fewshot_cls_name" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        cls_dir.mkdir(parents=True, exist_ok=True)
        fewshot_cls_name_dir.mkdir(parents=True, exist_ok=True)

        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            img = convert(file, output_file)
            for i in np.unique(img):
                if i == 255:
                    break
                cls_name = ADE20K_SEM_SEG_CATEGORIES[i]
                if cls_name in unseen_cls_names:
                    cls_dict.setdefault(cls_name, [])
                    base_name, extension = os.path.splitext(file.name)
                    cls_dict[cls_name].append(base_name)
                else:
                    ignore_cls_ids.append(i)
        for cls_name in tqdm.tqdm(cls_dict):
            if len(cls_dict[cls_name]) < 10:
                cls_ls_X.append(cls_name)
                ignore_cls_ids.append(i)
                continue
            support_files = random.sample(cls_dict[cls_name], 5)
            support_file_names[name].extend(support_files)
            for support_file in support_files:
                with open(osp.join(cls_dir, support_file+".json"), "w") as cls_anno_file:
                    json.dump({cls_name: ADE20K_SEM_SEG_CATEGORIES.index(cls_name)}, cls_anno_file)

        unseen_cls_names = list(cls_dict.keys())
        with open(osp.join(fewshot_cls_name_dir, "unseen_classnames.json"), "w") as fewshot_unseen_file:
            json.dump(unseen_cls_names, fewshot_unseen_file)
        print(f"classes: {cls_ls_X}, contain sample less than 10")
        with open(osp.join(fewshot_cls_name_dir, "ignore_classnames.json"), "w") as fewshot_ignore_file:
            json.dump(cls_ls_X, fewshot_ignore_file)
        
        cls_map = {i: (unseen_cls_names.index(cls_name) if cls_name in unseen_cls_names else 255) for i, cls_name in enumerate(ADE20K_SEM_SEG_CATEGORIES)}
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            img = convert(file, output_file, cls_map)


    for name in ["validation"]:
        annotation_dir = dataset_dir / "annotations" / name
        output_dir = dataset_dir / "query_annotations_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            file_base, _ = osp.splitext(file)
            if file_base in support_file_names[name]:
                continue
            output_file = output_dir / file.name
            convert(file, output_file, cls_map)