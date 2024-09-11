
**Benchmark Datasets**: Two refined versions of two publicly available datasets: 
  - *[Pascal-Part-116](https://drive.google.com/file/d/1f5kqrM2_iK_bWmQBW3rdSnGrnke4PUbX/view?usp=sharing)*
  - *[ADE20K-Part-234](https://drive.google.com/file/d/1EBVPW_tqzBOQ_DC6yLcouyxR7WrctRKi/view)*


### Data Preparation
  After downloading the two benchmark datasets, please extract the files by running the following command and place the extracted folder under the "Datasets" directory.
  ```bash
  tar -xzf PascalPart116.tar.gz
  tar -xzf ADE20KPart234.tar.gz
  ```
  The Datasets folder should follow this structure:
  ```shell
  Datasets/
  ├─Pascal-Part-116/
  │ ├─train_16shot.json
  │ ├─images/
  │ │ ├─train/
  │ │ └─val/
  │ ├─annotations_detectron2_obj/
  │ │ ├─train/
  │ │ └─val/
  │ └─annotations_detectron2_part/
  │   ├─train/
  │   └─val/
  └─ADE20K-Part-234/
    ├─images/
    │ ├─training/
    │ ├─validation/
    ├─train_16shot.json
    ├─ade20k_instance_train.json
    ├─ade20k_instance_val.json
    └─annotations_detectron2_part/
      ├─training/
      └─validation/
  ```
  Create `{train/val}_{obj/part}_label_count.json` files for Pascal-Part-116.
  ```bash
  python baselines/data/datasets/mask_cls_collect.py Datasets/Pascal-Part-116/annotations_detectron2_{obj/part}/{train/val} Datasets/Pascal-Part-116/annotations_detectron2_part/{train/val}_{obj/part}_label_count.json
  ```
