import os
import json
import torch

# # 抽取新的数据集的概念
# ann_file_path = '/gpfsdata/home/huangziyue/data/VOC/coco_annotations/test_coco_ann.json'
# concepts_dir = '/gpfsdata/home/yangshuai/open_vocabulary/RegionCLIP/concepts/VOC'

# ann_file_path = '/home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json'
# concepts_dir = '/gpfsdata/home/yangshuai/open_vocabulary/RegionCLIP/concepts/cityscapes'

ann_file_path = '/home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json'
concepts_dir = '/gpfsdata/home/yangshuai/open_vocabulary/RegionCLIP/concepts/foggy_cityscapes'

# os.makedirs(concepts_dir, exist_ok = True)

with open(ann_file_path, 'r') as ann_file:
    data = json.load(ann_file)

print(data['categories'])

# categories = data['categories']

# concepts_file_path = os.path.join(concepts_dir, 'concepts.txt')

# with open(concepts_file_path, 'w') as save_file:
#     for item in categories:
#         class_name = item['name']
#         save_file.write(class_name + '\n')


# 注册新的数据集时用到的，写在builtin.py
# img_dir = '/gpfsdata/home/huangziyue/data/VOC/VOC_test_corruptions'

# i = 0
# for dir_name in os.listdir(img_dir):
#     i += 1
#     # "coco_2017_ovd_all_train": ("coco/train2017", "coco/regionclip_ov_annotations/ovd_ins_train2017_all.json"),
#     print(f"\"{dir_name}\": (\"VOC/VOC_test_corruptions/{dir_name}\", \"VOC/coco_annotations/test_coco_ann.json\"),")

# print(f"总共{i}组")

# 注册coco-c时用到的，写在builtin.py
# img_dir = '/gpfsdata/home/huangziyue/data/coco_new/val2017_corruptions'

# i = 0
# for dir_name in os.listdir(img_dir):
#     i += 1
#     # "coco_2017_ovd_all_train": ("coco/train2017", "coco/regionclip_ov_annotations/ovd_ins_train2017_all.json"),
#     print(f"\"coco_c_{dir_name}\": (\"coco_new/val2017_corruptions/{dir_name}\", \"coco_new/annotations/instances_val2017.json\"),")

# print(f"总共{i}组")


# 注册cityscapes-c时用到的，写在builtin.py
# img_dir = '/gpfsdata/home/huangziyue/data/cityscapes/val_corruptions'

# i = 0
# for dir_name in os.listdir(img_dir):
#     i += 1
#     # "coco_2017_ovd_all_train": ("coco/train2017", "coco/regionclip_ov_annotations/ovd_ins_train2017_all.json"),
#     print(f"\"cityscapes_c_{dir_name}\": (\"gpfsdata/home/huangziyue/data/cityscapes/val_corruptions/{dir_name}\", \"home/DATASET_PUBLIC/domain_adaptation/cityscapes/coco_format/instancesonly_filtered_gtFine_val.json\"),")

# print(f"总共{i}组")


# 打印概念的embedding出来
# file_path = '/gpfsdata/home/yangshuai/open_vocabulary/RegionCLIP/weights/concept_emb/VOC/concept_embeds.pth'
# file_path = '/gpfsdata/home/yangshuai/open_vocabulary/RegionCLIP/weights/concept_emb/cityscapes/concept_embeds.pth'
# weights = torch.load(file_path)
# print(weights.shape)


