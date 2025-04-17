import os
import json

# concepts_dir = '/gpfsdata/home/yangshuai/open_vocabulary/RegionCLIP/concepts/object365'

# os.makedirs(concepts_dir, exist_ok = True)

ann_file = '/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets/cityscapes_in_voc/cityscapes_car_trainval.json'
# /home/DATASET_PUBLIC/Stanford_Dogs/stanford_dogs_val_clipemb.json
# /home/DATASET_PUBLIC/CUB_200_2011/CUB_200_2011/cub200_2011_val_clipemb.json
# /gpfsdata/home/buaa_liuchenguang/github_eval/OpenDataLab___CrowdHuman/raw/CrowdHuman/crowdhuman/annotations/crowdhuman_val.json
# /gpfsdata/home/buaa_liuchenguang/github_eval/OpenDataLab___OCHuman/raw/OCHuman/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json
# /gpfsdata/home/buaa_liuchenguang/github_eval/WiderPerson/val.json
# /gpfsdata/home/yangshuai/data/Objects365_v1/objects365_train.json

with open(ann_file, 'r') as file:
    data = json.load(file)

# print(data['categories'])
categories = data['categories'] # sorted(data['categories'], key=lambda x: x['id'])
# print(categories)
for item in categories:
    print(item['name'])
    print(item['id'])

# concepts_file_path = os.path.join(concepts_dir, 'concepts.txt')

# with open(concepts_file_path, 'w') as save_file:
#     for item in categories:
#         class_name = item['name']
#         save_file.write(class_name + '\n')
# break
        