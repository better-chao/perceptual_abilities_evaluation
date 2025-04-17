import os
import json

data_dir = "/home/yongchao/Transfer-Learning-Library-dev-tllib/examples/domain_adaptation/object_detection/datasets"
concepts_dir = '/gpfsdata/home/yangshuai/open_vocabulary/RegionCLIP/concepts/domain_adaptation'

ann_name = 'comic/comic_test.json' # kitti/kitti_trainval.json watercolor/watercolor_test.json comic/comic_test.json
concepts_subdir = 'comic' # kitti watercolor comic

os.makedirs(os.path.join(concepts_dir, concepts_subdir), exist_ok = True)

ann_file = os.path.join(data_dir, ann_name)
print(f"\"{concepts_subdir}\": (\"{concepts_subdir}\", \"{ann_name}\"),")
# print(ann_file)
with open(ann_file, 'r') as file:
    data = json.load(file)

print(data['categories'])
# categories = data['categories']

# concepts_file_path = os.path.join(concepts_dir, concepts_subdir, 'concepts.txt')

# with open(concepts_file_path, 'w') as save_file:
#     for item in categories:
#         class_name = item['name']
#         save_file.write(class_name + '\n')
# break
        