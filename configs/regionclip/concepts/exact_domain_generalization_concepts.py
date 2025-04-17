import os
import json

data_dir = "/home/DATASET_PUBLIC/domain_generalization/weather"
concepts_dir = '/gpfsdata/home/yangshuai/open_vocabulary/RegionCLIP/concepts/domain_generalization'

os.makedirs(concepts_dir, exist_ok = True)

for subdir in os.listdir(data_dir):
    if subdir[-3:] != 'zip':
        data_subdir = os.path.join(data_dir, subdir)
        ann_file = os.path.join(data_subdir, 'voc07_test.json')
        print(f"\"{subdir}\": (\"{subdir}/VOC2007/JPEGImages\", \"{subdir}/voc07_test.json\"),")
        # print(ann_file)
        with open(ann_file, 'r') as file:
            data = json.load(file)

        print(data['categories'])
        # categories = data['categories']

        # concepts_file_path = os.path.join(concepts_dir, 'concepts.txt')

        # with open(concepts_file_path, 'w') as save_file:
        #     for item in categories:
        #         class_name = item['name']
        #         save_file.write(class_name + '\n')
        # break
        