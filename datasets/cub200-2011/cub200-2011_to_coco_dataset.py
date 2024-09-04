import json
import argparse
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import os
import random
from scipy.io import loadmat
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='datasets/cub200_2011_train.json')
    parser.add_argument('--root_path', default='datasets/cub200_2011')

    args = parser.parse_args()

    # convert training set
    lines = open(os.path.join(args.root_path, 'images.txt'), 'r').readlines()
    img_paths = [tmp.lstrip().split()[1] for tmp in lines]
    lines = open(os.path.join(args.root_path, 'train_test_split.txt'), 'r').readlines()
    train_test_split = [int(tmp.lstrip().split()[1]) for tmp in lines]
    lines = open(os.path.join(args.root_path, 'bounding_boxes.txt'), 'r').readlines()
    boxes = []
    for line in lines:
        tmp = line.lstrip().split()[1:]
        box = [float(t) for t in tmp]
        boxes.append(box)
    lines = open(os.path.join(args.root_path, 'classes.txt'), 'r').readlines()
    categories = [tmp.lstrip().split()[1].split('.')[-1] for tmp in lines]
    lines = open(os.path.join(args.root_path, 'image_class_labels.txt'), 'r').readlines()
    img_categories = [categories[int(tmp.lstrip().split()[1])-1] for tmp in lines]

    train_img_paths = []
    train_boxes = []
    train_categories = []
    test_img_paths = []
    test_boxes = []
    test_categories = []
    for i in range(len(train_test_split)):
        if train_test_split[i]==1:
            train_img_paths.append(img_paths[i])
            train_boxes.append(boxes[i])
            train_categories.append(img_categories[i])
        elif train_test_split[i]==0:
            test_img_paths.append(img_paths[i])
            test_boxes.append(boxes[i])
            test_categories.append(img_categories[i])

    # convert training set
    coco_anno_all = {}
    coco_anno_all['categories'] = []
    coco_anno_all['images'] = []
    coco_anno_all['annotations'] = []

    categories_to_id = {}
    category_curr_id = 0
    images_to_id = {}
    image_curr_id = 0
    anno_curr_id = 0
    for i in range(len(train_img_paths)):
        img_path = train_img_paths[i]
        box = train_boxes[i]
        category = train_categories[i]
        image_item = {}
        img = cv2.imread(os.path.join(args.root_path, 'images', img_path))
        image_item['height'] = img.shape[0]
        image_item['width'] = img.shape[1]
        image_item['file_name'] = img_path
        if not image_item['file_name'] in images_to_id:
            images_to_id[image_item['file_name']] = image_curr_id
            image_curr_id += 1
        image_item['id'] = images_to_id[image_item['file_name']]
        coco_anno_all['images'].append(image_item)

        obj_info = {}
        if category not in obj_info:
            obj_info[category] = []
        obj_info[category].append([box[0], box[1], box[0]+box[2], box[1]+box[3]])

        cates = list(obj_info.keys())
        for cls in cates:
            anno_item = {}
            if not cls in categories_to_id:
                categories_to_id[cls] = category_curr_id
                category_curr_id += 1
            box = obj_info[cls][0]
            if box[0] > image_item['width']+1: box[0] = image_item['width']+1
            if box[2] > image_item['width']+1: box[2] = image_item['width']+1
            if box[1] > image_item['height']+1: box[1] = image_item['height']+1
            if box[3] > image_item['height']+1: box[3] = image_item['height']+1

            anno_item['bbox'] = [int(box[0]), int(box[1]), int(box[2])-int(box[0]), int(box[3])-int(box[1])]
            anno_item['area'] = anno_item['bbox'][-1]*anno_item['bbox'][-2]
            anno_item['iscrowd'] = 0
            anno_item["image_id"] = image_item['id']
            anno_item['category_id'] = categories_to_id[cls]
            anno_item['id'] = anno_curr_id
            anno_curr_id += 1
            coco_anno_all['annotations'].append(anno_item)

    for cls in categories_to_id:
        cate_item = {}
        cate_item["supercategory"] = cls
        cate_item["name"] = cls
        cate_item["id"] = categories_to_id[cls]
        coco_anno_all['categories'].append(cate_item)

    with open(args.output_path, 'w') as f:
        json.dump(coco_anno_all, f)

    # convert test set
    coco_anno_all = {}
    coco_anno_all['categories'] = []
    coco_anno_all['images'] = []
    coco_anno_all['annotations'] = []
    images_to_id = {}
    image_curr_id = 0
    anno_curr_id = 0
    for i in range(len(test_img_paths)):
        img_path = test_img_paths[i]
        box = test_boxes[i]
        category = test_categories[i]
        image_item = {}
        img = cv2.imread(os.path.join(args.root_path, 'images', img_path))
        image_item['height'] = img.shape[0]
        image_item['width'] = img.shape[1]
        image_item['file_name'] = img_path
        if not image_item['file_name'] in images_to_id:
            images_to_id[image_item['file_name']] = image_curr_id
            image_curr_id += 1
        image_item['id'] = images_to_id[image_item['file_name']]
        coco_anno_all['images'].append(image_item)

        obj_info = {}
        if category not in obj_info:
            obj_info[category] = []
        obj_info[category].append([box[0], box[1], box[0]+box[2], box[1]+box[3]])

        cates = list(obj_info.keys())
        for cls in cates:
            anno_item = {}
            if not cls in categories_to_id:
                categories_to_id[cls] = category_curr_id
                category_curr_id += 1
            box = obj_info[cls][0]
            if box[0] > image_item['width']+1: box[0] = image_item['width']+1
            if box[2] > image_item['width']+1: box[2] = image_item['width']+1
            if box[1] > image_item['height']+1: box[1] = image_item['height']+1
            if box[3] > image_item['height']+1: box[3] = image_item['height']+1

            anno_item['bbox'] = [int(box[0]), int(box[1]), int(box[2])-int(box[0]), int(box[3])-int(box[1])]
            anno_item['area'] = anno_item['bbox'][-1]*anno_item['bbox'][-2]
            anno_item['iscrowd'] = 0
            anno_item["image_id"] = image_item['id']
            anno_item['category_id'] = categories_to_id[cls]
            anno_item['id'] = anno_curr_id
            anno_curr_id += 1
            coco_anno_all['annotations'].append(anno_item)

    for cls in categories_to_id:
        cate_item = {}
        cate_item["supercategory"] = cls
        cate_item["name"] = cls
        cate_item["id"] = categories_to_id[cls]
        coco_anno_all['categories'].append(cate_item)

    with open(args.output_path.replace('train', 'val'), 'w') as f:
        json.dump(coco_anno_all, f)





