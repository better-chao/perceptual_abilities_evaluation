import json
import argparse
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import os
import random
from scipy.io import loadmat
import xml.etree.ElementTree as ET

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.

    Arguments:
        xml_files {list} -- A list of xml file paths.

    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='datasets/stanford_dogs_train.json')
    parser.add_argument('--root_path', default='datasets/stanford_dogs')

    args = parser.parse_args()

    # convert training set
    infos = loadmat(os.path.join(args.root_path, 'train_list.mat'))
    img_paths = infos['file_list']
    anno_paths = infos['annotation_list']

    coco_anno_all = {}
    coco_anno_all['categories'] = []
    coco_anno_all['images'] = []
    coco_anno_all['annotations'] = []

    categories_to_id = {}
    category_curr_id = 0
    images_to_id = {}
    image_curr_id = 0
    anno_curr_id = 0
    for i in range(len(anno_paths)):
        anno_path = os.path.join(args.root_path, 'Annotation',(str(anno_paths[i][0][0])))
        img_path = str(img_paths[i][0][0])
        image_item = {}

        tree = ET.parse(anno_path)
        root = tree.getroot()
        size = get_and_check(root, "size", 1)

        image_item['height'] = int(get_and_check(size, "height", 1).text)
        image_item['width'] = int(get_and_check(size, "width", 1).text)
        image_item['file_name'] = img_path
        if not image_item['file_name'] in images_to_id:
            images_to_id[image_item['file_name']] = image_curr_id
            image_curr_id += 1
        image_item['id'] = images_to_id[image_item['file_name']]
        coco_anno_all['images'].append(image_item)

        obj_info = {}
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) 
            ymin = int(get_and_check(bndbox, "ymin", 1).text)
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            bbox = [xmin, ymin, xmax, ymax]
            if category not in obj_info:
                obj_info[category] = []
            obj_info[category].append((bbox))

        cates = list(obj_info.keys())
        # cates = [_.lower() for _ in cates]
        for cls in cates:
            anno_item = {}
            if not cls in categories_to_id:
                categories_to_id[cls] = category_curr_id
                category_curr_id += 1
            box = obj_info[cls][0]
            # assert box[0] <= image_item['width']+1 and box[2] <= image_item['width']+1
            # assert box[1] <= image_item['height']+1 and box[3] <= image_item['height']+1
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

    # convert training set
    infos = loadmat(os.path.join(args.root_path, 'test_list.mat'))
    img_paths = infos['file_list']
    anno_paths = infos['annotation_list']

    coco_anno_all = {}
    coco_anno_all['categories'] = []
    coco_anno_all['images'] = []
    coco_anno_all['annotations'] = []
    images_to_id = {}
    image_curr_id = 0
    anno_curr_id = 0
    for i in range(len(anno_paths)):
        anno_path = os.path.join(args.root_path, 'Annotation',(str(anno_paths[i][0][0])))
        img_path = str(img_paths[i][0][0])
        image_item = {}

        tree = ET.parse(anno_path)
        root = tree.getroot()
        size = get_and_check(root, "size", 1)

        image_item['height'] = int(get_and_check(size, "height", 1).text)
        image_item['width'] = int(get_and_check(size, "width", 1).text)
        image_item['file_name'] = img_path
        if not image_item['file_name'] in images_to_id:
            images_to_id[image_item['file_name']] = image_curr_id
            image_curr_id += 1
        image_item['id'] = images_to_id[image_item['file_name']]
        coco_anno_all['images'].append(image_item)

        obj_info = {}
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) 
            ymin = int(get_and_check(bndbox, "ymin", 1).text)
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            bbox = [xmin, ymin, xmax, ymax]
            if category not in obj_info:
                obj_info[category] = []
            obj_info[category].append((bbox))

        cates = list(obj_info.keys())
        # cates = [_.lower() for _ in cates]
        for cls in cates:
            anno_item = {}
            if not cls in categories_to_id:
                categories_to_id[cls] = category_curr_id
                category_curr_id += 1
            box = obj_info[cls][0]
            # assert box[0] <= image_item['width']+1 and box[2] <= image_item['width']+1
            # assert box[1] <= image_item['height']+1 and box[3] <= image_item['height']+1
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





