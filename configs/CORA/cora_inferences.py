import datasets.transforms as T
import os
import argparse
import random
from pathlib import Path
import cv2
from tqdm import tqdm
import pickle, json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from util.visualizer import COCOVisualizer
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco import make_coco_transforms
from models import build_model
from main import get_args_parser
from torchvision.ops import batched_nms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from utils.pickle_handler import saveObject, loadObject
# from utils.json_handler import read_json, write_json

def save_object(obj, path):
    """"Save an object using the pickle library on a file
    
    :param obj: undefined. Object to save
    :param fileName: str. Name of the file of the object to save
    """
    print("Saving " + path)
    with open(path, 'wb') as fid:
        pickle.dump(obj, fid)
        
def load_object(path):
    """"Load an object from a file
    
    :param fileName: str. Name of the file of the object to load
    :return: obj: undefined. Object loaded
    """
    try:
        with open(path, 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None   

def read_json(file_name):
    #Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data

def write_json(data, file_name):
    # Write JSON file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)

device = 'cuda'

def get_category_name(id, categories):
    for category in categories:
        if id == category['id']:
            return category['name']
        
def get_image_filepath(id, images):
    for image in images:
        if id == image['id']:
            return image['file_name']

def create_vocabulary(ann, categories):
    vocabulary_id = [ann['category_id']] + ann['neg_category_ids']
    vocabulary = [get_category_name(id, categories) for id in vocabulary_id]
    
    return vocabulary, vocabulary_id

def adjust_out_id(output, vocabulary_id):
    for i in range(len(output['labels'])):
        output['labels'][i] = vocabulary_id[output['labels'][i]]
    return output


def get_image(img_path):
    """
    Read and normalize an image
    """
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]
    imm = Image.open(img_path)
    # Get the original width and height
    h, w = imm.size
    normalize = T.Compose([
        T.ToRGB(),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])
    compose = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])
    imm, _ = compose(imm, None)
    imm = imm.unsqueeze(0) # add batch dimension
    return imm, w, h

def apply_NMS(preds, iou=0.6):
    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']
    
    indexes_to_keep = batched_nms(torch.FloatTensor(boxes), 
                                  torch.FloatTensor(scores), 
                                  torch.IntTensor([0] * len(boxes)),
                                  iou)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    
    for x in range(len(boxes)):
        if x in indexes_to_keep:
            filtered_boxes.append(boxes[x])
            filtered_scores.append(scores[x])
            filtered_labels.append(labels[x])
    
    preds['boxes'] = filtered_boxes
    preds['scores'] = filtered_scores
    preds['labels'] = filtered_labels
    return preds

def evaluate_image(model, imm, post_processors, vocabulary, size, max_predictions=100, nms=True):
    w, h = size
    target_sizes = torch.tensor([[w, h]], device=device)
    
    outputs = model(imm, categories=vocabulary)
    results = post_processors['bbox'](outputs, target_sizes)[0]  
    
    scores = results['scores'].tolist()
    labels = results['labels'].tolist()
    boxes = results['boxes'].tolist()
    
    preds = {
        'scores': scores,
        'labels': labels,
        'boxes': boxes,
    } 
    preds = apply_NMS(preds)
    total_confidences = torch.sigmoid(outputs['pred_logits'])[0]
    max_score2total_scores = {max(confs.tolist()): confs.tolist() for confs in total_confidences}
    
    scores_final = []
    labels_final = []
    boxes_final = []
    total_scores = []
    for score, label, box in zip(preds['scores'],  preds['labels'], preds['boxes']):
        if score in max_score2total_scores:
            total_scores.append(max_score2total_scores[score])
            scores_final.append(score)
            labels_final.append(label)
            boxes_final.append(box)
            assert max(total_scores[-1]) == score, "Incongruent score"
            
        
        
    
    return {
        'scores': scores_final[:max_predictions],
        'labels': labels_final[:max_predictions],
        'boxes': boxes_final[:max_predictions],
        'total_scores': total_scores[:max_predictions]
    } 
    

def main(args):
    coco_path = '/gpfsdata/home/yangshuai/data/coco/'
    if args.n_hardnegatives == 0 and '1_attributes' not in args.dataset:
        return
    data = read_json(args.dataset)
    
    # fix the seed for reproducibility
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = 'cuda'
    
    model, criterion, post_processors = build_model(args)
    model.to(device)
    model.eval()
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_ema'])
        print("checkpoint loaded")

    categories_done = []
    complete_outputs = []
    
    for ann in tqdm(data['annotations']):
        # if the category is not done, we add it to the list
        if ann['category_id'] not in categories_done:
            categories_done.append(ann['category_id'])
        else:
            continue
        # check if a number of hardnegatives is setted to non-default values
        # if it is, the vocabulary is clipped and if it is too short, we skip that image
        vocabulary, vocabulary_id = create_vocabulary(ann, data['categories'])
        len_vocabulary = args.n_hardnegatives + 1
        if len(vocabulary) < len_vocabulary:
            continue
        vocabulary = vocabulary[:len_vocabulary]
        vocabulary_id = vocabulary_id[:len_vocabulary]
        image_filepath = coco_path + get_image_filepath(ann['image_id'], data['images'])
        imm, w, h = get_image(image_filepath)
        imm = imm.to(device)
        output = evaluate_image(model, imm, post_processors, vocabulary, (w, h))
        if output == None:
            continue
        output['category_id'] = ann['category_id']
        output['vocabulary'] = vocabulary_id
        output['image_filepath'] = get_image_filepath(ann['image_id'], data['images'])
        output = adjust_out_id(output, vocabulary_id)
        complete_outputs.append(output)
        
    # saveObject(complete_outputs, args.out)
    save_object(complete_outputs, args.out)

if __name__ == '__main__':
    # python inference_on_benchmark.py --backbone clip_RN50
    parser = argparse.ArgumentParser("CORA", parents=[get_args_parser()])
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
    parser.add_argument('--out', type=str, required=True, help='Out path')
    parser.add_argument('--n_hardnegatives', type=int, default=10, help="Number of hardnegatives in each vocabulary")
    args = parser.parse_args()
    
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

