import argparse
import torch

from tqdm import tqdm

import pickle, json

# Setup detectron2 logger
import detectron2

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

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

MAX_DETECTION_PER_CATEGORY = 100
SCORE_THRESH = 0.00
def create_detector(config_path, weight_path):
    # Build the detector and download our pretrained weights
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False # For better visualization purpose. Set to False for all classes.
    cfg.TEST.DETECTIONS_PER_IMAGE = 256
    predictor = DefaultPredictor(cfg)
    return predictor

def get_clip_embeddings(text_encoder, vocabulary, prompt=''):
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def evaluate_image(predictor, text_encoder, im, vocabulary):
    classifier = get_clip_embeddings(text_encoder, vocabulary)
    num_classes = len(vocabulary)
    reset_cls_test(predictor.model, classifier, num_classes)
    # Run model and show results
    outputs = predictor(im)
    #print(outputs['instances'])
    # instances = outputs['instances']
    # print("Available fields in Instances:", instances.get_fields().keys())
    
    
    # total_scores = outputs['instances'].total_scores
    # scores = outputs['instances'].scores
    # Calculate the maximum value along each row of total_scores
    # max_values, max_indices = total_scores.max(dim=1)
    # Check that the max values are equal to the corresponding elements in scores
    # assert torch.allclose(max_values, scores), "Assertion failed: Max values of total_scores do not match corresponding elements in scores."
    
    return outputs

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
    for i in range(len(output['instances'].pred_classes)):
        output['instances'].pred_classes[i] = vocabulary_id[output['instances'].pred_classes[i]]
    return output

def convert_to_standard_format(output):
    return {
            'labels': output['instances'].pred_classes.cpu().numpy().tolist(),
            'boxes': output['instances'].pred_boxes.tensor.cpu().numpy().tolist(),
            'scores': output['instances'].scores.cpu().numpy().tolist(),
            # 'total_scores': output['instances'].total_scores.cpu().numpy().tolist(),
            'category_id': output['category_id'],
            'image_filepath': output['image_filepath']
    }
    
    


def convert_to_standard_format_complete(outputs):
    std_out = []
    
    #print("Converting predictions in standard format")
    for output in outputs:
        std_out.append({
            'labels': output['instances'].pred_classes.cpu().numpy().tolist()[:MAX_DETECTION_PER_CATEGORY],
            'boxes': output['instances'].pred_boxes.tensor.cpu().numpy().tolist()[:MAX_DETECTION_PER_CATEGORY],
            'scores': output['instances'].scores.cpu().numpy().tolist()[:MAX_DETECTION_PER_CATEGORY],
            # 'total_scores': output['instances'].total_scores.cpu().numpy().tolist()[:MAX_DETECTION_PER_CATEGORY],
            'category_id': output['category_id'],
            'image_filepath': output['image_filepath']
        })

    return std_out

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
    parser.add_argument('--out', type=str, default="detic.pkl", help='Out path')
    parser.add_argument('--config_path', type=str, default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml", help='Path of the configuration file')
    parser.add_argument('--weight_path', type=str, default="weights/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth", help='Path of the weight file')
    parser.add_argument('--n_hardnegatives', type=int, default=5, help="Number of hardnegatives in each vocabulary")
    args = parser.parse_args()
    
    coco_path = '/gpfsdata/home/yangshuai/data/coco/'
    
    data = read_json(args.dataset)
    predictor = create_detector(args.config_path, args.weight_path)
    
    # building CLIP text encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    
    complete_outputs = []
    complete_outputs_gpu = []
    categories_done = []
    for ann in tqdm(data['annotations']):
        # if the category is not done, we add it to the list
        if ann['category_id'] not in categories_done:
            categories_done.append(ann['category_id'])
        else:
            continue
        
        vocabulary, vocabulary_id = create_vocabulary(ann, data['categories'])
        
        # check if a number of hardnegatives is setted to non-default values
        # if it is, the vocabulary is clipped and if it is too short, we skip that image
        len_vocabulary = args.n_hardnegatives + 1
        if len(vocabulary) < len_vocabulary:
            continue
        vocabulary = vocabulary[:len_vocabulary]
        vocabulary_id = vocabulary_id[:len_vocabulary]
        
        image_filepath = coco_path + get_image_filepath(ann['image_id'], data['images'])
        imm = cv2.imread(image_filepath)
        
        # if vocabulary_id[0] == 30426:
        #     continue
        
        output = evaluate_image(predictor, text_encoder, imm, vocabulary)
        output['category_id'] = ann['category_id']
        output['image_filepath'] = get_image_filepath(ann['image_id'], data['images'])
        output = adjust_out_id(output, vocabulary_id)
        complete_outputs_gpu.append(output)
        
        #to optimize the performances we move the outputs from gpu tu cpu every 50 predictions
        if len(complete_outputs_gpu) >= 50:
            complete_outputs += convert_to_standard_format_complete(complete_outputs_gpu)
            complete_outputs_gpu = []
    
    complete_outputs += convert_to_standard_format_complete(complete_outputs_gpu)
    
    # save_object(convert_to_standard_format_complete(complete_outputs), "out/output_%s" % dataset_name)
    save_object(complete_outputs, "backup.pkl")
    save_object(complete_outputs, args.out)

if __name__ == '__main__':
    main()