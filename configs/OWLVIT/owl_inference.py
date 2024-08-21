import argparse
import torch

from tqdm import tqdm

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from torchvision.ops import batched_nms

# import some common libraries
import sys
import numpy as np
import os, cv2, random
from skimage import io as skimage_io

from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTTextConfig

SCORE_THRESH = 0.1
import numpy as np

import pickle, json

# READ/WRITE UTILITIES

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

def convert_to_x1y1x2y2(bbox, img_width, img_height):
    """
    Convert bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        bbox (np.array): NumPy array of bounding boxes in the format [cx, cy, w, h].
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        np.array: NumPy array of bounding boxes in the format [x1, y1, x2, y2].
    """
    cx, cy, w, h = bbox
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return np.array([x1, y1, x2, y2])

def apply_NMS(boxes, scores, labels, total_scores, iou=0.5):
    indexes_to_keep = batched_nms(torch.stack([torch.FloatTensor(box) for box in boxes], dim=0),
                       torch.FloatTensor(scores),
                       torch.IntTensor(labels),
                       iou)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    filtered_total_scores = []
    deleted_boxes = []
    deleted_scores = []
    deleted_labels = []
    deleted_total_scores = []
    
    for x in range(len(boxes)):
        if x in indexes_to_keep:
            filtered_boxes.append(boxes[x])
            filtered_scores.append(scores[x])
            filtered_labels.append(labels[x])
            filtered_total_scores.append(total_scores[x])
        else:
            deleted_boxes.append(boxes[x])
            deleted_scores.append(scores[x])
            deleted_labels.append(labels[x])
            deleted_total_scores.append(total_scores[x])
    
    return filtered_boxes, filtered_scores, filtered_labels, filtered_total_scores



skipped_categories = 0
def evaluate_image_disentangled_inferences(model, processor, im, vocabulary, MAX_PREDICTIONS=100, nms=False):
    global skipped_categories    
        
    scores_filtered = []
    labels_filtered = []
    boxes_filtered = []
    total_scores_filtered = []
    for label, caption in enumerate(vocabulary):
        # preparing the inputs
        inputs = processor(text=caption, images=im, return_tensors="pt", padding=True).to(device)
        
        # if the tokens length is above 16, the model can't handle them
        if  inputs['input_ids'].shape[1] > 16:
            skipped_categories += 1
            return None
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            
            
        # Get prediction logits
        logits = torch.max(outputs['logits'][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()
        all_scores = torch.sigmoid(outputs['logits'][0]).cpu().detach().numpy()
        
        # Get prediction labels and boundary boxes
        boxes = outputs['pred_boxes'][0].cpu().detach().numpy()
        labels = np.array([label] * len(boxes))
        height = im.shape[0]
        width = im.shape[1]
        
        boxes = [convert_to_x1y1x2y2(box, width, height) for box in boxes]
        # Combine the lists into tuples using zip
        if nms:
            # apply NMS
            boxes, scores, labels, all_scores = apply_NMS(boxes, scores, labels, all_scores)
        data = list(zip(scores, boxes, labels, all_scores))

        # Sort the combined data based on the first element of each tuple (score) in decreasing order
        sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
        
        
        # filtering the predictions with low confidence
        for score, box, label, total_scores in sorted_data[:MAX_PREDICTIONS]:
            scores_filtered.append(score)
            labels_filtered.append(label)
            # boxes_filtered.append(convert_to_x1y1x2y2(box, width, height))
            boxes_filtered.append(box)
            total_scores_filtered.append(total_scores)
    
    return {
        'scores': scores_filtered,
        'labels': labels_filtered,
        'boxes': boxes_filtered,
        'total_scores': total_scores_filtered
    }

def evaluate_image(model, processor, im, vocabulary, MAX_PREDICTIONS=100, nms=False):
    global skipped_categories
    # preparing the inputs
    inputs = processor(text=vocabulary, images=im, return_tensors="pt", padding=True).to(device)
    
    # if the tokens length is above 16, the model can't handle them
    if  inputs['input_ids'].shape[1] > 16:
        skipped_categories += 1
        return None
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        
        
    # Get prediction logits
    logits = torch.max(outputs['logits'][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()
    all_scores = torch.sigmoid(outputs['logits'][0]).cpu().detach().numpy()
    
    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs['pred_boxes'][0].cpu().detach().numpy()    
        
    scores_filtered = []
    labels_filtered = []
    boxes_filtered = []
    total_scores_filtered = []
    height = im.shape[0]
    width = im.shape[1]
    
    boxes = [convert_to_x1y1x2y2(box, width, height) for box in boxes]
    # Combine the lists into tuples using zip
    if nms:
        # apply NMS
        boxes, scores, labels, all_scores = apply_NMS(boxes, scores, labels, all_scores)
    data = list(zip(scores, boxes, labels, all_scores))

    # Sort the combined data based on the first element of each tuple (score) in decreasing order
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
    
    
    # filtering the predictions with low confidence
    for score, box, label, total_scores in sorted_data[:MAX_PREDICTIONS]:
        scores_filtered.append(score)
        labels_filtered.append(label)
        # boxes_filtered.append(convert_to_x1y1x2y2(box, width, height))
        boxes_filtered.append(box)
        total_scores_filtered.append(total_scores)
    
    return {
        'scores': scores_filtered,
        'labels': labels_filtered,
        'boxes': boxes_filtered,
        'total_scores': total_scores_filtered
    }
        
    

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
    parser.add_argument('--out', type=str, required=True, help='Out path')
    parser.add_argument('--nms', default=False, action='store_true', help='If set it will be applied NMS with iou=0.5')
    parser.add_argument('--disentangled_inferences', default=False, action='store_true', help='If set, a vocabulary is decomposed in single captions')
    parser.add_argument('--large', default=False, action='store_true', help='If set, it will be loaded the large model')
    parser.add_argument('--n_hardnegatives', type=int, default=10, help="Number of hardnegatives in each vocabulary")
    args = parser.parse_args()
    global skipped_categories
    
    coco_path = '/gpfsdata/home/yangshuai/data/coco/'
    if args.n_hardnegatives == 0 and '1_attributes' not in args.dataset:
        return
    
    # data = read_json('/home/lorenzobianchi/PacoDatasetHandling/jsons/captioned_%s.json' % dataset_name)
    data = read_json(args.dataset)
    
    if args.large:
        model = OwlViTForObjectDetection.from_pretrained("/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlvit-large-patch14")
        processor = OwlViTProcessor.from_pretrained("/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlvit-large-patch14")
        print("Large model loaded")
    else:
        model = OwlViTForObjectDetection.from_pretrained("/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlvit-base-patch16")
        processor = OwlViTProcessor.from_pretrained("/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/owlvit-base-patch16")
        print("Base model loaded")
    if not args.disentangled_inferences:
        print("no disentanglement")
    model = model.to(device)
    model.eval()
    
    complete_outputs = []
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
        imm = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
        if not args.disentangled_inferences:
            output = evaluate_image(model, processor, imm, vocabulary, nms=args.nms)
        else:
            output = evaluate_image_disentangled_inferences(model, processor, imm, vocabulary, nms=args.nms)
        if output == None:
            continue
        output['category_id'] = ann['category_id']
        output['vocabulary'] = vocabulary_id
        output['image_filepath'] = get_image_filepath(ann['image_id'], data['images'])
        output = adjust_out_id(output, vocabulary_id)
        complete_outputs.append(output)
        
    save_object(complete_outputs, args.out)
    print("Skipped categories: %d/%d" % (skipped_categories, len(categories_done)))
if __name__ == '__main__':
    main()