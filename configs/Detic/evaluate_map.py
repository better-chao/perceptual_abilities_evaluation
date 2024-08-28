from tqdm import tqdm

from torch import BoolTensor, IntTensor, Tensor
import torch
import argparse
import pickle, json
import os, sys

from torchvision.ops import batched_nms

try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP
    MeanAveragePrecision = MAP

import time

DEVICE = 'cpu'

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

def transform_predslist_to_dict(preds):
    result = {}
    for pred in preds:
        image = pred['image_filepath']
        if image not in result:
            result[image] = []
        result[image].append(pred)
    return result  

def assert_box(boxes):
    """Check that the box is in [xmin, ymin, xmax, ymax] format"""
    for box in boxes:
        assert box[0] <= box[2] and box[1] <= box[3]

def convert_format(boxes):
    for box in boxes:
        box[2] += box[0]
        box[3] += box[1]
    return boxes

def get_image_ground_truth(data, image_id):
    """
    Given a dictionary 'data' and an 'image_id', returns a dictionary with 'boxes' and 'categories' information for
    that image.

    Args:
        data (dict): The data dictionary containing 'annotations'.
        image_id (int): The image_id for which to retrieve data.

    Returns:
        dict: A dictionary with 'boxes' and 'categories' information for the given image_id.
    """
    image_data = {'boxes': [], 'labels': []}  # Initialize the dictionary to store image data

    # Loop through each annotation in the 'annotations' list
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            # If the 'image_id' in the annotation matches the given 'image_id', append bbox and category_id to the lists
            image_data['boxes'].append(annotation['bbox'])
            image_data['labels'].append(annotation['category_id'])

    image_data['boxes'] = convert_format(image_data['boxes'])
    assert_box(image_data['boxes'])
    # tensorize elements
    image_data['boxes'] = Tensor(image_data['boxes']).to(DEVICE)
    image_data['labels'] = IntTensor(image_data['labels']).to(DEVICE)
    
    return image_data

def get_image_preds(preds):
    labels = []
    scores = []
    boxes = []
    for pred in preds:
        labels += [x for x in pred['labels']]
        scores += [x for x in pred['scores']]
        boxes += ([x for x in pred['boxes']])
        assert_box(boxes)
    # TODO: Check if the case where zero predictions are presents affects results
    
    # labels = labels if labels != [] else 0
    # scores = scores if scores != [] else 0    
    boxes = boxes if boxes != [] else [[0,0,0,0]]
    if type(boxes[0]) != torch.Tensor:
        return {
            'boxes': Tensor(boxes).to(DEVICE),
            'labels': IntTensor(labels).to(DEVICE),
            'scores': Tensor(scores).to(DEVICE)
        }
    else:
        return {
            'boxes': torch.stack(boxes, dim=0).to(DEVICE),
            'labels': IntTensor(labels).to(DEVICE),
            'scores': Tensor(scores).to(DEVICE)
        }

def remove_pacco(data, preds):
    anns = [ann for ann in data['annotations'] if 'query' in ann]
    image_ids = list(set([ann['image_id'] for ann in anns]))
    imgs = [img for img in data['images'] if img['id'] in image_ids]
    cat_ids = list(set(id for ann in anns for id in [ann['category_id']] + ann['neg_category_ids']))

    cats = [cat for cat in data['categories'] if cat['id'] in cat_ids]
    
    # adjusting preds
    preds = [pred for pred in preds if pred['category_id'] in cat_ids]
    
    data['annotations'] = anns
    data['images'] = imgs
    data['categories'] = cats
    
    return data, preds
    

def simplify_errors(pred, test_cat_ids):
    error_label = 1 if 1 not in test_cat_ids else test_cat_ids[-1] + 1
    for i, label in enumerate(pred['labels']):
        if label not in test_cat_ids:
            pred['labels'][i] = error_label
    
    return pred

# def remove_errors(pred, test_cat_ids):
#     for i, label in enumerate(pred['labels']):
#         if label not in test_cat_ids:
#             pred['labels'][i] = error_label
            
#     for x in range(len(boxes)):
#         if x in indexes_to_keep:
#             filtered_boxes.append(boxes[x])
#             filtered_scores.append(scores[x])
#             filtered_labels.append(labels[x])
    
#     preds['boxes'] = torch.stack(filtered_boxes, dim=0)
#     preds['scores'] = torch.stack(filtered_scores, dim=0)
#     preds['labels'] = torch.stack(filtered_labels, dim=0)
#     return preds

def apply_NMS(preds, iou=0.5):
    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']
    
    indexes_to_keep = batched_nms(boxes, 
                                  scores, 
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
    
    preds['boxes'] = torch.stack(filtered_boxes, dim=0)
    preds['scores'] = torch.stack(filtered_scores, dim=0)
    preds['labels'] = torch.stack(filtered_labels, dim=0)
    return preds

if __name__ == "__main__":
    """
    Boxes of Ground Truth in [x1, y1, w, h]
    Boxes of Predictions in [x1, y1, x2, y2]
    
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True, help='Path of the prediction that we want to evaluate')
    parser.add_argument('--ground_truth', type=str, default='ground_truth/captioned_paco_lvis_v1_test.json', help='Path of the prediction that we want to evaluate')
    parser.add_argument('--out', type=str, default='results/results.txt', help='Where results will be stored')
    parser.add_argument('--n_hardnegatives', type=int, default=5, help="Number of hardnegatives in each vocabulary")
    parser.add_argument('--simplify_errors', action='store_true', help='If setted, all the wrong label will have the same id')
    parser.add_argument('--disable_nms', action='store_true', help='If setted, the predictions are not preprocessed using class-agnostic nms')
    parser.add_argument('--remove_pacco', action='store_true', help='Keep only annotations from PACO')
    parser.add_argument('--evaluate_all_vocabulary', action='store_true', help='Evaulate even annotations where the number of negatives caption is lower then the number of evaluated negatives')
    args = parser.parse_args()
    
    pred_path = args.predictions
    dataset_path = args.ground_truth
     
    if os.path.exists(args.out):
        print(f"{args.out} already exist")
        sys.exit(0)
     
    print("reading %s" % pred_path)
    preds_list = load_object(pred_path)
    test_set = read_json(dataset_path)
    
    if args.remove_pacco:
        test_set, preds_list = remove_pacco(test_set, preds_list)
    
    # Initialize metric
    metric = MeanAveragePrecision().to(DEVICE)

    if not args.evaluate_all_vocabulary:
        n_neg = args.n_hardnegatives
        # assert 'pred' in pred_path.split('/')[1], "not found number of negative"
        # n_neg = int(''.join(filter(str.isdigit, pred_path.split('/')[1])))
        # assert str(n_neg) in pred_path.split('/')[1]
        # if not n_neg:
        #     print("Not found number of negatives!")
        #     n_neg = 10
        # print(f"Number of annotation before cut: {len(test_set['annotations'])}")
        test_set['annotations'] = [ann for ann in test_set['annotations'] if len(ann['neg_category_ids']) >= n_neg]
        # print(f"Number of annotation after cut: {len(test_set['annotations'])}")
        gt_ann_cats = list(set([ann['category_id'] for ann in test_set['annotations']]))
        # check if there are predictions for annotations removed
        there_are_leftovers = all([pred['category_id'] in gt_ann_cats for pred in preds_list])
        if not(there_are_leftovers):
            print(f"LEFTOVERS in {pred_path}")
            

    preds_per_image = transform_predslist_to_dict(preds_list)
    
    correct_ids = []
    if args.simplify_errors:
        for ann in test_set['annotations']:
            if ann['category_id'] not in correct_ids:
                correct_ids.append(ann['category_id'])
    
    targets = []
    preds = []
    
    n_images = 0
    # for imm in tqdm(test_set['images']):
    for imm in test_set['images']:
        target = get_image_ground_truth(test_set, imm['id'])
        # skipping image if empty after eliminating since the number of negatives was low
        if not args.evaluate_all_vocabulary and len(target['labels']) == 0:
            continue
        
        if imm['file_name'] in preds_per_image:
            if args.disable_nms:
                pred = get_image_preds(preds_per_image[imm['file_name']])
            else:
                # in case the ground truth for the image includes captions not processed by the detector, we remove them
                relevant_cats = [predictions['category_id'] for predictions in preds_per_image[imm['file_name']]]
                mask = torch.isin(target['labels'], torch.tensor(relevant_cats))
                target['labels'] = target['labels'][mask]
                target['boxes'] = target['boxes'][mask]
                preds_per_cat = [get_image_preds([pred_per_cat]) for pred_per_cat in preds_per_image[imm['file_name']]]
                preds_per_cat = [apply_NMS(pred_per_cat) for pred_per_cat in preds_per_cat]
                pred = {
                    'boxes': torch.cat([x['boxes'] for x in preds_per_cat], dim=0),
                    'labels': torch.cat([x['labels'] for x in preds_per_cat], dim=0),
                    'scores': torch.cat([x['scores'] for x in preds_per_cat], dim=0),
                }
        else:
            continue
        n_images += 1
        if args.simplify_errors:
            pred = simplify_errors(pred, correct_ids)
        targets.append(target)
        preds.append(pred)
        
    # Update metric with predictions and respective ground truth
    metric.update(preds, targets)
    
    # getting time of execution of the mAP
    print("Starting mAP computation")
    start_time = time.time()
    # Compute the results
    result = metric.compute()
    # print("--- %s seconds ---" % (time.time() - start_time))
    result['n_images'] = n_images
    # de-tensorize the results:
    result = {
        'map': float(result['map']),
        'map_50': float(result['map_50']),
        'map_75': float(result['map_75']),
        'map_small': float(result['map_small']),
        'map_medium': float(result['map_medium']),
        'map_large': float(result['map_large']),
        'mar_1': float(result['mar_1']),
        'mar_10': float(result['mar_10']),
        'mar_100': float(result['mar_100']),
        'mar_small': float(result['mar_small']),
        'mar_medium': float(result['mar_medium']),
        'mar_large': float(result['mar_large']),
        'map_per_class': float(result['map_per_class']),
        'mar_100_per_class': float(result['mar_100_per_class']),
        'n_images': int(result['n_images'])  
    }
    
    # print(result)
    print(f"Done {pred_path}. mAP: {result['map']}")
    
    
    with open(args.out, "w") as json_file:
        json.dump(result, json_file)  # The indent parameter is optional for pretty formatting