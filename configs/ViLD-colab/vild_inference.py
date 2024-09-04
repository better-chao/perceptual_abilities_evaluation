import json
import pickle

from tqdm import tqdm 
import pdb
import argparse

#@title Import libraries
from easydict import EasyDict

import numpy as np
import torch
import clip

from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import patches

import collections
import numpy as np

import os
import os.path as osp

from PIL import Image
from pprint import pprint
from scipy.special import softmax
# import yaml


import tensorflow.compat.v1 as tf

import cv2


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

def article(name):
  return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res


def get_category_name(cat_id, categories):
    for cat in categories:
        if cat['id'] == cat_id:
            return cat['name']

    return None

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

def convert_to_standard_format(output):
    return {
            'labels': output['instances'].pred_classes.cpu().numpy().tolist(),
            'boxes': output['instances'].pred_boxes.tensor.cpu().numpy().tolist(),
            'scores': output['instances'].scores.cpu().numpy().tolist(),
            'annotation_id': output['annotation_id'],
            'image_filepath': output['image_filepath']
    }

#@title Define hyperparameters
FLAGS = {
    'prompt_engineering': False,
    'this_is': True,

    'temperature': 100.0,
    'use_softmax': False,
}
FLAGS = EasyDict(FLAGS)


# Global matplotlib settings
SMALL_SIZE = 16#10
MEDIUM_SIZE = 18#12
BIGGER_SIZE = 20#14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Parameters for drawing figure.
display_input_size = (10, 10)
overall_fig_size = (18, 24)

line_thickness = 2
fig_size_w = 35
# fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
mask_color =   'red'
alpha = 0.5

single_template = [
    '{}'
]

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',


    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]

clip.available_models()
model, preprocess = clip.load("ViT-B/32")


# model loading
session = tf.Session(graph=tf.Graph())

saved_model_dir = '/gpfsdata/home/yangshuai/open_vocabulary/FG-OVD/weights/image_path_v2' #@param {type:"string"}
_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)

def build_text_embedding(categories):
  if FLAGS.prompt_engineering:
    templates = multiple_templates
  else:
    templates = single_template

  run_on_gpu = torch.cuda.is_available()

  with torch.no_grad():
    all_text_embeddings = []
    # print('Building text embeddings...')
    for category in categories:
      texts = [
        template.format(processed_name(category['name'], rm_dot=True),
                        article=article(category['name']))
        for template in templates]
      if FLAGS.this_is:
        texts = [
                 'This is ' + text if text.startswith('a') or text.startswith('the') else text
                 for text in texts
                 ]
      texts = clip.tokenize(texts) #tokenize
      if run_on_gpu:
        texts = texts.cuda()
      text_embeddings = model.encode_text(texts) #embed with text encoder
      text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
      text_embedding = text_embeddings.mean(dim=0)
      text_embedding /= text_embedding.norm()
      all_text_embeddings.append(text_embedding)
    all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
    if run_on_gpu:
      all_text_embeddings = all_text_embeddings.cuda()
  return all_text_embeddings.cpu().numpy().T

#@title NMS
def nms(dets, scores, thresh, max_dets=1000):
  """Non-maximum suppression.
  Args:
    dets: [N, 4]
    scores: [N,]
    thresh: iou threshold. Float
    max_dets: int.
  """
  y1 = dets[:, 0]
  x1 = dets[:, 1]
  y2 = dets[:, 2]
  x2 = dets[:, 3]

  areas = (x2 - x1) * (y2 - y1)
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0 and len(keep) < max_dets:
    i = order[0]
    keep.append(i)

    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h
    overlap = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)

    inds = np.where(overlap <= thresh)[0]
    order = order[inds + 1]
  return keep



def inference(image_path, category_name_string, params, show_res=False):
  #################################################################
  # Preprocessing categories and get params
  if type(category_name_string) is list:
    category_names = category_name_string
  else:
    category_names = [x.strip() for x in category_name_string.split(';')]
  categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
  category_indices = {cat['id']: cat for cat in categories}

  max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area = params
  fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)


  #################################################################
  # Obtain results and read image
  roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = session.run(
        ['RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0', 'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0'],
        feed_dict={'Placeholder:0': [image_path,]})

  roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
  # no need to clip the boxes, already done
  roi_scores = np.squeeze(roi_scores, axis=0)

  detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
  scores_unused = np.squeeze(scores_unused, axis=0)
  box_outputs = np.squeeze(box_outputs, axis=0)
  detection_masks = np.squeeze(detection_masks, axis=0)
  visual_features = np.squeeze(visual_features, axis=0)

  image_info = np.squeeze(image_info, axis=0)  # obtain image info
  image_scale = np.tile(image_info[2:3, :], (1, 2))
  image_height = int(image_info[0, 0])
  image_width = int(image_info[0, 1])

  rescaled_detection_boxes = detection_boxes / image_scale # rescale

  # Read image
  image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
  assert image_height == image.shape[0]
  assert image_width == image.shape[1]


  #################################################################
  # Filter boxes

  # Apply non-maximum suppression to detected boxes with nms threshold.
  nmsed_indices = nms(
      detection_boxes,
      roi_scores,
      thresh=nms_threshold
      )

  # Compute RPN box size.
  box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

  # Filter out invalid rois (nmsed rois)
  valid_indices = np.where(
      np.logical_and(
        np.isin(np.arange(len(roi_scores), dtype=int), nmsed_indices),
        np.logical_and(
            np.logical_not(np.all(roi_boxes == 0., axis=-1)),
            np.logical_and(
              roi_scores >= min_rpn_score_thresh,
              box_sizes > min_box_area
              )
        )
      )
  )[0]
  # print('number of valid indices', len(valid_indices))

  detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
  detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
  detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
  detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
  rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]


  #################################################################
  # Compute text embeddings and detection scores, and rank results
  text_features = build_text_embedding(categories)

  raw_scores = detection_visual_feat.dot(text_features.T)
  if FLAGS.use_softmax:
    scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
  else:
    scores_all = raw_scores

  indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
  indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])

  #################################################################
  # Plot detected boxes on the input image.
  ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
  processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
  #   segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

  if len(indices_fg) == 0:
    # display_image(np.array(image), size=overall_fig_size)
    print('ViLD does not detect anything belong to the given category')

  else:
    if show_res:
        image_with_detections = visualize_boxes_and_labels_on_image_array(
            np.array(image),
            rescaled_detection_boxes[indices_fg],
            valid_indices[:max_boxes_to_draw][indices_fg],
            detection_roi_scores[indices_fg],
            numbered_category_indices,
            instance_masks=segmentations[indices_fg],
            use_normalized_coordinates=False,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_rpn_score_thresh,
            skip_scores=False,
            skip_labels=True)
        plt.figure(figsize=overall_fig_size)
        plt.imshow(image_with_detections)
        plt.axis('off')
        plt.title('Detected objects and RPN scores')
        plt.show()


  #################################################################
  # Plot
  cnt = 0
  raw_image = np.array(image)
  n_boxes = rescaled_detection_boxes.shape[0]
  bboxes = []
  confidences = []
  predicted_categories = []
  total_scores = []
  for anno_idx in indices[0:int(n_boxes)]:
    rpn_score = detection_roi_scores[anno_idx]
    bbox = rescaled_detection_boxes[anno_idx]
    bboxes.append(bbox)
    scores = scores_all[anno_idx]
    confidences.append(max(scores))
    predicted_categories.append(np.argmax(scores))
    total_scores.append(scores)

    if np.argmax(scores) == 0:
      continue
    
    if show_res:
      y1, x1, y2, x2 = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
      img_w_mask = plot_mask(mask_color, alpha, raw_image, segmentations[anno_idx])
      crop_w_mask = img_w_mask[y1:y2, x1:x2, :]


      fig, axs = plt.subplots(1, 4, figsize=(fig_size_w, fig_size_h), gridspec_kw={'width_ratios': [3, 1, 1, 2]}, constrained_layout=True)

      # Draw bounding box.
      rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=line_thickness, edgecolor='r', facecolor='none')
      axs[0].add_patch(rect)

      axs[0].set_xticks([])
      axs[0].set_yticks([])
      axs[0].set_title(f'bbox: {y1, x1, y2, x2} area: {(y2 - y1) * (x2 - x1)} rpn score: {rpn_score:.4f}')
      axs[0].imshow(raw_image)

      # Draw image in a cropped region.
      crop = np.copy(raw_image[y1:y2, x1:x2, :])
      axs[1].set_xticks([])
      axs[1].set_yticks([])

      # axs[1].set_title(f'predicted: {category_names[np.argmax(scores)]}')
      axs[1].imshow(crop)

      # Draw segmentation inside a cropped region.
      axs[2].set_xticks([])
      axs[2].set_yticks([])
      axs[2].set_title('mask')
      axs[2].imshow(crop_w_mask)

      # Draw category scores.
      fontsize = max(min(fig_size_h / float(len(category_names)) * 45, 20), 8)
      for cat_idx in range(len(category_names)):
        axs[3].barh(cat_idx, scores[cat_idx],
                    color='orange' if scores[cat_idx] == max(scores) else 'blue')
      axs[3].invert_yaxis()
      axs[3].set_axisbelow(True)
      axs[3].set_xlim(0, 1)
      plt.xlabel("confidence score")
      axs[3].set_yticks(range(len(category_names)))
      axs[3].set_yticklabels(category_names, fontdict={
          'fontsize': fontsize})

      cnt += 1
      # fig.tight_layout()


  # print('Detection counts:', cnt)
  def convert_boxes_to_x1y1wh(bboxes):
    """
    Convert a list of bounding boxes from [y1, x1, y2, x2] format to [x1, y1, width, height] format.

    Args:
        bboxes (list of lists): List of bounding boxes in [y1, x1, y2, x2] format.

    Returns:
        list of lists: List of bounding boxes in [x1, y1, width, height] format.
    """
    x1y1wh_boxes = []
    for bbox in bboxes:
        y1, x1, y2, x2 = bbox
        width = x2 - x1
        height = y2 - y1
        x1y1wh_boxes.append([x1, y1, width, height])
    return x1y1wh_boxes
  def convert_boxes_to_x1y1x2y2(bboxes):
    """
    Convert a list of bounding boxes from [y1, x1, y2, x2] format to [x1, y1, x2, y2] format.

    Args:
        bboxes (list of lists): List of bounding boxes in [y1, x1, y2, x2] format.

    Returns:
        list of lists: List of bounding boxes in [x1, y1, x2, y2] format.
    """
    x1y1x2y2_boxes = []
    for bbox in bboxes:
        y1, x1, y2, x2 = bbox
        x1y1x2y2_boxes.append([x1, y1, x2, y2])
    return x1y1x2y2_boxes
  
  return convert_boxes_to_x1y1x2y2(bboxes), predicted_categories, confidences, total_scores



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
    parser.add_argument('--out', type=str, required=True, help='Out path')
    parser.add_argument('--n_hardnegatives', type=int, default=10, help="Number of hardnegatives in each vocabulary")
    args = parser.parse_args()
    
    data = read_json(args.dataset)
    
    
    
    
    complete_outputs_gpu = []
    coco_path = '/gpfsdata/home/yangshuai/data/coco/'
    if args.n_hardnegatives == 0 and '1_attributes' not in args.dataset:
        return
    max_boxes_to_draw = 100 #@param {type:"integer"}

    nms_threshold = 0.5 #@param {type:"slider", min:0, max:0.9, step:0.05}
    min_rpn_score_thresh = 0.1  #@param {type:"slider", min:0, max:1, step:0.01}
    min_box_area = 50 #@param {type:"slider", min:0, max:10000, step:1.0}


    params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area

    categories_done = []
    complete_outputs = []
    for ann in tqdm(data['annotations']):
        if ann['category_id'] not in categories_done:
            categories_done.append(ann['category_id'])
        else:
            continue
        
        image_filepath = coco_path + get_image_filepath(ann['image_id'], data['images'])
        vocabulary, vocabulary_id = create_vocabulary(ann, data['categories'])
        len_vocabulary = args.n_hardnegatives + 1
        if len(vocabulary) < len_vocabulary:
            continue
        vocabulary = vocabulary[:len_vocabulary]
        vocabulary_id = vocabulary_id[:len_vocabulary]
        boxes, categories, confidences, total_scores = inference(image_filepath, vocabulary, params)
        # boxes = [box.tolist() for box in boxes]
        total_scores = [total_score.tolist() for total_score in total_scores]
        output = {'scores': confidences,
                'labels': categories,
                'boxes': boxes,
                'total_scores': total_scores
        }
        output['category_id'] = ann['category_id']
        output['vocabulary'] = vocabulary_id
        output['image_filepath'] = get_image_filepath(ann['image_id'], data['images'])
        output = adjust_out_id(output, vocabulary_id)
        complete_outputs.append(output)
    
    save_object(complete_outputs, args.out)
    
if __name__ == "__main__":
    main()
