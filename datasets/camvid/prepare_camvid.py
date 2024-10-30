#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

RGBLabel2LabelName = {
    (128, 128, 128): "Sky",

    (0, 128, 64): "Building",
    (128, 0, 0): "Building",
    (64, 192, 0): "Building",
    (64, 0, 64): "Building",
    (192, 0, 128): "Building",

    (192, 192, 128): "Pole",
    (0, 0, 64): "Pole",

    (128, 64, 128): "Road",
    (128, 0, 192): "Road",
    (192, 0, 64): "Road",

    (0, 0, 192): "Sidewalk",
    (64, 192, 128): "Sidewalk",
    (128, 128, 192): "Sidewalk",

    (128, 128, 0): "Tree",
    (192, 192, 0): "Tree",

    (192, 128, 128): "SignSymbol",
    (128, 128, 64): "SignSymbol",
    (0, 64, 64): "SignSymbol",

    (64, 64, 128): "Fence",

    (64, 0, 128): "Car",
    (64, 128, 192): "Car",
    (192, 128, 192): "Car",
    (192, 64, 128): "Car",
    (128, 64, 64): "Car",

    (64, 64, 0): "Pedestrian",
    (192, 128, 64): "Pedestrian",
    (64, 0, 192): "Pedestrian",
    (64, 128, 64): "Pedestrian",

    (0, 128, 192): "Bicyclist",
    (192, 0, 192): "Bicyclist",

    (0, 0, 0): "Void"
}

labels = [
    "Bicyclist", 
    "Building", 
    "Car", 
    "Pole", 
    "Fence",  
    "Pedestrian", 
    "Road", 
    "Sidewalk", 
    "SignSymbol", 
    "Sky",  
    "Tree"
]

def rgb2label(img):
    # 获取数组的形状  
    shape = img.shape[:-1]  # 去掉最后一维的形状  
    
    # 初始化一个结果数组，用于存储映射后的值  
    result = np.zeros(shape, dtype=np.uint8)  
    
    # 遍历数组，应用映射  
    for i in np.ndindex(shape):  
        tuple_value = tuple(img[i])  # 将当前位置的子数组转换为元组  
        if tuple_value in RGBLabel2LabelName:  
            label_name = RGBLabel2LabelName[tuple_value]  # 如果元组在映射中，则替换为对应的值 
            if label_name == "Void":
                result[i] = 255
            else:
                result[i] = labels.index(label_name)
        else:  
            # 处理不在映射中的情况，这里设置为-1或其他适当的值  
            result[i] = 255

    return result

def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = rgb2label(img)  # 0 (ignore) becomes 255. others are shifted by 1
    try:
        Image.fromarray(img).save(output)
    except:
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) /  "camvid"
    for name in ["camvid_test_labels"]:
        annotation_dir = dataset_dir / name
        output_dir = dataset_dir / "annotations_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file)