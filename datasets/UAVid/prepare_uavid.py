#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

# UAVid 数据集的 RGB 到标签映射
RGBLabel2LabelName = {
    (0, 0, 0): "Background",         # 背景
    (128, 0, 0): "Building",         # 建筑物
    (0, 128, 0): "Tree",             # 树木
    (192, 0, 192): "Moving Car",     # 移动车辆
    (128, 64, 128): "Road",          # 道路
    (64, 64, 0): "Low Vegetation",   # 低矮植被
    (64, 0, 128): "Static Car",      # 静止车辆
    (128, 128, 0): "Human"           # 行人
}

labels = [
    "Background",    # 背景
    "Building",      # 建筑物
    "Tree",          # 树木
    "Moving Car",    # 移动车辆
    "Road",          # 道路
    "Low Vegetation",# 低矮植被
    "Static Car",    # 静止车辆
    "Human"          # 行人
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
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) /  "uavid"
    for name in ["val/Labels"]:
        annotation_dir = dataset_dir / name
        output_dir = dataset_dir / "annotations_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file)