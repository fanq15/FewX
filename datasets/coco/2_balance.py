#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:04:06 2020

@author: fanq15
"""

from pycocotools.coco import COCO
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import sys

def balance_coco(coco):
    all_cls_dict = {}
    for img_id, id in enumerate(coco.imgs):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))
        img_cls_dict = {}
        if len(anns) == 0:
            continue
        for ann in anns:
            category_id = ann['category_id']
            id = ann['id']
            
            if category_id in img_cls_dict.keys():
                img_cls_dict[category_id] += 1
            else:
                img_cls_dict[category_id] = 1
            
        for category_id, num in img_cls_dict.items():
            if category_id in all_cls_dict.keys():
                all_cls_dict[category_id] += 1 # count the image number containing the target category
            else:
                all_cls_dict[category_id] = 1 
    print('Image number of non-voc classes before class balancing: ', sorted(all_cls_dict.items(), key = lambda kv:(kv[1], kv[0])))  
    
    new_anns = []
    for img_id, id in enumerate(coco.imgs):
        save_flag = False
        anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))
        if len(anns) == 0:
            continue
        # in this image, if there is at least one category with less than 2000 images, save this image.
        # otherwise, remove this image.
        for ann in anns:
            category_id = ann['category_id']
            
            if all_cls_dict[category_id] <= 2000:
                save_flag = True
            
        if save_flag:
            for ann in anns:
                new_anns.append(ann)
        else:
            for ann in anns:
                category_id = ann['category_id']
                all_cls_dict[category_id] -= 1
    print('Instance number of non-voc classes after class balancing: ', len(new_anns))
    print('Image number of non-voc classes after class balancing: ', sorted(all_cls_dict.items(), key = lambda kv:(kv[1], kv[0])))  
    
    return new_anns

root_path = sys.argv[1]
#root_path = '/home/fanqi/data/COCO'
dataDir = os.path.join(root_path, 'new_annotations')
support_dict = {}

support_dict['support_box'] = []
support_dict['category_id'] = []
support_dict['image_id'] = []
support_dict['id'] = []
support_dict['file_path'] = []


for dataType in ['split_non_voc_instances_train2017.json']:
    annFile = join(dataDir, dataType)

    with open(annFile,'r') as load_f:
        dataset = json.load(load_f)
        print(dataset.keys())
        save_info = dataset['info']
        save_licenses = dataset['licenses']
        save_images = dataset['images']
        save_categories = dataset['categories']

    print(annFile)
    shot_num = 10
    coco = COCO(annFile)
    print(coco)
    
    annotations = balance_coco(coco)
    dataset_split = {
        'info': save_info,
        'licenses': save_licenses,
        'images': save_images,
        'annotations': annotations,
        'categories': save_categories}
    split_file = os.path.join(root_path, 'new_annotations/final_split_non_voc_instances_train2017.json')

    with open(split_file, 'w') as f:
        json.dump(dataset_split, f)
