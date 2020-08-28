#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:27:52 2020

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


def filter_coco(coco, cls_split):
    new_anns = []
    all_cls_dict = {}
    for img_id, id in enumerate(coco.imgs):
        img = coco.loadImgs(id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))
        skip_flag = False
        img_cls_dict = {}
        if len(anns) == 0:
            continue
        for ann in anns:
            segmentation = ann['segmentation']
            area = ann['area']
            iscrowd = ann['iscrowd']
            image_id = ann['image_id']
            bbox = ann['bbox']
            category_id = ann['category_id']
            id = ann['id']
            bbox_area = bbox[2] * bbox[3]
            
            # filter images with small boxes
            if category_id in cls_split:
                if bbox_area < 32 * 32:
                    skip_flag = True
                
        if skip_flag:
            continue
        else:
            for ann in anns:
                category_id = ann['category_id']
                if category_id in cls_split:
                    new_anns.append(ann)

                    if category_id in all_cls_dict.keys():
                        all_cls_dict[category_id] += 1
                    else:
                        all_cls_dict[category_id] = 1
                        
    print(len(new_anns))
    print(sorted(all_cls_dict.items(), key = lambda kv:(kv[1], kv[0])))     
    return new_anns


root_path = sys.argv[1]
print(root_path)
#root_path = '/home/fanqi/data/COCO'
dataDir = './annotations'
support_dict = {}

support_dict['support_box'] = []
support_dict['category_id'] = []
support_dict['image_id'] = []
support_dict['id'] = []
support_dict['file_path'] = []

voc_inds = (0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62)


for dataType in ['instances_train2017.json']: #, 'split_voc_instances_train2017.json']:
    annFile = join(dataDir, dataType)

    with open(annFile,'r') as load_f:
        dataset = json.load(load_f)
        print(dataset.keys())
        save_info = dataset['info']
        save_licenses = dataset['licenses']
        save_images = dataset['images']
        save_categories = dataset['categories']
        save_annotations = dataset['annotations']


    inds_split2 = [i for i in range(len(save_categories)) if i not in voc_inds]

    # split annotations according to categories
    categories_split1 = [save_categories[i] for i in voc_inds]
    categories_split2 = [save_categories[i] for i in inds_split2]
    cids_split1 = [c['id'] for c in categories_split1]
    cids_split2 = [c['id'] for c in categories_split2]
    print('Split 1: {} classes'.format(len(categories_split1)))
    for c in categories_split1:
        print('\t', c['name'])
    print('Split 2: {} classes'.format(len(categories_split2)))
    for c in categories_split2:
        print('\t', c['name'])
    
    coco = COCO(annFile)

    # for non-voc, there is no voc images
    annotations_split2 = filter_coco(coco, cids_split2)

    # for voc, there can be non_voc images
    annotations = dataset['annotations']
    annotations_split1 = []
    
    for ann in annotations:
        if ann['category_id'] in cids_split1: # voc 20
            annotations_split1.append(ann)

    dataset_split1 = {
        'info': save_info,
        'licenses': save_licenses,
        'images': save_images,
        'annotations': annotations_split1,
        'categories': save_categories}
    dataset_split2 = {
        'info': save_info,
        'licenses': save_licenses,
        'images': save_images,
        'annotations': annotations_split2,
        'categories': save_categories}
    new_annotations_path = os.path.join(root_path, 'new_annotations')
    if not os.path.exists(new_annotations_path):
        os.makedirs(new_annotations_path)
    split2_file = os.path.join(root_path, 'new_annotations/final_split_non_voc_instances_train2017.json')

    with open(split2_file, 'w') as f:
        json.dump(dataset_split2, f) 
