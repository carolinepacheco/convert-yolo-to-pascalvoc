#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 22:10:01 2018

@author: Caroline Pacheco do E. Silva
"""

import os
import cv2
import json
import argparse

import numpy as np

from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
from os.path import join

parser = argparse.ArgumentParser(description="Convert YOLO annotations to PASCAL VOC format",
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    '--base_folder',
    help='Folder to search for YOLO dataset. defaults to "yolo"',
    type=str,
    default="yolo",
    required=False
)
parser.add_argument(
    '--output_folder',
    help='Output folder name to store the PASCAL annotations. defaults to {base_folder}/pascal',
    type=str,
    default="pascal",
    required=False
)
parser.add_argument(
    '--label_folder',
    help='Folder containing labels in YOLO format. defaults to {base_folder}/labels',
    type=str,
    default="labels",
    required=False
)
parser.add_argument(
    '--image_folder',
    help='Folder containing dataset images. defaults to {base_folder}/images',
    type=str,
    default="images",
    required=False
)
parser.add_argument(
    '--class_map',
    help='Path to json file where an object of type { number: string } stores class values for custom dataset. defaults of all coco classes',
    type=str,
    default=None,
    required=False
)


## coco classes
YOLO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')


def load_classes(json_path):
    if json_path == None:
        return YOLO_CLASSES

    with open(json_path, 'r') as fp:
        class_map = json.load(fp)
    
    return { int(k): v for k, v in class_map.items() }


## converts the normalized positions into integer positions
def unconvert(class_id, width, height, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)


## converts coco into xml 
def main(args):
    base_folder = args.base_folder
    output_folder = args.output_folder
    label_folder = args.label_folder
    image_folder = args.image_folder
    class_map = args.class_map

    assert base_folder != None
    assert output_folder != None
    assert label_folder != None
    assert image_folder != None    

    classes = load_classes(class_map)

    class_path  = join(base_folder, 'labels')
    ids = list()
    l=os.listdir(class_path)
    
    check = '.DS_Store' in l
    if check == True:
        l.remove('.DS_Store')
        
    ids=[x.split('.')[0] for x in l]   

    annopath = join(base_folder, label_folder, '%s.txt')
    imgpath = join(base_folder, image_folder, '%s.jpg')
    
    os.makedirs(join(base_folder, output_folder), exist_ok=True)
    outpath = join(base_folder, output_folder, '%s.xml')

    for i in range(len(ids)):
        img_id = ids[i]

        if img_id == "classes":
            continue
        if os.path.exists(outpath % img_id):
            continue

        print(imgpath % img_id)

        img= cv2.imread(imgpath % img_id)
        height, width, channels = img.shape #get sizes and channels from images

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'VOC2007'
        img_name = img_id + '.jpg'
    
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_name
        
        node_source= SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = 'Coco database'
        
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)
    
        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(channels)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        target = (annopath % img_id)
        if os.path.exists(target):
            label_norm= np.loadtxt(target).reshape(-1, 5)

            for i in range(len(label_norm)):
                labels_conv = label_norm[i]
                new_label = unconvert(labels_conv[0], width, height, labels_conv[1], labels_conv[2], labels_conv[3], labels_conv[4])
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = classes[new_label[0]]
                
                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'
                
                
                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(new_label[1])
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(new_label[3])
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text =  str(new_label[2])
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(new_label[4])
                xml = tostring(node_root, pretty_print=True)  
                dom = parseString(xml)

        print(xml)  

        with open(outpath % img_id, "wb") as f:
            f.write(xml)
       

if __name__ == '__main__':
    main(parser.parse_args())
