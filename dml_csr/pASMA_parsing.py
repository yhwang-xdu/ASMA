#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author     :   Qingping Zheng
@Contact    :   qingpingzheng2014@gmail.com
@File       :   datasets.py
@Time       :   10/01/21 00:00 PM
@Desc       :   
@License    :   Licensed under the Apache License, Version 2.0 (the "License"); 
@Copyright  :   Copyright 2015 The Authors. All Rights Reserved.
"""

import os

import cv2
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F


from inplace_abn import InPlaceABN
from .networks import dml_csr


DATA_DIRECTORY = './datasets/Helen'
IGNORE_LABEL = 255
NUM_CLASSES = 19
SNAPSHOT_DIR = './snapshots/'
INPUT_SIZE = 473,473

dict_ = {'background':0, 'skin':1, 'nose':2, 'eye_g':3, 'l_eye':4, 'r_eye':5,
        'l_brow':6, 'r_brow':7, 'l_ear':8, 'r_ear':9, 'mouth':10, 'u_lip':11,
        'l_lip':12, 'hair':13, 'hat':14, 'ear_r':15, 'neck_l':16, 'neck':17,
        'cloth':18}

def parse(imgs, features = ['nose', 'l_brow', 'r_brow', 'mouth']):
    #cudnn.benchmark = True
    #cudnn.enabled = True

    
    torch.cuda.set_device(1)
    
    index = []
    for c in features:
        index.append(dict_[c]) 
    
    model = dml_csr.DML_CSR(NUM_CLASSES, InPlaceABN, False)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    restore_from = '/data/WangYH/DeepfakeBench/dml_csr/dml_csr_celebA.pth'
    state_dict = torch.load(restore_from,map_location='cpu')
    model.load_state_dict(state_dict)

    model.to('cuda:1')
    model.eval()

    bound = valid(model, imgs, index)
    #print(mask.shape)
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    
    return bound

def valid(model, imgs, index):

    model.eval()
    
    idx = 0
    bounds_list = []
    
    with torch.no_grad():
        imgs = imgs.squeeze(0)
        imgs = imgs.permute(1,2,0).cpu().numpy()
        image = (imgs*255).astype(np.float32)
        # store the image before parsing
        cv2.imwrite('./dml_csr/data/before_pASMA_parse.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        time.sleep(0.2)
        image_ = data('./dml_csr/data/before_pASMA_parse.jpg')

        image_= image_.unsqueeze(0).cuda(1)

        results = model(image_)

        outputs = results
        #print(image_.min(), image_.max()) #(1,19,119,119)
        outputs = torch.argmax(outputs, dim=1).float()

        outputs = F.interpolate(
                    outputs.unsqueeze(1), (255, 255), mode='nearest').squeeze(0)
        
        #print(outputs)
        parsing = np.asarray(outputs.squeeze(0).cpu(), dtype=np.uint8)
        #print(parsing)
        matplotlib.use('Agg')
        
        for idx in index:    #specific class 
            parsing_copy = parsing.copy()
            # 标记指定类别区域为255，其他区域为0
            parsing_copy[parsing_copy == idx] = 255
            
            parsing_copy[parsing_copy != 255] = 0  # 其他区域设为0
            #print(parsing_copy)
            
            plt.imshow(parsing_copy)
            plt.savefig(f'op_result/test{idx}.png')
            # 生成每个特征区域的bounds
            bounds = create_bounds(imgs, parsing_copy, 256, 256)
            bounds_list.extend(bounds)

    return bounds_list

def find_bounds_and_average(img, idx_where, dim_x, dim_y, mask=None):
    bounds = []
    
    # 初始化最大和最小的i, j值
    min_i, max_i = dim_x, 0
    min_j, max_j = dim_y, 0
    
   
    pixel_count = 0
    total_r = 0 
    total_g = 0
    total_b = 0
    for i, j in zip(*idx_where):
        if mask is None or mask[i, j]:  # 如果该像素在目标区域内
            # 更新最大和最小的i, j值
            min_i = min(min_i, i)
            max_i = max(max_i, i)
            min_j = min(min_j, j)
            max_j = max(max_j, j)
            # 获取当前像素的r, g, b值
            r, g, b = img[i, j]

            # 累加r, g, b的总和
            total_r += r
            total_g += g
            total_b += b
            pixel_count += 1


    # 计算r, g, b的平均值
    if pixel_count > 0:
        avg_r = total_r / pixel_count
        avg_g = total_g / pixel_count
        avg_b = total_b / pixel_count
    else:
        avg_r, avg_g, avg_b = 0, 0, 0  # 如果没有有效像素，设置为0

    # 返回最大/最小的i, j值以及r, g, b的平均值和bounds
    return (min_i, max_i, min_j, max_j), (avg_r, avg_g, avg_b)

def create_bounds(img, parsing, dim_x, dim_y, mask=None):
    """
    Create bounds for differential evolution based on parsing mask.

    :param parsing: 2D array that indicates the segmented part (the target area).
    :param dim_x: Image width
    :param dim_y: Image height
    :param pixel_count: Number of pixels to perturb
    :param mask: Optional mask indicating which pixels to perturb.
    :return: bounds for differential evolution
    """
    # Create an empty list for bounds
    bounds = []

    # Mask where parsing == 1e6 (targeted part)
    idx_where = np.where(parsing == 255)  # 255 is the mark for the targeted part
    #print(idx_where)
    
    (min_i, max_i, min_j, max_j), (avg_r, avg_g, avg_b) = find_bounds_and_average(img, idx_where, dim_x, dim_y)
    
    if min_i > max_i :
        return [(0,256), (0,256), (0,1), (0,1), (0,1)]

    bounds.append((min_i, max_i))  # x-coordinate range
    bounds.append((min_j, max_j))  # y-coordinate range
    

    bounds.append((max(0, avg_r*0.8), min(255, avg_r*1.2)))  # r value range
    bounds.append((max(0, avg_g*0.8), min(255, avg_g*1.2)))  # g value range
    bounds.append((max(0, avg_b*0.8), min(255, avg_b*1.2)))  # b value range
    return bounds

def data(im_path):


    im = cv2.imread(im_path, cv2.IMREAD_COLOR)
    im = cv2.resize(im,(473,473), interpolation=cv2.INTER_LINEAR)
    #trans = transforms.get_affine_transform((0,0), 0, 0, (473,473))
    image = cv2.warpAffine(
        im,
        np.float32([[1,0,0],[0,1,0]]),
        (473, 473),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return transform(image)