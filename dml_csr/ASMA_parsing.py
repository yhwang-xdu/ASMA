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
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import numpy as np
import cv2
import time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from inplace_abn import InPlaceABN
from .networks import dml_csr

NUM_CLASSES = 19

dict_ = {'background':0, 'skin':1, 'nose':2, 'eye_g':3, 'l_eye':4, 'r_eye':5,
        'l_brow':6, 'r_brow':7, 'l_ear':8, 'r_ear':9, 'mouth':10, 'u_lip':11,
        'l_lip':12, 'hair':13, 'hat':14, 'ear_r':15, 'neck_l':16, 'neck':17,
        'cloth':18}

def parse(imgs, attack, label, features = ['nose', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'mouth', 'hair']):

    torch.cuda.set_device(1)
    
    index = []
    for c in features:
        index.append(dict_[c]) 
    
    model = dml_csr.DML_CSR(NUM_CLASSES, InPlaceABN, False)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    restore_from = '/data/WangYH/DeepfakeBench/dml_csr/dml_csr_celebA.pth'
    state_dict = torch.load(restore_from,map_location='cpu')
    model.load_state_dict(state_dict)

    model.to('cuda:1')
    model.eval()
    mask = valid(model, imgs, index)
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    imgs.cuda(0)
    mask.cuda(0)
    attacked = attack(imgs, label, mask)

    return attacked


def data(im_path):

    im = cv2.imread(im_path, cv2.IMREAD_COLOR)
    im = cv2.resize(im,(473,473), interpolation=cv2.INTER_LINEAR)
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

def valid(model, imgs, index):

    model.eval()
    
    idx = 0

    batch_size = imgs.shape[0]
    #imgs = imgs.cuda(1)
    out_mask = []
    indexs = []
    with torch.no_grad():
        for i in range(batch_size):
            image = imgs[i]#(1, 3, 299, 299)

            image = image.permute(1,2,0).cpu().numpy()
            image = (image*255).astype(np.float32)
            # save the image before parsing for analyze.
            image_= cv2.imwrite(f'./dml_csr/data/before_parsing{i}.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            time.sleep(0.2) # time sleep for I/O
            image_ = data(f'./dml_csr/data/before_parsing{i}.jpg')  # read the image for parsing
            #image_ = data(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_= image_.unsqueeze(0).cuda(1)
            results = model(image_)
            outputs = results
            
            outputs = F.interpolate(
                     outputs, (imgs.shape[2], imgs.shape[3]), mode='bilinear', align_corners=True)
            
            parsing = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
            
            for idx in index:    #specific class 
                parsing[parsing == idx] = 1e6
            idx_where = np.where(parsing==1e6)
            
            indexs.append(idx_where)

    out_imgs = []
    for i in range(batch_size):   
        img = imgs[i]#.detach()
        out_img = img.clone().zero_()#.detach().zero_()
        index = indexs[i]

        out_img[:, index[0],index[1]] = img[:, index[0], index[1]] #+ img[:, index[0], index[1]] 
        out_imgs.append(out_img.cpu().numpy())
    out_imgs = np.array(out_imgs)  # change: first numpy.ndarray then tensor to GPU
    out_imgs = torch.tensor(out_imgs, requires_grad=True).cuda()
    out_mask = torch.abs(out_imgs).sign()
    
    return out_mask
