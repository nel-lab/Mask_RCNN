#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:01:14 2020
create training files
@author: nel
"""
#%%
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import zipfile

import caiman as cm
from caiman.base.rois import nf_read_roi
from caiman.base.rois import nf_read_roi_zip

#%%
img_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/data_voltage/summary_images'
masks_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/labels/combination_v1.2'
#save_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/datasets/voltage_v1.2_cross3'
#save_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/datasets/voltage_v1.2_L1_6'
#save_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/datasets/voltage_v1.2_L1_4'
#save_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/datasets/voltage_v1.2_L1_2'
#save_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/datasets/voltage_v1.2_L1_1'
#save_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/datasets/voltage_v1.2_TEG_2'
save_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/datasets/voltage_v1.2_TEG_1'

try:
    os.mkdir(save_folder)
    os.mkdir(save_folder+'/train')
    os.mkdir(save_folder+'/val')
except OSError:
    print ("already exist")
else:
    print ("make folders" )

files = sorted(os.listdir(img_folder))
files = np.array([file for file in files if '_summary.tif' in file])

#val_set = np.array([0, 3, 8, 10, 12, 15, 18, 21])
#val_set = np.array([1, 4, 7, 9, 13, 16, 19, 22])
#val_set = np.array([2, 5, 6, 11, 14, 17, 20, 23])
#val_set = np.array([3, 8, 10])
#train_set = np.array([4])
val_set = np.array([0])
train_set = np.array([1])

for file in files:
    if file in files[val_set]:
        group = 'val'
    elif file in files[train_set]:
        group = 'train'
    else:
        group = None
        
    if group is not None:
        #else:
        #    group = 'train'
        summary_img = cm.load(os.path.join(img_folder, file))
        input_img =  summary_img[[0, 0, 1], :, :]
    
        with zipfile.ZipFile(os.path.join(masks_folder, file.split('_summary')[0]+'.zip')) as zf:
            names = zf.namelist()
            coords = [nf_read_roi(zf.open(n)) for n in names]
            if input_img.shape[2] < 128:
                for idx, i in enumerate(coords):
                    coords[idx][:, 1] = list(np.array(i[:, 1]+ int((128-input_img.shape[2])/2)))
            aa = coords.copy()
            print('hha')
            polygons = [{'name': 'polygon','all_points_x':i[:,1],'all_points_y':i[:,0]} for i in coords]
            np.savez(save_folder+'/'+group+'/'+file.split('_summary')[0]+'_mask.npz', mask = polygons)      
        
        if input_img.shape[2] < 128:
            temp = input_img.copy()
            a = []
            for idx, img in enumerate(temp):
                a.append(cv2.copyMakeBorder(img,0, 0, int((128-input_img.shape[2])/2), int((128-input_img.shape[2])/2), borderType=cv2.BORDER_CONSTANT, value=0))
            input_img = np.stack(a)
        print(input_img.shape)    
        plt.figure();plt.imshow(input_img[0]);plt.show()
        input_img = input_img.transpose([1,2,0])
        np.savez(save_folder + '/' + group + '/' + file.split('_summary')[0]+'.npz', img = input_img)
        
"""
for i in aa:
    plt.figure();plt.plot(i[:,0], i[:,1])

import skimage
mask = np.zeros((284, 128, len(polygons)))
for i, p in enumerate(polygons):
    # Get indexes of pixels inside the polygon and set them to 1
    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
    mask[rr, cc, i] = 1            
"""           
        
