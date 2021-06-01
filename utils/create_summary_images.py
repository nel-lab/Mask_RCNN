#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 10:25:01 2020
Preparing summary images for voltage imaging
@author: caichangjia
"""
#%%
import matplotlib.pyplot as plt
import caiman as cm
import numpy as np
import cv2
import scipy.io as io
import os
import cv2 as cv
from caiman.summary_images import local_correlations_movie_offline

#%%
def normalize(img):
    return (img - np.mean(img)) / np.std(img)

#%%
root_dir = '/home/nel/Code/NEL_LAB/Mask_RCNN/data_voltage'
files = os.listdir(root_dir + '/raw_data')
files = np.array(sorted([file for file in files if '.mmap' in file]))
TEG_set = np.array([0, 1, 2])
L1_set = np.array([3, 6, 7, 8, 9, 10, 11, 12, 15])
HPC_set = np.arange(18, 30)
all_set = np.concatenate((TEG_set, L1_set, HPC_set))

#%%
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

#%%
for file in files[HPC_set]:
    if file in files[L1_set]:
        fr = 400
        transpose = True
        flip = True
        gaussian_blur = False
    elif file in files[TEG_set]:
        fr = 300
        transpose = False
        flip = True
        gaussian_blur = True
    elif file in files[HPC_set]:
        fr = 1000
        transpose = False
        flip = False
        gaussian_blur = False

    X = {}
    filename = root_dir +'/raw_data/' + file
    mov = cm.load(filename, fr=fr, in_memory=True)
    if transpose:
        mov = mov.transpose([0, 2, 1])
    mov = mov[1000:, :, :]
    plt.figure(); plt.plot(mov.mean((1,2))); plt.show()
    
    corr_img = local_correlations_movie_offline(filename, fr=fr, window=mov.shape[0], 
                                          stride=mov.shape[0], winSize_baseline=fr, 
                                          remove_baseline=True, gaussian_blur=gaussian_blur,
                                          dview=dview).max(axis=0)
    
    mov_rb =  mov.removeBL(fr)
    plt.figure(); plt.plot(mov_rb.mean((1,2))); plt.show()

    if transpose:
        corr_img = corr_img.T
    X['mean'] = np.mean(mov, axis=0)
    X['corr'] = corr_img
    X['std'] = np.std(mov_rb, 0)
    X['median'] = (mov_rb).bin_median()
    
    if flip == True:
        X['max'] = np.max(-mov_rb, 0)
    else:
        X['max'] = np.max(mov_rb, 0)
        
    np.save(root_dir + '/summary_images/' + file.split('.')[0] + '_summary.npy', X)
    
    X['mean'] = normalize(X['mean'])
    X['corr'] = normalize(X['corr'])
    X['std'] = normalize(X['std'])
    X['median'] = normalize(X['median'])
    X['max'] = normalize(X['max'])
    
    for key in X.keys():
        plt.figure(); plt.imshow(X[key]); plt.show()

    cm.movie(np.array(list(X.values()))).save(root_dir + '/summary_images/' + file.split('.')[0] + '_summary.tif')

    del mov_rb
    
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)






