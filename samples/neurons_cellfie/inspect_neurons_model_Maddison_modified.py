#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:36:25 2020

@author: nel
"""

#!/usr/bin/env python3
# # Mask R-CNN - Inspect Neurons Trained Model
# Code and visualizations to test, debug, and evaluate the Mask R-CNN model.

#%%
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = '42'
matplotlib.rcParams['ps.fonttype'] = '42'
import matplotlib.pyplot as plt
import matplotlib.patches as patches
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/nel/Code/NEL_LAB/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.neurons import neurons
from caiman.base.rois import nf_match_neurons_in_binary_masks

get_ipython().run_line_magic('matplotlib', 'auto')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# %%
# Device to load the neural network on.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

#%%

for detect in [.7]:
    
    # ## Configurations
    config = neurons.NeuronsConfig()
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.7
        IMAGE_RESIZE_MODE = "pad64"
        IMAGE_MAX_DIM=512
        #CLASS_THRESHOLD = 0.33
        #RPN_NMS_THRESHOLD = 0.7
        #DETECTION_NMS_THRESHOLD = 0.3
        #IMAGE_SHAPE=[512, 128,3]
        RPN_NMS_THRESHOLD = 0.9
        #TRAIN_ROIS_PER_IMAGE = 1000
        POST_NMS_ROIS_INFERENCE = 4000
        DETECTION_MAX_INSTANCES = 1000
    config = InferenceConfig()
    config.display()
    
    #%% Load  dataset
    #cross = 'cross3'
    #dataset_idx = 2
    #mode = ["train", "val"][1]
    for mode in ["val"]:
        for dataset_idx in [0]:
            """
            dataset_name = ["voltage_v1.2", "voltage_v1.2_cross2", "voltage_v1.2_cross3",
                            "voltage_v1.2_L1_6", "voltage_v1.2_L1_4", "voltage_v1.2_L1_2", "voltage_v1.2_L1_1", 
                            "voltage_v1.2_TEG_2", "voltage_v1.2_TEG_1", "voltage_v1.2_HPC_8", "voltage_v1.2_HPC_4",
                            "voltage_v1.2_HPC_2", "voltage_v1.2_HPC_1"] [dataset_idx]
            """
            dataset_name = ['maskrcnn_data_npz'] [dataset_idx]
        
            """
            weights = ["/neurons20200824T1032/mask_rcnn_neurons_0040.h5",
                       "/neurons20200825T0951/mask_rcnn_neurons_0040.h5", 
                       "/neurons20200825T1039/mask_rcnn_neurons_0040.h5",
                       '/neurons20200901T0906/mask_rcnn_neurons_0030.h5',
                       '/neurons20200901T1008/mask_rcnn_neurons_0030.h5',
                       '/neurons20200901T1058/mask_rcnn_neurons_0030.h5',
                       '/neurons20200902T1530/mask_rcnn_neurons_0030.h5',
                       "/neurons20200903T1124/mask_rcnn_neurons_0030.h5",
                       "/neurons20200903T1215/mask_rcnn_neurons_0030.h5",
                       '/neurons20200926T0919/mask_rcnn_neurons_0030.h5', 
                       "/neurons20200926T1036/mask_rcnn_neurons_0030.h5",
                       "/neurons20200926T1124/mask_rcnn_neurons_0030.h5", 
                       "/neurons20200926T1213/mask_rcnn_neurons_0030.h5"
                       ][dataset_idx]
            """
            weights = ['/neurons20201124T1903/mask_rcnn_neurons_0080.h5'][dataset_idx]
            
            NEURONS_DIR = os.path.join(ROOT_DIR, ("datasets/Maddison/" + dataset_name))
            dataset = neurons.NeuronsDataset()
            dataset.load_neurons(NEURONS_DIR, mode)
            
            # Must call before using the dataset
            dataset.prepare()
            print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
            
            # Load Model
            # Create model in inference mode
            with tf.device(DEVICE):
                model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                          config=config)
            
            # Set path to balloon weights file
            #weights_path = model.find_last()
            weights_path = MODEL_DIR + weights    
            
            # Load weights
            print("Loading weights ", weights_path)
            model.load_weights(weights_path, by_name=True)
            
            
            folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/result/'+os.path.split(os.path.split(weights_path)[0])[-1]+'/'+mode+'/'
            try:
                os.makedirs(folder)
                print('make folder')
            except:
                print('already exist')        
            
            # %% Run Detection
            run_single_detection = False
            if run_single_detection:
                image_id = random.choice(dataset.image_ids)
                image_id = 0
                image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
                info = dataset.image_info[image_id]
                print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                                       dataset.image_reference(image_id)))
                
                # Run object detection
                results = model.detect([image], verbose=1)
                
                # Display results
                ax = get_ax(1)
                r = results[0]
                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                            dataset.class_names, r['scores'], ax=ax,
                                            title="Predictions")
                log("gt_class_id", gt_class_id)
                log("gt_bbox", gt_bbox)
                log("gt_mask", gt_mask)
            # %%
            image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset, config, 0, use_mini_mask=False)
            image.shape
            
            # %%
            performance = {}
            F1 = {}
            recall = {}
            precision = {}
            number = {}
            for image_id in dataset.image_ids:
                image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
                info = dataset.image_info[image_id]
                print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                                       dataset.image_reference(image_id)))
                
                # Run object detection    
                results = model.detect([image], verbose=1)
                
                # Display results
                _, ax = plt.subplots(1,1, figsize=(16,16))
                r = results[0]
                mask_pr = r['masks'].copy().transpose([2,0,1])*1.
                
                if "IVQ" in info['id']:
                    neuron_idx = (mask_pr.sum((1, 2)) >= 400)
                elif "Fish" in info['id']:
                    neuron_idx = (mask_pr.sum((1, 2)) >= 100)
                else:
                    neuron_idx = (mask_pr.sum((1, 2)) >= 0)
                
                r['rois'] = r['rois'][neuron_idx]
                r['masks'] = r['masks'][:, :, neuron_idx]
                r['class_ids'] = r['class_ids'][neuron_idx]
                r['scores'] = r['scores'][neuron_idx]
            
                display_result = True
                if display_result:  
                    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                                dataset.class_names, r['scores'], ax=ax,
                                                title="Predictions")
                    plt.savefig(folder +dataset.image_info[image_id]['id'][:-4]+'_mrcnn_result.pdf')
                    plt.close()
                
                mask_pr = r['masks'].copy().transpose([2,0,1])*1.
                mask_gt = gt_mask.copy().transpose([2,0,1])*1.
                
                print("MASK GT INFO", mask_gt.shape,  mask_gt.dtype)
                print("MASK PR INFO", mask_pr.shape, mask_pr.dtype)
                print(mask_pr)
                tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                        mask_gt, mask_pr, thresh_cost=0.7, min_dist=10, print_assignment=True,
                        plot_results=True, Cn=image[:,:,0], labels=['GT', 'MRCNN'])
                plt.savefig(folder +dataset.image_info[image_id]['id'][:-4]+'_compare.pdf')
                plt.close()
                performance[info['id'][:-4]] = performance_cons_off
                F1[dataset.image_info[image_id]['id'][:-4]] = performance_cons_off['f1_score']
                recall[dataset.image_info[image_id]['id'][:-4]] = performance_cons_off['recall']
                precision[dataset.image_info[image_id]['id'][:-4]] = performance_cons_off['precision']
                number[dataset.image_info[image_id]['id'][:-4]] = dataset.image_info[image_id]['polygons'].shape[0]
                np.save(ROOT_DIR + '/result_f1/' + dataset_name + '_' + mode, performance)
                break