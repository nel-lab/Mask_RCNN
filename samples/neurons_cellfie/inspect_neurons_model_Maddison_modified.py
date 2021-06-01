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

from samples.neurons_cellfie import neurons_cellfie
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

#%% Load  dataset
for mode in ["val", "train"]:
    for dataset_idx in [0]:
        dataset_name = ['maskrcnn_data_npz'] [dataset_idx]
        #weights = ['/neurons20201124T1903/mask_rcnn_neurons_0080.h5'][dataset_idx]
        #weights = ['/neurons20201129T1828/mask_rcnn_neurons_0140.h5'][dataset_idx]
        weights = ['/neurons20201130T2159/mask_rcnn_neurons_0200.h5'][dataset_idx]
        
        NEURONS_DIR = os.path.join(ROOT_DIR, ("datasets/Maddison/" + dataset_name))
        dataset = neurons_cellfie.NeuronsDataset()
        dataset.load_neurons(NEURONS_DIR, mode)
        
        # Must call before using the dataset
        dataset.prepare()
        print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
        
        # Load Model
        # Create model in inference mode
        for detect in [0.7]:
            for rpn in [0.5]:
                # ## Configurations
                post = 1000
                config = neurons_cellfie.NeuronsConfig()
                class InferenceConfig(config.__class__):
                    # Run detection on one image at a time
                    GPU_COUNT = 1
                    IMAGES_PER_GPU = 1
                    DETECTION_MIN_CONFIDENCE = detect
                    IMAGE_RESIZE_MODE = "pad64"
                    IMAGE_MAX_DIM=512
                    #CLASS_THRESHOLD = 0.33
                    #RPN_NMS_THRESHOLD = 0.7
                    #DETECTION_NMS_THRESHOLD = 0.3
                    #IMAGE_SHAPE=[512, 128,3]
                    RPN_NMS_THRESHOLD = rpn
                    POST_NMS_ROIS_INFERENCE = post
                    DETECTION_MAX_INSTANCES = 1000
                config = InferenceConfig()
                config.display()
        
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
                
                

                #%%
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
                    #F1[dataset.image_info[image_id]['id'][:-4]] = performance_cons_off['f1_score']
                    #recall[dataset.image_info[image_id]['id'][:-4]] = performance_cons_off['recall']
                    #precision[dataset.image_info[image_id]['id'][:-4]] = performance_cons_off['precision']
                    #number[dataset.image_info[image_id]['id'][:-4]] = dataset.image_info[image_id]['polygons'].shape[0]
                    
                    number[dataset.image_info[image_id]['id'][:-4]] = [mask_gt.shape[0], mask_pr.shape[0]]
                
                
                folder = ROOT_DIR + f'/result_f1/CellFie/1130T2159_2_detect_{detect}_nms_{post}_rpn_{rpn}/' 
                try:
                    os.makedirs(folder)
                    print('make folder')
                except:
                    print('already exist') 
                
                np.save(ROOT_DIR + f'/result_f1/CellFie/1130T2159_2_detect_{detect}_nms_{post}_rpn_{rpn}/' + dataset_name + '_' + mode, performance)
                np.save(ROOT_DIR + f'/result_f1/CellFie/1130T2159_2_detect_{detect}_nms_{post}_rpn_{rpn}/' + dataset_name + '_' + mode + '_number', number)


#%%
result_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/result_f1/CellFie/1130T2159_detect_0.7_nms_1000_rpn_0.5'
a = np.load(os.path.join(result_folder, 'maskrcnn_data_npz_train.npy'), allow_pickle=True).item()
b = np.load(os.path.join(result_folder, 'maskrcnn_data_npz_val.npy'), allow_pickle=True).item()
a.update(b)

F1 = [a[key]['f1_score'] for key in a.keys()]
precision = [a[key]['precision'] for key in a.keys()]
recall = [a[key]['recall'] for key in a.keys()]

plt.figure()
plt.title('Mask R-CNN performance on all datasets')
plt.vlines(8.5, 0, 1, linestyles='dashed')
ax = plt.gca()
width = 0.20
labels = list(a.keys())
x = np.arange(len(labels))
rects1 = ax.bar(x - width, F1, width, label='F1')
rects2 = ax.bar(x , precision, width, label='precision')
rects3 = ax.bar(x + width, recall, width, label='recall')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_folder, 'mrcnn_all_datasets.pdf'))

#%%
a = np.load(os.path.join(result_folder, 'maskrcnn_data_npz_train.npy'), allow_pickle=True).item()
b = np.load(os.path.join(result_folder, 'maskrcnn_data_npz_val.npy'), allow_pickle=True).item()

F1 = []; precision = []; recall = []

for p in [a,b]:
    F1.append(np.round(np.array([p[key]['f1_score'] for key in p.keys()]).mean(), 2))
    precision.append(np.array([p[key]['precision'] for key in p.keys()]).mean())
    recall.append(np.array([p[key]['recall'] for key in p.keys()]).mean())

plt.figure()
plt.title('Mask R-CNN performance on train/val sets')
ax = plt.gca()
width = 0.20
labels = ['train', 'val']
x = np.arange(len(labels))
rects1 = ax.bar(x - width, F1, width, label=f'F1_{F1}')
rects2 = ax.bar(x , precision, width, label=f'precision')
rects3 = ax.bar(x + width, recall, width, label=f'recall')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()

plt.savefig(os.path.join(result_folder, 'mrcnn_train_val_datasets.pdf'))

#%%
result_folder = '/home/nel/Code/NEL_LAB/Mask_RCNN/result_f1/CellFie/140_epochs_detect_0.7_nms_1000_rpn_0.5'
a = np.load(os.path.join(result_folder, 'maskrcnn_data_npz_train_number.npy'), allow_pickle=True).item()
b = np.load(os.path.join(result_folder, 'maskrcnn_data_npz_val_number.npy'), allow_pickle=True).item()
a.update(b)

GT = [a[key][0] for key in a.keys()]
mrcnn = [a[key][1] for key in a.keys()]

plt.figure()
plt.title('Mask R-CNN performance on all datasets')
plt.vlines(8.5, 0, 1, linestyles='dashed')
ax = plt.gca()
width = 0.20
labels = list(a.keys())
x = np.arange(len(labels))
rects1 = ax.bar(x - width/2, GT, width, label='GT')
rects2 = ax.bar(x + width/2, mrcnn, width, label='mrcnn')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
plt.tight_layout()

plt.savefig(os.path.join(result_folder, 'mrcnn_numbers.pdf'))



#%%


#%%
folders = ['80_epochs_detect_0.7_nms_1000','80_epochs_detect_0.7_nms_2000','80_epochs_detect_0.7_nms_3000']
folders = [os.path.join(ROOT_DIR + f'/result_f1/CellFie/' + folder) for folder in folders]

F1_train = []
F1_val = []

for folder in folders:
    for file in os.listdir(folder):
        if 'train.npy' in file:
            p = np.load(os.path.join(folder, file), allow_pickle=True).item()
            F1_train.append(np.round(np.array([p[key]['f1_score'] for key in p.keys()]).mean(), 2))
        if 'val.npy' in file:
            p = np.load(os.path.join(folder, file), allow_pickle=True).item()
            F1_val.append(np.round(np.array([p[key]['f1_score'] for key in p.keys()]).mean(), 2))
    
plt.figure()
plt.title('Mask R-CNN performance nms')
ax = plt.gca()
width = 0.35
labels = ['1000', '2000', '3000']
x = np.arange(len(labels))
rects1 = ax.bar(x - width/2, F1_train, width, label=f'F1_train')
rects2 = ax.bar(x + width/2, F1_val, width, label=f'F1_val')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()

#%%
folders = ['80_epochs_detect_0.5_nms_1000','80_epochs_detect_0.6_nms_1000','80_epochs_detect_0.7_nms_1000']
folders = [os.path.join(ROOT_DIR + f'/result_f1/CellFie/' + folder) for folder in folders]

F1_train = []
F1_val = []

for folder in folders:
    for file in os.listdir(folder):
        if 'train.npy' in file:
            p = np.load(os.path.join(folder, file), allow_pickle=True).item()
            F1_train.append(np.round(np.array([p[key]['f1_score'] for key in p.keys()]).mean(), 2))
        if 'val.npy' in file:
            p = np.load(os.path.join(folder, file), allow_pickle=True).item()
            F1_val.append(np.round(np.array([p[key]['f1_score'] for key in p.keys()]).mean(), 2))
    
plt.figure()
plt.title('Mask R-CNN performance detect')
ax = plt.gca()
width = 0.35
labels = ['0.5', '0.6', '0.7']
x = np.arange(len(labels))
rects1 = ax.bar(x - width/2, F1_train, width, label=f'F1_train')
rects2 = ax.bar(x + width/2, F1_val, width, label=f'F1_val')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()

#%%
folders = ['80_epochs_detect_0.7_nms_1000_rpn_0.3', '80_epochs_detect_0.7_nms_1000_rpn_0.5', '80_epochs_detect_0.7_nms_1000_rpn_0.7']
folders = [os.path.join(ROOT_DIR + f'/result_f1/CellFie/' + folder) for folder in folders]

F1_train = []
F1_val = []

for folder in folders:
    for file in os.listdir(folder):
        if 'train.npy' in file:
            p = np.load(os.path.join(folder, file), allow_pickle=True).item()
            F1_train.append(np.round(np.array([p[key]['f1_score'] for key in p.keys()]).mean(), 2))
        if 'val.npy' in file:
            p = np.load(os.path.join(folder, file), allow_pickle=True).item()
            F1_val.append(np.round(np.array([p[key]['f1_score'] for key in p.keys()]).mean(), 2))
    
plt.figure()
plt.title('Mask R-CNN performance detect')
ax = plt.gca()
width = 0.35
labels = ['0.3', '0.5', '0.7']
x = np.arange(len(labels))
rects1 = ax.bar(x - width/2, F1_train, width, label=f'F1_train')
rects2 = ax.bar(x + width/2, F1_val, width, label=f'F1_val')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()

#%% Show ground truth and summary images
                for image_id in dataset.image_ids:
                    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
                    info = dataset.image_info[image_id]
               
                    fig, axs = plt.subplots(2, 2, figsize=(15,15))
                    #plt.suptitle(f'{info["id"]}')
                    axs[0, 0].imshow(image[:,:,0]); axs[0, 1].imshow(image[:,:,1]);
                    axs[1, 0].imshow(image[:,:,1]); axs[1,1].imshow(gt_mask.sum(2))
                    centers = []
                    for mask in gt_mask.transpose([2,0,1]):
                        a = np.where(mask>0)
                        center = [np.mean(a[0]), np.mean(a[1])] 
                        centers.append(center)
                    
                    centers = np.array(centers)
                    axs[0,0].scatter(centers[:,1], centers[:,0], color='red', s=3)
                    axs[0,1].scatter(centers[:,1], centers[:,0], color='red', s=3)
                    axs[1,0].scatter(centers[:,1], centers[:,0], color='red', s=3)
                    axs[1,1].scatter(centers[:,1], centers[:,0], color='red', s=3)

                    
                    axs[0,0].title.set_text('image[:,:,0]')
                    axs[0,1].title.set_text('image[:,:,1]')
                    axs[1,0].title.set_text('image[:,:,2]')
                    axs[1,1].title.set_text('gt_mask')
                    plt.tight_layout()
                    plt.savefig(os.path.join('/home/nel/Code/NEL_LAB/Mask_RCNN/datasets/Maddison/ground_truth', f'{info["id"].split("_")[0]}' + '.png'))

#%%
                    import random
                    from skimage.measure import find_contours
                    import colorsys
                    from matplotlib import patches,  lines
                    from matplotlib.patches import Polygon
                    def random_colors(N, bright=True):
                        """
                        Generate random colors.
                        To get visually distinct colors, generate them in HSV space then
                        convert to RGB.
                        """
                        brightness = 1.0 if bright else 0.7
                        hsv = [(i / N, 1, brightness) for i in range(N)]
                        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
                        random.shuffle(colors)
                        return colors
                    
                    def apply_mask(image, mask, color, alpha=0.5):
                        """Apply the given mask to the image.
                        """
                        for c in range(3):
                            image[:, :, c] = np.where(mask == 1,
                                                      image[:, :, c] *
                                                      (1 - alpha) + alpha * color[c] * 255,
                                                      image[:, :, c])
                        return image

                #%%
                rr = np.load('/home/nel/Code/NEL_LAB/Mask_RCNN/result_f1/CellFie/140_epochs_detect_0.7_nms_1000_rpn_0.5/maskrcnn_data_npz_train.npy', allow_pickle=True).item()

                for image_id in dataset.image_ids:
                    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
                    info = dataset.image_info[image_id]
                    
                    results = model.detect([image], verbose=1)
                    r = results[0]
                    mask_pr = r['masks'].copy().transpose([2,0,1])*1.
                    N = mask_pr.shape[0]
                    masked_image = np.zeros(image.copy().shape)#.astype(np.uint32).copy()
                    colors = random_colors(N)
                    
                    fig, axs = plt.subplots(1, 2, figsize=(15,15))
                    #plt.suptitle(f'{info["id"]}')
                    axs[0].imshow(image[:,:,0]); axs[1].imshow(image[:,:,1]);
                    centers = []
                    for mask in gt_mask.transpose([2,0,1]):
                        a = np.where(mask>0)
                        center = [np.mean(a[0]), np.mean(a[1])] 
                        centers.append(center)
                    
                    for i in range(N):
                        color = colors[i]
                        mask = mask_pr[i, :, :]
                        # Mask Polygon
                        # Pad to ensure proper polygons for masks that touch image edges.
                        padded_mask = np.zeros(
                            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                        padded_mask[1:-1, 1:-1] = mask
                        contours = find_contours(padded_mask, 0.5)
                        for verts in contours:
                            # Subtract the padding and flip (y, x) to (x, y)
                            verts = np.fliplr(verts) - 1
                            p = Polygon(verts, facecolor="none", edgecolor=color)
                            axs[0].add_patch(p)
                            
                    for i in range(N):
                        color = colors[i]
                        mask = mask_pr[i, :, :]
                        # Mask Polygon
                        # Pad to ensure proper polygons for masks that touch image edges.
                        padded_mask = np.zeros(
                            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                        padded_mask[1:-1, 1:-1] = mask
                        contours = find_contours(padded_mask, 0.5)
                        for verts in contours:
                            # Subtract the padding and flip (y, x) to (x, y)
                            verts = np.fliplr(verts) - 1
                            p = Polygon(verts, facecolor="none", edgecolor=color)
                            axs[1].add_patch(p)                    
                            
                    metric = rr[info['id'][:-4]]
                    for keys, values in metric.items():
                        metric[keys] = np.round(metric[keys], 2)                        
                    centers = np.array(centers)
                    axs[0].scatter(centers[:,1], centers[:,0], color='red', s=10)
                    axs[1].scatter(centers[:,1], centers[:,0], color='red', s=10)
                    axs[0].title.set_text(f'image[:,:,0], f1:{metric["f1_score"]}, precision:{metric["precision"]}, recall:{metric["recall"]}')
                    axs[1].title.set_text('image[:,:,1]')
                    plt.tight_layout()
                    plt.savefig(os.path.join('/home/nel/Code/NEL_LAB/Mask_RCNN/result_f1/CellFie/140_epochs_detect_0.7_nms_1000_rpn_0.5/images', f'{info["id"].split("_")[0]}' + '.png'))







