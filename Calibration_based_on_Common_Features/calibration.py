#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""

############### Lei ###################################
import os, glob, shutil, sys
sys.path.append("./") # add search path: sys.path.append("../../")
sys.path.append("./ImageYolo/")
######################################################
from tqdm import tqdm
import cv2
import io
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import gc

from get_img_ctd import parse_rosbag_to_img
#from get_radar_ctd import xml2rad
from csv_UNIX_time import extract_start_time
from ImageYolo.predict import process_imgs_to_get_bboxes, load_radimg_yolo_model
from common_feats_net_car_person_final.common_feats_compare import common_feats_find,load_common_feats_model

# Set the necessary environment variables before importing GPU-related libraries: solve RuntimeError: Physical devices cannot be modified after being initialized
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace "0" with the index of the GPU device you want to use
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_sorted_paths(directory, extension):
    files = glob.glob(os.path.join(directory, f"*.{extension}"))
    #sorted_paths = sorted([os.path.abspath(file) for file in files])
    sorted_paths = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return sorted_paths

def read_sorted_npy_files(sorted_paths, num=float('inf')):
    data = []
    for i, file_path in enumerate(sorted_paths):
        if i >= num:  # Check if the number of files read reaches the specified limit
            break
        # Load .npy file data using NumPy
        npy_data = np.load(file_path, allow_pickle=True)
        data.append(npy_data)
    return data

def read_sorted_png_files(sorted_paths, num=float('inf')):
    data = []
    for i, file_path in enumerate(sorted_paths):
        if i >= num:  # Check if the number of files read reaches the specified limit
            break
        # Read .png file using PIL (Pillow) library
        png_image = Image.open(file_path)
        # Convert image data to NumPy array
        png_data = np.array(png_image)
        data.append(png_data)
    return data

def get_sorted_files(directory, extension):
    files = glob.glob(os.path.join(directory, f"*.{extension}"))
    sorted_files = sorted(files)
    return sorted_files

def read_npy_files(directory,num):
    # Get sorted list of .npy files in the directory
    npy_files = get_sorted_files(directory, 'npy')
    data = []
    for i, file in enumerate(npy_files):
        if i >= num:  # Check if the number of files read reaches the specified limit
            break
        file_path = os.path.join(directory, file)
        # Load .npy file data using NumPy
        npy_data = np.load(file_path,allow_pickle=True)
        data.append(npy_data)
    return data

def read_png_files(directory,num):
    # Get sorted list of .png files in the directory
    png_files = get_sorted_files(directory, 'png')
    data = []
    for i, file in enumerate(png_files):
        if i >= num:  # Check if the number of files read reaches the specified limit
            break
        file_path = os.path.join(directory, file)
        # Read .png file using PIL (Pillow) library
        png_image = Image.open(file_path)
        # Convert image data to NumPy array
        png_data = np.array(png_image)
        data.append(png_data)
    return data

def get_subdirectories(parent_folder):
    subdirectories = []
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    return subdirectories

def filter_folders_with_names(folder_list, names):
    filtered_folders = []
    for folder_path in folder_list:
        folder_name = os.path.basename(folder_path)
        if folder_name in names:
            filtered_folders.append(folder_path)
    return filtered_folders

def filter_files_with_names(file_paths, names, mode='filter_file_name'):
    filtered_files = []
    for file_path in file_paths:
        if mode=='filter_file_name':
            # Check if the file_name contains the specified string
            file_name = os.path.basename(file_path)
            if any(filter_str in file_name for filter_str in names):
                filtered_files.append(file_path)
        elif mode=='filter_file_path':
            # Check if any of the filter strings exist in the complete file path
            if any(filter_str in file_path for filter_str in names):
                filtered_files.append(file_path)
    return filtered_files

# def calculate_bbox_centers(bbox_array):
#     centers = np.zeros((bbox_array.shape[0], 2))  # Initialize array to store centers
#     centers[:, 0] = (bbox_array[:, 0] + bbox_array[:, 2]) / 2  # Calculate x-coordinate of centers
#     centers[:, 1] = (bbox_array[:, 1] + bbox_array[:, 3]) / 2  # Calculate y-coordinate of centers
#     return centers

def calculate_bbox_centers(bbox_list):
    centers = []
    for bbox in bbox_list:
        xmin, ymin, xmax, ymax = np.transpose(bbox)
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center = np.column_stack((center_x, center_y))
        centers.append(center)
    return centers

def crop_3d_data(data, xmin, xmax, ymin, ymax):
    cropped_data = data[ymin:ymax, xmin:xmax, :]
    return cropped_data

def crop_data_for_bboxes(data_list, bbox_list):
    cropped_data_list = []
    for i in range(len(data_list)):
        data = data_list[i]
        bbox = bbox_list[i]
        
        # Check if bbox contains multiple bounding boxes
        if bbox.ndim == 1:
            bbox = bbox.reshape(1, -1)
        
        # Initialize list to store cropped data for each bounding box
        cropped_data_bbox = []
        
        # Crop data for each bounding box
        for single_bbox in bbox:
            xmin, ymin, xmax, ymax = np.transpose(single_bbox)
            cropped_data = crop_3d_data(data, int(xmin), int(xmax), data.shape[0] - int(ymax), data.shape[0] - int(ymin))
            cropped_data_bbox.append(cropped_data)
        
        # Append cropped data for current data and bbox to the main list
        cropped_data_list.append(cropped_data_bbox)
    
    return cropped_data_list

def find_nearest_betw_arr(known_array, match_array):
    '''
    Based on match_array, to find the value in an known_array which is closest to an element in match_array
    return match_value and inx in known_array, and indices size is len(match_array)
    Returns:
        indices (array): Indices of the nearest values in known_values.
        nearest_values (array): Nearest values in known_values corresponding to the match_array.
    '''
    # known_array=np.array([1, 9, 33, 26,  5 , 0, 18, 11])
    # match_array=np.array([-1, 0, 11, 15, 33, 35,10,31])
    # i.e.:For -1 in match_array, the nearest value in known_values is 0 at index 5. For 33, the nearest value in known_values is 33 at index 2.
    # indices = array([5, 5, 7, 5, 2, 2, 7, 4])
    # known_array[indices] = array([0, 0, 11, 5, 33, 33, 11, 5])

    # Sort the known_values array and get the sorted indices
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted] 
    # Find the insertion points in the sorted_known_array for each match_array value
    idx = np.searchsorted(known_array_sorted, match_array)
    idx1 = np.clip(idx, 0, len(known_array_sorted)-1)
    idx2 = np.clip(idx - 1, 0, len(known_array_sorted)-1)
    # Calculate the differences between the nearest values and the match_array values
    diff1 = known_array_sorted[idx1] - match_array
    diff2 = match_array - known_array_sorted[idx2]
    # Determine the indices of the nearest values based on the differences
    indices = index_sorted[np.where(diff1 <= diff2, idx1, idx2)]
    return indices,known_array[indices]

def process_radar_data(npy_path, png_path, confidence=0.5, nms_iou=0.3,yolo=None):
    # Read .npy files and store data in an array
    rad_fr_data = read_sorted_npy_files([npy_path])
    
    # Read .png files and store data in an array
    RA_img_data = read_sorted_png_files([png_path])
    
    # Get radar bboxes for num frames and their centers
    rad_bboxes = process_imgs_to_get_bboxes(RA_img_data, confidence=confidence, nms_iou=nms_iou,yolo=yolo)
    rad_centers = calculate_bbox_centers(rad_bboxes)
    
    # Get radar cropped bbox radar data based on bboxes for num frames
    bbox_rads_all = crop_data_for_bboxes(rad_fr_data, rad_bboxes)
    
    return rad_bboxes, rad_centers, bbox_rads_all

###########################################################################
if __name__ == "__main__":
    path = "/xdisk/caos/leicheng/my_rawdata_0624/0624/"
    names = ['jiahao-hao'] #['hao-jiahao'] #['lei-leicar'] #['jiahao-hao'] #['lei_alone']
    
    output_path = '/xdisk/caos/leicheng/calibration_based_on_common_features_07272024/' + names[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)       
    #############################  Load deep-learning model   #############################
    ### Try to load model first.
    radimg_yolo_model  = load_radimg_yolo_model(confidence = 0.5,nms_iou = 0.3)
    # Load the common_feats_model
    common_feats_model = load_common_feats_model()
    
    #############################  Load images   #############################
    conf_thr=0.8
    all_files_bag = glob.glob(path +'*.bag')
    file_names_bag = [os.path.basename(file) for file in all_files_bag]
    file_names_bag = filter_files_with_names(file_names_bag, names, mode='filter_file_name')
    sorted_files_bag = sorted(file_names_bag, reverse=False)
    for bag in sorted_files_bag:
        all_imgs,img_bbs_ctd,ts_img,bbox_imgs_all,img_bbs_all,img_classes_all = parse_rosbag_to_img(bag,path,conf_thr
                                                                                    ,filter_class = ['person','car']
                                                                                    ,filter_cls_flg = 1
                                                                                    ,mode='exclude_static_cars'
                                                                                    ,xmin_thr=390
                                                                                    ,ymin_thr=150
                                                                                    ,return_mode='centroid')    
    
    
    
    ###########################  Load radar  #############################
    num = 8 #specify the desired number of frames/files to be read
    frame_rate = 30  # Frame rate, 30 frames per second
    parentdir_name = path
    subdirectories = get_subdirectories(parentdir_name)
    subdirectories = sorted(subdirectories, reverse=False)
    
    
    subdirectories = filter_folders_with_names(subdirectories, names)
    
    for dir_name in subdirectories:
        # read frames and RA images     
        rad_fr_path = dir_name+"/frames/"
        RA_img_path = dir_name+"/RA/"
        
        # Get sorted list of radar .npy files in the directory
        npy_paths = get_sorted_paths(rad_fr_path, 'npy')
        # Get sorted list of radar .png files in the directory
        png_paths = get_sorted_paths(RA_img_path, 'png')
        
        # Read start time and get all radar frame timestamps
        start_time = extract_start_time(dir_name)
        total_frames = len(npy_paths) #num  # Total number of frames
        rad_fr_timestamps = np.linspace(start_time, start_time + total_frames / frame_rate, total_frames)
        
        # for i in range(len(npy_paths)):
        #     rad_bboxes, rad_centers, bbox_rads_all = process_radar_data(npy_paths[i], png_paths[i], confidence=0.5, nms_iou=0.3)
        
        
        # # Read .npy files and store data in an array
        # rad_fr_data = read_npy_files(rad_fr_path, num)
        # # Read .png files and store data in an array
        # RA_img_data = read_png_files(RA_img_path, num)
        # # Get radar bboxes for num frames and their centers
        # rad_bboxes = process_imgs_to_get_bboxes(RA_img_data,confidence = 0.5,nms_iou = 0.3)
        # rad_centers = calculate_bbox_centers(rad_bboxes)
        # # Get radar cropped bbox radar data based on bboxes for num frames
        # bbox_rads_all = crop_data_for_bboxes(rad_fr_data, rad_bboxes)
    ###########################  Paired radar and image  #############################  

    
    
    ## Associating the radar and image by timestamps
    idx_rad,nearest_rad_time = find_nearest_betw_arr(rad_fr_timestamps, ts_img)
    
    #filtered_bbox_imgs_all will contain tuples with the index and corresponding non-empty data, preserving the original indices from bbox_imgs_all
    filtered_bbox_imgs_all = [(i, bbox) for i, bbox in enumerate(bbox_imgs_all) if bbox != [[]]]
    filtered_bbox_list = [bbox for _, bbox in filtered_bbox_imgs_all]
    filtered_idx_list  = [idx for idx,_ in filtered_bbox_imgs_all]
    
    # # Get the corresponding elements based on the filtered indices
    # filtered_all_imgs    = [all_imgs[i] for i in filtered_idx_list]
    # filtered_img_bbs_ctd = [img_bbs_ctd[i] for i in filtered_idx_list]
    # filtered_ts_img      = [ts_img[i] for i in filtered_idx_list]
    
    # ********************Delete unused variables to free up memory********************##############
    del bbox_imgs_all, filtered_bbox_imgs_all, all_imgs
    
    
    
    
    # Get paired radar and image based on common feats
    paired_ctds = [] # List to store images and radar 's centers
    #for i in tqdm(range(len(filtered_bbox_list))):
    for i in tqdm(range(800,len(filtered_bbox_list))):
        print('img idx : ', i)
        # for each index i, obtain bbox_img_list and the corresponding bbox_rad_list
        bbox_img_list = filtered_bbox_list[i]

        
        # Get original index of the image
        img_idx = filtered_idx_list[i]
        # Get radar index based on the img_idx
        rad_idx = idx_rad[img_idx]
        # Get current radar path
        npy_path = npy_paths[rad_idx]
        png_path = png_paths[rad_idx]
        # Get current radar related data
        rad_bboxes, rad_centers, bbox_rads_all = process_radar_data(npy_path, png_path
                                                                    ,confidence=0.5, nms_iou=0.3
                                                                    ,yolo=radimg_yolo_model)
        
        bbox_rad_list   = bbox_rads_all[0]
        # Get rad_center_list
        rad_center_list = rad_centers[0]
        
        # Get corresponding data by index
        #img_fr     = all_imgs[img_idx]
        bbs_ctd_fr = img_bbs_ctd[img_idx]
        #ts_img_fr  = ts_img[img_idx]   #rad_fr_timestamps[rad_idx]
        
        # to avoid repeat compare
        #uses the already_compared_same_j and already_compared_same_k variables to keep track 
        #of the indices that have already been compared and found to be the same. 
        #If either of these variables matches the current indices j and k, 
        #the loop continues to the next iteration without performing the comparison again.
        already_compared_same_j = None
        already_compared_same_k = None
    
        for j in range(len(bbox_img_list)):  # iterate over each img bbox in the bbox_img_list
            bbox_img = bbox_img_list[j]
            images = [bbox_img]
            
            # Check if already compared and found to be the same, then skip this iteration
            if already_compared_same_j == j:
                continue
            
            for k in range(len(bbox_rad_list)):  # iterate over each radar bbox in the bbox_rad_list
                # Check if already compared and found to be the same, then skip this iteration
                if already_compared_same_k == k:
                    continue
                
                bbox_rad = bbox_rad_list[k]
                rad_complexs = [bbox_rad]
    
                pred_labels = common_feats_find(images, rad_complexs,model=common_feats_model)
                print('same' if pred_labels[0] == 1 else 'diff')
                # plt.imshow(images[0])
                # plt.show()

                
                if pred_labels == 1:
                    bbs_ctd_img = bbs_ctd_fr[j]   # img bbox center
                    bbs_ctd_rad = list(rad_center_list[k])  # rad bbox center
                    paired_ctds.append([bbs_ctd_img, bbs_ctd_rad])
                    # Mark the current indices as already compared and found to be the same
                    already_compared_same_j = j
                    already_compared_same_k = k
        ### loop end            
        gc.collect()
                    
                    
    ####   END   #############################################################            
    # Save paired_ctds as an .npy file
    file_name = 'paired_ctds.npy'
    np.save(os.path.join(output_path, file_name), np.array(paired_ctds))                   
        
        

        

