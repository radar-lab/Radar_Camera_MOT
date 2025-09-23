#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:54:45 2023

@author: lei
"""
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2
import io
import rosbag
from PIL import Image

import xml.etree.ElementTree as ET
from tqdm import tqdm

'''' parse xml and crop the RAD to npy for each obj '''
############### Lei ###################################
import os, glob, shutil, sys
sys.path.append("../../") # add search path: sys.path.append("../../")
#######################################################

######################################################################################################################
def resize_img(img, newsize=(64,64),filters=Image.Resampling.BICUBIC):
    new_img = img.resize(newsize, filters)#https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.LANCZOS
    return new_img


def filter_folders_with_keywords(folder_list, keywords):
    filtered_folders = []
    for folder_name in folder_list:
        if any(keyword in folder_name for keyword in keywords):
            filtered_folders.append(folder_name)
    return filtered_folders

def filter_folders_with_names(folder_list, names):
    filtered_folders = []
    for folder_path in folder_list:
        folder_name = os.path.basename(folder_path)
        if folder_name in names:
            filtered_folders.append(folder_path)
    return filtered_folders

def get_subdirectories(parent_folder):
    subdirectories = []
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    return subdirectories

def crop_img_data(data, xmin, xmax, ymin, ymax):
    cropped_data = data[ymin:ymax, xmin:xmax]# ymin:ymax for row; xmin:xmax for column
    return cropped_data

def xml2radimg(path,rad_ra_path,savepath):
    all_files = glob.glob('{}/*xml'.format(path))
    all_files.sort(key=lambda x: int(x.split('.')[0].split('/')[-1]))
    for xml_file in tqdm(all_files):
        tree    = ET.parse(xml_file)
        height  = int(tree.findtext('./size/height'))
        width   = int(tree.findtext('./size/width'))
        if height<=0 or width<=0:
            continue
        
        index = xml_file.split('.')[0].split('/')[-1]
        rad_path = rad_ra_path+ index + '.png'
        rad_data = np.array(Image.open(rad_path))
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin')))
            ymin = int(float(obj.findtext('bndbox/ymin')))
            xmax = int(float(obj.findtext('bndbox/xmax')))
            ymax = int(float(obj.findtext('bndbox/ymax')))
            
            bbox_img = crop_img_data(rad_data, xmin, xmax, ymin, ymax)
            ## save bbox img
            bbox_img = bbox_img.astype(np.uint8)
            img = Image.fromarray(bbox_img)
            img = resize_img(img, newsize=(64,64),filters=Image.Resampling.LANCZOS)
            # Display the image
            # plt.imshow(img)
            # plt.axis('off')  # Optional: Turn off axes
            # plt.show()
            # #img.show()                        
            img.save(savepath + index + '.png',quality=100, optimize=True)



##################################################################
''' ************************* Main ***********************************'''

if __name__ == '__main__':

    parentdir_name = "/xdisk/caos/leicheng/my_rawdata_0519/0519/"
    subdirectories = get_subdirectories(parentdir_name)
    subdirectories = sorted(subdirectories, reverse=False)
    
    #names = {'hao', 'jiahao', 'lei', 'shuting', 'sijie'}
    names = {'car-lei'} #{'car-shuting'}
    subdirectories = filter_folders_with_names(subdirectories, names)

    for dir_name in subdirectories:
        # RA_objnpy folder
        save_folder = '/RA_objimg/'
        if os.path.exists(dir_name+save_folder):
            shutil.rmtree(dir_name+save_folder)  #Delete the existing folder and its contents
        os.makedirs(dir_name+save_folder, exist_ok=True) 
        
        xmlpath     = dir_name+"/RA_label"
        rad_ra_path = dir_name+"/RA/"
        savepath    = dir_name+save_folder
        print('Load xmls.')
        data = xml2radimg(xmlpath,rad_ra_path,savepath)
        print('Load xmls done.')
        