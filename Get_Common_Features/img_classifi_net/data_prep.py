#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lei
"""
import tensorflow as tf
import numpy as np
import glob
import re
import os
import datetime
from PIL import Image
from scipy.signal import find_peaks



def resize_img(img, newsize=(64,64),filters=Image.Resampling.BICUBIC):
    new_img = img.resize(newsize, filters)
    return new_img

#imgs[int(selected_path[image_indexes[0]])]
def show_img(img_arr):
    img_arr = img_arr.astype(np.uint8)
    img     = Image.fromarray(img_arr, 'RGB')
    img     = resize_img(img, newsize=(64*5,64*5),filters=Image.Resampling.LANCZOS)
    img.show()

def npyToimg(imgs, dir_save_root = './imgs_for_annotation/',filters=Image.Resampling.LANCZOS):
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')#%Y-%m-%d %H:%M:%S
    dir_save_final = dir_save_root + now_time + '/'    
    os.makedirs(dir_save_final, exist_ok=True)
    for i in range(len(imgs)):
        imgs=imgs.astype(np.uint8)
        img = Image.fromarray(imgs[i], 'RGB')
        #img = resize_img(img, newsize=(64*5,64*5),filters=filters)
        #img.show()
        img.save(dir_save_final + str(i) + '.png',quality=100, optimize=True)
        
def concat_npy(tra_folder):       
    ### concatenate all npy files
    all_files_tra=glob.glob(tra_folder+'*.npy')
    #all_files_tra.sort(reverse=True)
    all_files_tra.sort()
    len4one=int(len(all_files_tra)/3)# numbers of files for one class of npy
    radar_data = []
    img_label  = []
    images     = []
    for k in range(len4one):#k< len(all_files_tra)/3
        images.append(np.load(all_files_tra[k])) 
        img_label.append(np.load(all_files_tra[k+len4one]))
        radar_data.append(np.load(all_files_tra[k+2*len4one]))        
    rad_data    = np.concatenate(radar_data)
    imgs_label  = np.concatenate(img_label)
    imgs        = np.concatenate(images)
    return rad_data, imgs_label, imgs          

########################### get index based on img folders: index[person][img_idx] *Adopted* second implementation
def idxFormimg(idx_file_name='person_imgidx.txt', datasets_path="./tracking_datasets/person/",out_type=1):
    folders_name      = next(os.walk(datasets_path))[1] # list all subfolders exclude files
    folders_name      = sorted(folders_name,key=int)# folders_name is subfolder_name
    idx_file_name     = datasets_path+idx_file_name
    idx_file = open(idx_file_name, 'w')
    for obj_id, folder_name in enumerate(folders_name):
        imgs_path = os.path.join(datasets_path, folder_name)
        imgs_name = os.listdir(imgs_path)
        if len(imgs_name):# exclude empty folder
            #nums = [int(n) for strings in imgs_name for n in strings.split('.') if n.isdigit()]# extract numbers from fname
            nums = [n for img_name in imgs_name for n in img_name.split('.') if n.isdigit()]# extract numbers from fname
            nums = sorted(nums,key=int)
            if out_type==1: #one object one row
                for num in nums:
                    idx_file.write(str(obj_id) + ";" + num)
                    idx_file.write('\n')
            elif out_type==2: #one img one row
                idx_file.write(str(obj_id) + ":" + ','.join(nums))
                idx_file.write('\n')
    idx_file.close()


# Find the peak element in the array:https://www.geeksforgeeks.org/find-a-peak-in-a-given-array/
def findPeak(arr,left_thr=0,right_thr=1) :
    peaks=[]
    n=len(arr)
    # first element is peak element
    if (n == 1) : #ONLY ONE element
        peaks.append(0)
        return
    if (arr[0] > arr[1]) :# first
        peaks.append(0)
    # check for every element except first and last one
    for i in range(1, n - 1) :
        # check if the neighbors are smaller
        if (arr[i]-arr[i - 1] >= left_thr and arr[i]-arr[i + 1] >= right_thr) :
            peaks.append(i)
    # #   last element is peak element  
    # if (arr[n - 1] >= arr[n - 2]) : #last
    #     peaks.append(n - 1)  
    return peaks     

# arr = [ 1, 0, 3, 20, 4, 10, 0 ]
# print("Index of a peak point is", findPeak(arr,right_thr=10))

################### extract radar img data from bags ######################
def data_prep_clf(radar_data,img_label):
    ### classification
    ## radar
    fr_idx_rad=radar_data[:,0]
    class_rad=radar_data[:,1]
    x_rad=radar_data[:,2:2+256]
    y_rad=radar_data[:,2+256:2+256*2]
    z_rad=radar_data[:,2+256*2:2+256*3]
    dop_rad=radar_data[:,2+256*3:2+256*4]
    snr_rad=radar_data[:,2+256*4:2+256*5]
    az_rad=radar_data[:,2+256*5:2+256*6]
    el_rad=radar_data[:,2+256*6:2+256*7]
    ts_rad=radar_data[:,2+256*7:2+256*8]
    r_rad=radar_data[:,2+256*8:2+256*9]
    vx_rad=radar_data[:,2+256*9:2+256*10]
    vy_rad=radar_data[:,2+256*10:2+256*11]
    vz_rad=radar_data[:,2+256*11:2+256*12]
    x_cent=radar_data[:,2+256*12]
    y_cent=radar_data[:,2+256*12+1]
    z_cent=radar_data[:,2+256*12+2]
    ## img
    fr_idx_img=img_label[:,0]
    class_img=img_label[:,1]
    x_min=img_label[:,2]
    y_min=img_label[:,3]
    x_max=img_label[:,4]
    y_max=img_label[:,5]
    ## return
    ctr=[fr_idx_rad,class_rad,x_cent,y_cent,z_cent]
    xyzv=[x_rad,y_rad,z_rad,dop_rad]
    saervc=[snr_rad,az_rad,el_rad,r_rad,vx_rad,vy_rad,vz_rad]
    bbox=[x_min,y_min,x_max,y_max]
    return np.array(ctr),np.array(xyzv),np.array(saervc),np.array(bbox)

def data_prep_tra(radar_data,img_label):
    ### tracking 
    # radar_data=radar_data.tolist()
    # img_label=img_label.tolist()    
    ## radar
    fr_idx_rad=radar_data[:,0]
    tid_rad=radar_data[:,1]
    class_rad=radar_data[:,2]
    x_rad=radar_data[:,3:3+256]
    y_rad=radar_data[:,3+256:3+256*2]
    z_rad=radar_data[:,3+256*2:3+256*3]
    dop_rad=radar_data[:,3+256*3:3+256*4]
    snr_rad=radar_data[:,3+256*4:3+256*5]
    az_rad=radar_data[:,3+256*5:3+256*6]
    el_rad=radar_data[:,3+256*6:3+256*7]
    ts_rad=radar_data[:,3+256*7:3+256*8]
    r_rad=radar_data[:,3+256*8:3+256*9]
    vx_rad=radar_data[:,3+256*9:3+256*10]
    vy_rad=radar_data[:,3+256*10:3+256*11]
    vz_rad=radar_data[:,3+256*11:3+256*12]
    x_cent=radar_data[:,3+256*12]
    y_cent=radar_data[:,3+256*12+1]
    z_cent=radar_data[:,3+256*12+2]
    ## img
    fr_idx_img=img_label[:,0]
    tid_img=img_label[:,1]#it's same with tid_rad
    class_img=img_label[:,2]
    x_min=img_label[:,3]
    y_min=img_label[:,4]
    x_max=img_label[:,5]
    y_max=img_label[:,6]
    ## return
    ctr=[fr_idx_rad,tid_rad,class_rad,x_cent,y_cent,z_cent]
    xyzv=[x_rad,y_rad,z_rad,dop_rad]
    saervc=[snr_rad,az_rad,el_rad,r_rad,vx_rad,vy_rad,vz_rad]
    bbox=[x_min,y_min,x_max,y_max]
    return np.array(ctr),np.array(xyzv),np.array(saervc),np.array(bbox)


def gen_txt_idx_rad(folder_path = "/xdisk/caos/leicheng/my_rawdata_0519/0519/",txt_filename = 'radar_npy_idx.txt', filter_name='NPY'):
    # TXT filename to save the paths   
    # Open the TXT file for writing
    with open(folder_path+txt_filename, 'w') as f:
        # Traverse the folder and its subfolders
        for root, dirs, files in os.walk(folder_path):
            dirs  = sorted(dirs)
            for folder in dirs:
                if len(folder):# exclude empty folder
                    folder_path = os.path.join(root, folder)
                    npy_folder_path = os.path.join(folder_path, filter_name)
                    # Check if the grandchild folder 'NPY' exists
                    if os.path.exists(npy_folder_path):
                        # Traverse the files in the 'NPY' grandchild folder
                        all_files = os.listdir(npy_folder_path)
                        all_files = sorted(all_files, key=lambda x: int(x.split('.')[0]))
                        for file in all_files:
                            file_path = os.path.join(npy_folder_path, file)
                            # Write the folder and file path to the TXT file
                            f.write(f'{folder};{file_path}\n')

def gen_txt_idx_img(folder_path = "/xdisk/caos/leicheng/my_rawdata_0519/0519_camera/",txt_filename = 'img_idx.txt'):
    # TXT filename to save the paths   
    # Open the TXT file for writing
    with open(folder_path+txt_filename, 'w') as f:
        # Traverse the folder and its subfolders
        for root, dirs, files in os.walk(folder_path):
            dirs  = sorted(dirs)
            for folder in dirs:
                if len(folder):# exclude empty folder
                    folder_path = os.path.join(root, folder)
                    # Traverse the files in the 'NPY' grandchild folder
                    all_files = os.listdir(folder_path)
                    all_files = sorted(all_files, key=lambda x: int(x.split('.')[0]))
                    for file in all_files:
                        file_path = os.path.join(folder_path, file)
                        # Write the folder and file path to the TXT file
                        f.write(f'{folder};{file_path}\n')
                            
if __name__ == "__main__":
    
    gen_txt_idx_rad(folder_path = "/xdisk/caos/leicheng/my_rawdata_0519/0519/",txt_filename = 'radar_npy_idx.txt', filter_name='RA_objnpy')
    print('Radar Done!')
    
    gen_txt_idx_img(folder_path = "/xdisk/caos/leicheng/my_rawdata_0519/0519_camera/",txt_filename = 'img_idx.txt')
    print('Image Done!')
