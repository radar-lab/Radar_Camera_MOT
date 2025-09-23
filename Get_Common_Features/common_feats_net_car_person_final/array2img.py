#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:09:00 2022

@author: lei
"""
import numpy as np
import glob
import os
import re
import datetime
import shutil


from PIL import Image
import matplotlib.pyplot as plt



########################## create 100 folders, each folder for each obj ##########################
create=1
if create:
    for i in range(1000): # create num=100 folders
        #root_dir= './tracking_datasets/person/'
        root_dir= './tracking_datasets/car_person/'
        os.makedirs(root_dir+str(i), exist_ok=True)


def rename_file(path):  #also for folders
    file_list=os.listdir(path)
    file_list = sorted(file_list,key=int)
    for i,fi_name in enumerate(file_list):
        old_name=os.path.join(path,fi_name)
        new_name=os.path.join(path,str(i))
        os.rename(old_name,new_name)
        #shutil.rmtree(old_name)  #delete non empty folder      
########################################################################

########################### convert npy to imgs for only one npy , use npyToimg for all npys not this####################################
def resize_img(img, newsize=(64,64),filters=Image.BICUBIC):
    new_img = img.resize(newsize, filters)#https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.LANCZOS
    return new_img



tra_car_folder = './tracking_data/tracking/car_day/'
tra_person_folder = './tracking_data/tracking/person/'
tra_folder = tra_person_folder

dir_save_root='./imgs_for_annotation/'
now_time=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')#%Y-%m-%d %H:%M:%S
dir_save=dir_save_root + now_time + '/'
#os.makedirs(dir_save, exist_ok=True)


all_files_tra=glob.glob(tra_folder+'*.npy')
#all_files_tra.sort(reverse=True)
all_files_tra.sort()
len4one=int(len(all_files_tra)/3)
for k in range(len4one):#k< len(all_files_tra)/3
    #radar_data  = np.load(all_files_tra[k])
    #img_label  = np.load(all_files_tra[k+len4one])
    images = np.load(all_files_tra[k+2*len4one])
    #bag_dir = re.findall('car.*?(?=.npy)',all_files_tra[0])# (?=exp) 匹配exp前面的位置;(?<=exp) 匹配exp后面的位置:https://blog.csdn.net/weixin_43890704/article/details/126100731
    bag_dir = re.findall('(?<=tracking/).*?(?=.npy)',all_files_tra[k+2*len4one])
    dir_save_final = dir_save + bag_dir[0] + '/'
    os.makedirs(dir_save_final, exist_ok=True)
    for i in range(len(images)):
        images=images.astype(np.uint8)
        img = Image.fromarray(images[i], 'RGB')
        img = resize_img(img, newsize=(64*5,64*5),filters=Image.LANCZOS)
        #img.show()
        img.save(dir_save_final + str(i) + '.png',quality=100, optimize=True)
######################################################################################################

####################### get index based on img folders: index[person][img_idx]: All imgs one row #######################
'''
datasets_path   = "./tracking_datasets/person/"
#folders_name    = os.listdir(datasets_path)# list all subfolders include files
folders_name    = next(os.walk(datasets_path))[1] # list all subfolders exclude files
folders_name    = sorted(folders_name,key=int)# folders_name is subfolders_name

list_file = open(datasets_path+'person_imgidx_array.txt', 'w')#https://blog.csdn.net/weixin_32093519/article/details/112733922
for obj_id, folder_name in enumerate(folders_name):
    imgs_path = os.path.join(datasets_path, folder_name)
    if not os.path.isdir(imgs_path):# eliminate README.md in the folder
        continue
    imgs_name = os.listdir(imgs_path)
    if len(imgs_name):# exclude empty folder
        #nums = [int(n) for strings in imgs_name for n in strings.split('.') if n.isdigit()]# extract numbers from fname
        nums = [n for strings in imgs_name for n in strings.split('.') if n.isdigit()]# extract numbers from fname
        nums = sorted(nums,key=int)
        #list_file.write(str(obj_id) + ":" + '%s'%(nums))
        list_file.write(str(obj_id) + ":" + ','.join(nums))
        list_file.write('\n')
        ### parse
        # test_list = list(map(int, (string.split(':')[1]).split(',')))
list_file.close()
'''
########################### get index based on img folders: index[person][img_idx] *Adopted* second implementation: one img one row
def idxFromimg(idx_file_name='person_imgidx.txt', datasets_path="./tracking_datasets/car_person/"):
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
            for num in nums:
                idx_file.write(str(obj_id) + ";" + num)
                idx_file.write('\n')
    idx_file.close()


########################### convert imgs_folders to index ####################################
'''
1) Manually move the pictures that have been generated above into folders (use code to generate these folders).
2) Then generate these indexes based on these folders.
3) after complete running the above codes, uncomment below code to generate index
'''
datasets_path   = "./tracking_datasets/car_person/" 
idxFromimg(idx_file_name='person_imgidx.txt', datasets_path=datasets_path)       