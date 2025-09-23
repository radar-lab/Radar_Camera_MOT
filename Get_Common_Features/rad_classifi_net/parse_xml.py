## parse xml and crop the RAD to npy for each obj
############### Lei ###################################
import os, glob, shutil, sys
sys.path.append("../../") # add search path: sys.path.append("../../")
#######################################################
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

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

def crop_3d_data(data, xmin, xmax, ymin, ymax):
    cropped_data = data[xmin:xmax, ymin:ymax, :]
    return cropped_data

def xml2rad(path,rad_fr_path,savepath):
    all_files = glob.glob('{}/*xml'.format(path))
    all_files.sort(key=lambda x: int(x.split('.')[0].split('/')[-1]))
    for xml_file in tqdm(all_files):
        tree    = ET.parse(xml_file)
        height  = int(tree.findtext('./size/height'))
        width   = int(tree.findtext('./size/width'))
        if height<=0 or width<=0:
            continue
        
        index = xml_file.split('.')[0].split('/')[-1]
        rad_path = rad_fr_path+ index + '.npy'
        rad_data = np.load(rad_path,allow_pickle=True)
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin')))
            ymin = int(float(obj.findtext('bndbox/ymin')))
            xmax = int(float(obj.findtext('bndbox/xmax')))
            ymax = int(float(obj.findtext('bndbox/ymax')))
            
            cropped_data = crop_3d_data(rad_data, xmin, xmax, ymin, ymax)
            
        ### save x_cent,y_cent,w,h,class
        with open(savepath+ index +'.npy' , 'wb') as f:
            np.save(f, cropped_data) 


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def load_data(path):
    gt_label=[]
    
    data = []
    #-------------------------------------------------------------#
    #   对于每一个xml都寻找box
    #-------------------------------------------------------------#
    all_files = glob.glob('{}/*xml'.format(path))
    all_files.sort(key=lambda x: int(x.split('.')[0].split('/')[-1]))
    for xml_file in tqdm(all_files):
        gt_label_obj=[]
        tree    = ET.parse(xml_file)
        height  = int(tree.findtext('./size/height'))
        width   = int(tree.findtext('./size/width'))
        if height<=0 or width<=0:
            continue
        
        #-------------------------------------------------------------#
        #   对于每一个目标都获得它的宽高
        #-------------------------------------------------------------#
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin')))
            ymin = int(float(obj.findtext('bndbox/ymin')))
            xmax = int(float(obj.findtext('bndbox/xmax')))
            ymax = int(float(obj.findtext('bndbox/ymax')))
            
            #cropped_data = crop_3d_data(data, xmin, ymin, xmax, ymax)

            # 得到宽高
            data.append([xmax - xmin, ymax - ymin]) 
            #class
            classes_path    = 'model_data/radar_classes.txt'
            classes, num_classes = get_classes(classes_path)
            cls = obj.find('name').text
            if cls not in classes:
                print('wrong class',xml_file)
                continue
            cls_id = classes.index(cls)
            gt_label_obj.append([(xmax + xmin)//2, (ymax + ymin)//2, xmax - xmin, ymax - ymin,cls_id])
        gt_label.append(gt_label_obj)
    ### save x_cent,y_cent,w,h,class
    with open(path +'/RA_label.npy' , 'wb') as f:
        np.save(f, gt_label) 
    #a=np.load("/xdisk/caos/leicheng/my_rawdata/RA_label/RA_label.npy",allow_pickle=True)           
    return np.array(data)


if __name__ == '__main__':
    xml_label = True
    
    # xmlpath        = "/xdisk/caos/leicheng/my_rawdata/RA_label"
    # print('Load xmls.')
    # data = load_data(xmlpath)
    # print('Load xmls done.')

    parentdir_name = "/xdisk/caos/leicheng/my_rawdata_0519/0519/"
    subdirectories = get_subdirectories(parentdir_name)
    subdirectories = sorted(subdirectories, reverse=False)
    
    names = {'hao', 'jiahao', 'lei', 'shuting', 'sijie'}
    subdirectories = filter_folders_with_names(subdirectories, names)

    for dir_name in subdirectories:
        # RA_objnpy folder
        if os.path.exists(dir_name+'/RA_objnpy'):
            shutil.rmtree(dir_name+'/RA_objnpy')  #Delete the existing folder and its contents
        os.makedirs(dir_name+'/RA_objnpy', exist_ok=True) 
        
        xmlpath     = dir_name+"/RA_label"
        rad_fr_path = dir_name+"/frames/"
        savepath    = dir_name+'/RA_objnpy/'
        print('Load xmls.')
        data = xml2rad(xmlpath,rad_fr_path,savepath)
        print('Load xmls done.')

        

