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
import os
import glob
import shutil
import rosbag
from PIL import Image

######################################################################################################################
def resize_img(img, newsize=(64,64),filters=Image.Resampling.BICUBIC):
    new_img = img.resize(newsize, filters)#https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.LANCZOS
    return new_img

def parse_rosbag_to_img(bag='track_1002_02.bag'):
    """This function is used for parse_rosbag_to_img. input: image and bbox"""
    """
    ### Read ros_bag
    """    
    #################### load bounding_box_bag use rosbag  ,for getting xmin etc. ##################################################
    bbs = []
    ts_bbs = []
    bbs_img_bag = rosbag.Bag('./0519/'+bag)
    for topic, msg, t in bbs_img_bag.read_messages(topics=['/darknet_ros/bounding_boxes']):
        #print(msg.bounding_boxes[0])
        bbox = []
        #breakpoint()    
        #ts_bbs.append(msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9)
        ts_bbs.append(msg.image_header.stamp.secs+msg.image_header.stamp.nsecs*1e-9) #use imageheader.timestamps
        for i in range(len(msg.bounding_boxes)):
            bbox.append(msg.bounding_boxes[i])
        bbs.append(bbox) #usage: bbs[0][0].xmin; bbs[img_idx][bbx_idx].xmin
    
    ####################      load img_bag use rosbag  ,it is more faster ##################################################
    all_imgs = []
    ts_img = []
    for topic, msg, t in bbs_img_bag.read_messages(topics=['/usb_webcam/image_rect_color/compressed']):    
        ts_img.append(msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9)
        #decoded_im = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), -1) # for compressed image, we need decompress(decode) image(bytes) first, then we can get image even though there isn't width and height of image.
        #decoded_im = cv2.cvtColor(decoded_im, cv2.COLOR_BGR2RGB) 
        # The following is an alternative to the above decoding process
        decoded_im = np.array(PIL.Image.open(io.BytesIO(msg.data))) #io.BytesIO to Convert bytes to binary stream ; Image.open to Create Image object and input should be binary data        
        #bi=io.BytesIO(msg.data)
        #decoded_im = np.array(Image.open(msg.data)) #io.BytesIO to Convert bytes to binary stream ; Image.open to Create Image object and input should be binary data
        #im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)# -1 is in accordance with the original channel,the default is three-channel picture
        all_imgs.append(decoded_im)
        #plt.imshow(decoded_im)
        #print('result:\n', decoded_im.shape)
    
    bbs_img_bag.close()
    
    """### Associate timesteps of bounding_boxes and imgs
    
    #### First method: Split ts_bbx between two consecutive ts_img and Assign it to the first img
    """
    ############################ associate bounding_boxes and all imgs #############################################
    ts_bbs=np.array(ts_bbs)
    ts_img=np.array(ts_img)
    bbs=np.array(bbs,dtype=object)
    
    img_bbs = []
    for i in range(len(ts_img)-1):# len(ts_img)==len(all_imgs)
        ts_img_start=ts_img[i]
        ts_img_last=ts_img[i+1]
        # #split ts_bbx between two consecutive ts_img
        # img_bbs.append(ts_bbs[(ts_bbs>=ts_img_start) & (ts_bbs<ts_img_last)])
        # Get the index of elements with condition: [(ts_bbs>=ts_img_start) & (ts_bbs<ts_img_last)]
        indexes = np.where((ts_bbs>=ts_img_start) & (ts_bbs<ts_img_last))
        img_bbs.append(bbs[indexes])
    ### since ts_bbs larger than ts_img, need to append the remaining timestamps,with an interval between ts_img[-1] with ts_bbs[-1]
    #img_bbs.append(ts_bbs[(ts_bbs>=ts_img[-1]) & (ts_bbs<=ts_bbs[-1])])
    indexes = np.where((ts_bbs>=ts_img[-1]) & (ts_bbs<=ts_bbs[-1]))
    img_bbs.append(bbs[indexes])   #usage: img_bbs[0][0][0].xmin  ,img_bbs[i][j][k]--> i is img num;j is img num;k is different bbx num
    
    
    """### Filter desired Class"""
    ############# populate centroids list  #############
    bag_name = os.path.basename(bag).replace('.bag', '')
    if os.path.exists(bag_name):
        #Delete the existing folder and its contents
        shutil.rmtree(bag_name)
    os.makedirs(bag_name, exist_ok=True)
    img_idx = 0
    
    filter_flg = 1
    filter_class = ['person']#['person','car']
    img_bbs_ctd = [] #all centroids
    img_bbs_all = [] #all bboxes
    bbox_imgs_all = [] #all bbox_imgs
    for i in range(len(ts_img)):
        centroid  = [] #centroids for one img
        bboxes    = [] #bboxes for one img
        bbox_imgs = [] #bbox_imgs for one img
        #img_feats = [] #img_feats for one img
        img_arr   = all_imgs[i] #one img for cropping
        if (len(img_bbs[i])>0):
            for k in range(len(img_bbs[i])):  # k for bbs frame_img num
                for j in range(len(img_bbs[i][k])):  # j for bbs num
                    if (filter_flg) & (img_bbs[i][k][j].Class in filter_class) & (img_bbs[i][k][j].probability >=0.99):
                        x_mid=(img_bbs[i][k][j].xmin+img_bbs[i][k][j].xmax)/2
                        centroid.append([x_mid,img_bbs[i][k][j].ymax])
                        bboxes.append([img_bbs[i][k][j].xmin,img_bbs[i][k][j].ymin,img_bbs[i][k][j].xmax,img_bbs[i][k][j].ymax,img_bbs[i][k][j].probability])
                        ## crop bbox img
                        #bbox_img = img_arr.crop((img_bbs[i][k][j].xmin,img_bbs[i][k][j].ymin,img_bbs[i][k][j].xmax,img_bbs[i][k][j].ymax)) #crop((xmin,ymin,xmax,ymax)), img should be PIL image object
                        bbox_img = img_arr[img_bbs[i][k][j].ymin:img_bbs[i][k][j].ymax, img_bbs[i][k][j].xmin:img_bbs[i][k][j].xmax]
                        #bbox_imgs.append(bbox_img)
                        ## save bbox img
                        bbox_img = bbox_img.astype(np.uint8)
                        img = Image.fromarray(bbox_img, 'RGB')
                        img = resize_img(img, newsize=(64,64),filters=Image.Resampling.LANCZOS)
                        #img.show()                        
                        img.save(bag_name + '/' + str(img_idx) + '.png',quality=100, optimize=True)
                        img_idx = img_idx + 1
        else:
            centroid.append([])
            bboxes.append([])
            #bbox_imgs.append([])

            
        img_bbs_ctd.append(centroid)
        img_bbs_all.append(bboxes)
        #bbox_imgs_all.append(bbox_imgs)

        
    #return all_imgs,img_bbs_ctd,ts_img,bbox_imgs_all,img_bbs_all
    return all_imgs,img_bbs_ctd,ts_img,img_bbs_all


##################################################################
''' ************************* Main ***********************************'''

all_files_bag = glob.glob('./0519/'+'*.bag')
file_names_bag = [os.path.basename(file) for file in all_files_bag]
sorted_files_bag = sorted(file_names_bag, reverse=False)
# for bag in sorted_files_bag:
#     all_imgs,img_bbs_ctd,ts_img,img_bbs_all = parse_rosbag_to_img(bag)

# Filter list
filter_list = ['inters_data_2023-05-19-11-44-08.bag', 'inters_data_2023-05-19-15-02-28.bag']
for bag in sorted_files_bag:
    if bag in filter_list:
        all_imgs,img_bbs_ctd,ts_img,img_bbs_all = parse_rosbag_to_img(bag)