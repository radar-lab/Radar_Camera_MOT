#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2
import io
import rosbag
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import Image
import joblib

######################################################################################################################
def preprocessing_camera_data(bag_name='./inters_data_2023-03-23-17-41-16.bag'):
    """This function is used for generate centroids. input: image and bbox"""
    """
    ### Read ros_bag
    """    
    #################### load bounding_box_bag use rosbag  ,for getting xmin etc. ##################################################
    bbs = []
    ts_bbs = []
    #bbs_img_bag = rosbag.Bag('./track_1002_calib.bag')
    bbs_img_bag = rosbag.Bag(bag_name)
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
        decoded_im = np.array(PIL.Image.open(io.BytesIO(msg.data))) #io.BytesIO to Convert bytes to binary stream ; Image.open to Create Image object and input should be binary data        
        all_imgs.append(decoded_im)
    
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
    filter_flg = 1
    filter_class = ['person']#['person','car']
    img_bbs_ctd = [] #all centroids
    img_bbs_all = [] #all bboxes
    bbox_imgs_all = [] #all bbox_imgs
    img_feats_all = [] #all img_feats
    for i in range(len(ts_img)):
        centroid  = [] #centroids for one img
        bboxes    = [] #bboxes for one img
        bbox_imgs = [] #bbox_imgs for one img
        #img_feats = [] #img_feats for one img
        img_arr   = all_imgs[i] #one img for cropping
        if (len(img_bbs[i])>0):
            for k in range(len(img_bbs[i])):  # k for bbs frame_img num
                for j in range(len(img_bbs[i][k])):  # j for bbs num
                    if (filter_flg) & (img_bbs[i][k][j].Class in filter_class) & (img_bbs[i][k][j].probability >=0.2):
                        x_mid=(img_bbs[i][k][j].xmin+img_bbs[i][k][j].xmax)/2
                        centroid.append([x_mid,img_bbs[i][k][j].ymax])
                        bboxes.append([img_bbs[i][k][j].xmin,img_bbs[i][k][j].ymin,img_bbs[i][k][j].xmax,img_bbs[i][k][j].ymax,img_bbs[i][k][j].probability])
                        ## crop bbox img
                        #bbox_img = img_arr.crop((img_bbs[i][k][j].xmin,img_bbs[i][k][j].ymin,img_bbs[i][k][j].xmax,img_bbs[i][k][j].ymax)) #crop((xmin,ymin,xmax,ymax)), img should be PIL image object
                        bbox_img = img_arr[img_bbs[i][k][j].ymin:img_bbs[i][k][j].ymax, img_bbs[i][k][j].xmin:img_bbs[i][k][j].xmax]
                        bbox_imgs.append(bbox_img)
                        ## save bbx_img_feats
                        # feat_img = np.expand_dims(process_img(bbox_img), axis=0) #add batchsize_dim
                        # img_feat = img_model.predict(feat_img)[1] # feat is been l2_normalize
                        # img_feats.append(img_feat)
        else:
            centroid.append([])
            bboxes.append([])
            bbox_imgs.append([])
            #img_feats.append([])
            
        img_bbs_ctd.append(centroid)
        img_bbs_all.append(bboxes)
        bbox_imgs_all.append(bbox_imgs)
        #img_feats_all.append(img_feats)
        
    return all_imgs,img_bbs_ctd,ts_img,bbox_imgs_all,img_feats_all,img_bbs_all
##################################################################
def main():
    # Put all the code need to execute directly when this script run directly.
    ####################################################### main #########################    
    '''
    ########################## RUN  ##########################
    '''
        
    # Load img_model
    img_model_path = "model/assonet_model_hpc.h5" #change per need
    img_model=load_img_model(img_model_path,triplet_loss)
    # Load rnn_model
    rnn_model_path,scl_path = "model/prednet_low_model_hpc.h5" , 'model/bbox_low_scaler_hpc.joblib'
    rnn_model,scl=load_rnn_model(rnn_model_path,scl_path)
    
    all_imgs,img_bbs_ctd,ts_img,bbox_imgs_all,img_feats_all,img_bbs_all = preprocessing_camera_data()
    

################################################################################################

#####################################################################################
if __name__ == '__main__':
    ####################################################### main #########################
    main()