# -*- coding: utf-8 -*-
'''
Tracking_birds_eye_view_LC.ipynb
conda search ros-rosbag --channel conda-forge # check available version to python
conda install -c conda-forge ros-rosbag #so you can import rosbag
!pip3 install bagpy #or to Install bagpy first so you can import rosbag

remember to set use_BEV=1
'''
############### Lei ###################################
import os, glob, shutil, sys
sys.path.append("./common_features_for_tracking/") # add search path: sys.path.append("../../")
#######################################################

from DetectionAndTrackingProject import DetectionAndTrackingProject

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


### common feature
from get_img_ctd import parse_rosbag_to_img
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
############
def compare_2_imgs(img1,img2,img_model): 
    feat_img = np.expand_dims(process_img(img1), axis=0) #add batchsize_dim
    img_feat1 = img_model.predict(feat_img)[1] # feat is been l2_normalize
    feat_img = np.expand_dims(process_img(img2), axis=0) #add batchsize_dim
    img_feat2 = img_model.predict(feat_img)[1] # feat is been l2_normalize
    diff = np.linalg.norm(img_feat1 - img_feat2, axis=-1)
    return diff

def write_video(file_path, frames, fps):
    """
    Writes frames to an video file
    :param file_path: Path to output video, must end with suffix(.mp4,.avi)
    :param frames: List of Image objects
    :param fps: Desired frame rate
    """
    h,w,d = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
    for frame in frames:
        video.write(frame)
    video.release()

def load_rnn_model(model_path,scl_path):
    print('Loading RNN model...')
    scl = joblib.load(scl_path) #'model/bbox_low_scaler_hpc.joblib'
    model = tf.keras.models.load_model(model_path) #"model/prednet_low_model_hpc.h5"
    print('{} model loaded.'.format(model_path))
    return model,scl


    
def load_img_model(model_path,triplet_loss):
    print('Loading feat model...')
    model = tf.keras.models.load_model(model_path,custom_objects={'_triplet_loss': triplet_loss,'K':K}, compile=False)
    print('{} model loaded.'.format(model_path))
    return model

def triplet_loss(alpha = 0.2):#as the custom_loss value in custom_objects
    def _triplet_loss(y_true, y_pred):
        batch_size = tf.shape(y_pred)[0] // 3
        anchor, positive, negative = y_pred[:batch_size], y_pred[batch_size:2 * batch_size], y_pred[-batch_size:]

        pos_dist    = K.sqrt(K.sum(K.square(anchor - positive), axis=-1))
        neg_dist    = K.sqrt(K.sum(K.square(anchor - negative), axis=-1))

        basic_loss  = pos_dist - neg_dist + alpha
        
        idxs        = tf.where(basic_loss > 0)
        select_loss = tf.gather_nd(basic_loss, idxs)

        loss        = K.sum(K.maximum(basic_loss, 0)) / tf.cast(tf.maximum(1, tf.shape(select_loss)[0]), tf.float32)
        return loss
    return _triplet_loss #as the custom_loss key

   
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)    
        image   = image.resize((nw,nh), Image.Resampling.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.Resampling.BICUBIC)
    return new_image
  
def preprocess_input(image,rescale_type=0): #scale
    if rescale_type==0:
        image /= 255.0 # rescale to [0,1] 
    elif rescale_type==1:  
        image = (image/ 127.5) - 1 # rescale to [-1,1]
    return image

def process_img(image,img_target_shape=(160,160)):
    ### old_version
    image = Image.fromarray(image.astype(np.uint8))
    image = resize_image(image, img_target_shape, letterbox_image=True)
    image = preprocess_input(np.array(image, dtype='float32'),rescale_type=0)
    # ### new_version
    # image=tf.image.resize_with_pad(image,target_height=img_target_shape[0],target_width=img_target_shape[1],method='bicubic',antialias=False)
    # image = self.preprocess_input(np.array(image, dtype='float32'),rescale_type=1)
    return image

def diff_each_elements(array,flag_iou=0):
    '''
    #Difference between each elements in an numpy array---stackoverflow
    array_1   = np.array([1,2,3,4])
    a         =array([0, 0, 0, 1, 1, 2], dtype=int64)#index
    b         =array([1, 2, 3, 2, 3, 3], dtype=int64)
    array_1[a]=array([1,  1,  1,  2,  2,  3])
    array_1[b]=array([2,  3,  4,  3,  4,  4])
    diff      =array([-1, -2, -3, -1, -2, -1])
    '''
    array=np.array(array)
    #diff = [abs(i-j) for i in array for j in array if i != j]
    a, b = np.triu_indices(len(array), 1)
    if flag_iou:
        iou,overlape1,overlape2=get_iou(box1=array[a], box2=array[b])
        return a,b,iou,overlape1,overlape2
    else:
        #diff = array[a] - array[b]
        diff = np.linalg.norm(array[a] - array[b], axis=-1)
        return a,b,diff



def get_iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
        
    Arguments:
        box1 -- first box, numpy array with coordinates (ymin, xmin, ymax, xmax)
        box2 -- second box, numpy array with coordinates (ymin, xmin, ymax, xmax)
    """
    # ymin, xmin, ymax, xmax = box

    # y11, x11, y21, x21 = box1
    # y12, x12, y22, x22 = box2
    
    # yi1 = max(y11, y12)
    # xi1 = max(x11, x12)
    # yi2 = min(y21, y22)
    # xi2 = min(x21, x22)
    # inter_area = max( ((xi2 - xi1) * (yi2 - yi1)) , 0)
    # # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    # box1_area = (x21 - x11) * (y21 - y11)
    # box2_area = (x22 - x12) * (y22 - y12)
    # union_area = box1_area + box2_area - inter_area
    # ### compute the IoU
    # iou = inter_area / union_area
    # return iou
    w_intersection = np.maximum(0, (np.minimum(box1[:,2], box2[:,2]) - np.maximum(box1[:,0], box2[:,0])))
    h_intersection = np.maximum(0, (np.minimum(box1[:,3], box2[:,3]) - np.maximum(box1[:,1], box2[:,1])))
    inter_area = np.multiply(w_intersection , h_intersection)
    
    box1_area = np.multiply( (box1[:,2] - box1[:,0]) , (box1[:,3] - box1[:,1]) )
    box2_area = np.multiply( (box2[:,2] - box2[:,0]) , (box2[:,3] - box2[:,1]) )
    
    iou = np.divide( inter_area , (box1_area + box2_area - inter_area) )
    
    ### computing the overlaping, which different with IoU 
    overlape1= inter_area / box1_area
    overlape2= inter_area / box2_area
    return iou,overlape1,overlape2 

def remove_duplicates(cent,img_feat,bbxs,prob,thr1=5,thr2=0.55): 
    a1,b1,diff1=diff_each_elements(cent)
    a2,b2,diff2=diff_each_elements(img_feat)
    a3,b3,iou,overlape1,overlape2=diff_each_elements(bbxs,flag_iou=1)
    ## according diff to remove duplicates
    overlape=((overlape1>0.8)|(overlape2>0.8))
    similar= (diff2<thr2)
    #closer= (diff1<thr1)
    low_prob_a=(prob[a2]<0.55)
    low_prob_b=(prob[b2]<0.55)
    #list_idx=list(range(len(img_feat)))
    list_idx=np.array(range(len(img_feat)))
    dupl_del_a    =a2[ (similar& (overlape1>0.8)) | (low_prob_a & (overlape1>0.8))] #duplicates; dupl_del_a is indexes of img_feat
    dupl_del_b    =b2[ (similar& (overlape2>0.8)) | (low_prob_b & (overlape2>0.8))] #duplicates
    dupl_del=np.unique(np.concatenate((dupl_del_a,dupl_del_b)))
    list_idx=np.setdiff1d(list_idx, dupl_del) #from list_idx remove dupl_del
    return list_idx

def unique_without_sorted(arr, axis=0, return_index=True): 
    _,unique_indices = np.unique(arr, axis=axis, return_index=return_index)
    retain_order = np.sort(unique_indices)
    return arr[retain_order], retain_order

def append_1_to_homo(mat):
    hommat = np.append(mat, np.ones((mat.shape[0],1)), axis=1)
    return hommat

def project_3d_to_2d(polyline_world, K):
    pl_uv_cam = (K @ (polyline_world.T)).T
    u = pl_uv_cam[:,0] / pl_uv_cam[:,2]
    v = pl_uv_cam[:,1] / pl_uv_cam[:,2]
    return np.stack((u,v)).T

"""## Create birds eye view image"""

def create_birds_eye_view(K, img, d_min=3., d_max=20., cr_lf=-6., cr_rt=6., height=1.3, yaw_deg=0, pitch_deg=-5, roll_deg=0,img_w_param=30):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rotation_road_to_cam = np.array([[cr*cy+sp*sr+sy, cr*sp*sy-cy*sr, -cp*sy],
                                     [cp*sr, cp*cr, sp],
                                     [cr*sy-cy*sp*sr, -cr*cy*sp -sr*sy, cp*cy]])
    rotation_cam_to_road = rotation_road_to_cam.T # for rotation matrices, taking the transpose is the same as inversion
    translation_road_to_cam = np.array([0,height,0])
    road_3d = np.array([[cr_lf,  0,    d_max],#unit is 'm'
                        [cr_rt,  0,    d_max],
                        [cr_lf,  0,    d_min],
                        [cr_rt,  0,    d_min]]) 
    cam_3d = (rotation_road_to_cam @ (road_3d.T)).T + translation_road_to_cam
    #The resulting pixel coordinates can be negative, which does not affect the creation of the final bird's-eye view.
    uv = project_3d_to_2d(cam_3d, K)
    
    depth=d_max-d_min
    cross_range=cr_rt-cr_lf
    hw_ratio=depth/cross_range
    IMAGE_W = int(cross_range*img_w_param) #output image width
    IMAGE_H = int(hw_ratio * IMAGE_W)  #output image height
    
    x_pixel2meter_ratio=cross_range/IMAGE_W
    y_pixel2meter_ratio=depth/IMAGE_H    
    
    src = np.float32(uv)
    dst = np.float32([[0, 0], [IMAGE_W, 0], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])

    M = cv2.getPerspectiveTransform(src, dst) # The Plane transformation matrix
    
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping,# dsize: size of the output imag
    return warped_img,M,IMAGE_W,IMAGE_H,x_pixel2meter_ratio,y_pixel2meter_ratio


"""## Transform the points on the original image to the corresponding points on BEV image"""
def distance_on_BEV_cam(rnn_model,scl,img_model,image,img_bbs_ctd,img_bbs_all,bbox_imgs_all,i,frame_idx,dtp):
    image_height=image.shape[0]
    image_width=image.shape[1]
    height=1.635 #1.0 #1.635 #camera height
    pitch_deg=-5 #-5 #-3.2    
    K = np.array([[545.7881532041737, 0,                314.9817965777312],#unit is 'mm'
                  [0,                 544.7004623674142,250.4216021567457],
                  [0,                 0,                                1]])
    
    d_min=3.  #depth_min
    d_max=20 #20.  #depth_max
    cr_lf= -6. #-6.  #cross range left
    cr_rt=-cr_lf  #cross range right
    
    BEV_img,plane_tm,IMAGE_W,IMAGE_H,x_pixel2meter_ratio,y_pixel2meter_ratio=create_birds_eye_view(K, image, d_min, d_max, cr_lf, cr_rt, 
                                                                            height=height, yaw_deg=0, pitch_deg=pitch_deg, roll_deg=0,img_w_param=30)
    
    label_sf  = []
    xy        = np.array([[]])
    bbx_imgs  = [[]]
    img_feats = np.array([[]])
    feat_imgs = []
    classes   = [[]]
    if any(img_bbs_ctd[i]): # check if there no bbs in i-th img
        ####### Transform the points on the original image to the corresponding points on BEV image
        #centers = np.array([[300,250], [310, 260]])
        centers = np.array(img_bbs_ctd[i])
        centers,unique_indices = np.unique(centers, axis=0, return_index=True)#remove duplicates bbx_coor
        
        uv_1=append_1_to_homo(centers)
        xy = project_3d_to_2d(uv_1, plane_tm) #centroids for final usage
        #xy = xy[(xy >= 0).all(axis=1)] # when ctd is beyond of rang[d_min,d_max], it will be negative and should be deleted
        not_neg_filter  = (xy >= 0).all(axis=1)# condition
        xy = xy[not_neg_filter] ## need to remove duplicate img in the same frame by using img_model
        if np.size(xy):
            # bbox_imgs=np.array(bbox_imgs_all[i],dtype=object)[unique_indices] #The numpy arrays within the list must also be the same size.
            # bbx_imgs = bbox_imgs[not_neg_filter]
            bbox_imgs=[bbox_imgs_all[i][idx] for idx in unique_indices]
            bbx_imgs = [bbox_imgs[idx] for idx in np.where(not_neg_filter)[0]] #np.where()[0] convert Boolean Array to index
            
            
            bbxs = np.array(img_bbs_all[i])[:,0:4]
            bbxs = bbxs[unique_indices]
            bbxs = bbxs[not_neg_filter]
            prob = np.array(img_bbs_all[i])[:,4]
            prob = prob[unique_indices]
            prob = prob[not_neg_filter]
            
            classes = np.array(img_bbs_all[i])[:,5]
            classes = classes[unique_indices]
            classes = classes[not_neg_filter]
            
            if img_model is not None:
                for k in range(len(bbx_imgs)):
                    feat_imgs.append( process_img(np.array(bbx_imgs[k])) ) #plt.imshow(bbx_imgs[0])            
                img_feats = img_model.predict( np.array(feat_imgs) )[1] # feat is been l2_normalize
                
            # list_idx = remove_duplicates(xy[:,0].reshape(-1,1),img_feats,bbxs,prob) #compare_2_imgs(bbx_imgs[0],bbx_imgs[1],img_model)  
            
            # xy=xy[list_idx]
            # img_feats=img_feats[list_idx]
            # #bbx_imgs=bbox_imgs_all[i]       
            # x,y = xy[:,0], xy[:,1]  #pixel points on BEV image
        
    # Process images for tracking
    img_cp=BEV_img.copy()
    #out_img = dtp.DetectionByTracking(img_model,img_cp,x_pixel2meter_ratio,y_pixel2meter_ratio,bbx_imgs=img_feats,centroids=xy,d_max=d_max)
    out_img = dtp.DetectionByTracking(rnn_model,scl,img_model,img_cp,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,bbx_imgs,img_feats,classes,label_sf,centroids=xy,d_max=d_max)
    plt.imshow(out_img)
    plt.gcf().set_dpi(400)
    plt.title('C-Frame Idx:%d'%i)
    ##plt.title('C-Frame Idx:%d'%frame_idx)
    plt.show()        
    return BEV_img,out_img,xy,x_pixel2meter_ratio,y_pixel2meter_ratio

"""## Transform the points on the original image to the corresponding points on radar image"""
def distance_on_cam2rad(rnn_model,scl,img_model,image,img_bbs_ctd,img_bbs_all,bbox_imgs_all,i,frame_idx,dtp):
    label_sf = []
    image_height=image.shape[0]
    image_width=image.shape[1]
    height=1.635 #1.0 #1.635 #camera height
    pitch_deg=-5 #-5 #-3.2    
    K = np.array([[545.7881532041737, 0,                314.9817965777312],#unit is 'mm'
                  [0,                 544.7004623674142,250.4216021567457],
                  [0,                 0,                                1]])
    
    d_min=3.  #depth_min
    d_max=20.  #depth_max
    cr_lf=-6.  #cross range left
    cr_rt=-cr_lf  #cross range right
    
    BEV_img,plane_tm,IMAGE_W,IMAGE_H,x_pixel2meter_ratio,y_pixel2meter_ratio=create_birds_eye_view(K, image, d_min, d_max, cr_lf, cr_rt, 
                                                                                              height=height, yaw_deg=0, pitch_deg=pitch_deg, roll_deg=0,img_w_param=30)
    xy        = np.array([[]])
    bbx_imgs  = [[]]
    classes  = [[]]
    img_feats = np.array([[]])
    feat_imgs = []
    if np.any(img_bbs_ctd[i]): # check if there no bbs in i-th img
        ####### Transform the points on the original image to the corresponding points on BEV image
        #centers = np.array([[300,250], [310, 260]])
        centers = np.array(img_bbs_ctd[i])
        centers,unique_indices = unique_without_sorted(centers, axis=0, return_index=True)#remove duplicates bbx_coor
        

        xy = centers ## need to remove duplicate img in the same frame by using img_model
        if np.size(xy):
            # bbox_imgs=np.array(bbox_imgs_all[i],dtype=object)[unique_indices] #The numpy arrays within the list must also be the same size.
            # bbx_imgs = bbox_imgs[not_neg_filter]
            bbx_imgs=[bbox_imgs_all[i][idx] for idx in unique_indices]

            
            bbxs = np.array(img_bbs_all[i])[:,0:4]
            bbxs = bbxs[unique_indices]

            prob = np.array(img_bbs_all[i])[:,4]
            prob = prob[unique_indices]
            
            classes = np.array(img_bbs_all[i])[:,5]
            classes = classes[unique_indices]

            
            if img_model is not None:
                for k in range(len(bbx_imgs)):
                    feat_imgs.append( process_img(np.array(bbx_imgs[k])) ) #plt.imshow(bbx_imgs[0])            
                img_feats = img_model.predict( np.array(feat_imgs) )[1] # feat is been l2_normalize
                
            # list_idx = remove_duplicates(xy[:,0].reshape(-1,1),img_feats,bbxs,prob) #compare_2_imgs(bbx_imgs[0],bbx_imgs[1],img_model)    
            # xy=xy[list_idx]
            # img_feats=img_feats[list_idx]
            # #bbx_imgs=bbox_imgs_all[i]       
            # x,y = xy[:,0], xy[:,1]  #pixel points on BEV image
        
    # Process images for tracking
    img_cp= image.copy() #BEV_img.copy()
    #out_img = dtp.DetectionByTracking(img_model,img_cp,x_pixel2meter_ratio,y_pixel2meter_ratio,bbx_imgs=img_feats,centroids=xy,d_max=d_max)
    out_img = dtp.DetectionByTracking(rnn_model,scl,img_model,img_cp,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,bbx_imgs,img_feats,classes,label_sf,centroids=xy,d_max=d_max)
    
    # plt.imshow(out_img)
    # plt.gcf().set_dpi(100)
    # plt.title('C-Frame Idx:%d'%i)
    # ##plt.title('C-Frame Idx:%d'%frame_idx)
    # plt.show()        
    return BEV_img,out_img,xy,x_pixel2meter_ratio,y_pixel2meter_ratio
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
                        bboxes.append([img_bbs[i][k][j].xmin,img_bbs[i][k][j].ymin,img_bbs[i][k][j].xmax,img_bbs[i][k][j].ymax,img_bbs[i][k][j].probability, img_bbs[i][k][j].Class])
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

def modify_array(img_bbs_ctd):
    if not img_bbs_ctd:  # Check if the list of lists is empty
        return img_bbs_ctd

    # Define a new list to store the modified points
    modified_points = []

    for sublist in img_bbs_ctd:
        if not any(sublist):  # Check if the sublist is empty
            modified_points.append(sublist)
        else:
            # Modify the points in the sublist based on y conditions
            modified_sublist = np.array(sublist)
            mask = (modified_sublist[:, 1] > 225) #& (modified_sublist[:, 1] < 240)
            modified_sublist[:, 1][mask] -= 5
            mask = modified_sublist[:, 1] < 125
            modified_sublist[:, 1][mask] -= 30
            
            # Append the modified sublist to the final list
            modified_points.append(modified_sublist.tolist())

    return modified_points


def cam2rad_proj_updown(img_points,homography_matrix_up, homography_matrix_down, Y_thr = 320):
    # Convert to arrays
    img_points = np.array(img_points)
   
    if img_points[1] <= Y_thr:
        ### project points UP
        projected_point = cv2.perspectiveTransform(img_points.reshape(-1, 1, 2), homography_matrix_up)    
    else:
        ### project points DOWN
        projected_point = cv2.perspectiveTransform(img_points.reshape(-1, 1, 2), homography_matrix_down)    
    projected_point = np.squeeze(projected_point).tolist()    
    return projected_point

def cam2radar_proj(img_points, proj_matrix_path = "/xdisk/caos/leicheng/my_rawdata_0624/0624/homography_matrix_cam2rad_LMEDS.npy",):
    projection_matrix = np.load(proj_matrix_path, allow_pickle=True)  #affine_matrix_path = "/xdisk/caos/leicheng/my_rawdata_0624/0624/homography_matrix.npy"
    # Convert to arrays
    points = np.array(img_points)

    if len(projection_matrix)==3: #mode=='homography':
        # Apply the perspective transformation using homography_matrix
        projected_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), projection_matrix)
    elif len(projection_matrix)==2:    
        # Apply the affine transformation using affine_matrix
        projected_points = cv2.transform(points.reshape(-1, 1, 2), projection_matrix)
        
    return projected_points    

def convert_points(img_bbs_ctd):
    if not img_bbs_ctd:  # Check if the list of lists is empty
        return img_bbs_ctd

    # Define a new list to store the converted points
    converted_points = []

    # Iterate through each sublist in img_bbs_ctd and convert the points
    for sublist in img_bbs_ctd:
        if not any(sublist):  # Check if the sublist is empty
            converted_points.append(sublist)
        else:
            # Apply cam2radar_proj function to each point in the sublist
            converted_sublist =  np.squeeze(cam2radar_proj(sublist), axis=1).tolist() #[cam2radar_proj(point) for point in sublist]
            converted_points.append(converted_sublist)

    return converted_points       
##################################################################
def camera_main(path = "/xdisk/caos/leicheng/my_rawdata_0624/0624/", names = {'jiahao-hao'}, conf_thr=0.8, xmin_thr=390):
    # Put all the code need to execute directly when this script run directly.
    ####################################################### main #########################    
    '''
    ########################## RUN  ##########################
    '''
    # Whether to save video
    SAVE_VIDEO_FLAG=1
    if SAVE_VIDEO_FLAG:
        video_frames=[]
        
    # Load img_model
    img_model_path = "model/assonet_model_hpc.h5" #change per need
    img_model=load_img_model(img_model_path,triplet_loss)
    # Load rnn_model
    rnn_model_path,scl_path = "model/prednet_low_model_hpc.h5" , 'model/bbox_low_scaler_hpc.joblib'
    rnn_model,scl=load_rnn_model(rnn_model_path,scl_path)
    ############################
    ## Create the blank image
    blank_image = 255 * np.ones(shape=[256, 256, 3], dtype=np.uint8)          
    #############################  Load images   #############################
    
    all_files_bag = glob.glob(path +'*.bag')
    file_names_bag = [os.path.basename(file) for file in all_files_bag]
    file_names_bag = filter_files_with_names(file_names_bag, names, mode='filter_file_name')
    sorted_files_bag = sorted(file_names_bag, reverse=False)
    for bag in sorted_files_bag:
        all_imgs,img_bbs_ctd,ts_img,bbox_imgs_all,img_bbs_all,img_classes_all = parse_rosbag_to_img(bag,path,conf_thr
                                                                                    ,filter_class = ['person','car']
                                                                                    ,filter_cls_flg = 1
                                                                                    ,mode='exclude_static_cars'
                                                                                    ,xmin_thr=xmin_thr
                                                                                    ,return_mode='centroid')   #return_mode='centroid'
        
    
    #all_imgs,img_bbs_ctd,ts_img,bbox_imgs_all,img_feats_all,img_bbs_all = preprocessing_camera_data()

    ### Convert img to radar
    #img_bbs_ctd = convert_points(img_bbs_ctd)    

    
    # Create DetectionAndTracking instances
    dtp = DetectionAndTrackingProject(mode=0)
    for i in range(len(all_imgs)):
    #for i in range(3):
        image = all_imgs[i] #blank_image  #
        _,out_img_cam,_,_,_=distance_on_BEV_cam(rnn_model,scl,img_model,image,img_bbs_ctd,img_bbs_all,bbox_imgs_all,i,0,dtp)  # img_bbs_ctd is cam_ctd
        #_,out_img_cam,_,_,_=distance_on_cam2rad(rnn_model,scl,img_model,image,img_bbs_ctd,img_bbs_all,bbox_imgs_all,i,0,dtp)  # img_bbs_ctd is cam_ctd
        ## save video
        if SAVE_VIDEO_FLAG:
            video_frames.append(cv2.cvtColor(out_img_cam, cv2.COLOR_BGR2RGB))
    ### write video
    if SAVE_VIDEO_FLAG:
        file_path='./out_videos/cam_track_out_rnn.avi'    
        write_video(file_path, video_frames, fps=30)   
    ### Using cv2.imwrite() method to Save the image into filename
    #cv2.imwrite(filename='./image4radar.jpg', img=cv2.cvtColor(all_imgs[0], cv2.COLOR_BGR2RGB))
    ### print accuray
    print('Accuray: ', (dtp.acc_rnn-dtp.acc_kal)/len(all_imgs))
################################################################################################

#####################################################################################
if __name__ == '__main__':
    ####################################################### main #########################
    path = "/xdisk/caos/leicheng/my_rawdata_0624/0624/"
    names = {'jiahao-hao'} #{'lei-leicar'} #{'jiahao-hao'} #
    conf_thr=0.8
    xmin_thr=390
    camera_main(path,names,conf_thr,xmin_thr)
       
