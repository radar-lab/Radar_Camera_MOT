#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 01:22:24 2022

@author: kevalen

remember to set use_BEV=0
"""
############### Lei ###################################
import os, glob, shutil, sys
sys.path.append("./common_features_for_tracking/") # add search path: sys.path.append("../../")
sys.path.append("./ImageYolo/")
######################################################

from DetectionAndTrackingProject import DetectionAndTrackingProject
from tracking_birds_eye_view_lc import *
from radar_tracking import *

from GPS_Process import *

from get_img_ctd import parse_rosbag_to_img
from common_features_for_tracking.common_feats_net_car_person_final.common_feats_compare import common_feats_find,load_common_feats_model


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
import rosbag
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial import distance
from scipy import stats
import gc
import time

def cam2rad_proj_updown_frame(img_points, homography_matrix_up, homography_matrix_down, rad_img_shape, Y_thr=320):
    # Convert to arrays
    img_points = np.array(img_points) 
    # Initialize an array to store the projected points
    projected_points = []
    if img_points.size > 0 :
        # Get the height and width of the radar image
        height, width = rad_img_shape[:2]   
        # Iterate over each point
        for point in img_points:
            if point[1] <= Y_thr:
                # Project points UP
                projected_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), homography_matrix_up)    
            else:
                # Project points DOWN
                projected_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), homography_matrix_down)
            
            projected_point = np.squeeze(projected_point).tolist()
            
            # Clamp the projected_point within the radar image dimensions
            projected_point = [
                min(max(projected_point[0], 0), width - 1), # w
                min(max(projected_point[1], 0), height - 1) # h
            ]
            
            projected_points.append(projected_point)   
    return np.array(projected_points)

def save_variable_as_npy(variable, filename='./name.npy'):
    np.save(filename, variable)
    print(f"{variable} has been successfully saved as a .npy file")
    
def save_plot_to_folder(fig, folder_path, file_name, dpi=300):
    # Create the target folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save the figure to the specified path  f"{file_name}_{index}.png"
    plot_path = os.path.join(folder_path, f"{file_name}.png")
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    
    print(f"Plot saved to: {plot_path}")    
    
def save_image_to_folder(image, folder_path, image_name, suffix='.png'):
    # Create the target folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save the image to the specified path
    image_path = os.path.join(folder_path, str(image_name) + suffix)
    ## Convert the NumPy array to a PIL Image
    ## image = Image.fromarray(image)
    ##image.save(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, image)
    
    print(f"Image saved to: {image_path}")

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
        #frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) # Convert PIL image back to OpenCV format before writing
        video.write(frame)
    video.release()

def get_img_from_fig(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi='figure')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img

def plot_muti_imgs(i,img_fr,rad_fr,sf_fr, folder_path):
    my_dpi=100
    fig = plt.figure(figsize=(1280 / my_dpi, 960 / my_dpi), dpi=my_dpi) #fig = Figure(figsize=(1024, 512), dpi=1) To render an image of a specific size
    #####if we don't want multiple plot we can comment the code for ax1 and ax2
    ax = fig.add_subplot(131)
    ax.margins(0)
    ax.imshow(img_fr)
    ax.set_title('Image Frame:%d'%i, fontsize=16)
    
    ax2 = fig.add_subplot(132)
    ax2.margins(0)
    ax2.imshow(rad_fr)
    ax2.set_title('Radar Frame:%d'%i, fontsize=16)
    
    ax3 = fig.add_subplot(133)  #fig.add_subplot(111)
    ax3.margins(0)
    ax3.imshow(sf_fr)
    ax3.set_title('SensorFusion Frame:%d'%i, fontsize=16)
    plt.tight_layout()
    #plt.suptitle('SF-Frame Idx:%d'%i, fontsize=20) #main title
    
    # Save the plot
    #save_plot_to_folder(fig, folder_path, i, dpi=300)
    
    plot_img_np = get_img_from_fig(fig, dpi=300)   
    # plt.tight_layout()
    # plt.show()
    print('Frame Idx:%d'%i)
    # # nowTime = datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
    # # file_path_name = directory+'walk-'+nowTime+'.avi'
    # cv2.waitKey(10)
    # all_imgs.append(plot_img_np)
    # frames=all_imgs
    # fps=30
    # write_video(file_path_name, frames, fps)
    return plot_img_np
        
def find_nearest_betw_arr(known_array, match_array):
    '''
    Based on match_array, to find the value in an known_array which is closest to an element in match_array
    return match_value and inx in known_array, and arr size is len(match_array)
    '''
    # known_array=np.array([1, 9, 33, 26,  5 , 0, 18, 11])
    # match_array=np.array([-1, 0, 11, 15, 33, 35,10,31])
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted] 
    idx = np.searchsorted(known_array_sorted, match_array)
    idx1 = np.clip(idx, 0, len(known_array_sorted)-1)
    idx2 = np.clip(idx - 1, 0, len(known_array_sorted)-1)
    diff1 = known_array_sorted[idx1] - match_array
    diff2 = match_array - known_array_sorted[idx2]
    indices = index_sorted[np.where(diff1 <= diff2, idx1, idx2)]
    return indices,known_array[indices]

def form_bbx_img_feat(img_model,image,img_bbs_ctd,img_bbs_all,bbox_imgs_all,i):
    image_height=image.shape[0]
    image_width=image.shape[1]
    height=1.635 #camera height
    pitch_deg=-5 #-3.2    
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
    img_feats = np.array([[]])
    feat_imgs = []
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
        
        
        for k in range(len(bbx_imgs)):
            feat_imgs.append( process_img(bbx_imgs[k]) ) #plt.imshow(bbx_imgs[0])            
        img_feats = img_model.predict( np.array(feat_imgs) )[1] # feat is been l2_normalize
        
        list_idx = remove_duplicates(xy[:,0].reshape(-1,1),img_feats,bbxs,prob) #compare_2_imgs(bbx_imgs[0],bbx_imgs[1],img_model)  
        
        xy=xy[list_idx]
        img_feats=img_feats[list_idx]
    return xy, bbx_imgs,img_feats


def Generate_bbx_img_feat(img_model,img_bbs_ctd,img_bbs_all,bbox_imgs_all,i):             
    xy        = np.array([[]])
    bbx_imgs  = [[]]
    classes  = [[]]
    img_feats = np.array([[]])
    feat_imgs = []
    if np.any(img_bbs_ctd[i]): # check if there no bbs in i-th img
        ########remove duplicates bbx_coor
        #centers = np.array([[300,250], [310, 260]])
        centers = np.array(img_bbs_ctd[i])
        centers,unique_indices = unique_without_sorted(centers, axis=0, return_index=True)
        

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
             
    return xy, bbx_imgs,img_feats,classes
##########################################################################

def rad_to_bev_coordinate(centroids,img_w_param=30, d_min=3., d_max=20., cr_lf=-6., cr_rt=6.):
    ''' Transform radar to BEV image pixel position'''    
    d_min=d_min  #depth_min
    d_max=d_max  #depth_max
    cr_lf=cr_lf  #cross range left
    cr_rt=cr_rt  #cross range right
    
    depth=d_max-d_min
    cross_range=cr_rt-cr_lf
    hw_ratio=depth/cross_range
    IMAGE_W = int(cross_range*img_w_param) #output image width
    IMAGE_H = int(hw_ratio * IMAGE_W)  #output image height
    
    x_pixel2meter_ratio=cross_range/IMAGE_W
    y_pixel2meter_ratio=depth/IMAGE_H

    ####### Transform distance to pixel position
    centroids = np.array(centroids)
    if centroids.size:#detections is not empty
        if centroids.ndim>1:
            xy_centroids=centroids[:,0:2]#Select only the first two columns:x,y
            ######## convert radar_coordinates to img_coordinates
            centroids=xy_centroids[:,[1,0]]#Swap the columns x and y
            #centroids[:,0]= -centroids[:,0]#flip the x-axis
            cr_distance_mid=x_pixel2meter_ratio*(IMAGE_W/2)
            pix_ctd_x = (centroids[:,0]+cr_distance_mid)/x_pixel2meter_ratio
            pix_ctd_y = (d_max-centroids[:,1])/y_pixel2meter_ratio
            centroids[:,0]=pix_ctd_x  ###centroids is np.array([[]]),this will change the original centroids array to pixel values
            centroids[:,1]=pix_ctd_y  # return centroids will return the pixel positions
        else:
            xy_centroids=centroids[0:2]#Select only the first two columns:x,y
            ######## convert radar_coordinates to img_coordinates
            centroids=xy_centroids[[1,0]]#Swap the columns x and y
            #centroids[0]= -centroids[0]#flip the x-axis
            cr_distance_mid=x_pixel2meter_ratio*(IMAGE_W/2)
            pix_ctd_x = (centroids[0]+cr_distance_mid)/x_pixel2meter_ratio
            pix_ctd_y = (d_max-centroids[1])/y_pixel2meter_ratio
            centroids[0]=pix_ctd_x  ###centroids is np.array([[]]),this will change the original centroids array to pixel values
            centroids[1]=pix_ctd_y  # return centroids will return the pixel positions
    return centroids,x_pixel2meter_ratio,y_pixel2meter_ratio   
########################
def one_frame_acc(dtp,gps_all_for_radar,total_dist,i,theta=[-91,3.2,0]):
    gps_one_for_radar = np.array(gps_all_for_radar[i])
    ## one frame acc
    if gps_one_for_radar.size:
        trk_position = []
        if dtp.tracker_list:
            for tracker in dtp.tracker_list:
                if len(tracker.distance_pos)>=2:
                    trk_position.append(tracker.distance_pos[-2])
                else:
                    trk_position.append(tracker.distance_pos[-1])
        
        ### transform_radar_under_gps_coordinate_system
        trk_position = np.array(trk_position)
        loc_to_gps_mat=rotation_matrix(theta[0],theta[1],theta[2], order='zyx') #90.9
        loc_homo = append_1_to_homo(trk_position)
        loc_to_gps_res = project_3d_to_2d(loc_homo, loc_to_gps_mat) #centroids for final usage
        loc_to_gps_res[:,[0,1]] = loc_to_gps_res[:,[1,0]]
        loc_to_gps_res[:,1] = -loc_to_gps_res[:,1]

        dist = match_gps_to_rad( loc_to_gps_res[:,0:2],gps_one_for_radar[:,0:2], max_dis_cost=20)
        total_dist.append(dist)
    return total_dist 
##############################################################################
########################
def org_img_to_BEV_coordinate(image,img_bbs_ctd,i):
    """## Transform the points on the original image to the corresponding points on BEV image"""
    height=1.635 #camera height
    pitch_deg=-3.2    
    K = np.array([[545.7881532041737, 0,                314.9817965777312],#unit is 'mm'
                  [0,                 544.7004623674142,250.4216021567457],
                  [0,                 0,                                1]])    
    d_min=3.  #depth_min
    d_max=20.  #depth_max
    cr_lf=-6.  #cross range left
    cr_rt=-cr_lf  #cross range right
    
    BEV_img,plane_tm,IMAGE_W,IMAGE_H,x_pixel2meter_ratio,y_pixel2meter_ratio=create_birds_eye_view(K, image, d_min, d_max, cr_lf, cr_rt, 
                                                                                              height=height, yaw_deg=0, pitch_deg=pitch_deg, roll_deg=0,img_w_param=30)
    xy=np.array([[]])
    if any(img_bbs_ctd[i]): # check if there no bbs in i-th img
        ####### Transform the points on the original image to the corresponding points on BEV image
        #centers = np.array([[300,250], [310, 260]])
        centers = np.array(img_bbs_ctd[i])
        centers = np.unique(centers, axis=0)#remove duplicates 
        uv_1=append_1_to_homo(centers)
        xy = project_3d_to_2d(uv_1, plane_tm) #centroids
        #x,y = xy[:,0], xy[:,1]  #pixel points on BEV image        
    return xy 
##############################################################################
def match_cam_to_rad(trackers, detections, max_dis_cost=40, matched_index=[]):
    ## tracker--camera; detection--radar

    # Filter out the matched indices and create new trackers and detections, and index_Map
    new_trackers = []
    new_detections = []
    matched_tra_idx = [idx[0] for idx in matched_index] #img_idx
    matched_det_idx = [idx[1] for idx in matched_index]
    old_to_new_trackers = {}
    old_to_new_detections = {}
    for i, tracker in enumerate(trackers):
        if i not in matched_tra_idx:
            new_idx = len(new_trackers)  # Get the index in the new_trackers list
            old_to_new_trackers[new_idx] = i  # Map new index to old index
            new_trackers.append(tracker)

    for i, detection in enumerate(detections):
        if i not in matched_det_idx:
            new_idx = len(new_detections)  # Get the index in the new_detections list
            old_to_new_detections[new_idx] = i  # Map new index to old index
            new_detections.append(detection)
            
    # Initialize 'cost_matrix'
    cost_matrix = np.zeros(shape=(len(new_trackers), len(new_detections)), dtype=np.float32)

    # Populate 'cost_matrix'
    for t, tracker in enumerate(new_trackers):#trackers, detections just positions
        for d, detection in enumerate(new_detections):
            cost_matrix[t,d] = distance.euclidean(tracker, detection)  
            
    ###  Run Hungarian algorithm to find matches for remaining elements
    # Produce matches by using the Hungarian algorithm to minimize the cost_distance
    # Since linear_assignment try to minimize the cost default, to maximize the sum of IOU will need to negative IOU cost 
    row_ind, col_ind = linear_assignment(cost_matrix) #`row_ind array` for `tracks`,`col_ind` for `detections`

    # Populate 'unmatched_trackers'
    unmatched_trackers = []
    for t in np.arange(len(new_trackers)):
        if t not in row_ind: #`row_ind` for `tracks`
            original_t_idx = old_to_new_trackers[t]
            unmatched_trackers.append(original_t_idx)

    # Populate 'unmatched_detections'
    unmatched_detections = []
    for d in np.arange(len(new_detections)):
        if d not in col_ind:#`col_ind` for `detections`
            original_d_idx = old_to_new_detections[d]
            unmatched_detections.append(original_d_idx)

    # Populate 'matches'
    matches = []
    for t_idx, d_idx in zip(row_ind, col_ind):
        original_t_idx = old_to_new_trackers[t_idx]
        original_d_idx = old_to_new_detections[d_idx]
        # Create tracker if cost is less than 'max_dis_cost'
        # Check for cost distance threshold.
        # If cost is very high then unmatched (delete) the track
        if cost_matrix[t_idx,d_idx] < max_dis_cost:
            matches.append([original_t_idx, original_d_idx])
        else:
            unmatched_trackers.append(original_t_idx)
            unmatched_detections.append(original_d_idx)
            
    # Add matched_index at the beginning of matches
    matches = matched_index + matches
    # Or to Add matched_index at the end of matches         
    # matches.extend(matched_index)

    # Return matches, unmatched detection and unmatched trackers
    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)
#######
def radarTocameraAssociation(radar_ctd,cam_ctd,dtp, bbx_imgs,img_feats,classes, matched_index, max_dis_cost=120):
    '''
    Ideally(the matching algorithm is very accurate), unmatched radar represent objects only detected by radar alone, 
    unmatched camera represent objects only detected by cameras alone,
    and  matched represent objects detected by both the radar and the camera
    '''

    # Match radar(like det) to camera(like trk), set max_dis_cost per your need
    matched, unmatched_radar, unmatched_camera = \
                match_cam_to_rad(cam_ctd, radar_ctd, max_dis_cost, matched_index)
              
                          
    sensorfusion_ctd=[]
    bbx_imgs_sf  = []
    img_feats_sf = []
    classes_sf = []
    label_sf = []
    # Deal with matched: [cam_x, rad_y]
    if len(matched) > 0:      
        for cam_idx, rad_idx in matched:
            #cam_x=cam_ctd[cam_idx,0] #extract camera's x as sensorfusion's x
            rad_x=radar_ctd[rad_idx,0] #extract radar's x as sensorfusion's x
            rad_y=radar_ctd[rad_idx,1] #extract radar's y as sensorfusion's y
            #sensorfusion_ctd.append([cam_x,rad_y])
            sensorfusion_ctd.append([rad_x,rad_y])
            bbx_imgs_sf.append(bbx_imgs[cam_idx])
            if np.array(img_feats).size > 0 :
                img_feats_sf.append(img_feats[cam_idx])
            else:
                img_feats_sf.append([])
            classes_sf.append(classes[cam_idx])
            label_sf.append(2)  # 1-radar; 0-cam; 2-SF
    
    # Deal with unmatched radar
    if len(unmatched_radar) > 0:        
        for i in unmatched_radar:
            sensorfusion_ctd.append(radar_ctd[i].tolist())
            bbx_imgs_sf.append([])
            img_feats_sf.append([])
            classes_sf.append('')
            label_sf.append(1)  # 1-radar; 0-cam; 2-SF

    # Deal with unmatched camera
    if len(unmatched_camera) > 0:
        for i in unmatched_camera:
            sensorfusion_ctd.append(cam_ctd[i].tolist())
            bbx_imgs_sf.append(bbx_imgs[i])
            if np.array(img_feats).size > 0 :
                img_feats_sf.append(img_feats[i])
            else:
                img_feats_sf.append([])                        
            classes_sf.append(classes[i])
            label_sf.append(0)  # 1-radar; 0-cam; 2-SF

    
    return np.array(sensorfusion_ctd),bbx_imgs_sf,np.array(img_feats_sf),np.array(classes_sf),np.array(label_sf)
    
##############################################################################
def construct_sensorfusion_centroids(radar_ctd,cam_ctd,dtp, bbx_imgs,img_feats,classes, matched_index):
    ####### Transform distance to pixel position, also we can use the ctd from radar return directly
    #radar_ctd,x_pixel2meter_ratio,y_pixel2meter_ratio=rad_to_bev_coordinate(radar_ctd,img_w_param=30, d_min=3., d_max=20., cr_lf=-6., cr_rt=6.)
    radar_ctd=radar_ctd[:,0:2] #xy
    
    sensor_label=''
    ####### get sensorfusion centroids     
    if radar_ctd.size and cam_ctd.size: #radar and camera's detections are all not empty
        sensorfusion_ctd,bbx_imgs_sf,img_feats_sf,classes_sf,label_sf=radarTocameraAssociation(radar_ctd,cam_ctd,dtp, bbx_imgs,img_feats,classes, matched_index)
        #sensor_label='SF-'
    elif radar_ctd.size: #radar's detections are not empty
        sensorfusion_ctd=radar_ctd
        #sensor_label='R-'
        bbx_imgs_sf  = []
        img_feats_sf = []
        classes_sf = []
        label_sf = []
        for i in range(len(sensorfusion_ctd)):
            bbx_imgs_sf.append([])
            img_feats_sf.append([])
            classes_sf.append('')
            label_sf.append(1)  # 1-radar; 0-cam; 2-SF
        bbx_imgs_sf,img_feats_sf,classes_sf,label_sf = bbx_imgs_sf,np.array(img_feats_sf),np.array(classes_sf),np.array(label_sf)
        #bbx_imgs_sf,img_feats_sf = np.array([[]]), np.array([[]])
    elif cam_ctd.size: #camera's detections are not empty
        sensorfusion_ctd=cam_ctd
        label_sf = []
        for i in range(len(sensorfusion_ctd)):
            label_sf.append(0)
        bbx_imgs_sf,img_feats_sf,classes_sf = bbx_imgs,img_feats,classes
        #sensor_label='C-'
    else: #radar and camera's detections are all empty
        sensorfusion_ctd=np.array([[]])
        #sensor_label=''
        bbx_imgs_sf  = []
        img_feats_sf = []
        classes_sf = []
        label_sf = []
        
    #return sensorfusion_ctd,bbx_imgs_sf,img_feats_sf,sensor_label,x_pixel2meter_ratio,y_pixel2meter_ratio
    return sensorfusion_ctd,bbx_imgs_sf,img_feats_sf,classes_sf,label_sf,sensor_label
##############################################################################
def common_feature_match(radar_ctd,cam_ctd,dtp, bbox_rad_list, bbox_img_list,img_feats,classes, max_dis_cost=50):
    paired_ctds = [] 
    matched_idx = []
    ### To avoid repeat compare
    #uses the already_compared_same_j and already_compared_same_k variables to keep track 
    #of the indices that have already been compared and found to be the same. 
    #If either of these variables matches the current indices j and k, 
    #the loop continues to the next iteration without performing the comparison again.
    already_compared_same_j = None
    already_compared_same_k = None    
    
    if radar_ctd.size and cam_ctd.size:
        for j in range(len(bbox_img_list)):  # iterate over each img bbox in the bbox_img_list
            # Check if already compared and found to be the same, then skip this iteration
            if already_compared_same_j == j:
                continue        
        
            bbox_img = bbox_img_list[j]
            images = [bbox_img]  
            
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
                    bbs_ctd_img = cam_ctd[j]   # img bbox center
                    bbs_ctd_rad = radar_ctd[k]  # rad bbox center
                    # paired_ctds.append([bbs_ctd_img, bbs_ctd_rad])
                    dis_r2c = distance.euclidean(bbs_ctd_img, bbs_ctd_rad)
                    if dis_r2c < max_dis_cost:
                        matched_idx.append([j, k]) #[img, rad]

                        # Mark the current indices as already compared and found to be the same
                        already_compared_same_j = j
                        already_compared_same_k = k                    
                

    return matched_idx
################################################################################################
"""## Transform the points of sensorfusion to the corresponding points on BEV image"""
def distance_on_BEV_sensorfusion(rnn_model,scl,img_model,image,i,centroids,dtp, bbx_imgs,img_feats,classes,label_sf,sensor_label):
    height=1.635 #1.0#1.635 #camera height
    pitch_deg=-5#-3.2    
    K = np.array([[545.7881532041737, 0,                314.9817965777312],#unit is 'mm'
                  [0,                 544.7004623674142,250.4216021567457],
                  [0,                 0,                                1]])
    
    d_min=3.  #depth_min
    d_max=20.  #depth_max
    cr_lf=-6.  #cross range left
    cr_rt=-cr_lf  #cross range right
    ## below function_call can be removed and recieve the BEV image directly from cam_tracking or radar_tracking
    BEV_img,plane_tm,IMAGE_W,IMAGE_H,x_pixel2meter_ratio,y_pixel2meter_ratio=create_birds_eye_view(K, image, d_min, d_max, cr_lf, cr_rt, 
                                                                                              height=height, yaw_deg=0, pitch_deg=pitch_deg, roll_deg=0,img_w_param=30)



    ####### Transform distance to pixel position

    ####### Process centroids
    #img_cp = 255 * np.ones_like(BEV_img , dtype = np.uint8)#blank_image same size as BEV_img
    img_cp=image.copy() #BEV_img.copy()
    #out_img = dtp.DetectionByTracking(img_cp,x_pixel2meter_ratio,y_pixel2meter_ratio,centroids=centroids,d_max=d_max,sensor_label=sensor_label)
    out_img = dtp.DetectionByTracking(rnn_model,scl,img_model,img_cp,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,bbx_imgs,img_feats,classes,label_sf,centroids=centroids,d_max=d_max,sensor_label=sensor_label)
    # plt.imshow(out_img)
    # plt.gcf().set_dpi(100)
    # plt.title('SF-Frame Idx:%d'%i)
    # plt.show()        
    return BEV_img,out_img,centroids,x_pixel2meter_ratio,y_pixel2meter_ratio
#################################################################################################################

''' #######################              RUN             ####################################################'''
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
########################### load DL model #######################
img_model = None
rad_model = None
rnn_model_cam, scl_cam = None, None
rnn_model_rad, scl_rad = None, None
# ### Load img_model
# img_model_path = "model/assonet_model_hpc.h5" #change per need
# img_model=load_img_model(img_model_path,triplet_loss)
# # Load rnn_model
# rnn_model_path,scl_path = "model/prednet_low_model_hpc.h5" , 'model/bbox_low_scaler_hpc.joblib'
# rnn_model_cam,scl_cam=load_rnn_model(rnn_model_path,scl_path)

# ### Load rad_model
# # rad_model_path = "model/assonet_model_hpc.h5" #change per need
# # rad_model=load_rad_model(rad_model_path)
# rad_model=[]
# # Load rnn_model
# rnn_model_path,scl_path = "model/prednet_rad_model_3d_3ts.h5" , 'model/radar_scaler_3d_3ts.joblib'
# rnn_model_rad,scl_rad=load_rnn_model(rnn_model_path,scl_path)

#############################  Load deep-learning model   #############################
### Try to load model first.
radimg_yolo_model  = load_radimg_yolo_model(confidence = 0.5,nms_iou = 0.3)
# Load the common_feats_model
common_feats_model = load_common_feats_model(full_model_path = "common_features_for_tracking/common_feats_net_car_person_final/model_data/final_model.h5")
##########################################################################################

start = time.perf_counter()

############################# Create DetectionAndTrackingProject instances   #################
dtp_cam = DetectionAndTrackingProject(mode=0)
dtp_radar = DetectionAndTrackingProject(mode=1)
dtp_sensorfusion = DetectionAndTrackingProject(mode=2)

#############################  Load images   #############################
path = "/xdisk/caos/leicheng/my_rawdata_0624/0624/"
names = ['lei-leicar'] #['hao-jiahao'] #['lei-leicar'] #['jiahao-hao'] #['lei_alone']#['2person-car']    

matrix_path = './common_features_for_tracking/' + names[0]
homography_matrix_up   = np.load(os.path.join(matrix_path, 'homography_matrix_cam2rad_up.npy'), allow_pickle=True)
homography_matrix_down = np.load(os.path.join(matrix_path, 'homography_matrix_cam2rad_down.npy'), allow_pickle=True)


conf_thr=0.8
xmin_thr=390
ymin_thr= 150 #210
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
                                                                                , ymin_thr=ymin_thr
                                                                                ,return_mode='centroid')  #,return_mode='centroid'

### Convert img to radar
img_bbs_ctd_cp = img_bbs_ctd
# img_bbs_ctd = convert_points(img_bbs_ctd)

# ### Pruning the img_bbs_ctd
# img_bbs_ctd = modify_array(img_bbs_ctd)

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
    

##################################################     Traverse all radar to find img   ################## 
###########################  Paired radar and image  #############################  
## Associating the radar and image by timestamps
idx_img,nearest_rad_time = find_nearest_betw_arr(ts_img, rad_fr_timestamps)    

############################### Tracking  ###########################################################
blank_image = 255 * np.ones(shape=[256, 256, 3], dtype=np.uint8)
total_loc_cam=[]
rcsf_frames=[]
total_dist_cam=[]
total_dist_rad=[]
total_dist_sf=[]
# Traverse all radar to find img
for i in range(len(npy_paths)):  
#for i in range(3400, len(npy_paths)):
    # if i>2500:
    #     break
    print('img idx : ', i)
    # for each index i, obtain bbox_img_list and the corresponding bbox_rad_list
    #bbox_img_list = filtered_bbox_list[i]
    bbox_img_list = bbox_imgs_all[i]


    ### Radar only Tracking
    # Get radar index based on the img_idx
    rad_idx = i
    # Get current radar path
    npy_path = npy_paths[rad_idx]
    png_path = png_paths[rad_idx]
    # Read .png file using PIL (Pillow) library
    image = Image.open(png_path)  
    # Convert image data to NumPy array
    image = np.array(image)    #blank_image
    # Get current radar related data
    rad_bboxes, rad_centers, bbox_rads_all = process_radar_data(npy_path, png_path
                                                                ,confidence=0.5, nms_iou=0.3
                                                                ,yolo=radimg_yolo_model)
    
    bbox_rad_list   = bbox_rads_all[0]
    # Get radar_ctd
    radar_ctd = rad_centers[0]
    radar_ctd_cp=radar_ctd.copy()

    _,out_img_rad,_,_,_=distance_on_BEV_radar(rnn_model_rad,scl_rad,rad_model,image.copy(),i,radar_ctd,dtp_radar)
    #one_frame_acc(dtp_radar,gps_all_for_radar,total_dist_rad,i)
    


    ### Camera only Tracking
    # Get original index of the image
    img_idx = idx_img[i]
    # Get corresponding data by index
    bbs_ctd_fr = img_bbs_ctd[img_idx]
    image_cam = all_imgs[img_idx]
    #ts_img_fr  = ts_img[img_idx]   #rad_fr_timestamps[rad_idx]
    #BEV_img,out_img_cam,cam_ctd,_,_ = distance_on_BEV_cam(rnn_model_cam,scl_cam,img_model,image.copy(),img_bbs_ctd,img_bbs_all,bbox_imgs_all,img_idx,i,dtp_cam)#i=img_idx
    BEV_img,out_img_cam,cam_ctd,_,_ = distance_on_cam2rad(rnn_model_cam,scl_cam,img_model,image.copy(),img_bbs_ctd,img_bbs_all,bbox_imgs_all,img_idx,i,dtp_cam)#i=img_idx
    #one_frame_acc(dtp_cam,gps_all_for_radar,total_dist_cam,i)
    total_loc_cam.append(img_bbs_ctd_cp[img_idx])

    

    
    ### Sensor Fusion Tracking
    #cam_ctd, bbx_imgs,img_feats = form_bbx_img_feat(img_model,image,img_bbs_ctd,img_bbs_all,bbox_imgs_all,img_idx)
    cam_ctd, bbx_imgs,img_feats,classes = Generate_bbx_img_feat(img_model,img_bbs_ctd,img_bbs_all,bbox_imgs_all,img_idx)
    cam_ctd = cam2rad_proj_updown_frame(cam_ctd, homography_matrix_up, homography_matrix_down, image.shape, Y_thr=320)
    matched_index = common_feature_match(radar_ctd_cp,cam_ctd,dtp_sensorfusion,bbox_rad_list, bbx_imgs,img_feats,classes)#get sensorfusion_ctd
    sensorfusion_ctd, bbx_imgs_sf,img_feats_sf,classes_sf,label_sf,sensor_label = construct_sensorfusion_centroids(radar_ctd_cp,cam_ctd,dtp_sensorfusion, bbx_imgs,img_feats,classes, matched_index)#get sensorfusion_ctd
    _,out_img_sf,sf_ctd,_,_= distance_on_BEV_sensorfusion(rnn_model_cam,scl_cam,img_model,image.copy(),i,sensorfusion_ctd,dtp_sensorfusion, bbx_imgs_sf,img_feats_sf,classes_sf,label_sf,sensor_label)
    #one_frame_acc(dtp_sensorfusion,gps_all_for_radar,total_dist_sf,i)
    
    # ## save video
    rcsf_frame = plot_muti_imgs(i,out_img_cam,out_img_rad,out_img_sf, 
                                folder_path= names[0]+'/out_images/')
    rcsf_frames.append(rcsf_frame)# save video
    #rcsf_frames.append(cv2.cvtColor(out_img_sf, cv2.COLOR_BGR2RGB))# save video without idx
    
    # Save the image to a folder
    save_image_to_folder(rcsf_frame, folder_path= names[0]+'/out_images/', image_name=i, suffix='.png')
    save_image_to_folder(cv2.cvtColor(image_cam, cv2.COLOR_BGR2RGB), 
                         folder_path= names[0]+'/out_images_cam/', image_name=i, suffix='.png')
    
    ## clear mem
    gc.collect()    

end = time.perf_counter()
print('run time:{} secs'.format(end-start))    

# # # write video
file_path='./out_videos/SensorFusion_rcsf.avi'    
write_video(file_path, rcsf_frames, fps=30)

# # save npy file
if not os.path.exists(names[0]+ "/npy_files/"):
    os.makedirs(names[0]+ "/npy_files/")
save_variable_as_npy(dtp_cam.cam_trc_loc, names[0]+ "/npy_files/cam_trc_loc.npy")
save_variable_as_npy(dtp_radar.rad_trc_loc, names[0]+ "/npy_files/rad_trc_loc.npy")
save_variable_as_npy(dtp_sensorfusion.sf_trc_loc, names[0]+ "/npy_files/sf_trc_loc.npy")
save_variable_as_npy(total_loc_cam, names[0]+ "/npy_files/total_loc_cam.npy")

# cam_trc_loc = np.load("./npy_files/cam_trc_loc.npy",allow_pickle=True)
# rad_trc_loc = np.load("./npy_files/rad_trc_loc.npy",allow_pickle=True)
# sf_trc_loc = np.load("./npy_files/sf_trc_loc.npy",allow_pickle=True)
# total_loc_cam = np.load("./npy_files/total_loc_cam.npy",allow_pickle=True)



# ##################################################     Traverse all img to find radar   ################## 
# ###########################  Paired radar and image  #############################  
# ## Associating the radar and image by timestamps
# idx_rad,nearest_rad_time = find_nearest_betw_arr(rad_fr_timestamps, ts_img)

# # #filtered_bbox_imgs_all will contain tuples with the index and corresponding non-empty data, preserving the original indices from bbox_imgs_all
# # filtered_bbox_imgs_all = [(i, bbox) for i, bbox in enumerate(bbox_imgs_all) if bbox != [[]]]
# # filtered_bbox_list = [bbox for _, bbox in filtered_bbox_imgs_all]
# # filtered_idx_list  = [idx for idx,_ in filtered_bbox_imgs_all]

# # # Get the corresponding elements based on the filtered indices
# #filtered_all_imgs    = [all_imgs[i] for i in filtered_idx_list]
# # filtered_img_bbs_ctd = [img_bbs_ctd[i] for i in filtered_idx_list]
# # filtered_ts_img      = [ts_img[i] for i in filtered_idx_list]

# # ********************Delete unused variables to free up memory********************##############
# #del filtered_bbox_imgs_all        

# ############################### Tracking  ###########################################################
# blank_image = 255 * np.ones(shape=[256, 256, 3], dtype=np.uint8)
# rcsf_frames=[]
# total_dist_cam=[]
# total_dist_rad=[]
# total_dist_sf=[]
# # Traverse all img to find radar
# for i in range(len(bbox_imgs_all)):  #len(filtered_bbox_list)
# #for i in range(948, len(filtered_bbox_list)):
#     if i>5215:
#         break
#     print('img idx : ', i)
#     # for each index i, obtain bbox_img_list and the corresponding bbox_rad_list
#     #bbox_img_list = filtered_bbox_list[i]
#     bbox_img_list = bbox_imgs_all[i]


#     ### Camera only Tracking
#     # Get original index of the image
#     #img_idx = filtered_idx_list[i]
#     img_idx = i
#     # Get corresponding data by index
#     image      = blank_image   #all_imgs[img_idx]
#     bbs_ctd_fr = img_bbs_ctd[img_idx]
#     #ts_img_fr  = ts_img[img_idx]   #rad_fr_timestamps[rad_idx]
#     #BEV_img,out_img_cam,cam_ctd,_,_ = distance_on_BEV_cam(rnn_model_cam,scl_cam,img_model,image.copy(),img_bbs_ctd,img_bbs_all,bbox_imgs_all,img_idx,i,dtp_cam)#i=img_idx
#     BEV_img,out_img_cam,cam_ctd,_,_ = distance_on_cam2rad(rnn_model_cam,scl_cam,img_model,image.copy(),img_bbs_ctd,img_bbs_all,bbox_imgs_all,img_idx,i,dtp_cam)#i=img_idx
#     #one_frame_acc(dtp_cam,gps_all_for_radar,total_dist_cam,i)

    
#     ### Radar only Tracking
#     # Get radar index based on the img_idx
#     rad_idx = idx_rad[img_idx]
#     # Get current radar path
#     npy_path = npy_paths[rad_idx]
#     png_path = png_paths[rad_idx]
#     # Get current radar related data
#     rad_bboxes, rad_centers, bbox_rads_all = process_radar_data(npy_path, png_path
#                                                                 ,confidence=0.5, nms_iou=0.3
#                                                                 ,yolo=radimg_yolo_model)
    
#     bbox_rad_list   = bbox_rads_all[0]
#     # Get radar_ctd
#     radar_ctd = rad_centers[0]
#     radar_ctd_cp=radar_ctd.copy()

#     _,out_img_rad,_,_,_=distance_on_BEV_radar(rnn_model_rad,scl_rad,rad_model,image.copy(),i,radar_ctd,dtp_radar)
#     #one_frame_acc(dtp_radar,gps_all_for_radar,total_dist_rad,i)
    
#     ### Sensor Fusion Tracking
#     #cam_ctd, bbx_imgs,img_feats = form_bbx_img_feat(img_model,image,img_bbs_ctd,img_bbs_all,bbox_imgs_all,img_idx)
#     cam_ctd, bbx_imgs,img_feats,classes = Generate_bbx_img_feat(img_model,img_bbs_ctd,img_bbs_all,bbox_imgs_all,img_idx)
#     #matched_idx = common_feature_match(radar_ctd_cp,cam_ctd,dtp_sensorfusion,bbox_rad_list, bbx_imgs,img_feats,classes)#get sensorfusion_ctd
#     sensorfusion_ctd, bbx_imgs_sf,img_feats_sf,classes_sf,sensor_label = construct_sensorfusion_centroids(radar_ctd_cp,cam_ctd,dtp_sensorfusion, bbx_imgs,img_feats,classes)#get sensorfusion_ctd
#     _,out_img_sf,sf_ctd,_,_= distance_on_BEV_sensorfusion(rnn_model_cam,scl_cam,img_model,image.copy(),i,sensorfusion_ctd,dtp_sensorfusion, bbx_imgs_sf,img_feats_sf,classes_sf,sensor_label)
#     #one_frame_acc(dtp_sensorfusion,gps_all_for_radar,total_dist_sf,i)
    
#     ## save video
#     #rcsf_frame=plot_muti_imgs(i,out_img_cam,out_img_rad,out_img_sf)
#     #rcsf_frames.append(rcsf_frame)# save video
#     rcsf_frames.append(out_img_sf)# save video
    
#     ## clear mem
#     gc.collect()    

# end = time.perf_counter()
# print('run time:{} secs'.format(end-start))    

# # # write video
# file_path='./out_videos/SensorFusion_rcsf.avi'    
# write_video(file_path, rcsf_frames, fps=30)
    

