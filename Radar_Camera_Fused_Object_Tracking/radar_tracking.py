#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lei
"""
############### Lei ###################################
import os, glob, shutil, sys
sys.path.append("./common_features_for_tracking/") # add search path: sys.path.append("../../")
sys.path.append("./ImageYolo/")
######################################################
import rosbag
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from DetectionAndTrackingProject import DetectionAndTrackingProject
from tracking_birds_eye_view_lc import create_birds_eye_view
import tensorflow as tf
import tensorflow.keras.backend as K
import joblib
#from RADYolo.predict_location_RAD import RADYolo_main

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#####################################################################################
from PIL import Image
from csv_UNIX_time import extract_start_time
from ImageYolo.predict import process_imgs_to_get_bboxes, load_radimg_yolo_model

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
   
def load_rad_model(model_path):
    print('Loading feat model...')
    model = tf.keras.models.load_model(model_path,custom_objects={'K':K}, compile=False)
    print('{} model loaded.'.format(model_path))
    return model
############################################## Compute centroids of clusters
def generate_clusters(data,eps=1.5,min_samples=8, metric='euclidean'):
    """Use DBSCAN or OPTICS to generate clusters.
    see:https://scikit-learn.org/stable/modules/clustering.html
    Parameters:
        data : array_like(dtype='float', shape=(n, 2))
            points: (x, y)
    Returns:
        labels : array(dtype='float', shape=(n, 1))
            clusters_labels of every points in the data.
    """
    #generate clusters
    clustering = DBSCAN(eps=eps,min_samples=min_samples,metric=metric).fit(data)
    #get labels
    labels = clustering.labels_
    return labels

def centroids_of_clusters(data,eps=1.5,min_samples=8, metric='euclidean'):
    """
    Parameters:
        data : array(shape=(n, 2))
            points: (x, y)
    Returns:
        centroids : array(shape=(m, 2))
            centroids of m clusters: (x, y).
    """
    #get labels, or say detections
    labels = generate_clusters(data,eps=eps,min_samples=min_samples,metric=metric)
    #print('labels=',labels)
    #get number of clusters : Noisy samples are given the label -1.
    #n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_clusters = np.max(labels)+1
    #print('n_clusters=',n_clusters)
    #get centroids
    centroids = []
    for i in range(n_clusters):
        cluster = data[labels == i]
        centroid=np.mean(cluster, axis=0)
        centroids.append(centroid.tolist())

    return np.array(centroids)


def append_1_to_homo(mat):
    hommat = np.append(mat, np.ones((mat.shape[0],1)), axis=1)
    return hommat

def project_3d_to_2d(polyline_world, K):
    pl_uv_cam = (K @ (polyline_world.T)).T
    u = pl_uv_cam[:,0] / pl_uv_cam[:,2]
    v = pl_uv_cam[:,1] / pl_uv_cam[:,2]
    return np.stack((u,v)).T



def proj_rad2img(rad_xyz, cali_method=1):
    rad_xyz = np.array(rad_xyz)
    ## intrinsic mat
    # cameraMatrix = np.array([[554.4203610089122, 0.,               299.0464166708532],
    #                           [  0.,              556.539219672516, 265.177086523325 ],
    #                           [  0.,              0.,               1.              ]])
    
    cameraMatrix = np.array([[545.7881532041737, 0,                314.9817965777312],#unit is 'mm'
                              [0,                 544.7004623674142,250.4216021567457],
                              [0,                 0,                                1]])
    #cameraMatrix = cameraMatrix.T
    ############# distortion coefficents
    dist_coeffs = np.array([-0.3941065587817811, 0.1667170598953747, -0.003527054281471521, 0.001866412711427509, 0]).reshape(5,1)
    #dist_coeffs = np.array([0., 0., 0., 0., 0.]).reshape(5,1)
    if cali_method:
        ##### my own calibration
        rotation_vector=np.array([[ 1.30927651],
                                  [-1.29183232],
                                  [ 1.09104368]])
        translation_vector=np.array([[-0.00115773],
                                     [-0.0608874 ],
                                     [-0.01496503]])
        r2imgpoints, jacobian = cv2.projectPoints(rad_xyz, rotation_vector, translation_vector, cameraMatrix, dist_coeffs)
        r2imgpoints=np.squeeze(r2imgpoints,axis=1)
    else:  #  cali_method=0
        ## use physical measurements
        radarpoints_a=rad_xyz.copy()
        R1 = np.eye(3)
        R1_vector = cv2.Rodrigues(R1)[0] #convert to rotatin vector
        t1 = np.array([0.,-0.04,0.])*1000
        radarpoints_a[:,0] = -rad_xyz[:,1]
        radarpoints_a[:,1] = -rad_xyz[:,2]
        radarpoints_a[:,2] =  rad_xyz[:,0]
        r2imgpoints, jacobian = cv2.projectPoints(radarpoints_a, R1_vector, np.expand_dims(t1/1000,axis=1), cameraMatrix, dist_coeffs)
        r2imgpoints = np.squeeze(r2imgpoints,axis=1)
        
        # Rt1 = np.concatenate([R1, np.expand_dims(t1,axis=1)], axis=-1)  # Extrinsic_matrix=[R|t]
        # P_meas = np.matmul(cameraMatrix, Rt1)  # Homo_matrix=A[R|t]
        # radar_world_points = np.insert((radarpoints_a*1000), 3, 1, axis=1) #insert 1 to final column #for milimeter
        # r2img_points = np.dot(radar_world_points, np.array(P_meas.T))
        # r2img_points_m = np.asmatrix(r2img_points) #convert arr to matrix so as to use Matrix Operations
        # r2img_xy_points=r2img_points_m[:,0:2]/r2img_points_m[:,2] # [:,0:2] is right open,mean to chose 0 and 1 column.
        # r2imgpoints= np.array(r2img_xy_points)
    
    return r2imgpoints
################################################################################################
"""## Transform the points of radar to the corresponding points on BEV image"""
#def distance_on_BEV_radar(image,i,centroids,dtp):
def distance_on_BEV_radar(rnn_model,scl,rad_model,image,i,img_ctd,dtp):
    image_height=image.shape[0]
    image_width=image.shape[1]
    height=1.635 #1.0 #1.635 #camera height
    pitch_deg=-5 #-3.2 #-5 #-3.2    
    K = np.array([[545.7881532041737, 0,                314.9817965777312],#unit is 'mm'
                  [0,                 544.7004623674142,250.4216021567457],
                  [0,                 0,                                1]])
    # K = np.array([[554.4203610089122, 0.,               299.0464166708532],
    #               [  0.,              556.539219672516, 265.177086523325 ],
    #               [  0.,              0.,               1.              ]])
    
    d_min=3.  #depth_min
    d_max=20.  #depth_max
    cr_lf=-6.  #cross range left
    cr_rt=-cr_lf  #cross range right
    
    BEV_img,plane_tm,IMAGE_W,IMAGE_H,x_pixel2meter_ratio,y_pixel2meter_ratio=create_birds_eye_view(K, image, d_min, d_max, cr_lf, cr_rt, 
                                                                                              height=height, yaw_deg=0, pitch_deg=pitch_deg, roll_deg=0,img_w_param=30)

    bbx_imgs  = [[]] # for radar, it should be empty 
    img_feats = np.array([[]]) # for radar, it should be empty 
    xy        = np.array([[]])
    
    ####### Transform distance to BEV pixel position, NOT Original img
    centroids= np.array(img_ctd)
    
    ####### Process radar centroids for tracking
    #img_cp=BEV_img.copy()
    img_cp=image.copy()
    out_img = dtp.DetectionByTracking_rad(rnn_model,scl,rad_model,img_cp,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,bbx_imgs,img_feats,centroids=centroids,d_max=d_max)
    ##cv2.putText(out_img, 'R-Frame Idx:%d'%i, (5, 5),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2., color=(255,0,0), thickness=1)
    # plt.imshow(out_img)
    # plt.gcf().set_dpi(100)
    # plt.title('R-Frame Idx:%d'%i)
    # plt.show()        
    return BEV_img,out_img,centroids,x_pixel2meter_ratio,y_pixel2meter_ratio
########################################################################################
def preprocessing_radar_data(bag_name='./track_1002_02.bag'):
    """This function is used for generate centroids. input: radar bag(or radar_frame_data)"""
    #################### load radar_data_bag use rosbag and split radar data into frames ##################################################
    total_frame_data = [] #all frames
    frame_data = []  #one frame
    frame_ts = []  #one frame's timestamp
    #bag = rosbag.Bag('./track_1002_calib.bag')
    bag = rosbag.Bag(bag_name)
    for topic, msg, t in bag.read_messages(topics=['/sony_radar/radar_scan']):
        if ((msg.point_id == 0) and (frame_data)):# initialize or refresh frame_list for first frame or next frame
            total_frame_data.append(frame_data) #append one frame, index by total_frame_data[frame_idx][point_idx][x,y,z,doppler,time]
            #frame_ts.append(frame_data[0][-1]) # set the frame's timestamp with the timestamp of the first radar point in that frame
            frame_ts.append((np.array(frame_data)[:,-1]).mean()) # set the frame's timestamp with the mean value of timestamps of all radar points in that frame
            frame_data = []
        if msg.target_idx <253: #Remove Unassociated/Weak/Noise points    
            #populate one frame with point whose target_idx <253  
            frame_data.append([msg.x, msg.y, msg.z, msg.doppler,msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9])    
    bag.close()
    ################################################ index total_frame_data[frame_idx][point_idx][x,y,z,doppler,time] with total_frame_data[1][0][1]
    ## list is not good for indexing, Should convert it to array
    
    xy_radar=[]
    for i in range(len(total_frame_data)):
        xy_radar2img= np.array([[]])#initialization
        one_frame=np.array(total_frame_data[i])
        xyzd_frame=one_frame[:,0:4]#Select only the first four columns:x,y,z,doppler
        #get centroids in one frame
        centroids=centroids_of_clusters(xyzd_frame,eps=1.5,min_samples=8, metric='euclidean')
        
        if centroids.size != 0:
            xy_radar2img=centroids[:,0:3]
        xy_radar.append(xy_radar2img)
        
    return xy_radar,frame_ts
############################################################################################
def radar_main():    
    '''
    ############################################   RUN   ##########################
    '''
    ############ Whether to save video
    SAVE_VIDEO_FLAG=1
    if SAVE_VIDEO_FLAG:
        video_frames=[]
        
    ############################ Load rad_model
    # rad_model_path = "model/assonet_model_hpc.h5" #change per need
    # rad_model=load_rad_model(rad_model_path)
    rad_model=[]
    # Load rnn_model
    rnn_model_path,scl_path = "model/prednet_rad_model_3d_3ts.h5" , 'model/radar_scaler_3d_3ts.joblib'
    rnn_model,scl=load_rnn_model(rnn_model_path,scl_path)
    #############################  Load deep-learning model   #############################
    ### Try to load model first.
    radimg_yolo_model  = load_radimg_yolo_model(confidence = 0.5,nms_iou = 0.3)
    
    ########################################################     
    ## Create the blank image
    blank_image = 255 * np.ones(shape=[256, 256, 3], dtype=np.uint8) #shape=(h, w, c),blank_image with self-defined size
    #blank_image = 255 * np.ones(shape=[480, 640, 3], dtype=np.uint8)
    # Create DetectionAndTracking instances
    dtp_radar = DetectionAndTrackingProject(mode=1)
    
    ###########################  Load radar  #############################
    path = "/xdisk/caos/leicheng/my_rawdata_0624/0624/"
    names = {'jiahao-hao'} #{'lei-leicar'} #{'jiahao-hao'} #
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
        
        for i in range(len(npy_paths)):
            # Read .png file using PIL (Pillow) library
            image = Image.open(png_paths[i])  
            # Convert image data to NumPy array
            image = np.array(image) #blank_image
            
            rad_bboxes, rad_centers, bbox_rads_all = process_radar_data(npy_paths[i], png_paths[i], confidence=0.5, nms_iou=0.3, yolo=radimg_yolo_model)
            
            # Get rad_center_list
            centroids = rad_centers[0]
            bbox_rad_list   = bbox_rads_all[0]  
            
            _,out_img,_,_,_=distance_on_BEV_radar(rnn_model,scl,rad_model,image,i,centroids,dtp_radar)
            ## save video
            if SAVE_VIDEO_FLAG:
                video_frames.append(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
        ### write video
        if SAVE_VIDEO_FLAG:
            file_path='./out_videos/rad_track_out_rnn.avi'    
            write_video(file_path, video_frames, fps=30)   
        ### print accuray
        print('Accuray: ', (dtp_radar.acc_rnn - dtp_radar.acc_kal)/len(npy_paths))
        
    ########################################################      
    # ##xy_radar_ctd,ts_radar_fr=preprocessing_radar_data()
    # # xy_radar_ctd,ts_radar_fr=RADYolo_main()
    # # with open("/xdisk/caos/leicheng/my_rawdata/RA_label" +'/RA_XY_label.npy' , 'wb') as f:
    # #     np.save(f, xy_radar_ctd)
    # with open("/xdisk/caos/leicheng/my_rawdata/RA_label" +'/RA_YX_label.npy' , 'rb') as f:
    #     xy_radar_ctd = np.load(f,allow_pickle=True)
    # with open("/xdisk/caos/leicheng/my_rawdata/RA_label" +'/RA_ts_label.npy' , 'rb') as f:
    #     ts_radar_fr = np.load(f,allow_pickle=True)  
    ########################################################     
    # for i in range(len(xy_radar_ctd)):
    #     centroids= xy_radar_ctd[i]    
    #     #distance_on_BEV_radar(blank_image,i,centroids,dtp_radar)
    #     _,out_img,_,_,_=distance_on_BEV_radar(rnn_model,scl,rad_model,blank_image,i,centroids,dtp_radar)
    #     ## save video
    #     if SAVE_VIDEO_FLAG:
    #         video_frames.append(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    # ### write video
    # if SAVE_VIDEO_FLAG:
    #     file_path='./out_videos/rad_track_out_rnn1.avi'    
    #     write_video(file_path, video_frames, fps=30)   
    # ### print accuray
    # print('Accuray: ', (dtp_radar.acc_rnn - dtp_radar.acc_kal)/len(xy_radar_ctd))
#####################################################################################
if __name__ == '__main__':
    ####################################################### main #########################
    radar_main()