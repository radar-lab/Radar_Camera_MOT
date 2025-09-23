#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lei Cheng
"""
import rosbag
from bagpy import bagreader
import pandas as pd
import numpy as np
import cv2
from scipy import stats
import matplotlib.pyplot as plt
import datetime
import io
import math
import statistics


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


def element_wise_euclidian_dist(a, b):
    '''
    Element-wise Euclidian distances between 2 numpy arrays
    return nx1 numpy array
    '''
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b, axis=1)
    return dist
#######################################################################################
'''
Load ROS Bag Data
'''
################################### load radar_bag
b = bagreader('/home/kevalen/AWR1843_correction_ws/test_radar_2022-01-23-14-27-01.bag')
# replace the topic name as per your need: '/sony_radar/radar_scan' or '/ti_mmwave/radar_scan'
RADAR_MSG = b.message_by_topic('/ti_mmwave/radar_scan')  #return csv file
df_radar = pd.read_csv(RADAR_MSG)
################################### load img_bag
img_timesp = []
img_bag = rosbag.Bag('/home/kevalen/AWR1843_correction_ws/test_cam_2022-01-23-14-26-55.bag')
for topic, msg, t in img_bag.read_messages(topics=['/usb_webcam/image_raw/compressed']):     
    img_timesp.append(msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9)
    if len(img_timesp)>0:
        break;
img_bag.close()

img_start_time=img_timesp[0]  #img_start_time=1661136097.393281
#df_radar.radar_frame_idx[0]
#################################### filtering radar data per your need
#df_f=df_radar
df_f=df_radar[(df_radar.velocity==0)&(df_radar.range>1.5)]

#################################### radar & img timestamp
#### complete time: sec+msec from header
img_time=img_start_time+np.array(timestamps, float)/1000  # the timestamps need to be generated first by using the mouse click.
radar_time=(df_f['header.stamp.secs']+df_f['header.stamp.nsecs']*1e-9).values
# #Align the time of two bags
# img_time_start=img_time[0]
# img_time_last=img_time[-1]
# radar_time_crop=radar_time[(radar_time>img_time_start) & (radar_time<img_time_last)]
#################################### associate img and radar
idx,time_associate=find_nearest_betw_arr(radar_time, img_time)
#### integer time: only sec, without msec
time_associate_i=(time_associate//1).astype(int) #exact division(integer division) and rounding down
#################################### filter radar data by using INTeger time_associate:  Adopted
index_time=time_associate_i.tolist()
radarpoints_i=[]
thres=1
for i in range(len(index_time)):
    ind=index_time[i]    

    df_t=df_f[((df_f['header.stamp.secs']+df_f['header.stamp.nsecs']*1e-9)>=ind-1) & ((df_f['header.stamp.secs']+df_f['header.stamp.nsecs']*1e-9)<=ind+1)]# data within 3 secs
    ######remove outlier
    if (np.abs(stats.zscore(df_t.x)).notna().all())&(not (df_t[(np.abs(stats.zscore(df_t.x)) < thres)]).empty):      
        df_t=df_t[(np.abs(stats.zscore(df_t.x)) < thres)] 
    if (np.abs(stats.zscore(df_t.y)).notna().all())&(not (df_t[(np.abs(stats.zscore(df_t.y)) < thres)]).empty):
        df_t=df_t[(np.abs(stats.zscore(df_t.y)) < thres)]
    if (np.abs(stats.zscore(df_t.z)).notna().all())&(not (df_t[(np.abs(stats.zscore(df_t.z)) < thres)]).empty):
        df_t=df_t[(np.abs(stats.zscore(df_t.z)) < thres)]        
        
    radarpoints_i.append([df_t.x.mean(),df_t.y.mean(),df_t.z.mean()])
    
######################################################################################
################ forming radar points
radar_3d=np.array(radarpoints_i)
radar_3d=radar_3d[~np.isnan(radar_3d).any(axis=1), :]#Remove rows containing missing values (NaN)
################ forming img_pixel_points
img_pixel_points=(np.array(pix_xy).astype(float)).tolist() # the pix_xy need to be generated first by using the mouse click.

##########################################################################################################################

#######################################################################################
'''
calcuate Extrinsic_matrix
'''
#######################################################################################
############ The intrinsics matrix needs to be given in pixel units
cameraMatrix = np.array([[554.4203610089122, 0.        , 299.0464166708532],
                         [  0.       ,  556.539219672516, 265.177086523325],
                         [  0.       ,    0.         ,  1.        ]])
############# distortion coefficents
dist_coeffs = np.array([-0.3941065587817811, 0.1667170598953747, -0.003527054281471521, 0.001866412711427509, 0]).reshape(5,1) # for the usb_webcam1


'''
####################solvePnPRansac uses Random Sample Consensus ( RANSAC ) for robust estimation
'''
#The reprojectionError value is the maximum allowed distance between the observed and computed point projections to consider it an inlier.
success_ITERATIVE, rotation_vector_ITERATIVE, translation_vector_ITERATIVE, inliers_ITERATIVE = cv2.solvePnPRansac(radar_3d, np.array(img_pixel_points), cameraMatrix, dist_coeffs, iterationsCount = 10000, flags=cv2.SOLVEPNP_ITERATIVE)
projected_rad2d, jacobian = cv2.projectPoints(radar_3d, rotation_vector_ITERATIVE, translation_vector_ITERATIVE, cameraMatrix, dist_coeffs)
error_ITERATIVE = cv2.norm(np.expand_dims(np.array(img_pixel_points),axis=1), projected_rad2d, cv2.NORM_L2)/len(projected_rad2d)# also can use AED

success_SQPNP, rotation_vector_SQPNP, translation_vector_SQPNP, inliers_SQPNP = cv2.solvePnPRansac(radar_3d, np.array(img_pixel_points), cameraMatrix, dist_coeffs,flags=cv2.SOLVEPNP_SQPNP)
projected_rad2d, jacobian = cv2.projectPoints(radar_3d, rotation_vector_SQPNP, translation_vector_SQPNP, cameraMatrix, dist_coeffs)
error_SQPNP = cv2.norm(np.expand_dims(np.array(img_pixel_points),axis=1), projected_rad2d, cv2.NORM_L2)/len(projected_rad2d)
if error_ITERATIVE<error_SQPNP:
    rotation_vector, translation_vector, inliers =rotation_vector_ITERATIVE, translation_vector_ITERATIVE, inliers_ITERATIVE
else:
    rotation_vector, translation_vector, inliers =rotation_vector_SQPNP, translation_vector_SQPNP, inliers_SQPNP    
rotation_vector, translation_vector = cv2.solvePnPRefineLM(radar_3d, np.array(img_pixel_points), cameraMatrix, dist_coeffs, rotation_vector, translation_vector)
#rotation_vector, translation_vector = cv2.solvePnPRefineVVS(radar_3d, np.array(img_pixel_points), cameraMatrix, dist_coeffs, rotation_vector, translation_vector)
##########  get inliers from radar and img
inliers =np.squeeze(inliers)
radar_inliers =[]
img_inliers =[]
for i in range(len(inliers)):
    idx=inliers[i]
    radar_inliers.append(radar_3d[idx].tolist())
    img_inliers.append(img_pixel_points[idx])

'''
########################################################### Re-projection Error
'''
projected_rad2d, jacobian = cv2.projectPoints(radar_3d, rotation_vector, translation_vector, cameraMatrix, dist_coeffs)
ed=element_wise_euclidian_dist(np.array(img_pixel_points), np.squeeze(projected_rad2d))
aed=np.mean(ed)
cdsd=statistics.stdev(ed) #the corrected distance standard deviation(CDSD)
print("AED: {}; CDSD_error: {}".format(aed,cdsd))
##########for inliers from radar and img
# projected_rad2d, jacobian = cv2.projectPoints(np.array(radar_inliers), rotation_vector, translation_vector, cameraMatrix, dist_coeffs)
# ed=element_wise_euclidian_dist(np.array(img_inliers), np.squeeze(projected_rad2d))
# aed=np.mean(ed)
# cdsd=statistics.stdev(ed) #the corrected distance standard deviation(CDSD)
# print("For Inliers:\n")
# print("AED: {}; CDSD_error: {}".format(aed,cdsd))

#################   Counstruct Extrinsic_matrix  
# rotation_vector is 3x1 which needs to be converted to 3x3 using Rodrigues method of opencv.
R = cv2.Rodrigues(rotation_vector)[0]  #cv2.Rodrigues Converts a rotation matrix to a rotation vector or vice versa. [0] is dst,and[1] is jacobian
t = translation_vector
Rt = np.concatenate([R, t], axis=-1)  # Extrinsic_matrix=[R|t]
P = np.matmul(cameraMatrix, Rt)  # Homo_matrix=K[R|t]

'''
######################################################### Visulization ###################
'''
img = cv2.imread('/home/kevalen/Desktop/images/background.jpg')
img_copy = img.copy()
for p in np.array(img_pixel_points):
    image_original = cv2.circle(img_copy, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
image_original_cp = image_original.copy()
for pp in np.squeeze(projected_rad2d):
    image_projection = cv2.circle(img_copy, (int(pp[0]), int(pp[1])), 3, (255, 0, 0), -1)
##########for inliers from radar and img   
# img_copy = img.copy() 
# for p in np.array(img_inliers):
#     image_original = cv2.circle(img_copy, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

# for pp in np.squeeze(projected_rad2d):
#     image_projection = cv2.circle(img_copy, (int(pp[0]), int(pp[1])), 3, (255, 0, 0), -1)
  

'''    
############################################# Show the resulting image  #####################################3    
'''
## show
cv2.imshow('points_original is Red; points_project is Blue', image_projection)
cv2.waitKey(10000)  
directory = '/home/kevalen/Desktop/images/'
nowTime = datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
# save original points
fileName = directory+'org-'+nowTime+'.jpg'
#cv2.imwrite(fileName,image_original_cp)
# save projection image
cv2.imwrite(directory+'re-projection-'+nowTime+'.jpg', image_projection)
cv2.destroyAllWindows()


'''    
############################################# physical measurement method  #####################################3    
'''
# R1 = np.eye(3)
# R1_vector = cv2.Rodrigues(R1)[0] #convert to rotation vector
# t1=np.array([0.,-0.045,0.])*1000;# (Note *1000 to convert to mm)
# Rt1 = np.concatenate([R1, np.expand_dims(t1,axis=1)], axis=-1)  # Extrinsic_matrix=[R|t]
# P_meas = np.matmul(cameraMatrix, Rt1)  # Homo_matrix=A[R|t]
# radarpoints_a = radar_3d.copy()
# radarpoints_a[:,0]= -radar_3d[:,1]
# radarpoints_a[:,1]= -radar_3d[:,2]
# radarpoints_a[:,2]= radar_3d[:,0]
# radar_world_points = np.insert((radarpoints_a*1000), 3, 1, axis=1) #insert 1 to final column #for milimeter
# r2img_points = np.dot(radar_world_points, np.array(P_meas.T))
# r2img_points_m = np.asmatrix(r2img_points) #convert arr to matrix so as to use Matrix Operations
# r2img_xy_points=r2img_points_m[:,0:2]/r2img_points_m[:,2] # [:,0:2] is right open,mean to chose 0 and 1 column.
# projected_rad2d= np.array(r2img_xy_points)
# ed=element_wise_euclidian_dist(np.array(img_pixel_points), np.array(r2img_xy_points))
# aed=np.mean(ed)
# cdsd=statistics.stdev(ed) #the corrected distance standard deviation(CDSD)
# print(" AED: {}; CDSD_error: {}".format(aed,cdsd))


'''
############## Code for real-world data projection and performence evaluation  ##################################################
'''
################################################ split radar data into frames
b = bagreader('/home/kevalen/sony_radar_2022-01-17-17-09-45.bag')# real-world car long
#b = bagreader('/home/kevalen/sony_radar_2022-01-17-16-54-45.bag')# real-world car short
#b = bagreader('/home/kevalen/sony_radar_2022-01-10-14-19-01.bag')# Walking on the rooftop
#b = bagreader('/home/kevalen/test_radar_2022-01-05-00-01-39.bag')# indoor walking 
# replace the topic name as per your need
##get the list of topics
print(b.topic_table)
RADAR_MSG = b.message_by_topic('/sony_radar/radar_scan')  #return csv file
df_radar = pd.read_csv(RADAR_MSG)
indx_fr=(df_radar[(df_radar.point_id==0)].index).tolist()
indx_fr.append(len(df_radar))  # so need not to change sth in for loop below
radar_fr=[]
ts_radar_fr=[]
x_radar_fr=[]
y_radar_fr=[]
z_radar_fr=[]
time_radar_fr=[]
for i in range(len(indx_fr)-1):
    radar_fr.append(df_radar[indx_fr[i]:indx_fr[i+1]])
    df_i=df_radar[indx_fr[i]:indx_fr[i+1]][df_radar.target_idx<253]#Remove Unassociated/Weak/Noise points
    if not df_i.empty:
        ts_radar_fr.append((df_i['header.stamp.secs']+df_i['header.stamp.nsecs']*1e-9).values)
        time_radar_fr.append(ts_radar_fr[-1]) #set last point in a frame as this frame's own time
        x_radar_fr.append(df_i.x.values)
        y_radar_fr.append(df_i.y.values)
        z_radar_fr.append(df_i.z.values)  
#################### load bounding_box_bag use rosbag  ,for getting xmin etc. ##################################################
# we can use cv2.rectangle to draw bounding box but not recommend
filter_flg = 1
filter_class = ['person','car','bus','truck'] # Filter desired Class
bbs = []
ts_bbs = []
ts_bbs_img= []
#bbs_bag = rosbag.Bag('/home/kevalen/test_yolo_2022-01-17-16-54-52.bag')# real-world car short
bbs_bag = rosbag.Bag('/home/kevalen/test_yolo_2022-01-17-17-09-43.bag')# real-world car long
#bbs_bag = rosbag.Bag('/home/kevalen/test_yolo_2022-01-10-14-19-07.bag')# Walking on the rooftop
for topic, msg, t in bbs_bag.read_messages(topics=['/darknet_ros/bounding_boxes']):
    bbox = []    
    ts_bbs.append(msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9)
    ts_bbs_img.append(msg.image_header.stamp.secs+msg.image_header.stamp.nsecs*1e-9)
    for i in range(len(msg.bounding_boxes)):
        if filter_flg & (msg.bounding_boxes[i].Class in filter_class): # will lead to have empty list bbox
            bbox.append(msg.bounding_boxes[i])
        elif filter_flg==0:
            bbox.append(msg.bounding_boxes[i])
    bbs.append(bbox) #usage: bbs[0][0].xmin
bbs_bag.close() 

####################      load img_bag use rosbag  ,it is more faster ##################################################
all_imgs = []
ts_img = []
#img_bag = rosbag.Bag('/home/kevalen/yolo_detect_img_2022-01-17-16-54-42.bag')# real-world car short
img_bag = rosbag.Bag('//home//kevalen//yolo_detect_img_2022-01-17-17-09-49.bag')# real-world car long
#img_bag = rosbag.Bag('//home//kevalen//yolo_detect_img_2022-01-10-14-18-50.bag')# Walking on the rooftop
for topic, msg, t in img_bag.read_messages(topics=['/darknet_ros/detection_image']):        
    ts_img.append(msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9)   
    im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)# -1 is in accordance with the original channel,the default is three-channel picture
    all_imgs.append(im)
img_bag.close()
########################################################################################################
######################associate detect_img and all radar points, not radar frame,resorts to bbx_img_ts
df_target=df_radar[df_radar.target_idx<253] #all radar points that target_idx<253 use for projection
df_xyz=df_target[['x','y','z']].values
ts_radar=(df_target['header.stamp.secs']+df_target['header.stamp.nsecs']*1e-9).values
############### associate bounding_boxes img_ts and all radar points
#Align the time of two bags i.e. crop the time to same size, for bbx,img,radar all should do this,but only need to crop the base_array(match_array)
ts_bbs_img_start=ts_bbs_img[0]
ts_bbs_img_last=ts_bbs_img[-1]
ts_radar_crop_bb=ts_radar[(ts_radar>ts_bbs_img_start) & (ts_radar<ts_bbs_img_last)]
idx_bb,ts_bbs_img_assoc=find_nearest_betw_arr(np.array(ts_bbs_img), ts_radar_crop_bb)
ts_bbs_assoc=np.array(ts_bbs)[idx_bb]# use idx_bb to extract ts_bbs_assoc directly
################associate bounding_boxes to  detect_img and then to all radar points
ts_img=np.array(ts_img)
idx_img,ts_img_assoc=find_nearest_betw_arr(ts_img, ts_bbs_assoc)
idx=idx_img
#################################################################################################
####################### Project radar to img by using cv2.projectPoints
# rotation_vector=np.array([[ 1.30927651],
#                           [-1.29183232],
#                           [ 1.09104368]])#test_2021-12-11-17-18-12.bag#main
# translation_vector=np.array([[-0.00115773],
#                               [-0.0608874 ],
#                               [-0.01496503]])
#test_2021-12-11-17-18-12.bag#main
r2imgpoints, jacobian = cv2.projectPoints(df_xyz, rotation_vector, translation_vector, cameraMatrix, dist_coeffs)
r2imgpoints=np.squeeze(r2imgpoints)
r2img_x_points=r2imgpoints[:,0]
r2img_y_points=r2imgpoints[:,1]

#########$################################## use physical measurements
radarpoints_a=df_xyz.copy()
R1 = np.eye(3)
R1_vector = cv2.Rodrigues(R1)[0] #convert to rotatin vector
t1=np.array([0.,-0.045,0.])*1000;
Rt1 = np.concatenate([R1, np.expand_dims(t1,axis=1)], axis=-1)  # Extrinsic_matrix=[R|t]
P1 = np.matmul(cameraMatrix, Rt1)  # Homo_matrix=A[R|t]
radarpoints_a[:,0]= -df_xyz[:,1]
radarpoints_a[:,1]= -df_xyz[:,2]
radarpoints_a[:,2]= df_xyz[:,0]
r2imgpoints, jacobian = cv2.projectPoints(radarpoints_a, R1_vector, np.expand_dims(t1/1000.,axis=1), cameraMatrix, dist_coeffs)
r2imgpoints=np.squeeze(r2imgpoints)
r2img_x_points_pm=r2imgpoints[:,0]
r2img_y_points_pm=r2imgpoints[:,1]
######################### with chess board for projection
R_che=np.array([[-0.02115827, -0.99855784, -0.04934136],
       [-0.99963428,  0.02029812,  0.01786897],
       [-0.01684166,  0.04970139, -0.99862212]])
t_che=np.array([0.12121375,1.74179495,1.48803506])
R_che_vector = cv2.Rodrigues(R_che)[0] #convert to rotatin vector
r2imgpoints, jacobian = cv2.projectPoints(df_xyz, R_che_vector, t_che, cameraMatrix, dist_coeffs)
r2imgpoints=np.squeeze(r2imgpoints)
r2img_x_points_che=r2imgpoints[:,0]
r2img_y_points_che=r2imgpoints[:,1]
###############################################################################
h,w,d = all_imgs[0].shape#img.shape[0] is height,
def write_video(file_path, frames, fps):
    """
    Writes frames to an video file
    :param file_path: Path to output video, must end with suffix(.mp4,.avi)
    :param frames: List of Image objects
    :param fps: Desired frame rate
    """
    h,w,d = frames[0].shape
    #AVI: cv2.VideoWriter_fourcc('M','J','P','G');  MP4: cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
    for frame in frames:
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
def plot3(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im
def canvas2rgb_array(canvas):
    """Adapted from: https://stackoverflow.com/a/21940031/959926"""
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()
    scale = round(math.sqrt(buf.size / 3 / nrows / ncols))
    return buf.reshape(scale * nrows, scale * ncols, 3)  
##################################### show only frames of image those have the corresponding radar points ##########################################################
dot_size=50  #10
all_imgs_copy=all_imgs.copy()#use for showing all images
test_img=[]
plot_imgs=[]
for i in range(len(ts_radar_crop_bb)): #we use len(ts_radar_crop) since we crop data and not use len(df_target)
#for i in range(1000):    
    if i==0:
        img_fr=all_imgs[idx[i]]
        j=i   #j is the start_idx of the radar points those belong to one img_frame
    elif idx[i-1]<idx[i]:
        ###Each time this if branch is run, the previous frame of image and the corresponding radar points are output first.
        plt.style.use('default')
        my_dpi=100
        fig = plt.figure(figsize=(1280 / my_dpi, 960 / my_dpi), dpi=my_dpi) #fig = Figure(figsize=(1024, 512), dpi=1) To render an image of a specific size
        #####if we don't want multiple plot we can comment the code for ax1 and ax2
        ax = fig.add_subplot(223)
        ax.margins(0)
        ax.imshow(img_fr)
        ax.set_xlim(0, w)
        ax.set_ylim(bottom=h, top=0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Radar projection(a)', fontsize=16)
        ax.scatter(r2img_x_points[j:i],r2img_y_points[j:i],s=dot_size)
        
        ax1 = fig.add_subplot(224)
        ax1.margins(0)
        ax1.imshow(img_fr)
        ax1.set_xlim(0, w)
        ax1.set_ylim(bottom=h, top=0)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title('Radar projection(b)', fontsize=16)  
        ax1.scatter(r2img_x_points_pm[j:i],r2img_y_points_pm[j:i],s=dot_size)
        # ax1.margins(0)
        # ax1.imshow(img_fr)
        # ax1.set_xlim(0, w)
        # ax1.set_ylim(bottom=h, top=0)
        # ax1.get_xaxis().set_visible(False)
        # ax1.get_yaxis().set_visible(False)
        # ax1.set_title('Radar projection(b)', fontsize=16)
        
        
        ax2 = fig.add_subplot(222)  #adopted
        ax2.margins(0)
        ax2.scatter(-df_xyz[j:i,1],df_xyz[j:i,0],s=dot_size)
        ax2.set_xlim(-4, 6)#road
        ax2.set_ylim(2, 21)        
        # ax2.set_xlim(-2, 2)# walking
        # ax2.set_ylim(0, 8)        
        ax2.set_xlabel('Azimuth', rotation=0, fontsize=14, labelpad=2)
        ax2.set_ylabel('Range', rotation=90, fontsize=14)
        #ax2.tick_params(size=14,labelsize=14)        
        ax2.grid()
        ax2.set_title('Radar 2D representation', fontsize=16) 
        
        ax3 = fig.add_subplot(221, projection='3d')
        ax3.margins(0)
        ax3.scatter(-df_xyz[j:i,1],df_xyz[j:i,0],df_xyz[j:i,2],s=dot_size)#yxz
        ax3.grid()        
        ax3.set_xlabel('Azimuth', rotation=0, labelpad=-15, fontsize=14)
        ax3.set_ylabel('Range', rotation=0, labelpad=-15, fontsize=14)
        ax3.set_zlabel('Elevation', rotation=0, labelpad=-15, fontsize=14)
        ax3.set_xlim(-4, 6)#road
        ax3.set_ylim(1, 19) 
        #ax3.set_zlim(-5, 5)        
        # ax3.set_xlim(-2, 2)# walking
        # ax3.set_ylim(0, 8) 
        # ax3.set_zlim(-5, 6)
        # Turn off tick labels
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_zticklabels([])
        ax3.set_title('Radar 3D detection', fontsize=16)        

        plot_img_np = get_img_from_fig(fig, dpi=300)
        test_img.append(plot_img_np)#use for showing images those have radarpoints
        all_imgs_copy[idx[i-1]]=plot_img_np #use for showing all images
        plt.tight_layout()
        plt.show()
        ###set img_frame as current frame
        img_fr=all_imgs[idx[i]]
        j=i #j is been reseted as the start_idx of the radar points those belong to next img_frame


nowTime = datetime.datetime.now().strftime('%Y%m%d-%H_%M_%S')
directory = '/home/kevalen/usb_webcam_ws/test5/'
file_path_name = directory+'walk-'+nowTime+'.avi'
cv2.waitKey(50)
#frames=all_imgs_copy
frames=test_img
fps=30
write_video(file_path_name, frames, fps)


###########################################  Count how many radar points are inside the bounding box to get calibration accuracy
def in_bbox(x,y,xmin,xmax,ymin,ymax):
    '''
    To determine whether the radar point (x, y) is located in the bounding box, and when it is in then return 1 otherwise 0.
    '''
    if((xmin<=x) & (x<=xmax) & (ymin<=y) & (y<=ymax)):        
        ret=1
    else:
        ret=0
    return ret


################# accuracy for each frame not all frames (((adopted
fr_total=len(bbs)
in_sum=0
cnt=0
accuracy_fr=[]
cnt_in_fr_all=[] #radar points num in one frame
for i in range(len(ts_radar_crop_bb)):
#for i in range(5000):
    decision=0    
    bbs_num_in_fr=len(bbs[idx_bb[i]])
    for j in range(bbs_num_in_fr): #Traverse all bounding boxes in a frame
        x=r2img_x_points[i]
        y=r2img_y_points[i]        
        # x=r2img_x_points_ma[i]
        # y=r2img_y_points_ma[i]     
        # x=r2img_x_points_pm[i]#phycial
        # y=r2img_y_points_pm[i]        
        xmin=bbs[idx_bb[i]][j].xmin
        xmax=bbs[idx_bb[i]][j].xmax
        ymin=bbs[idx_bb[i]][j].ymin
        ymax=bbs[idx_bb[i]][j].ymax
        ret=in_bbox(x,y,xmin,xmax,ymin,ymax)
        decision=decision+ret
    if decision>0: # one radar point may correspond to multiple bboxes
        in_sum=in_sum+1
        
    if i==0:
        k=i
        in_sum_temp=0
    elif idx_bb[i-1]<idx_bb[i]:  #new frame
        cnt_in_fr=i-k 
        k=i
        #accuracy_fr=(in_sum-in_sum_temp)/cnt_in_fr*100
        accuracy_fr.append((in_sum-in_sum_temp)/cnt_in_fr*100)
        cnt_in_fr_all.append(cnt_in_fr)
        print('the accuracy of frame %d is %f' % (cnt,accuracy_fr[cnt]))
        in_sum_temp=in_sum
        cnt=cnt+1

######## total accuracy for all frames by using accuracy_fr   adopted
accuracy_fr_old=np.array(accuracy_fr)
accuracy_fr_old=accuracy_fr_old[accuracy_fr_old!=0]#remove zeros
accuracy_all=accuracy_fr_old.mean()
print('accuracy_all',accuracy_all)
########################################## Visualizing Histograms  adopted
scene='g'
n, bins, patches = plt.hist(x=accuracy_fr_old, bins=10, color='#607c8e',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Accuracy(%)')
plt.ylabel('Frequency')
plt.title('Co-calibration Accuracy Distribution(%s)'%(scene))
plt.gcf().set_dpi(300)
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,}

plt.text(bins[0]+2, n.max()-2, 'Acc=%.2f%%'%(accuracy_all), fontdict=font)
plt.savefig('/home/kevalen/Desktop/images/hist_%s.png'%(scene), dpi = 300)









