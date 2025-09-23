import cv2, os
import numpy as np
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial import distance
from PIL import Image
import matplotlib.pyplot as plt

from utilities.Tracker import Tracker

import math

#***#########################################################################################***#
## use global variable for performance evaluation, just for evaluation, should be deleted when releasing the codes
use_loc_match=1 #only use location to do match
use_kalman=1
### kal+loc_match; kal+feat_match; rnn++loc_match; rnn+feat_match
## use BEV
use_BEV=0
##########################
names = ['lei-leicar'] #['hao-jiahao'] #['lei-leicar'] #['jiahao-hao'] #['lei_alone']#['2person-car']    
path = './common_features_for_tracking/' + names[0]
homography_matrix_up   = np.load(os.path.join(path, 'homography_matrix_cam2rad_up.npy'), allow_pickle=True)
homography_matrix_down = np.load(os.path.join(path, 'homography_matrix_cam2rad_down.npy'), allow_pickle=True)
#***#########################################################################################***#


def cam2rad_proj_updown(img_points,homography_matrix_up, homography_matrix_down, rad_img_shape, Y_thr = 320):
    # Convert to arrays    #Y_thr =  img_h - (img_h / 3)
    img_points = np.array(img_points)
   
    if img_points[1] <= Y_thr:
        ### project points UP
        projected_point = cv2.perspectiveTransform(img_points.reshape(-1, 1, 2), homography_matrix_up)    
    else:
        ### project points DOWN
        projected_point = cv2.perspectiveTransform(img_points.reshape(-1, 1, 2), homography_matrix_down)    
    projected_point = np.squeeze(projected_point).tolist() 
    # Clamp the projected_point within the radar image dimensions
    height, width = rad_img_shape[:2]
    projected_point = [
        min(max(projected_point[0], 0), width - 1), #w
        min(max(projected_point[1], 0), height - 1)  
    ]
    return projected_point

def calculate_distance(x, y, range_resolution=0.244, azimuth_resolution_deg=15, r=256, a=256,maximum_unambiguous_range=25.0):
    # r, a is the size of the range-azimuth map
    # Calculate angle
    angle = (x - (a / 2)) * (180 / a) 
    
    range_quantization = maximum_unambiguous_range / r
    #actual_distance = range_resolution * y
    actual_distance = range_quantization * (r - y)
    
    # Calculate radial/depth distance
    y_distance = actual_distance * math.cos(math.radians(angle))

    # Calculate actual horizontal/cross distance
    x_distance = actual_distance * math.sin(math.radians(angle))  

    return x_distance, y_distance
#***#########################################################################################***#
#***#########################################################################################***#

class DetectionAndTrackingProject:
    def __init__(self, min_conf=0.6, max_unmatched=10, min_matched=5, age_threshold=5,mode=0):
        # Initialize constants
        self.mode = mode  # mode=0: camera; 1: radar; 2:sensor_fusion
        if self.mode==0:
            #self.max_unmatched = 10
            self.cam_trc_loc = []
        elif self.mode==1:
            self.rad_trc_loc = []            
        else:
            #self.max_unmatched = max_unmatched
            self.sf_trc_loc = []
        self.max_unmatched = max_unmatched       # no. of consecutive unmatched detection before a track is deleted
        self.min_matched = min_matched           # no. of consecutive matches needed to establish a track
        self.age_threshold = age_threshold                   
        self.tracker_list = []
        self.count = 0
        self.classes = []
        #leicheng
        self.trackId = 0
        self.removed_trackID = []
        #self.tracker_img_list = []
        self.acc_rnn = 0
        self.acc_kal = 0
        self.match_count = 0
        self.acc_rnn_gps = 0
        self.acc_kal_gps = 0
        
   
    @staticmethod    
    def rad3d_to_img2d(centroids,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max):
        centroids = np.array(centroids)
        if centroids.size:#detections is not empty
            if centroids.ndim>1:
                xy_centroids=centroids[:,0:2]#Select only the first two columns:x,y
                ######## convert radar_coordinates to img_coordinates
                centroids=xy_centroids[:,[1,0]]#Swap the columns x and y
                centroids[:,0]= -centroids[:,0]#flip the x-axis
                cr_distance_mid=x_pixel2meter_ratio*(IMAGE_W/2)
                pix_ctd_x = (centroids[:,0]+cr_distance_mid)/x_pixel2meter_ratio
                pix_ctd_y = (d_max-centroids[:,1])/y_pixel2meter_ratio
                centroids[:,0]=pix_ctd_x  ###centroids is np.array([[]]),this will change the original centroids array to pixel values
                centroids[:,1]=pix_ctd_y  # return centroids will return the pixel positions
            else:
                xy_centroids=centroids[0:2]#Select only the first two columns:x,y
                ######## convert radar_coordinates to img_coordinates
                centroids=xy_centroids[[1,0]]#Swap the columns x and y
                centroids[0]= -centroids[0]#flip the x-axis
                cr_distance_mid=x_pixel2meter_ratio*(IMAGE_W/2)
                pix_ctd_x = (centroids[0]+cr_distance_mid)/x_pixel2meter_ratio
                pix_ctd_y = (d_max-centroids[1])/y_pixel2meter_ratio
                centroids[0]=pix_ctd_x  ###centroids is np.array([[]]),this will change the original centroids array to pixel values
                centroids[1]=pix_ctd_y  # return centroids will return the pixel positions
        return centroids
            
            
    def RNN_KalM_Acc(self,rnn_pos, kal_pos,real_pos,mode=1):
        self.match_count = self.match_count + 1
        if mode: #rnn and kal
            dist_rnn=distance.euclidean(rnn_pos, real_pos)
            dist_kal=distance.euclidean(kal_pos, real_pos)
            self.acc_rnn = self.acc_rnn + dist_rnn
            self.acc_kal = self.acc_kal + dist_kal
        else: #only kalm
            dist_rnn=0
            dist_kal=distance.euclidean(kal_pos, real_pos)
            self.acc_rnn = 0
            self.acc_kal = self.acc_kal + dist_kal            
        return dist_rnn,dist_kal
        
    @staticmethod
    def split_RNN_dataset(data,timestamps, pred_n=1, with_label_flag=1):
        # Take the data of every timestamps as X; the data of the timestamps+1 as label-Y
        data = np.array(data)
        X,Y = [],[]
        if with_label_flag:# with labels
            if (len(data)>=4): 
                for i in range(len(data) - timestamps):
                    X.append(data[i : (i+timestamps)])
                    Y.append(data[(i+timestamps) : (i+timestamps+pred_n)])
                ## last one
                i=i+1
                X.append(data[i : (i+timestamps)])
                Y.append(data[(i+timestamps-pred_n) : (i+timestamps)])#make the last value as the label, remember to filter out this one in any post-process
                return np.array(X),np.array(Y)
            else: # first one, without labels
                for i in range(len(data) - timestamps +1):
                    X.append(data[i : (i+timestamps)])
                    #Y.append(data[(i+timestamps-pred_n) : (i+timestamps)])
                    Y.append(np.array([[]]))# make the empty value as the label
                return np.array(X),np.array(Y) 
        else: #without labels
            for i in range(len(data) - timestamps +1):
                X.append(data[i : (i+timestamps)])
            return np.array(X) 
            
        
    @staticmethod    
    def pred_use_rnn_model(model,scl,cent,rnn_label, batch_size=1):
        predicted_value = model.predict(cent, batch_size=batch_size)
        predicted_value = np.concatenate(predicted_value,axis=1)
        predicted_value = scl.inverse_transform(predicted_value)
        
        rnn_label=np.concatenate(rnn_label,axis=0)
        if rnn_label.size==0:  # for first element
            real_value = predicted_value
        else:
            real_value = scl.inverse_transform(rnn_label)
        # plt.plot(real_value[:,0], color='red', label='Real Values')
        # plt.plot(predicted_value[:,0], color='blue', label='Predicted Values')
        return predicted_value,real_value       
    
    @staticmethod    
    def l2_normalize(arr,epsilon=1e-16): 
        # tf.math.l2_normalize
        arr=np.array(arr)
        # a_norm = np.linalg.norm(arr)
        # a_normalized = arr/(a_norm+epsilon) # a_normalized = arr/(np.linalg.norm(arr) + 1e-16)
        a_normalized = arr / ( np.sqrt(np.sum(arr**2)) + epsilon )
        return a_normalized
    @staticmethod    
    def resize_image(image, size, letterbox_image):
        iw, ih  = image.size
        w, h    = size
        if letterbox_image:
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)    
            image   = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image
    
    @staticmethod    
    def preprocess_input(image,rescale_type=0): #scale
        if rescale_type==0:
            image /= 255.0 # rescale to [0,1] 
        elif rescale_type==1:  
            image = (image/ 127.5) - 1 # rescale to [-1,1]
        return image
    
    def process_img(self,image,img_target_shape=(160,160)):
        ### old_version
        image = Image.fromarray(image.astype(np.uint8))
        image = self.resize_image(image, img_target_shape, letterbox_image=True)
        image = self.preprocess_input(np.array(image, dtype='float32'),rescale_type=0)
        # ### new_version
        # image=tf.image.resize_with_pad(image,target_height=img_target_shape[0],target_width=img_target_shape[1],method='bicubic',antialias=False)
        # image = self.preprocess_input(np.array(image, dtype='float32'),rescale_type=1)
        return image
#########################################################################################################################################       
    global use_loc_match
    if use_loc_match:
        # Method: Used to match detections to trackers
        def match_detections_to_trackers(self,img_model,trackers, detections, trk_bbx_imgs, det_bbx_imgs,trk_img_feats, det_img_feats,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max,max_dis_cost=25, img_dist_thr=[0.45,0.7]):
            # Initialize 'cost_matrix'
            cost_matrix = np.zeros(shape=(len(trackers), len(detections)), dtype=np.float32)
            #cost_matrix_img = cost_matrix.copy()
            #cost_matrix_total = cost_matrix_img.copy()
            #cost_matrix_feat = cost_matrix.copy()
            row_ind, col_ind = [], []
            
            
            # if self.mode==1:#radar to img
            #     trackers  = self.rad3d_to_img2d(trackers,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)
            #     detections= self.rad3d_to_img2d(detections,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)## RNN posi
            
    
                    
            # Populate 'cost_matrix'
            for t, tracker in enumerate(trackers):#trackers, detections just positions
                for d, detection in enumerate(detections):
                    cost_matrix[t,d] = distance.euclidean(tracker, detection)
                    #cost_matrix1 = self.l2_normalize(cost_matrix)/2.0
    
            if cost_matrix.size:                
                row_ind_bbx, col_ind_bbx = linear_assignment(cost_matrix) #`row_ind array` for `tracks`,`col_ind` for `detections`                
                row_ind, col_ind = row_ind_bbx, col_ind_bbx
    
            # Populate 'unmatched_trackers' with index
            unmatched_trackers = []  #(row_ind_total!=row_ind_img).all()
            for t in np.arange(len(trackers)):
                if t not in row_ind: #`row_ind` for `tracks`
                    unmatched_trackers.append(t)#plt.imshow(trk_bbx_imgs[t])
    
            # Populate 'unmatched_detections'
            unmatched_detections = []
            for d in np.arange(len(detections)):
                if d not in col_ind:#`col_ind` for `detections`
                    unmatched_detections.append(d)#plt.imshow(det_bbx_imgs[d].astype(np.uint8))
    
            # Populate 'matches'
            matches = []
            for t_idx, d_idx in zip(row_ind, col_ind):
                # Create tracker if cost is less than 'max_dis_cost'
                # Check for cost distance threshold.
                # If cost is very high then unmatched (delete) the track
                if (cost_matrix[t_idx,d_idx] >= max_dis_cost) :
                    unmatched_trackers.append(t_idx)
                    unmatched_detections.append(d_idx) 
                elif (np.array(self.classes).size) and (self.classes[d_idx] != '') and (np.array(self.tracker_list[t_idx].classes).size) and (self.tracker_list[t_idx].classes != '') and (self.tracker_list[t_idx].classes != self.classes[d_idx]):
                        unmatched_trackers.append(t_idx)
                        unmatched_detections.append(d_idx)                    
                else:
                    matches.append([t_idx, d_idx])
    
            # Return matches, unmatched detection and unmatched trackers
            return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)
        
    else: #feat_match
        # Method: Used to match detections to trackers
        def match_detections_to_trackers(self,img_model,trackers, detections, trk_bbx_imgs, det_bbx_imgs,trk_img_feats, det_img_feats,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max,max_dis_cost=25, img_dist_thr=[0.45,0.7]):
            # Initialize 'cost_matrix'
            cost_matrix = np.zeros(shape=(len(trackers), len(detections)), dtype=np.float32)
            cost_matrix_img = cost_matrix.copy()
            cost_matrix_total = cost_matrix_img.copy()
            cost_matrix_feat = cost_matrix.copy()
            row_ind, col_ind = [], []
            
            
            if self.mode==0:#img    
                # Populate 'cost_matrix_img'
                for t, tra_img in enumerate(trk_img_feats):#trackers, detections just positions
                    for d, det_img in enumerate(det_img_feats):
                        if np.array(tra_img).size and np.array(det_img).size:
                            dists = np.linalg.norm(tra_img - det_img, axis=-1)#or dists=distance.euclidean(feat_tra[0],feat_det[0])
                            cost_matrix_img[t,d] = dists
                        else:
                            cost_matrix_img[t,d] = 10000.
                        
            elif self.mode==2:# sensor fusion
                # Populate 'cost_matrix_img'
                for t, tra_img in enumerate(trk_img_feats):#trackers, detections just positions
                    for d, det_img in enumerate(det_img_feats):
                        if np.array(tra_img).size and np.array(det_img).size:
                            dists = np.linalg.norm(tra_img - det_img, axis=-1)#or dists=distance.euclidean(feat_tra[0],feat_det[0])
                            cost_matrix_img[t,d] = dists
                        else:
                            cost_matrix_img[t,d] = 10000.
           
                        
            # elif self.mode==1:#radar to img
            #     trackers  = self.rad3d_to_img2d(trackers,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)
            #     detections= self.rad3d_to_img2d(detections,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)## RNN posi
            
    
                    
            # Populate 'cost_matrix'
            for t, tracker in enumerate(trackers):#trackers, detections just positions
                for d, detection in enumerate(detections):
                    cost_matrix[t,d] = distance.euclidean(tracker, detection)
                    #cost_matrix1 = self.l2_normalize(cost_matrix)/2.0
    
            if cost_matrix.size:
                # ###11/29 how to Compare 2 variables of different scales; mahalanobis?
                #max_bbx=np.amax(cost_matrix)+1e-16
                #max_img=np.amax(cost_matrix_img)+1e-16
                #cost_matrix2=cost_matrix_img*(max_bbx/max_img)
                #cost_matrix_img1=cost_matrix*(max_img/max_bbx)
                interval_bbx=np.amax(cost_matrix)-np.amin(cost_matrix)+1e-16
                interval_img=np.amax(cost_matrix_img)-np.amin(cost_matrix_img)+1e-16
                loc2im_ratio=interval_img/interval_bbx		
                #cost_matrix2=cost_matrix_img*(interval_bbx/interval_img)
                cost_matrix_loc2img=cost_matrix*loc2im_ratio# map cost_matrix to cost_matrix_img
                
                ## build cost_martix_total: best_threshold is determined by test_dataset
                # when img_dist<=img_dist_thr[0] & >=img_dist_thr[1], the costs are only determined by img_dist; otherwise, determined by img_dist*ratio+bbx_dist*(1-ratio)
                cost_matrix_total = cost_matrix_img.copy()
                #max_dist_cond=(cost_matrix_img<(max_dis_cost*loc2im_ratio))
                same_im_cond=(cost_matrix_img<=img_dist_thr[0])
                diff_im_cond=(cost_matrix_img>=img_dist_thr[1])
                uncertain_im_cond=(cost_matrix_img>img_dist_thr[0]) & (cost_matrix_img<img_dist_thr[1])# (~ same_im_cond) & (~ diff_im_cond)
                cost_ratio1=0.9
                cost_matrix_total[same_im_cond] = cost_matrix_img[same_im_cond]*(cost_ratio1) + (1-cost_ratio1)*cost_matrix_loc2img[same_im_cond]
                cost_matrix_total[diff_im_cond] = cost_matrix_img[diff_im_cond]*(cost_ratio1) + (1-cost_ratio1)*cost_matrix_loc2img[diff_im_cond]
                cost_ratio2=0.8
                cost_matrix_total[uncertain_im_cond] = cost_matrix_img[uncertain_im_cond]*(cost_ratio2) + (1-cost_ratio2)*cost_matrix_loc2img[uncertain_im_cond]
                
                if self.mode==0:
                    # Produce matches by using the Hungarian algorithm to minimize the cost_distance
                    # Since linear_assignment try to minimize the cost default, to maximize the sum of IOU will need to negative IOU cost 
                    row_ind_bbx, col_ind_bbx = linear_assignment(cost_matrix) #`row_ind array` for `tracks`,`col_ind` for `detections`
                    row_ind_img, col_ind_img = linear_assignment(cost_matrix_img)
                    row_ind_total, col_ind_total = linear_assignment(cost_matrix_total)
                    
                    row_ind, col_ind = row_ind_bbx, col_ind_bbx
                    row_ind, col_ind = row_ind_img, col_ind_img
                    row_ind, col_ind = row_ind_total, col_ind_total
                    
                elif self.mode==1:
                    row_ind_bbx, col_ind_bbx = linear_assignment(cost_matrix) #`row_ind array` for `tracks`,`col_ind` for `detections`                
                    row_ind, col_ind = row_ind_bbx, col_ind_bbx

                elif self.mode==2 :  #sensor fusion
                    ### only loc match for sf
                    # row_ind_bbx, col_ind_bbx = linear_assignment(cost_matrix) #`row_ind array` for `tracks`,`col_ind` for `detections`                
                    # row_ind, col_ind = row_ind_bbx, col_ind_bbx
                    ### feat match for sf
                    row_ind_bbx, col_ind_bbx = linear_assignment(cost_matrix) #`row_ind array` for `tracks`,`col_ind` for `detections`
                    row_ind_img, col_ind_img = linear_assignment(cost_matrix_img)
                    row_ind_total, col_ind_total = linear_assignment(cost_matrix_total)
                    
                    row_ind, col_ind = row_ind_bbx, col_ind_bbx
                    row_ind, col_ind = row_ind_img, col_ind_img
                    row_ind, col_ind = row_ind_total, col_ind_total
                    
            # Populate 'unmatched_trackers' with index
            unmatched_trackers = []  #(row_ind_total!=row_ind_img).all()
            for t in np.arange(len(trackers)):
                if t not in row_ind: #`row_ind` for `tracks`
                    unmatched_trackers.append(t)#plt.imshow(trk_bbx_imgs[t])
    
            # Populate 'unmatched_detections'
            unmatched_detections = []
            for d in np.arange(len(detections)):
                if d not in col_ind:#`col_ind` for `detections`
                    unmatched_detections.append(d)#plt.imshow(det_bbx_imgs[d].astype(np.uint8))
    
            # Populate 'matches'
            matches = []
            if self.mode==0:
                for t_idx, d_idx in zip(row_ind, col_ind):
                    # Create tracker if cost is less than 'max_dis_cost'
                    # Check for cost distance threshold.
                    # If cost is very high then unmatched (delete) the track
                    if (cost_matrix[t_idx,d_idx] >= max_dis_cost) or (cost_matrix_img[t_idx,d_idx] >=img_dist_thr[1]):
                        unmatched_trackers.append(t_idx)
                        unmatched_detections.append(d_idx)                
                    else:
                        matches.append([t_idx, d_idx])
            elif self.mode==1 or self.mode==2:
                for t_idx, d_idx in zip(row_ind, col_ind):
                    # Create tracker if cost is less than 'max_dis_cost'
                    # Check for cost distance threshold.
                    # If cost is very high then unmatched (delete) the track
                    if (cost_matrix[t_idx,d_idx] >= max_dis_cost):
                        unmatched_trackers.append(t_idx)
                        unmatched_detections.append(d_idx)                
                    else:
                        matches.append([t_idx, d_idx])    
            # Return matches, unmatched detection and unmatched trackers
            return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)


########################################################************************************************#######################################################################
    global use_kalman
    if use_kalman:
        ############################################################## camera ################################################################
        # Detect first and then track
        def DetectionByTracking(self,rnn_model,scl, img_model, image,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,bbx_imgs,img_feats,classes,label_sf,centroids,d_max=20.,sensor_label=''):
            dims = image.shape[:2]
            self.count += 1
            self.classes = classes
            #timestamps = 3
            # Get list of tracker bounding boxes and Get list of tracker bbox_imgs
            trk_position = []
            trk_bbx_imgs = []
            trk_img_feats = []
            if self.tracker_list:
                for tracker in self.tracker_list:
                    trk_position.append(tracker.position)## Kalman posi
                    trk_bbx_imgs.append(tracker.bbx_img)
                    trk_img_feats.append(tracker.img_feat)
    
            # # Load img_model
            # img_model=self.load_img_model(self.img_model_path)
            
            # Match detected to trackers, set max_dis_cost per your need
            if centroids.size:#detections is not empty
                matched, unmatched_dets, unmatched_trks = \
                      self.match_detections_to_trackers(img_model,trk_position, centroids, trk_bbx_imgs, bbx_imgs,trk_img_feats, img_feats,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max,max_dis_cost=80)
            else: #detections is empty, only do prediction; Set all tracks as unmatched
                matched = np.empty((0, 2), dtype=int)
                unmatched_dets = np.empty((0, 1), dtype=int)
                unmatched_trks = np.arange(len(trk_position))
                
    
            # Deal with Matched detections:predict_and_update; need to count age of matched of the tracker and reset the age of unmatched for the matched tracker
            if len(matched) > 0:
                for trk_idx, det_idx in matched:
                    temp_trk = self.tracker_list[trk_idx] #take out the corresponding track                
                    temp_trk.num_matched += 1 # set flag for this track, num_matched is how many times(ages) of this same tracker are been matched
                    temp_trk.num_unmatched = 0 # set 0 for Continuous counting unmatched, num_unmatched is how many times(ages) of this same tracker are been Continuous unmatched, once it mtached, the unmathed is been reset to 0                

    
                    ### location
                    #z = det_boxes[det_idx]
                    z = centroids[det_idx]# det_centroids of next frame
                    z = np.expand_dims(z, axis=0).T
                    temp_trk.predict_and_update(z) # based on z, to do update and pred
                    xx = temp_trk.x_previous.T[0].tolist() # get pred_location
                    xx = [xx[0], xx[2]]
                    temp_trk.position = xx # put pred_location into track_list
                    if classes[det_idx] != '':  #when only exists radar detection, keep original track classes 
                        temp_trk.classes = classes[det_idx] # put class label into track_list
                    if (self.mode ==2):
                        temp_trk.sf_label = label_sf[det_idx]
    
                    if np.array(img_feats).size and np.array(img_feats[det_idx]).size:
                        ### img: update img as z_img(new det_img), no prediction for img
                        z_img = bbx_imgs[det_idx]                
                        temp_trk.bbx_img = z_img #org_img assign to track, and will be used to calculate the img cost matrix 
                        temp_trk.img_feat = img_feats[det_idx]
                    
                    #### RNN and Kalman accuracy after do position prediction
                    _,dist_kal=self.RNN_KalM_Acc(0, temp_trk.position[0:2], centroids[det_idx,0:2],mode=0)
                    temp_trk.dist_rnn.append(dist_kal)
                    print('cam-sf_rnn-kal_label-diff:',dist_kal,'\n')
    
            # Deal with Unmatched Detections: only predict(); need to assign track_ID to the tracker
            if len(unmatched_dets) > 0:
                for i in unmatched_dets:# i is index of centroids or say det_idx
                    # Create a new tracker
                    if  (any(self.removed_trackID)) and self.trackId>50: #reuse the ID  when removed_trackID is non-empty,and the ID exceeds 50
                        temp_trk = Tracker(trackId=self.removed_trackID.pop(), P= 100.0)  # Create a new tracker and reuse the ID 
                    else:
                        temp_trk = Tracker(trackId=self.trackId, P= 100.0)  # Create a new tracker
                        self.trackId += 1  # ID incremented by 1 
                                        
            
                    ### location
                    #z = det_boxes[i]
                    z = centroids[i]
                    z = np.expand_dims(z, axis=0).T
                    x = np.array([[z[0], 0, z[1], 0]],dtype=object).T #x=[x,x',y,y']
                    temp_trk.x_previous = x
                    temp_trk.predict()
                    xx = temp_trk.x_previous
                    xx = xx.T[0].tolist()
                    xx = [xx[0], xx[2]] #only extract [x,y] from [x,x',y,y']
                    temp_trk.position = xx #prediction assign to track, and will be used to calculate the cost matrix
                    temp_trk.classes = classes[i] # put class label into track_list
                    if (self.mode ==2):
                        temp_trk.sf_label = label_sf[i]
   
                    if np.array(img_feats).size and np.array(img_feats[i]).size:
                        ### img: keep org_img, no prediction for img
                        z_img = bbx_imgs[i]                
                        temp_trk.bbx_img = z_img #org_img assign to track, and will be used to calculate the img cost matrix                  
                        temp_trk.img_feat = img_feats[i]
                    ### populate the tracker_list
                    self.tracker_list.append(temp_trk)#plt.imshow(temp_trk.bbx_img)
    
            # Deal with Unmatched Tracks: only predict(); need to count age of unmatched of the tracker
            if len(unmatched_trks) > 0:
                for i in unmatched_trks:
                    temp_trk = self.tracker_list[i]
                    temp_trk.num_unmatched += 1 # Continuous num(times) of unmatched of this tracker
    
                    ### location
                    temp_trk.predict()
                    xx = temp_trk.x_previous
                    xx = xx.T[0].tolist()
                    xx = [xx[0], xx[2]]
                    temp_trk.position = xx          
                    ### img: keep img of this tracker unchanged, no prediction for img               
                    #temp_trk.bbx_img = temp_trk.bbx_img
    
    
            #leicheng-Find list of trackers to be deleted    
            # If tracks are not associated for long time, remove them        
            del_tracks = []
            for i in range(len(self.tracker_list)):
                #set track age condition
                track_age=self.tracker_list[i].num_unmatched+self.tracker_list[i].num_matched
                ##track_age_condition = track_age!=0 and track_age<=self.age_threshold and (self.tracker_list[i].num_matched/track_age)<0.5 # In the initial stage of the tracker, determine whether the tracker is reliable
                track_age_condition = track_age>=3 and track_age<=self.age_threshold and (self.tracker_list[i].num_matched/track_age)<0.5 # In the initial stage of the tracker, determine whether the tracker is reliable
                if (self.mode ==2) and ( (self.tracker_list[i].sf_label==1) or (self.tracker_list[i].sf_label==2)):  #
                    if self.tracker_list[i].num_unmatched > self.max_unmatched or track_age_condition:
                        del_tracks.append(i)
                        self.removed_trackID.append(self.tracker_list[i].id) #Populate 'removed_trackID'
                else:
                    if self.tracker_list[i].num_unmatched > self.max_unmatched:
                        del_tracks.append(i)
                        self.removed_trackID.append(self.tracker_list[i].id) #Populate 'removed_trackID'
            # del lost_track
            self.tracker_list=np.delete(np.array(self.tracker_list), del_tracks, None).tolist()
    
    
    
    
            # kalman pos : tracker.position    
            # Populate the list of Reliable trackers to be displayed on the image
            trc_loc = []            
            reliable_tracker_list = []
            for tracker in self.tracker_list:
                # Leicheng-Draw centroid on the image for debugging
                trackID_text = "%d" % tracker.id
                trackClass_text = tracker.classes
                center = tracker.position # predicted_center for Drawing not the measured_center(real detected)
                ####### Transform tracker.position_pixel to distance
                if use_BEV:
                    cr_distance_mid=x_pixel2meter_ratio*(dims[1]/2)
                    cr_distance = x_pixel2meter_ratio*center[0] - cr_distance_mid
                    dep_distance = d_max-y_pixel2meter_ratio*center[1]
                else:
                    if self.mode == 0:  #camera
                        center = cam2rad_proj_updown(center,homography_matrix_up, homography_matrix_down, image.shape, Y_thr = 320)
                    cr_distance, dep_distance = calculate_distance(center[0], center[1])
                tracker.distance_pos.append([cr_distance,dep_distance])
                position_text = "[%.2fm,%.2fm]" % (cr_distance, dep_distance)
                trc_loc.append([cr_distance,dep_distance])
                #########################################################################
                if tracker.num_matched >= self.min_matched and tracker.num_unmatched <=  (self.max_unmatched / 2):
                    #leicheng-reliable_tracker
                    reliable_tracker_list.append(tracker)
                    # Draw centroid on the image  #center[1]-radius
                    cv2.putText(image, sensor_label+'R'+trackID_text+'-'+ trackClass_text+':', (round(center[0]-20), round(center[1]-16)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=0.7, color=(0,0,255), thickness=1)
                    cv2.putText(image, position_text, (round(center[0]-20), round(center[1]-6)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=0.7, color=(0,0,255), thickness=1)
                    image = cv2.circle(image, center=[round(item) for item in center], radius=3, color=(0, 255, 0), thickness=-1)               
                else: #SHOW unreliable_tracker, comment these to make us only show reliable tracks 
                    # Leicheng-Draw centroid on the image for debugging
                    #leicheng-unreliable_tracker  #center[1]-radius
                    cv2.putText(image, sensor_label+'U'+trackID_text+'-'+ trackClass_text+':', (round(center[0]-20), round(center[1]-16)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=0.7, color=(255,0,0), thickness=1)
                    cv2.putText(image, position_text, (round(center[0]-20), round(center[1]-6)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=0.7, color=(255,0,0), thickness=1)
                    image = cv2.circle(image, center=[round(item) for item in center], radius=3, color=(0, 255, 0), thickness=-1)
    
            if self.mode==0:
                self.cam_trc_loc.append(trc_loc)
            elif self.mode==2:
                self.sf_trc_loc.append(trc_loc)  
                
            return image
    
        ##############################################################  radar ###################################
        # Detect first and then track
        def DetectionByTracking_rad(self,rnn_model,scl, img_model, image,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,bbx_imgs,img_feats,centroids,d_max=20.,sensor_label=''):
            dims = image.shape[:2]
            self.count += 1
            #timestamps = 3
            # Get list of tracker bounding boxes and Get list of tracker bbox_imgs
            trk_position = []
            trk_bbx_imgs = []
            trk_img_feats = []
            if self.tracker_list:
                for tracker in self.tracker_list:
                    trk_position.append(tracker.position)## Kalman posi
                    trk_bbx_imgs.append(tracker.bbx_img)
                    trk_img_feats.append(tracker.img_feat)
    
            # if self.mode==1:#radar to img for RNN
            #     #trk_position  = self.rad3d_to_img2d(trk_position,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)
            #     centroids= self.rad3d_to_img2d(centroids,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)## RNN posi
            
            # Match detected to trackers, set max_dis_cost per your need
            if centroids.size:#detections is not empty
                matched, unmatched_dets, unmatched_trks = \
                      self.match_detections_to_trackers(img_model,trk_position, centroids, trk_bbx_imgs, bbx_imgs,trk_img_feats, img_feats,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max,max_dis_cost=80)
            else: #detections is empty, only do prediction; Set all tracks as unmatched
                matched = np.empty((0, 2), dtype=int)
                unmatched_dets = np.empty((0, 1), dtype=int)
                unmatched_trks = np.arange(len(trk_position))
    
            # Deal with Matched detections:predict_and_update; need to count age of matched of the tracker and reset the age of unmatched for the matched tracker
            if len(matched) > 0:
                for trk_idx, det_idx in matched:
                    temp_trk = self.tracker_list[trk_idx] #take out the corresponding track                
                    temp_trk.num_matched += 1 # set flag for this track, num_matched is how many times(ages) of this same tracker are been matched
                    temp_trk.num_unmatched = 0 # set 0 for Continuous counting unmatched, num_unmatched is how many times(ages) of this same tracker are been Continuous unmatched, once it mtached, the unmathed is been reset to 0                

                        
                    ### location
                    z = centroids[det_idx,0:2]# det_centroids of next frame# det_centroids of next frame
                    z = np.expand_dims(z, axis=0).T
                    temp_trk.predict_and_update(z) # based on z, to do update and pred
                    xx = temp_trk.x_previous.T[0].tolist() # get pred_location
                    xx = [xx[0], xx[2]]
                    temp_trk.position = xx # put pred_location into track_list
                    
                    #### RNN and Kalman accuracy after do position prediction
                    _,dist_kal=self.RNN_KalM_Acc(0, temp_trk.position[0:2], centroids[det_idx,0:2],mode=0)
                    temp_trk.dist_rnn.append(dist_kal)
                    print('rad_rnn-kal_label-diff:',dist_kal,'\n')
               
    
            # Deal with Unmatched Detections: only predict(); need to assign track_ID to the tracker
            if len(unmatched_dets) > 0:
                for i in unmatched_dets:# i is index of centroids or say det_idx
                    # Create a new tracker
                    if  (any(self.removed_trackID)) and self.trackId>50: #reuse the ID  when removed_trackID is non-empty,and the ID exceeds 50
                        temp_trk = Tracker(trackId=self.removed_trackID.pop(), P= 100.0)  # Create a new tracker and reuse the ID 
                    else:
                        temp_trk = Tracker(trackId=self.trackId, P= 100.0)  # Create a new tracker
                        self.trackId += 1  # ID incremented by 1 
                    
                    ### location
                    z = centroids[i,0:2]
                    z = np.expand_dims(z, axis=0).T
                    x = np.array([[z[0], 0, z[1], 0]],dtype=object).T #x=[x,x',y,y']
                    temp_trk.x_previous = x
                    temp_trk.predict()
                    xx = temp_trk.x_previous
                    xx = xx.T[0].tolist()
                    xx = [xx[0], xx[2]] #only extract [x,y] from [x,x',y,y']
                    temp_trk.position = xx #prediction assign to track, and will be used to calculate the cost matrix

                                
                    ### populate the tracker_list
                    self.tracker_list.append(temp_trk)#plt.imshow(temp_trk.bbx_img)
    
            # Deal with Unmatched Tracks: only predict(); need to count age of unmatched of the tracker
            if len(unmatched_trks) > 0:
                for i in unmatched_trks:
                    temp_trk = self.tracker_list[i]
                    temp_trk.num_unmatched += 1 # Continuous num(times) of unmatched of this tracker
                  
                    ### location
                    temp_trk.predict()
                    xx = temp_trk.x_previous
                    xx = xx.T[0].tolist()
                    xx = [xx[0], xx[2]]
                    temp_trk.position = xx
   
    
            #leicheng-Find list of trackers to be deleted    
            # If tracks are not associated for long time, remove them        
            del_tracks = []
            for i in range(len(self.tracker_list)):
                #set track age condition
                track_age=self.tracker_list[i].num_unmatched+self.tracker_list[i].num_matched
                track_age_condition = track_age!=0 and track_age<=self.age_threshold and (self.tracker_list[i].num_matched/track_age)<0.5 # In the initial stage of the tracker, determine whether the tracker is reliable
                if self.tracker_list[i].num_unmatched > self.max_unmatched or track_age_condition:
                    del_tracks.append(i)
                    self.removed_trackID.append(self.tracker_list[i].id) #Populate 'removed_trackID'
            # del lost_track
            self.tracker_list=np.delete(np.array(self.tracker_list), del_tracks, None).tolist()
    
    
            # kalman pos : tracker.position    
            # Populate the list of Reliable trackers to be displayed on the image
            rad_trc_loc = []
            reliable_tracker_list = []
            for tracker in self.tracker_list:
                # Leicheng-Draw centroid on the image for debugging
                trackID_text = "%d" % tracker.id            
                center = tracker.position # predicted_center for Drawing not the measured_center(real detected)
                ####### Transform tracker.position_pixel to distance
                # cr_distance_mid=x_pixel2meter_ratio*(dims[1]/2)
                # cr_distance = x_pixel2meter_ratio*center[0] - cr_distance_mid
                # dep_distance = d_max-y_pixel2meter_ratio*center[1]
                cr_distance, dep_distance = calculate_distance(center[0], center[1])
                tracker.distance_pos.append([cr_distance,dep_distance])
                position_text = "[%.2fm,%.2fm]" % (cr_distance, dep_distance)
                rad_trc_loc.append([cr_distance,dep_distance])
                #########################################################################
                if tracker.num_matched >= self.min_matched and tracker.num_unmatched <= (self.max_unmatched / 2):
                    #leicheng-reliable_tracker
                    reliable_tracker_list.append(tracker)
                    # Draw centroid on the image  #center[1]-radius
                    cv2.putText(image, sensor_label+'R'+trackID_text+':', (round(center[0]-20), round(center[1]-16)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=0.7, color=(0,0,255), thickness=1)
                    cv2.putText(image, position_text, (round(center[0]-20), round(center[1]-6)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=0.7, color=(0,0,255), thickness=1)
                    image = cv2.circle(image, center=[round(item) for item in tracker.position], radius=3, color=(0, 255, 0), thickness=-1)               
                else: #SHOW unreliable_tracker, comment these to make us only show reliable tracks 
                    # Leicheng-Draw centroid on the image for debugging
                    #leicheng-unreliable_tracker  #center[1]-radius
                    cv2.putText(image, sensor_label+'U'+trackID_text+':', (round(center[0]-20), round(center[1]-16)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=0.7, color=(255,0,0), thickness=1)
                    cv2.putText(image, position_text, (round(center[0]-20), round(center[1]-6)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=0.7, color=(255,0,0), thickness=1)
                    image = cv2.circle(image, center=[round(item) for item in tracker.position], radius=3, color=(0, 255, 0), thickness=-1)
            
            self.rad_trc_loc.append(rad_trc_loc) 
            
            return image
        
    else:
        ############################################################## camera ################################################################
        # Detect first and then track
        def DetectionByTracking(self,rnn_model,scl, img_model, image,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,bbx_imgs,img_feats,centroids,d_max=20.,sensor_label=''):
            dims = image.shape[:2]
            self.count += 1
            timestamps = 3
            # Get list of tracker bounding boxes and Get list of tracker bbox_imgs
            trk_position = []
            trk_bbx_imgs = []
            trk_img_feats = []
            if self.tracker_list:
                for tracker in self.tracker_list:
                    trk_position.append(tracker.rnn_position)## RNN posi
                    #print('rnn-kal_pos-diff:',distance.euclidean(tracker.rnn_position[0:2], tracker.position),'\n')
                    trk_bbx_imgs.append(tracker.bbx_img)
                    trk_img_feats.append(tracker.img_feat)
    
            # # Load img_model
            # img_model=self.load_img_model(self.img_model_path)
            
            # Match detected to trackers, set max_dis_cost per your need
            if centroids.size:#detections is not empty
                matched, unmatched_dets, unmatched_trks = \
                      self.match_detections_to_trackers(img_model,trk_position, centroids, trk_bbx_imgs, bbx_imgs,trk_img_feats, img_feats,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max,max_dis_cost=40)
            else: #detections is empty, only do prediction; Set all tracks as unmatched
                matched = np.empty((0, 2), dtype=int)
                unmatched_dets = np.empty((0, 1), dtype=int)
                unmatched_trks = np.arange(len(trk_position))
    
            # Deal with Matched detections:predict_and_update; need to count age of matched of the tracker and reset the age of unmatched for the matched tracker
            if len(matched) > 0:
                for trk_idx, det_idx in matched:
                    temp_trk = self.tracker_list[trk_idx] #take out the corresponding track                
                    temp_trk.num_matched += 1 # set flag for this track, num_matched is how many times(ages) of this same tracker are been matched
                    temp_trk.num_unmatched = 0 # set 0 for Continuous counting unmatched, num_unmatched is how many times(ages) of this same tracker are been Continuous unmatched, once it mtached, the unmathed is been reset to 0                

    
                    ### location
                    #z = det_boxes[det_idx]
                    z = centroids[det_idx]# det_centroids of next frame
                    z = np.expand_dims(z, axis=0).T
                    temp_trk.predict_and_update(z) # based on z, to do update and pred
                    xx = temp_trk.x_previous.T[0].tolist() # get pred_location
                    xx = [xx[0], xx[2]]
                    temp_trk.position = xx # put pred_location into track_list
                    ### RNN : need at least 3 positions to do RNN pred, otherwise, only append new detection
                    if len(temp_trk.rnn_position_list)>=timestamps-1: # timestamps-1 since below append another one
                        temp_trk.rnn_position_list.append( centroids[det_idx].tolist() ) #use new detection to do pred, then it will be 3
                        scaled_cnt = scl.transform( np.array(temp_trk.rnn_position_list)[-3:] )
                        rnn_seq_pos,rnn_label = self.split_RNN_dataset(scaled_cnt,timestamps=timestamps, pred_n=1)                    
                        rnn_seq_pos = [ rnn_seq_pos[:,:,0],  rnn_seq_pos[:,:,1]]
                        rnn_pred_pos,rnn_real_pos = self.pred_use_rnn_model(rnn_model,scl,rnn_seq_pos,rnn_label, batch_size=1)
                        rnn_pred_pos_last = rnn_pred_pos[-1,:].tolist()
                        #rnn_pred_pos_last=rnn_pred_pos_last+np.mean(temp_trk.dist_rnn)
                        #temp_trk.rnn_position_list.append( rnn_pred_pos_last ) # when measurment exists, not need to append pred_pos.  pos_list for creating RNN dataset
                        temp_trk.rnn_position = rnn_pred_pos_last
                    else:
                        temp_trk.rnn_position_list.append( centroids[det_idx].tolist() ) # pos_list for creating RNN dataset
                        temp_trk.rnn_position = centroids[det_idx].tolist()
    
                    if np.array(img_feats).size and np.array(img_feats[det_idx]).size:
                        ### img: update img as z_img(new det_img), no prediction for img
                        z_img = bbx_imgs[det_idx]                
                        temp_trk.bbx_img = z_img #org_img assign to track, and will be used to calculate the img cost matrix 
                        temp_trk.img_feat = img_feats[det_idx]
                        
                    #### RNN and Kalman accuracy after do position prediction
                    dist_rnn,dist_kal=self.RNN_KalM_Acc(temp_trk.rnn_position[0:2], temp_trk.position[0:2], centroids[det_idx,0:2])
                    temp_trk.dist_rnn.append(dist_rnn)
                    print('cam_sf_rnn-kal_label-diff:',dist_rnn,'\n')
    
            # Deal with Unmatched Detections: only predict(); need to assign track_ID to the tracker
            if len(unmatched_dets) > 0:
                for i in unmatched_dets:# i is index of centroids or say det_idx
                    # Create a new tracker
                    if  (any(self.removed_trackID)) and self.trackId>50: #reuse the ID  when removed_trackID is non-empty,and the ID exceeds 50
                        temp_trk = Tracker(trackId=self.removed_trackID.pop(), P= 100.0)  # Create a new tracker and reuse the ID 
                    else:
                        temp_trk = Tracker(trackId=self.trackId, P= 100.0)  # Create a new tracker
                        self.trackId += 1  # ID incremented by 1 
                                        
            
                    ### location
                    #z = det_boxes[i]
                    z = centroids[i]
                    z = np.expand_dims(z, axis=0).T
                    x = np.array([[z[0], 0, z[1], 0]],dtype=object).T #x=[x,x',y,y']
                    temp_trk.x_previous = x
                    temp_trk.predict()
                    xx = temp_trk.x_previous
                    xx = xx.T[0].tolist()
                    xx = [xx[0], xx[2]] #only extract [x,y] from [x,x',y,y']
                    temp_trk.position = xx #prediction assign to track, and will be used to calculate the cost matrix
                    ### RNN : since unmatched_dets means new_tracker, and it doesn't have 3 positions to do RNN pred
                    temp_trk.rnn_position_list.append( centroids[i].tolist() ) # pos_list for creating RNN dataset
                    temp_trk.rnn_position = centroids[i].tolist() # centroids assign to track as output_position directly: [np(1500,3),np(1500,3)]
    
                    if np.array(img_feats).size and np.array(img_feats[i]).size:
                        ### img: keep org_img, no prediction for img
                        z_img = bbx_imgs[i]                
                        temp_trk.bbx_img = z_img #org_img assign to track, and will be used to calculate the img cost matrix                  
                        temp_trk.img_feat = img_feats[i]
                    ### populate the tracker_list
                    self.tracker_list.append(temp_trk)#plt.imshow(temp_trk.bbx_img)
    
            # Deal with Unmatched Tracks: only predict(); need to count age of unmatched of the tracker
            if len(unmatched_trks) > 0:
                for i in unmatched_trks:
                    temp_trk = self.tracker_list[i]
                    temp_trk.num_unmatched += 1 # Continuous num(times) of unmatched of this tracker
    
                    ### location
                    temp_trk.predict()
                    xx = temp_trk.x_previous
                    xx = xx.T[0].tolist()
                    xx = [xx[0], xx[2]]
                    temp_trk.position = xx
                    ### RNN : need at least 3 positions to do RNN pred, otherwise, no action                   
                    if len(temp_trk.rnn_position_list)>=timestamps:
                        scaled_cnt = scl.transform( np.array(temp_trk.rnn_position_list)[-3:] )
                        rnn_seq_pos,rnn_label = self.split_RNN_dataset(scaled_cnt,timestamps=timestamps, pred_n=1)
                        rnn_seq_pos = [ rnn_seq_pos[:,:,0],  rnn_seq_pos[:,:,1]]
                        rnn_pred_pos,rnn_real_pos  = self.pred_use_rnn_model(rnn_model,scl,rnn_seq_pos,rnn_label, batch_size=1)
                        rnn_pred_pos_last = rnn_pred_pos[-1,:].tolist()
                        temp_trk.rnn_position_list.append( rnn_pred_pos_last ) # pos_list for creating RNN dataset
                        temp_trk.rnn_position = rnn_pred_pos_last              
                    ### img: keep img of this tracker unchanged, no prediction for img               
                    #temp_trk.bbx_img = temp_trk.bbx_img
    
    
            #leicheng-Find list of trackers to be deleted    
            # If tracks are not associated for long time, remove them        
            del_tracks = []
            for i in range(len(self.tracker_list)):
                #set track age condition
                track_age=self.tracker_list[i].num_unmatched+self.tracker_list[i].num_matched
                #track_age_condition = track_age!=0 and track_age<=self.age_threshold and (self.tracker_list[i].num_matched/track_age)<0.5 # In the initial stage of the tracker, determine whether the tracker is reliable
                track_age_condition = track_age>=3 and track_age<=self.age_threshold and (self.tracker_list[i].num_matched/track_age)<0.5 # In the initial stage of the tracker, determine whether the tracker is reliable
                if self.tracker_list[i].num_unmatched > self.max_unmatched or track_age_condition:
                    del_tracks.append(i)
                    self.removed_trackID.append(self.tracker_list[i].id) #Populate 'removed_trackID'
            # del lost_track
            self.tracker_list=np.delete(np.array(self.tracker_list), del_tracks, None).tolist()
    
    
    
    
            # # kalman pos : tracker.position    
            # # Populate the list of Reliable trackers to be displayed on the image
            # reliable_tracker_list = []
            # for tracker in self.tracker_list:
            #     # Leicheng-Draw centroid on the image for debugging
            #     trackID_text = "%d" % tracker.id            
            #     center = tracker.position # predicted_center for Drawing not the measured_center(real detected)
            #     ####### Transform tracker.position_pixel to distance
            #     cr_distance_mid=x_pixel2meter_ratio*(dims[1]/2)
            #     cr_distance = x_pixel2meter_ratio*center[0] - cr_distance_mid
            #     dep_distance = d_max-y_pixel2meter_ratio*center[1]
            #     tracker.distance_pos.append([cr_distance,dep_distance])
            #     position_text = "[%.2fm,%.2fm]" % (cr_distance, dep_distance)
            #     #########################################################################
            #     if tracker.num_matched >= self.min_matched and tracker.num_unmatched <= self.max_unmatched:
            #         #leicheng-reliable_tracker
            #         reliable_tracker_list.append(tracker)
            #         # Draw centroid on the image  #center[1]-radius
            #         cv2.putText(image, sensor_label+'R'+trackID_text+':'+position_text, (round(center[0]-8), round(center[1]-3)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1., color=(0,0,255), thickness=1)
            #         image = cv2.circle(image, center=[round(item) for item in tracker.position], radius=3, color=(0, 255, 0), thickness=-1)               
            #     else: #SHOW unreliable_tracker, comment these to make us only show reliable tracks 
            #         # Leicheng-Draw centroid on the image for debugging
            #         #leicheng-unreliable_tracker  #center[1]-radius
            #         cv2.putText(image, sensor_label+'U'+trackID_text+':'+position_text, (round(center[0]-8), round(center[1]-3)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1., color=(255,0,0), thickness=1)
            #         image = cv2.circle(image, center=[round(item) for item in tracker.position], radius=3, color=(0, 255, 0), thickness=-1)
    
            # # RNN pos : tracker.position    
            # Populate the list of Reliable trackers to be displayed on the image
            reliable_tracker_list = []
            for tracker in self.tracker_list:
                # Leicheng-Draw centroid on the image for debugging
                trackID_text = "%d" % tracker.id            
                center = tracker.rnn_position # predicted_center for Drawing not the measured_center(real detected)
                ####### Transform tracker.position_pixel to distance
                cr_distance_mid=x_pixel2meter_ratio*(dims[1]/2)
                cr_distance = x_pixel2meter_ratio*center[0] - cr_distance_mid
                dep_distance = d_max-y_pixel2meter_ratio*center[1]
                tracker.distance_pos.append([cr_distance,dep_distance])
                position_text = "[%.2fm,%.2fm]" % (cr_distance, dep_distance)
                #########################################################################
                if tracker.num_matched >= self.min_matched and tracker.num_unmatched <= self.max_unmatched:
                    #leicheng-reliable_tracker
                    reliable_tracker_list.append(tracker)
                    # Draw centroid on the image  #center[1]-radius
                    cv2.putText(image, sensor_label+'R'+trackID_text+':'+position_text, (round(center[0]-8), round(center[1]-3)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1., color=(0,0,255), thickness=1)
                    image = cv2.circle(image, center=[round(item) for item in tracker.rnn_position], radius=3, color=(0, 255, 0), thickness=-1)               
                else: #SHOW unreliable_tracker, comment these to make us only show reliable tracks 
                    # Leicheng-Draw centroid on the image for debugging
                    #leicheng-unreliable_tracker  #center[1]-radius
                    cv2.putText(image, sensor_label+'U'+trackID_text+':'+position_text, (round(center[0]-8), round(center[1]-3)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1., color=(255,0,0), thickness=1)
                    image = cv2.circle(image, center=[round(item) for item in tracker.rnn_position], radius=3, color=(0, 255, 0), thickness=-1)
    
    
            return image
    
        ##############################################################  radar ###################################
        # Detect first and then track
        def DetectionByTracking_rad(self,rnn_model,scl, img_model, image,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,bbx_imgs,img_feats,centroids,d_max=20.,sensor_label=''):
            dims = image.shape[:2]
            self.count += 1
            timestamps = 3
            # Get list of tracker bounding boxes and Get list of tracker bbox_imgs
            trk_position = []
            trk_bbx_imgs = []
            trk_img_feats = []
            if self.tracker_list:
                for tracker in self.tracker_list:
                    trk_position.append(tracker.rnn_position)## RNN posi
                    #print('rnn-kal_pos-diff:',distance.euclidean(tracker.rnn_position[0:2], tracker.position),'\n')
                    trk_bbx_imgs.append(tracker.bbx_img)
                    trk_img_feats.append(tracker.img_feat)
    
            if self.mode==1:#radar to img for RNN
                trackers  = self.rad3d_to_img2d(trk_position,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)
                detections= self.rad3d_to_img2d(centroids,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)## RNN posi
            # Match detected to trackers, set max_dis_cost per your need
            if centroids.size:#detections is not empty
                matched, unmatched_dets, unmatched_trks = \
                      self.match_detections_to_trackers(img_model,trackers, detections, trk_bbx_imgs, bbx_imgs,trk_img_feats, img_feats,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max,max_dis_cost=40)
            else: #detections is empty, only do prediction; Set all tracks as unmatched
                matched = np.empty((0, 2), dtype=int)
                unmatched_dets = np.empty((0, 1), dtype=int)
                unmatched_trks = np.arange(len(trk_position))
    
            # Deal with Matched detections:predict_and_update; need to count age of matched of the tracker and reset the age of unmatched for the matched tracker
            if len(matched) > 0:
                for trk_idx, det_idx in matched:
                    temp_trk = self.tracker_list[trk_idx] #take out the corresponding track                
                    temp_trk.num_matched += 1 # set flag for this track, num_matched is how many times(ages) of this same tracker are been matched
                    temp_trk.num_unmatched = 0 # set 0 for Continuous counting unmatched, num_unmatched is how many times(ages) of this same tracker are been Continuous unmatched, once it mtached, the unmathed is been reset to 0                
                    
                        
                    ### location
                    z = centroids[det_idx,0:2]# det_centroids of next frame# det_centroids of next frame
                    z = np.expand_dims(z, axis=0).T
                    temp_trk.predict_and_update(z) # based on z, to do update and pred
                    xx = temp_trk.x_previous.T[0].tolist() # get pred_location
                    xx = [xx[0], xx[2]]
                    temp_trk.position = xx # put pred_location into track_list
                    ### RNN : need at least 3 positions to do RNN pred, otherwise, only append new detection           
                    if len(temp_trk.rnn_position_list)>=timestamps-1: # timestamps-1 since below append another one
                        temp_trk.rnn_position_list.append( centroids[det_idx].tolist() ) #use new detection to do pred, then it will be 3
                        scaled_cnt = scl.transform( np.array(temp_trk.rnn_position_list)[-3:] )
                        rnn_seq_pos,rnn_label = self.split_RNN_dataset(scaled_cnt,timestamps=timestamps, pred_n=1)                    
                        rnn_seq_pos = [ rnn_seq_pos[:,:,0],  rnn_seq_pos[:,:,1],  rnn_seq_pos[:,:,2]]
                        rnn_pred_pos,rnn_real_pos = self.pred_use_rnn_model(rnn_model,scl,rnn_seq_pos,rnn_label, batch_size=1)
                    
                        rnn_pred_pos_last = rnn_pred_pos[-1,:].tolist()
                        #rnn_pred_pos_last=rnn_pred_pos_last+np.mean(temp_trk.dist_rnn)
                        #temp_trk.rnn_position_list.append( rnn_pred_pos_last ) # when measurment exists, not need to append pred_pos.  pos_list for creating RNN dataset
                        temp_trk.rnn_position = rnn_pred_pos_last
                    else:
                        temp_trk.rnn_position_list.append( centroids[det_idx].tolist() ) # pos_list for creating RNN dataset
                        temp_trk.rnn_position = centroids[det_idx].tolist()
                    #### RNN and Kalman accuracy after do position prediction
                    trackers_rnn  = self.rad3d_to_img2d(temp_trk.rnn_position,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)
                    detections_rnn= self.rad3d_to_img2d(temp_trk.position,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)## RNN posi
                    centroids_rnn= self.rad3d_to_img2d(centroids[det_idx],x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)
                    dist_rnn,dist_kal=self.RNN_KalM_Acc(trackers_rnn, detections_rnn, centroids_rnn)
                    #dist_rnn,dist_kal=self.RNN_KalM_Acc(temp_trk.rnn_position[0:2], temp_trk.position[0:2], centroids[det_idx,0:2])
                    temp_trk.dist_rnn.append(dist_rnn)
                    print('rad_rnn-kal_label-diff:',dist_rnn,'\n')
               
    
            # Deal with Unmatched Detections: only predict(); need to assign track_ID to the tracker
            if len(unmatched_dets) > 0:
                for i in unmatched_dets:# i is index of centroids or say det_idx
                    # Create a new tracker
                    if  (any(self.removed_trackID)) and self.trackId>50: #reuse the ID  when removed_trackID is non-empty,and the ID exceeds 50
                        temp_trk = Tracker(trackId=self.removed_trackID.pop(), P= 100.0)  # Create a new tracker and reuse the ID 
                    else:
                        temp_trk = Tracker(trackId=self.trackId, P= 100.0)  # Create a new tracker
                        self.trackId += 1  # ID incremented by 1 
                    
                    ### location
                    z = centroids[i,0:2]
                    z = np.expand_dims(z, axis=0).T
                    x = np.array([[z[0], 0, z[1], 0]],dtype=object).T #x=[x,x',y,y']
                    temp_trk.x_previous = x
                    temp_trk.predict()
                    xx = temp_trk.x_previous
                    xx = xx.T[0].tolist()
                    xx = [xx[0], xx[2]] #only extract [x,y] from [x,x',y,y']
                    temp_trk.position = xx #prediction assign to track, and will be used to calculate the cost matrix
                    ### RNN : since unmatched_dets means new_tracker, and it doesn't have 3 positions to do RNN pred
                    temp_trk.rnn_position_list.append( centroids[i].tolist() ) # pos_list for creating RNN dataset
                    temp_trk.rnn_position  = centroids[i].tolist()
                                
                    ### populate the tracker_list
                    self.tracker_list.append(temp_trk)#plt.imshow(temp_trk.bbx_img)
    
            # Deal with Unmatched Tracks: only predict(); need to count age of unmatched of the tracker
            if len(unmatched_trks) > 0:
                for i in unmatched_trks:
                    temp_trk = self.tracker_list[i]
                    temp_trk.num_unmatched += 1 # Continuous num(times) of unmatched of this tracker
                  
                    ### location
                    temp_trk.predict()
                    xx = temp_trk.x_previous
                    xx = xx.T[0].tolist()
                    xx = [xx[0], xx[2]]
                    temp_trk.position = xx
                    ### RNN : need at least 3 positions to do RNN pred, otherwise, no action
                    if len(temp_trk.rnn_position_list)>=timestamps:
                        scaled_cnt = scl.transform( np.array(temp_trk.rnn_position_list)[-3:] )
                        rnn_seq_pos,rnn_label = self.split_RNN_dataset(scaled_cnt,timestamps=timestamps, pred_n=1)
                        rnn_seq_pos = [ rnn_seq_pos[:,:,0],  rnn_seq_pos[:,:,1],  rnn_seq_pos[:,:,2]]
                        rnn_pred_pos,rnn_real_pos  = self.pred_use_rnn_model(rnn_model,scl,rnn_seq_pos,rnn_label, batch_size=1)
                        #rnn_pred_pos_last  = np.squeeze(rnn_pred_pos_last)
                        rnn_pred_pos_last = rnn_pred_pos[-1,:].tolist()                    
                        temp_trk.rnn_position_list.append( rnn_pred_pos_last ) # pos_list for creating RNN dataset
                        temp_trk.rnn_position = rnn_pred_pos_last 
    
    
    
    
    
            #leicheng-Find list of trackers to be deleted    
            # If tracks are not associated for long time, remove them        
            del_tracks = []
            for i in range(len(self.tracker_list)):
                #set track age condition
                track_age=self.tracker_list[i].num_unmatched+self.tracker_list[i].num_matched
                track_age_condition = track_age!=0 and track_age<=self.age_threshold and (self.tracker_list[i].num_matched/track_age)<0.5 # In the initial stage of the tracker, determine whether the tracker is reliable
                if self.tracker_list[i].num_unmatched > self.max_unmatched or track_age_condition:
                    del_tracks.append(i)
                    self.removed_trackID.append(self.tracker_list[i].id) #Populate 'removed_trackID'
            # del lost_track
            self.tracker_list=np.delete(np.array(self.tracker_list), del_tracks, None).tolist()
    
    
            ## RNN pos : tracker.rnn_position 
            # Populate the list of Reliable trackers to be displayed on the image
            reliable_tracker_list = []
            for tracker in self.tracker_list:
                # Leicheng-Draw centroid on the image for debugging
                trackID_text = "%d" % tracker.id            
                center = tracker.rnn_position # predicted_center for Drawing not the measured_center(real detected)
                center  = self.rad3d_to_img2d(center,x_pixel2meter_ratio,y_pixel2meter_ratio,IMAGE_W,d_max)
                ####### Transform tracker.position_pixel to distance
                cr_distance_mid=x_pixel2meter_ratio*(dims[1]/2)
                cr_distance = x_pixel2meter_ratio*center[0] - cr_distance_mid
                dep_distance = d_max-y_pixel2meter_ratio*center[1]
                tracker.distance_pos.append([cr_distance,dep_distance])
                position_text = "[%.2fm,%.2fm]" % (cr_distance, dep_distance)
                #########################################################################
                if tracker.num_matched >= self.min_matched and tracker.num_unmatched <= self.max_unmatched:
                    #leicheng-reliable_tracker
                    reliable_tracker_list.append(tracker)
                    # Draw centroid on the image  #center[1]-radius
                    cv2.putText(image, sensor_label+'R'+trackID_text+':'+position_text, (round(center[0]-8), round(center[1]-3)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1., color=(0,0,255), thickness=1)
                    image = cv2.circle(image, center=[round(item) for item in center], radius=3, color=(0, 255, 0), thickness=-1)               
                else: #SHOW unreliable_tracker, comment these to make us only show reliable tracks 
                    # Leicheng-Draw centroid on the image for debugging
                    #leicheng-unreliable_tracker  #center[1]-radius
                    cv2.putText(image, sensor_label+'U'+trackID_text+':'+position_text, (round(center[0]-8), round(center[1]-3)),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1., color=(255,0,0), thickness=1)
                    image = cv2.circle(image, center=[round(item) for item in center], radius=3, color=(0, 255, 0), thickness=-1)
    
            return image
        
