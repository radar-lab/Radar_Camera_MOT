#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:20:16 2022

@author: kevalen
"""
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter #Use a format string to format the tick with significant digits.
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import statistics
from scipy import stats
# outlier/anomaly detection
from sklearn.neighbors import LocalOutlierFactor
import scipy.spatial.transform
import pandas as pd
#!pip3 install pyproj  #Install pyproj first for doing wsg-84 conversion
from pyproj import Transformer

#########################
def rotation_matrix(theta1, theta2, theta3, order='xyz'):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        oreder = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
        Each matrix is meant to operate by pre-multiplying column vectors 
    output
        3x3 rotation matrix (numpy array)
    ref: https://en.wikipedia.org/wiki/Euler_angles, the intrinsic rotations​
    Any extrinsic rotation is equivalent to an intrinsic rotation by the same angles but with inverted order of elemental rotations, and vice versa.
    
    clockwise-posi,conter-clock-nega
    intri-rot-->xyz(will keep value in the rotaion axis unchanged);
    ex-rot-->zyx,need to invert order of angel argum
    """
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])
    return matrix    
def coor3d_tran(phi,lamb,theta=0):  #for enu to radar face cs
    # rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -phi, degrees=True).as_matrix()#angle*-1 : left handed *-1;rotate over east(x)
    # rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -lamb, degrees=True).as_matrix()#angle*-1 : left handed *-1;rotate over z
    # rotMatrix = rot1.dot(rot3)    #rot1(x) @ rot3(z)
    #rotMatrix = scipy.spatial.transform.Rotation.from_euler('zyx', [-(90+lamb),theta,-(90-phi)], degrees=True).as_matrix()
    rotMatrix = scipy.spatial.transform.Rotation.from_euler('zyx', [-(lamb),theta,-(90-phi)], degrees=True).as_matrix()
    return rotMatrix

def geodetic2enu(LLH, LLH_org):
    # transformer = Transformer.from_crs(
    #     {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    #     {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    #     )
    # The local coordinate origin (RADAR)
    lat_org = LLH_org[0] # deg
    lon_org = LLH_org[1]  # deg
    alt_org = LLH_org[2]     # meters
    # The point of interest
    lat = LLH[0]  # deg
    lon = LLH[1]   # deg
    alt = LLH[2]      # meters    
    transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)#EPSG:4979 (LLH) -> EPSG:4978 (XYZ)
    x, y, z = transformer.transform( lon,lat,  alt,radians=False)
    x_org, y_org, z_org = transformer.transform( lon_org,lat_org,  alt_org,radians=False)
    vec=np.array([[ x-x_org, y-y_org, z-z_org]]).T
    #firstly,counterclockwise rotate x to east over z,positive deg; then counterclockwise rotate z to up over east(x),neg deg. Opposite of from_euler.
    rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()#angle*-1 : left handed *-1;rotate over east(x)
    rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()#angle*-1 : left handed *-1;rotate over z

    rotMatrix = rot1.dot(rot3)    #rot1(x) @ rot3(z)
    #rotMatrix = scipy.spatial.transform.Rotation.from_euler('zyx', [-(90+lon_org),0,-(90-lat_org)], degrees=True).as_matrix()
    enu = rotMatrix.dot(vec).T.ravel()
    return enu.T

def enu2geodetic(x,y,z, lat_org, lon_org, alt_org):
    transformer1 = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)#EPSG:4979 (LLH) -> EPSG:4978 (XYZ)
    transformer2 = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)#EPSG:4978 (XYZ) -> EPSG:4979 (LLH)
    
    x_org, y_org, z_org = transformer1.transform( lon_org,lat_org,  alt_org,radians=False)
    ecef_org=np.array([[x_org,y_org,z_org]]).T
    
    rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()#angle*-1 : left handed *-1
    rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()#angle*-1 : left handed *-1

    rotMatrix = rot1.dot(rot3)

    ecefDelta = rotMatrix.T.dot( np.array([[x,y,z]]).T )
    ecef = ecefDelta+ecef_org
    lon, lat, alt = transformer2.transform( ecef[0,0],ecef[1,0],ecef[2,0],radians=False)

    return [lat,lon,alt]

def trans_LLH_to_XYZ(LLH):
    '''
    Transform EPSG:4979 (LLH) to EPSG:4978 (XYZ), NOTE: EPSG:4326 is 2D(LL)
    INPUT: array
    list of tuples = [tuple(x) for x in lst]
    List of Lists = [list(x) for x in tuples]
    '''
    LLH=np.array(LLH)    
    trans_GPS_to_XYZ = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True) #EPSG:4979 (LLH) -> EPSG:4978 (XYZ), NOTE: EPSG:4326 is 2D(LL)
    if LLH.ndim==1:
        XYZ=trans_GPS_to_XYZ.transform(LLH[1],LLH[0],LLH[2])
        XYZ=np.array(XYZ)
    else:
        XYZ=trans_GPS_to_XYZ.transform(LLH[:,1],LLH[:,0],LLH[:,2]) #longitude, latitude, height(meters) instead latitude,longitude, height(meters)
        XYZ=np.vstack((XYZ[0], XYZ[1],XYZ[2])).T
    return XYZ
# =============================================================================
# def trans_LLH_to_XYZ(LLH):
#     '''
#     Transform EPSG:4979 (LLH) to EPSG:4978 (XYZ), NOTE: EPSG:4326 is 2D(LL)
#     list of tuples = [tuple(x) for x in lst]
#     List of Lists = [list(x) for x in tuples]
#     '''
#     trans_GPS_to_XYZ = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True) #EPSG:4979 (LLH) -> EPSG:4978 (XYZ), NOTE: EPSG:4326 is 2D(LL)
#     XYZ=trans_GPS_to_XYZ.transform(LLH[1],LLH[0],LLH[2]) #longitude, latitude, height(meters) instead latitude,longitude, height(meters)
#     return np.array(XYZ)
# =============================================================================
################################### project 3d to 3d
def append_1_to_homo(mat):
    #Note that the second argument is the index before which you want to insert.
    #the third argument is the Values to insert into arr 
    #And the axis = 1 indicates that you want to insert as a column without flattening the array.
    #hommat = np.insert(mat, -1, 1, axis=1)
    hommat = np.append(mat, np.ones((mat.shape[0],1)), axis=1)
    #hommat = np.stack((mat,np.ones_like(mat[:,0])))
    #hommat = np.hstack((mat,np.ones((mat.shape[0],1))))
    return hommat


def project_3d_to_3d(point_mat, M):
    # M is 3x4
    result_mat = (M @ (point_mat.T)).T
    return result_mat
####################################################################
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

def element_wise_euclidian_dist(a, b):
    '''
    Element-wise Euclidian distances between 2 numpy arrays
    return nx1 numpy array
    '''
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b, axis=1)
    return dist
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
        #centroids.append(np.average(cluster, axis=0))
        #x=np.mean(cluster[:,0])
        #y=np.mean(cluster[:,1])
        #centroids.append([x,y])
    return np.array(centroids)

def preprocessing_gps_data(bag_path,ts_centroids_fr,translation=1,base_llh=[32.23534993,-110.95383927,710.459]):
    ####################  load 3 GPS rover bags by using rosbag
    ts_rover1,all_gps_enu_arr1,x1,y1 = load_gps_data(bag_path,ros_topic=['/rover1/tcpfix'],translation=translation,base_llh=base_llh)
    ts_rover2,all_gps_enu_arr2,x2,y2 = load_gps_data(bag_path,ros_topic=['/rover2/tcpfix'],translation=translation,base_llh=base_llh)
    ts_rover3,all_gps_enu_arr3,x3,y3 = load_gps_data(bag_path,ros_topic=['/rover3/tcpfix'],translation=translation,base_llh=base_llh)
    
    gps_all_for_radar = asso_gps_to_radar(ts_centroids_fr,ts_rover1,ts_rover2,ts_rover3,all_gps_enu_arr1,all_gps_enu_arr2,all_gps_enu_arr3)
    return gps_all_for_radar

def load_gps_data(bag_path,ros_topic=['/rover3/tcpfix'],translation=1,base_llh=[32.23534993,-110.95383927,710.459]):
    ####################  load GPS rover bag by using rosbag
    rover = []
    ts_rover = []
    bag = rosbag.Bag(bag_path)
    for topic, msg, t in bag.read_messages(topics=ros_topic):  #reach_m_Rover  
        ts_rover.append(msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9) 
        rover.append([msg.latitude, msg.longitude, msg.altitude]) #usage: rover[0][0]-->latitude
    bag.close()    
    ################## transform to ENU
    all_gps_enu=[]
    for i in range(len(rover)):
        #base_pos_llh= base[idx_base[i]]
        base_pos_llh= base_llh
        rover_pos_llh = rover[i]
        ### LLH to ENU
        rover_pos=geodetic2enu(rover_pos_llh,base_pos_llh)
        all_gps_enu.append(rover_pos.tolist())
    ################## transform to ENU
    all_gps_enu_arr=np.array(all_gps_enu)
    all_gps_enu_arr[:,0] = all_gps_enu_arr[:,0] + translation # translation    
    # gps_to_rad_mat=rotation_matrix(-1.220887,0,0, order='zyx') 
    # all_gps_enu_arr=(gps_to_rad_mat@(all_gps_enu_arr.T)).T   
    x = all_gps_enu_arr[:,0] # X is E
    y = all_gps_enu_arr[:,1] # Y is S, or, -N
    return ts_rover,all_gps_enu_arr,x,y



def process_radar_for_gps(centroids_radar,theta=[-91,3.2,0]):
    ### this func is only useful for one track or one cluster
    ### filter_radar_for_gps
    radar_data_list=[]
    for i in range (len(centroids_radar)):
        radar_data_list.append(centroids_radar[i][0].tolist())#only select the *First* cluster    
    radar_data_arr=np.array(radar_data_list)
    ################################### remove the spikes and outliers 
    df = pd.DataFrame(radar_data_arr)
    #################using Local Outlier Factor  Method of Outlier Detection
    # model specification
    model1 = LocalOutlierFactor(n_neighbors = 40, metric = "manhattan", contamination = 'auto')#contamination = 0.02,metric = "euclidean"
    # model fitting
    y_pred = model1.fit_predict(df)
    # filter outlier index
    #outlier_index = np.where(y_pred == -1)
    inlier_index = np.where(y_pred == 1) # negative values(-1) are outliers and positives(1) inliers
    # filter inlier values
    df_inliers = df.iloc[inlier_index]
    ts_centroids_rad_inlier=np.array(ts_centroids_rad)[inlier_index]
    radar_data_arr=df_inliers.to_numpy()
    
    ### transform_radar_under_gps_coordinate_system
    rad_to_gps_mat=rotation_matrix(theta[0],theta[1],theta[2], order='zyx') #90.9
    rad_to_gps_res=(rad_to_gps_mat@(radar_data_arr.T)).T
    return radar_data_arr,ts_centroids_rad_inlier,rad_to_gps_res

def asso_radar_to_gps(rad_to_gps_res,all_gps_enu_arr,ts_centroids_rad_inlier, ts_rover):
    '''###############################################  Associate GPS and Radar '''
    idx_radar,radar_matched_val = find_nearest_betw_arr(ts_centroids_rad_inlier, np.array(ts_rover)) # assign gps to radar, or say to find each gps's corresponding radar_position
    # smooth radar data
    rad_to_gps_ass=[]
    for i in range(len(all_gps_enu_arr)):
        if (i-2)>=0 and (i+2) <= (len(all_gps_enu_arr)-1): # smooth radar data
            rad_to_gps_ass.append(rad_to_gps_res[idx_radar[i-2:i+2]].mean(axis=0).tolist()) #all associated radar points
        else:
            rad_to_gps_ass.append(rad_to_gps_res[idx_radar[i]].tolist()) #all associated radar points
    
    #################  association error  ##########################################
    ed=element_wise_euclidian_dist(np.array(rad_to_gps_ass)[:,0:2], np.squeeze(all_gps_enu_arr)[:,0:2])
    aed=np.mean(ed)
    udsd=stats.tstd(ed) #the corrected distance standard deviation(CDSD),Unbiased distance standard deviation(UDSD)
    #print("AED: {}; UDSD_error: {}".format(aed,udsd))
    return rad_to_gps_ass,aed,udsd
    
    
def asso_gps_to_radar(ts_centroids_rad,ts_rover1,ts_rover2,ts_rover3,all_gps_enu_arr1,all_gps_enu_arr2,all_gps_enu_arr3):
    #################### group_3_gps_as_1_frame by using radar as anchor
    idx_radar1,radar_matched_val1 = find_nearest_betw_arr(np.array(ts_centroids_rad), np.array(ts_rover1)) # find each gps's corresponding radar_position
    idx_radar2,radar_matched_val2 = find_nearest_betw_arr(np.array(ts_centroids_rad), np.array(ts_rover2))
    idx_radar3,radar_matched_val3 = find_nearest_betw_arr(np.array(ts_centroids_rad), np.array(ts_rover3))
    gps_all_for_radar=[]
    for i in range(len(ts_centroids_rad)): #same with len(ts_radar_fr)  
        gps_in_one_rad_fr=[]
        if i in idx_radar1:
            gps_in_one_rad_fr.append(all_gps_enu_arr1[ np.where(idx_radar1==i)[0][0] ])#First index is np.where()[0][0]; All index is np.where()[0]
        if i in idx_radar2:
            gps_in_one_rad_fr.append(all_gps_enu_arr2[ np.where(idx_radar2==i)[0][0] ])            
        if i in idx_radar3:
            gps_in_one_rad_fr.append(all_gps_enu_arr3[ np.where(idx_radar3==i)[0][0] ])   
        gps_all_for_radar.append(gps_in_one_rad_fr) 
    
    # #################  association error  ##########################################
    # for i in range(len(ts_centroids_rad)):
    #     if np.array(gps_all[i]).size:
    #         avg_dist=match_gps_to_rad( radar_cent,gps_all[i], max_dis_cost=20)
    #         print("AED: {}".format(avg_dist))
    return gps_all_for_radar


def match_gps_to_rad(trackers, detections, max_dis_cost=20):
    ''' trackers=radar, detections=gps '''
    # Initialize 'cost_matrix'
    cost_matrix = np.zeros(shape=(len(trackers), len(detections)), dtype=np.float32)

    # Populate 'cost_matrix'
    for t, tracker in enumerate(trackers):#trackers, detections just positions
        for d, detection in enumerate(detections):
            cost_matrix[t,d] = distance.euclidean(tracker, detection) 
    # find mini_cost
    cost_arr = np.sort(cost_matrix, axis=None)     # sort the flattened array
    min_len = min(len(trackers),len(detections))
    # cost_arr = cost_arr[0:len(trackers)]
    cost_arr = cost_arr[0:min_len]
    avg_dist = np.sum(cost_arr)/len(cost_arr)    #len(trackers)  avg_dist in one frame    
    # # Do match
    # row_ind, col_ind = linear_assignment(cost_matrix) #`row_ind array` for `tracks`,`col_ind` for `detections`
    # # Populate 'unmatched_trackers'
    # unmatched_trackers = []
    # for t in np.arange(len(trackers)):
    #     if t not in row_ind: #`row_ind` for `tracks`
    #         unmatched_trackers.append(t)
    # # Populate 'unmatched_detections'
    # unmatched_detections = []
    # for d in np.arange(len(detections)):
    #     if d not in col_ind:#`col_ind` for `detections`
    #         unmatched_detections.append(d)
    # # Populate 'matches'
    # matches = []
    # for t_idx, d_idx in zip(row_ind, col_ind):
    #     if cost_matrix[t_idx,d_idx] < max_dis_cost:
    #         matches.append([t_idx, d_idx])
    #     else:
    #         unmatched_trackers.append(t_idx)
    #         unmatched_detections.append(d_idx)

    # return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)
    return avg_dist


if __name__ == '__main__':
    ######################################################### GPS Tracks ####################################################
    #################################   Read Ros Bag For Tracks ################################################   
    #################### load radar_data_bag by referencing timestamps Directly ##################################################
    rad_bag = rosbag.Bag('./12182022/tracking_data_2022-12-18-16-14-12.bag')
    ## for radar frame
    total_frame_data = [] #all frames
    frame_data = []  #one frame
    frame_ts = []  #one frame's timestamp
    for topic, msg, t in rad_bag.read_messages(topics=['/sony_radar/radar_scan']):
        if ((msg.point_id == 0) and (frame_data)):# initialize or refresh frame_list for first frame or next frame
            total_frame_data.append(frame_data) #append one frame, index by total_frame_data[frame_idx][point_idx][x,y,z,doppler,time]
            #frame_ts.append(frame_data[0][-1]) # set the frame's timestamp with the timestamp of the first radar point in that frame
            frame_ts.append((np.array(frame_data)[:,-1]).mean()) # set the frame's timestamp with the mean value of timestamps of all radar points in that frame
            frame_data = []
        if msg.target_idx <253: #Remove Unassociated/Weak/Noise points    
            #populate one frame with point whose target_idx <253  
            frame_data.append([msg.x, msg.y, msg.z, msg.doppler,msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9])    
    rad_bag.close()
    ############# get centroids
    centroids_radar=[]
    ts_centroids_rad=[]
    for i in range(len(total_frame_data)):
        xyz_centroids= np.array([[]])#initialization
        one_frame=np.array(total_frame_data[i])
        xyzd_frame=one_frame[:,0:4]#Select only the first four columns:x,y,z,doppler for DBSCAN
        #get centroids in one frame
        centroids=centroids_of_clusters(xyzd_frame,eps=1.5,min_samples=8, metric='euclidean')
        if centroids.size != 0:
            #store centroids of one frame
            xyz_centroids=centroids[:,0:3]#Select only the first three columns:x,y,z
            centroids_radar.append(xyz_centroids)# only store non-empty centroids
            ts_centroids_rad.append(frame_ts[i])        
        #centroids_radar.append(xyz_centroids)# also store empty centroids when out of loop
        
    ################################################ Obtain Track for ENU, this for using compass to associate with radar #############################
    ####################  load GPS rover bag by using rosbag ##################################################
    base_llh=[32.23534993,-110.95383927,710.459]
    bag_path='./12182022/gps_data_2022-12-18-16-14-15.bag'
    ts_rover1,all_gps_enu_arr1,x1,y1 = load_gps_data(bag_path,ros_topic=['/rover1/tcpfix'],translation=1,base_llh=base_llh)
    ts_rover2,all_gps_enu_arr2,x2,y2 = load_gps_data(bag_path,ros_topic=['/rover2/tcpfix'],translation=1,base_llh=base_llh)
    ts_rover3,all_gps_enu_arr3,x3,y3 = load_gps_data(bag_path,ros_topic=['/rover3/tcpfix'],translation=1,base_llh=base_llh)
    
    #################### group_3_gps_as_1_frame by using one gps as anchor
    len_ts_rover = [len(ts_rover1),len(ts_rover2),len(ts_rover3)]
    ts_rover_list = [ts_rover1,ts_rover2,ts_rover3]
    all_gps_enu_arr_list = [all_gps_enu_arr1,all_gps_enu_arr2,all_gps_enu_arr3]
    min_idx  = np.argmin(len_ts_rover)
    ts_rover_min        = ts_rover_list[min_idx]
    all_gps_enu_arr_min = all_gps_enu_arr_list[min_idx]
    idx_list = [0,1,2]
    idx_list.remove(min_idx)
    ts_rover_a = ts_rover_list[idx_list[0]]
    ts_rover_b = ts_rover_list[idx_list[1]]
    all_gps_enu_arr_a = all_gps_enu_arr_list[idx_list[0]]
    all_gps_enu_arr_b = all_gps_enu_arr_list[idx_list[1]]
    idx_a,_ = find_nearest_betw_arr(np.array(ts_rover_a), np.array(ts_rover_min))
    idx_b,_ = find_nearest_betw_arr(np.array(ts_rover_b), np.array(ts_rover_min))
    # group
    ts_rover = ts_rover_min
    all_gps_enu_arr=[]
    for i in range(len(ts_rover_min)): #timestamp also can be the avg_val of these three gps
        all_gps_enu_arr.append( [all_gps_enu_arr_min[i].tolist(),all_gps_enu_arr_a[idx_a[i]].tolist(),all_gps_enu_arr_b[idx_b[i]].tolist()] ) #all associated radar points
    
    all_gps_enu_arr=np.array(all_gps_enu_arr) #this arr may not contain only one track, so the below plot may be unreal
    #################### group_3_gps_as_1_frame by using radar as anchor
    idx_radar1,radar_matched_val1 = find_nearest_betw_arr(np.array(ts_centroids_rad), np.array(ts_rover1)) # find each gps's corresponding radar_position
    idx_radar2,radar_matched_val2 = find_nearest_betw_arr(np.array(ts_centroids_rad), np.array(ts_rover2))
    idx_radar3,radar_matched_val3 = find_nearest_betw_arr(np.array(ts_centroids_rad), np.array(ts_rover3))
    gps_all=[]
    for i in range(len(ts_centroids_rad)): #same with len(ts_radar_fr)  
        gps_in_one_rad_fr=[]
        if i in idx_radar1:
            gps_in_one_rad_fr.append(all_gps_enu_arr1[ np.where(idx_radar1==i)[0][0] ])#First index is np.where()[0][0]; All index is np.where()[0]
        if i in idx_radar2:
            gps_in_one_rad_fr.append(all_gps_enu_arr2[ np.where(idx_radar2==i)[0][0] ])            
        if i in idx_radar3:
            gps_in_one_rad_fr.append(all_gps_enu_arr3[ np.where(idx_radar3==i)[0][0] ])   
        gps_all.append(gps_in_one_rad_fr)    
    #################### process_radar_for_gps
    radar_data_arr,ts_centroids_rad_inlier,rad_to_gps_res = process_radar_for_gps(centroids_radar,theta=[-91,3.2,0])

 
    ###############################################  Associate GPS and Radar 
    rad_to_gps_ass,aed,udsd = asso_radar_to_gps(rad_to_gps_res,all_gps_enu_arr,ts_centroids_rad_inlier, ts_rover)
    print("AED: {}; UDSD_error: {}".format(aed,udsd))
    
    
    
    
    ############################################# Plot for visulization ###################################################
    
    '''############################################### for GPS '''
    x = all_gps_enu_arr[:,0] # X is E
    y = all_gps_enu_arr[:,1] # Y is S, or, -N
    plt.figure(dpi=400)
    plt.xlim(x.max()+0.2, x.min()-0.2)
    plt.ylim(y.max()+0.2, y.min()-0.2) #plt.gca().invert_yaxis()
    for i in range (np.size(x)):
        plt.plot(x[i:i+2], y[i:i+2], 'ro-', ms = 2, mec = 'g', mfc = 'g') #x[start:end:step]
        #plt.plot(x[i], y[i], ls='', marker='o', color='r')
        #plt.pause(0.2)
    plt.title("GPS Groudtruth Track")
    plt.xlabel("Direction(m)")
    plt.ylabel("Distance(m)")
    # Setting the interval of ticks of x-axis and y-axis to 0.5.
    plt.xticks(np.arange(x.max()+0.2, x.min()-0.2, -0.2), rotation=90) 
    #plt.xticks(np.arange(3, -4, -0.2), rotation=90) 
    plt.yticks(np.arange(y.max()+0.2, y.min()-0.2, -0.4))
    # fmt = lambda x: "{:.2f}%".format(x) # formating x tick
    # plt.xticks(np.arange(x.max()+0.2, x.min()-0.2, -0.2),[fmt(i) for i in np.arange(x.max()+0.2, x.min()-0.2, -0.2)], rotation=90) 
    # plt.yticks(np.arange(y.max()+0.2, y.min()-0.2, -0.4), [fmt(i) for i in np.arange(y.max()+0.2, y.min()-0.2, -0.4)])
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # No decimal places
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 2 decimal places 
    plt.tight_layout()
    plt.grid()
    plt.show()
    
    '''############################################# for Radar '''
    y = radar_data_arr[:,0] #inverse x and y for aligning GPS
    x = radar_data_arr[:,1]
    plt.figure(dpi=400)
    plt.xlim(x.max()+0.2, x.min()-0.2)
    plt.ylim(y.min()-0.2, y.max()+0.2) #plt.gca().invert_yaxis()
    for i in range(0, len(x), 1): #for i in range(0, len(x), 1):
    #inverse x and y for aligning GPS
        plt.plot(x[i:i+2], y[i:i+2], 'ro-', ms = 2, mec = 'g', mfc = 'g')#markersize,markeredgecolor,markerfacecolor
    #plt.title("Radar Track before outliers removing")
    plt.title("Radar Track after outliers removing")
    plt.xlabel("Direction(m)")
    plt.ylabel("Distance(m)")    
    plt.xticks(np.arange(x.max()+0.2, x.min()-0.2, -0.2), rotation=90) 
    plt.yticks(np.arange(y.min()-0.2, y.max()+0.2, 0.4))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # No decimal places
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 2 decimal places 
    plt.tight_layout()
    plt.grid()
    plt.show()
    
    '''################################################### radar to gps '''    
    ############# Plot GPS_for_radar
    x = rad_to_gps_res[:,0]
    y = rad_to_gps_res[:,1]
    plt.figure(dpi=400)
    plt.xlim(x.max()+0.2, x.min()-0.2)
    plt.ylim(y.max()+0.2, y.min()-0.2) #plt.gca().invert_yaxis()
    for i in range (np.size(x)):
    #for i in range (np.size(x)): #for i in range(0, len(x), 2):
        plt.plot(x[i:i+2], y[i:i+2], 'ro-', ms = 2, mec = 'g', mfc = 'g') #x[start:end:step]
    plt.title("Radar Track under GPS ref-frame")
    plt.xlabel("Direction(m)")
    plt.ylabel("Distance(m)") 
    plt.xticks(np.arange(x.max()+0.2, x.min()-0.2, -0.2), rotation=90) 
    plt.yticks(np.arange(y.max()+0.2, y.min()-0.2, -0.4))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # No decimal places
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 2 decimal places 
    plt.tight_layout()
    plt.grid()
    plt.show()
    
    '''###############################################  Associate GPS and Radar '''
    ############# Plot associated GPS_for_radar
    rad_to_gps_ass=np.array(rad_to_gps_ass)
    x = rad_to_gps_ass[:,0]
    y = rad_to_gps_ass[:,1]
    plt.figure(dpi=400)
    plt.xlim(x.max()+0.2, x.min()-0.2)
    plt.ylim(y.max()+0.2, y.min()-0.2) #plt.gca().invert_yaxis()
    for i in range (np.size(x)):
    #for i in range (np.size(x)): #for i in range(0, len(x), 2):
        plt.plot(x[i:i+2], y[i:i+2], 'ro-', ms = 2, mec = 'g', mfc = 'g') #x[start:end:step]
    plt.title("Associated Radar to GPS Track")
    plt.xlabel("Direction(m)")
    plt.ylabel("Distance(m)") 
    plt.xticks(np.arange(x.max()+0.2, x.min()-0.2, -0.2), rotation=90) 
    plt.yticks(np.arange(y.max()+0.2, y.min()-0.2, -0.4))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # No decimal places
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 2 decimal places 
    plt.tight_layout()
    plt.grid()
    plt.show()
    
    '''############# Plot associated radar and GPS in the same plot '''
    rad_and_gps_concat=np.vstack((all_gps_enu_arr,rad_to_gps_ass))
    x = rad_and_gps_concat[:,0]
    y = rad_and_gps_concat[:,1]
    #y = -rad_and_gps_concat[:,1]
    plt.figure(dpi=400)
    plt.xlim(x.max()+0.2, x.min()-0.2)
    plt.ylim(y.max()+0.2, y.min()-0.2) #plt.gca().invert_yaxis()
    for i in range (np.size(x)):
        if i<len(all_gps_enu_arr): #plot gps
            if i==0: #add legend
                plt.plot(x[i:i+2], y[i:i+2], 'ro-', ms = 2, mec = 'g', mfc = 'g',label='GPS') #x[start:end:step]
            else:
                plt.plot(x[i:i+2], y[i:i+2], 'ro-', ms = 2, mec = 'g', mfc = 'g') #x[start:end:step]
        else:  #plot radar
            if i==len(all_gps_enu_arr): #add legend
                plt.plot(x[i:i+2], y[i:i+2], 'b^-', ms = 2, mec = 'm', mfc = 'm',label='Radar') #x[start:end:step]
            else:
                plt.plot(x[i:i+2], y[i:i+2], 'b^-', ms = 2, mec = 'm', mfc = 'm') #x[start:end:step]
    plt.title("Associated Radar and GPS Two Tracks")
    plt.xlabel("Direction(m)")
    plt.ylabel("Distance(m)")
    #plt.legend(['GPS', 'Radar'],loc="upper right") 
    plt.legend(loc="upper right")
    #plt.xticks(np.arange(3, -4, -0.2), rotation=90) 
    plt.xticks(np.arange(x.max()+0.2, x.min()-0.2, -0.2), rotation=90) 
    plt.yticks(np.arange(y.max()+0.2, y.min()-0.2, -0.4))
    # plt.yticks(np.arange(y.min()-0.2, y.max()+0.2, 0.4))
    # plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # No decimal places
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places 
    plt.tight_layout()
    plt.grid()
    plt.show()
