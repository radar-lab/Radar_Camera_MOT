#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""

import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore


def remove_outliers_zscore(data, threshold=3):
    # remove outliers using z-score and return filtered data's indices
    z_scores = np.abs(zscore(data, axis=0))
    filtered_indices = np.where((z_scores < threshold).all(axis=1))[0]
    #filtered_data = data[filtered_indices]
    return filtered_indices

def rad_cam_Homography(radar_points, img_points,validation_radar_points, validation_img_points, 
                       method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995):
    # Use cv2.findHomography() to calculate the perspective transformation matrix: cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    # #RANSAC
    homography_matrix, inliers = cv2.findHomography(radar_points, img_points, method=method, 
                                                    ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters, confidence=confidence)
    
    # #LMEDS
    # homography_matrix, inliers = cv2.findHomography(radar_points, img_points, method=cv2.LMEDS
    #                                               , ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995)
    
    # #RHO
    # homography_matrix, inliers = cv2.findHomography(radar_points, img_points, method=cv2.RHO
    #                                               , ransacReprojThreshold=10.0, maxIters=2000, confidence=0.995)
    
    # Output the homography matrix
    print("Homography Matrix:",homography_matrix)
    
    
    ## Outliers and Inliers
    # Apply the perspective transformation using homography_matrix
    transformed_points = cv2.perspectiveTransform(radar_points.reshape(-1, 1, 2), homography_matrix)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points - img_points.reshape(-1, 1, 2)
    mean_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Mean Reprojection Error:", mean_reprojection_error)
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("RMSE:", rmse)
    
    
    ## Inliers
    inliers_points = radar_points[inliers.ravel()==1] #inliers.ravel().astype(bool)
    # Get Projected Points corresponding to inliers
    transformed_inliers = transformed_points[inliers.ravel()==1]
    inliers_reprojection_errors = reprojection_errors[inliers.ravel()==1]
    mean_reprojection_error_inliers = np.mean(np.linalg.norm(inliers_reprojection_errors, axis=-1))
    # Print the inliers and their Mean Reprojection Error
    #print("Inliers:", inliers_points)
    print("Number of inliers:", len(inliers_points))
    print("Mean Reprojection Error (inliers):", mean_reprojection_error_inliers)
    # Calculate Inliers RMSE
    inliers_rmse = np.sqrt(np.mean(np.square(inliers_reprojection_errors)))
    print("Inliers RMSE:", inliers_rmse)
    
    ## Validation points
    # Apply the perspective transformation using homography_matrix
    transformed_points_val = cv2.perspectiveTransform(validation_radar_points.reshape(-1, 1, 2), homography_matrix)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_val - validation_img_points.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Validation-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("Validation-RMSE:", val_rmse)
    
    return homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val

def rad_cam_Affine(radar_points, img_points,validation_radar_points, validation_img_points, 
                       method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995):
    # solve error: (-215:Assertion failed) count >= 0 && to.checkVector(2) == count
    radar_points = radar_points.astype(np.float32)
    img_points = img_points.astype(np.float32)

    # Use cv2.estimateAffine2D() to calculate the affine transformation matrix
    # #RANSAC
    affine_matrix, inliers = cv2.estimateAffine2D(radar_points, img_points, method=method, 
                                                    ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters, confidence=confidence)
    
    # # #LMEDS
    # affine_matrix, inliers = cv2.estimateAffine2D(radar_points, img_points, method=cv2.LMEDS
    #                                               , ransacReprojThreshold=30.0, maxIters=2000, confidence=0.995)
    
    # Output the affine matrix
    print("Affine Matrix:",affine_matrix)
    
    
    ## Outliers and Inliers
    # Apply the affine transformation using affine_matrix
    transformed_points = cv2.transform(radar_points.reshape(-1, 1, 2), affine_matrix) 
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points - img_points.reshape(-1, 1, 2)
    mean_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Mean Reprojection Error:", mean_reprojection_error)
    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("RMSE:", rmse)
    
    
    ## Inliers
    inliers_points = radar_points[inliers.ravel()==1] #inliers.ravel().astype(bool)
    # Get Projected Points corresponding to inliers
    transformed_inliers = transformed_points[inliers.ravel()==1]
    inliers_reprojection_errors = reprojection_errors[inliers.ravel()==1]
    mean_reprojection_error_inliers = np.mean(np.linalg.norm(inliers_reprojection_errors, axis=-1))
    # Print the inliers and their Mean Reprojection Error
    #print("Inliers:", inliers_points)
    print("Number of inliers:", len(inliers_points))
    print("Mean Reprojection Error (inliers):", mean_reprojection_error_inliers)
    # Calculate Inliers RMSE
    inliers_rmse = np.sqrt(np.mean(np.square(inliers_reprojection_errors)))
    print("Inliers RMSE:", inliers_rmse)
    
    ## Validation points
    # Apply the affine transformation using affine_matrix
    transformed_points_val = cv2.transform(validation_radar_points.reshape(-1, 1, 2), affine_matrix) 
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_val - validation_img_points.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("Validation-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("Validation-RMSE:", val_rmse)
    
    return affine_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val


def show_calib_result(radar_points, img_points, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_radar_points, transformed_points_val):
    # Plot the transformed_points and img_points with Connect the corresponding points with lines
    # Create a new figure
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    # Plot the transformed_points and img_points
    ax.scatter(img_points[:, 0], img_points[:, 1], c='blue', label='Image Points')
    ax.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='Radar Points')
    # Connect the corresponding points with lines
    for i in range(len(img_points)):
        x = [img_points[i, 0], transformed_points[i, 0, 0]]
        y = [img_points[i, 1], transformed_points[i, 0, 1]]
        ax.plot(x, y, color='green', alpha=0.5)
    # Set plot title and labels
    ax.set_title('Radar-Image Point Correspondences')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Invert the Y-axis
    ax.invert_yaxis()
    # Show legend in the upper left corner
    ax.legend(loc='upper left')
    ax.grid(True)
    # Show the plot
    plt.show()
    
    
    # Plot the transformed_points and img_points
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(img_points[:, 0], img_points[:, 1], c='blue', label='Image Points')
    plt.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='Radar Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Radar Points vs Image Points')
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(img_points[:, 0], img_points[:, 1], c='blue', label='Image Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Image Points')
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='Radar Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Radar Points')
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(radar_points[:, 0], radar_points[:, 1], c='red', label='Radar Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original Radar Points')
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    # Plot the transformed_inliers and inliers_points for inliers
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(inliers_points[:, 0], inliers_points[:, 1], c='blue', label='Original Radar Points')
    plt.scatter(transformed_inliers[:, 0, 0], transformed_inliers[:, 0, 1], c='red', label='Radar Points')
    # Connect the corresponding points with lines
    for i in range(len(transformed_inliers)):
        x = [inliers_points[i, 0], transformed_inliers[i, 0, 0]]
        y = [inliers_points[i, 1], transformed_inliers[i, 0, 1]]
        plt.plot(x, y, color='green', alpha=0.5)
    # Set plot title and labels
    plt.title('Radar-Original Radar Inlier Correspondences')
    plt.xlabel('X')
    plt.ylabel('Y')
    # Show legend and grid
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    # Show the plot
    plt.show()
    
    
    # Plot the transformed_inliers and inliers_points for inliers
    plt.figure(figsize=(8, 6), dpi=300)
    img_inliers = img_points[inliers.ravel() == 1, :]
    plt.scatter(img_inliers[:, 0], img_inliers[:, 1], c='blue', label='Image Points')
    plt.scatter(transformed_inliers[:, 0, 0], transformed_inliers[:, 0, 1], c='red', label='Radar Points')
    # Connect the corresponding points with lines
    for i in range(len(transformed_inliers)):
        x = [img_inliers[i, 0], transformed_inliers[i, 0, 0]]
        y = [img_inliers[i, 1], transformed_inliers[i, 0, 1]]
        plt.plot(x, y, color='green', alpha=0.5)
    # Set plot title and labels
    plt.title('Radar-Image Inlier Correspondences')
    plt.xlabel('X')
    plt.ylabel('Y')
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    # Show legend in the upper left corner
    plt.legend(loc='upper left')
    plt.grid(True)
    # Show the plot
    plt.show()
    
    
    # Plot the transformed_points and img_points with Connect the corresponding points with lines
    # Create a new figure
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    # Plot the transformed_points and img_points
    ax.scatter(img_points[:, 0], img_points[:, 1], c='blue', label='Image Points')
    ax.scatter(transformed_points[:, 0, 0], transformed_points[:, 0, 1], c='red', label='Radar Points')
    # Connect the corresponding points with lines
    for i in range(len(img_points)):
        x = [img_points[i, 0], transformed_points[i, 0, 0]]
        y = [img_points[i, 1], transformed_points[i, 0, 1]]
        ax.plot(x, y, color='green', alpha=0.5)
    
    # Mark the inliers with hollow circles
    ax.scatter(img_points[inliers.ravel() == 1, 0], img_points[inliers.ravel() == 1, 1], c='none', label='Inliers', facecolors='none', edgecolors='cyan', marker='o', s=50)
    ax.scatter(transformed_points[inliers.ravel() == 1, 0, 0], transformed_points[inliers.ravel() == 1, 0, 1], c='none', facecolors='none', edgecolors='cyan', marker='o', s=50) #marker='^'
    
    # Set plot title and labels
    ax.set_title('Radar-Image Point Correspondences')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Invert the Y-axis
    ax.invert_yaxis()
    # Show legend in the upper left corner
    ax.legend(loc='upper left')
    ax.grid(True)
    # Show the plot
    plt.show()
	
    # Plot the transformed_points_val and validation_img_points with Connect the corresponding points with lines
    # Create a new figure
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    # Plot the transformed_points_val and validation_img_points
    ax.scatter(validation_img_points[:, 0], validation_img_points[:, 1], c='blue', label='Image Points')
    ax.scatter(transformed_points_val[:, 0, 0], transformed_points_val[:, 0, 1], c='red', label='Radar Points')
    # Connect the corresponding points with lines
    for i in range(len(validation_img_points)):
        x = [validation_img_points[i, 0], transformed_points_val[i, 0, 0]]
        y = [validation_img_points[i, 1], transformed_points_val[i, 0, 1]]
        ax.plot(x, y, color='green', alpha=0.5)
    
    # Set plot title and labels
    ax.set_title('Validation Radar-Image Point Correspondences')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Invert the Y-axis
    ax.invert_yaxis()
    # Show legend in the upper left corner
    ax.legend(loc='upper left')
    ax.grid(True)
    # Show the plot
    plt.show()	
    
    
def up_down_Validation(homography_matrix_up, homography_matrix_down, validation_radar_points, validation_img_points,Y_thr):
    ### Split to 2 plane to do calibration seperatly
    # Split the points based on the Y-coordinate threshold
    #Y_thr = 320  # img_h - (img_h / 3)
    up_idx = validation_img_points[:, 1] <= Y_thr
    down_idx = validation_img_points[:, 1] > Y_thr
    img_points_up = validation_img_points[up_idx]
    img_points_down = validation_img_points[down_idx]
    radar_points_up = validation_radar_points[up_idx]
    radar_points_down = validation_radar_points[down_idx]    
    
    ### Validation points UP
    # Apply the perspective transformation using homography_matrix
    transformed_points_up = cv2.perspectiveTransform(radar_points_up.reshape(-1, 1, 2), homography_matrix_up)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_up - img_points_up.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("UP-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("UP-RMSE:", val_rmse)
    
    ### Validation points DOWN
    # Apply the perspective transformation using homography_matrix
    transformed_points_down = cv2.perspectiveTransform(radar_points_down.reshape(-1, 1, 2), homography_matrix_down)    
    # Calculate the reprojection error using homography_matrix
    reprojection_errors = transformed_points_down - img_points_down.reshape(-1, 1, 2)
    val_reprojection_error = np.mean(np.linalg.norm(reprojection_errors, axis=-1))
    # Output  mean reprojection error
    print("DOWN-Reprojection Error:", val_reprojection_error)
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean(np.square(reprojection_errors)))
    print("DOWN-RMSE:", val_rmse)
    
    # Plot the transformed_points_val and validation_img_points with Connect the corresponding points with lines
    # Create a new figure
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)
    # Plot the transformed_points_up and validation_img_points
    ax.scatter(img_points_up[:, 0], img_points_up[:, 1], c='blue', label='Image Points')
    ax.scatter(transformed_points_up[:, 0, 0], transformed_points_up[:, 0, 1], c='red', label='Radar Points')
    # Connect the corresponding points with lines
    for i in range(len(img_points_up)):
        x = [img_points_up[i, 0], transformed_points_up[i, 0, 0]]
        y = [img_points_up[i, 1], transformed_points_up[i, 0, 1]]
        ax.plot(x, y, color='green', alpha=0.5)
        
    # Plot the transformed_points_down and validation_img_points
    ax.scatter(img_points_down[:, 0], img_points_down[:, 1], c='blue')
    ax.scatter(transformed_points_down[:, 0, 0], transformed_points_down[:, 0, 1], c='red')
    # Connect the corresponding points with lines
    for i in range(len(img_points_down)):
        x = [img_points_down[i, 0], transformed_points_down[i, 0, 0]]
        y = [img_points_down[i, 1], transformed_points_down[i, 0, 1]]
        ax.plot(x, y, color='orange', alpha=0.5)        
    
    # Set plot title and labels
    ax.set_title('Validation Radar-Image Point Correspondences')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Invert the Y-axis
    ax.invert_yaxis()
    # Show legend in the upper left corner
    ax.legend(loc='upper left')
    ax.grid(True)
    # Show the plot
    plt.show()	    
    
    
    
#####################################################################################

if __name__ == "__main__":
    ###########################  Load Centers  #########################################
    names = ['lei_alone'] #['hao-jiahao'] #['lei-leicar'] #['jiahao-hao'] #['lei_alone']#['2person-car']    
    path = '/xdisk/caos/leicheng/calibration_based_on_common_features_07272024/' + names[0]
    
    ####################################################################################
    # Define additional Validation points
    filename = os.path.join(path,'selected_pairs_array_1.npy')
    # Load the data from the npy file
    data = np.load(filename, allow_pickle=True)
    # Define camera image coordinates and radar image coordinates
    validation_img_points   = data[:,0]
    validation_radar_points = data[:,1]
    
    
    ##########################  Points used for Calibration    ##########################################################
    filename = os.path.join(path,'selected_pairs_array_3.npy')
    # Load the data from the npy file
    data = np.load(filename, allow_pickle=True)
    # Define camera image coordinates and radar image coordinates
    img_points   = data[:,0]
    radar_points = data[:,1]
    # Remove outliers from the data points
    filtered_indices = remove_outliers_zscore(img_points, threshold=3)
    img_points       = img_points[filtered_indices]
    radar_points     = radar_points[filtered_indices]
    ####################################################################################
    
    
    USE_findHomography   = 1
    USE_estimateAffine2D = 0
    if USE_findHomography:
        ### For all points
        homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
            rad_cam_Homography(radar_points, img_points, validation_radar_points, validation_img_points, 
                              method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
        # Save  as an .npy file
        np.save(path+'/homography_matrix_rad2cam.npy', homography_matrix) 
        # Plot calibration results
        show_calib_result(radar_points, img_points, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_radar_points, transformed_points_val)
        
        
        ### Split to 2 plane to do calibration seperatly
        # Split the points based on the Y-coordinate threshold
        Y_thr = 320  # img_h - (img_h / 3)
        up_idx = img_points[:, 1] <= Y_thr
        down_idx = img_points[:, 1] > Y_thr
        img_points_up = img_points[up_idx]
        img_points_down = img_points[down_idx]
        radar_points_up = radar_points[up_idx]
        radar_points_down = radar_points[down_idx]
        ### For UP
        homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
            rad_cam_Homography(radar_points_up, img_points_up, validation_radar_points, validation_img_points, 
                              method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
        # Save  as an .npy file
        np.save(path+'/homography_matrix_rad2cam_up.npy', homography_matrix) 
        # Plot calibration results
        show_calib_result(radar_points_up, img_points_up, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_radar_points, transformed_points_val)
        
        
        ### For DOWN
        homography_matrix, inliers, inliers_points, transformed_points, transformed_inliers, transformed_points_val = \
            rad_cam_Homography(radar_points_down, img_points_down, validation_radar_points, validation_img_points, 
                              method=cv2.RANSAC, ransacReprojThreshold=50.0, maxIters=4000, confidence=0.995)
        # Save  as an .npy file
        np.save(path+'/homography_matrix_rad2cam_down.npy', homography_matrix) 
        # Plot calibration results
        show_calib_result(radar_points_down, img_points_down, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_radar_points, transformed_points_val)    

        ### Validation For UP and DOWN
        homography_matrix_up   = np.load(path+'/homography_matrix_rad2cam_up.npy', allow_pickle=True)
        homography_matrix_down = np.load(path+'/homography_matrix_rad2cam_down.npy', allow_pickle=True)
        up_down_Validation(homography_matrix_up, homography_matrix_down, validation_radar_points, validation_img_points,Y_thr)    
        
        
    elif USE_estimateAffine2D:
        affine_matrix, inliers, inliers_points, transformed_points, transformed_inliers = \
            rad_cam_Affine(radar_points, img_points,validation_radar_points, validation_img_points, 
                              method=cv2.RANSAC, ransacReprojThreshold=30.0, maxIters=4000, confidence=0.995)
        # Save  as an .npy file
        np.save(path+'affine_matrix_rad2cam.npy', affine_matrix)    
        # Plot calibration results
        show_calib_result(radar_points, img_points, inliers, inliers_points, transformed_points, transformed_inliers,validation_img_points,validation_radar_points, transformed_points_val)
    
    
    # Output the Projected Points
    #print("Projected Points:",transformed_points)


