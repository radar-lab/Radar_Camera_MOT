#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:03:35 2023

@author: leicheng
"""

import cv2
import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def images_to_video(input_folder, output_file, fps=30):
    # Get the list of images in the folder and sort based on numeric values in filenames
    image_list = sorted(os.listdir(input_folder), key=natural_sort_key)
    
    
    # Read the first image to get dimensions
    img = cv2.imread(os.path.join(input_folder, image_list[0]))
    height, width, _ = img.shape

    # Define the video codec and create a VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change the codec to XVID for .avi format
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Iterate through the images and write to video
    for image in image_list:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release resources
    cv2.destroyAllWindows()
    video.release()

# Example usage
input_folder_path = '/home/u14/leicheng/Desktop/RADYolo_Enhanced_Radar_Camera_Fused_Object_Tracking/out_images'  # Replace with your folder path
output_video_path = '/home/u14/leicheng/Desktop/RADYolo_Enhanced_Radar_Camera_Fused_Object_Tracking/output_video_1.avi'  # Replace with desired video file name
images_to_video(input_folder_path, output_video_path, fps=30)  # Generate video
