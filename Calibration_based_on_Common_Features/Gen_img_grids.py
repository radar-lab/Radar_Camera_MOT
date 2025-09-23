#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 02:10:42 2023

@author: leicheng
"""
import math
import numpy as np
import random
import glob, re, os
# Image size and number of regions
image_width = 640
image_height = 480
num_cols = 128#320#160#128#80#64#32#16
num_rows = 96#240#120#96#60#48#24#12

################# Randomly generate pixel coordinates for TEST  #####################
# num = 100
# pixels = np.random.randint(0, image_width * image_height, size=(num,))  # Generate num random pixels
# pixel_coords = [(pixel // image_width, pixel % image_width) for pixel in pixels]

# # Convert pixel_coords to the desired format [[pixel_coord], [pixel_coord], [pixel_coord], ...]
# pixel_coords = [list(pixel_coord) for pixel_coord in pixel_coords]

# # Print the generated pixel coordinates
# print("Pixel Coordinates:")
# #print(pixel_coords)

###########################  Load Centers  #########################################
names = ['2person-car'] #['hao-jiahao'] #['lei-leicar'] #['jiahao-hao'] #['lei_alone'] #['2person-car']   
path = '/xdisk/caos/leicheng/calibration_based_on_common_features_07272024/' + names[0]
        
npy_paths = glob.glob(path + "/paired*.npy")
npy_paths = sorted(npy_paths, key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else float('inf'))        
# Load and concatenate all .npy files
all_data = []

for npy_path in npy_paths:
    data = np.load(npy_path, allow_pickle=True)
    all_data.append(data)

# Concatenate all loaded data along the first axis
data = np.concatenate(all_data, axis=0)
del all_data


# filename = path + 'paired_centroids.npy'  #jiahao-hao
# data = np.load(filename, allow_pickle=True)
pixel_coords = data[:,0]
####################################################################################
# Calculate the region size
region_width = image_width // num_cols
region_height = image_height // num_rows
print("Region Width:", region_width)
print("Region Height:", region_height)

# Calculate the coordinates of region centers
region_centers = []
for row in range(num_rows):
    for col in range(num_cols):
        center_x = col * region_width + region_width // 2
        center_y = row * region_height + region_height // 2
        region_centers.append((center_x, center_y))



# # Assign pixels to the nearest region
# assigned_regions = {}
# for pixel_coord in zip(*pixel_coords):
#     distances = [np.sqrt((pixel_coord[0] - center[1])**2 + (pixel_coord[1] - center[0])**2) for center in region_centers]
#     closest_region = np.argmin(distances)
#     if closest_region in assigned_regions:
#         # If the closest region is already in the assigned_regions dictionary,
#         # append the current pixel_coord to the list of assigned pixels for that region.
#         assigned_regions[closest_region].append(pixel_coord)
#     else:
#         # If the closest region is not yet present in the assigned_regions dictionary,
#         # create a new entry in the dictionary with the closest_region as the key
#         # and a list containing only the current pixel_coord as the value.
#         assigned_regions[closest_region] = [pixel_coord]

# Assign pixels to the respective regions they fall into
assigned_regions = {}
for pixel_coord, img_rad_coord in zip(pixel_coords, data):
    region_index = (pixel_coord[0] // region_height) * num_cols + (pixel_coord[1] // region_width)
    
    if region_index in assigned_regions:
        # If the closest region is already in the assigned_regions dictionary,
        # append the current pixel_coord to the list of assigned pixels for that region.
        assigned_regions[region_index].append(img_rad_coord)
    else:
        # If the closest region is not yet present in the assigned_regions dictionary,
        # create a new entry in the dictionary with the closest_region as the key
        # and a list containing only the current pixel_coord as the value.
        assigned_regions[region_index] = [img_rad_coord]


# Retrieve pixels from each region in a specific pattern
selected_pixels = []
for row in range(0, num_rows, 2):  # Iterate over rows with a step of 2
    for col in range(0, num_cols, 2):  # Iterate over columns with a step of 2
        current_region = row * num_cols + col  # Calculate the current region index
        if current_region in assigned_regions:  # Check if the current region has assigned pixels
            region_pixels = assigned_regions[current_region]  #print(assigned_regions.keys())
            if region_pixels: # Check if the region has any pixels
                #selected_pixel = random.choice(region_pixels)  # Select a random pixel from the region
                # Calculate the distance between each pixel and the region center
                distances = [math.sqrt((pixel[0][0] - region_centers[current_region][0])**2 +
                                       (pixel[0][1] - region_centers[current_region][1])**2)
                             for pixel in region_pixels]
                # Find the index of the pixel with the minimum distance
                closest_pixel_index = distances.index(min(distances))
                selected_pixel = region_pixels[closest_pixel_index]  # Select the closest pixel
                #selected_pixel = region_pixels[0]  # Select the closest pixel
                selected_pixels.append(selected_pixel)
                # # Remove the selected pixel from the region's pixels
                # region_pixels.remove(selected_pixel)  


# Check the number of selected pixels
print("Number of selected paired points:", len(selected_pixels))
if len(selected_pixels) < 9:
    print("Insufficient pixels selected. Please collect more data.")

# Convert the selected pixels to a numpy array
selected_pixels_array = np.array(selected_pixels)

# Print the selected pixels
#print("Selected Pixels:")
#print(selected_pixels_array)

# Save  as an .npy file
np.save(os.path.join(path,'selected_pairs_array_3.npy'), selected_pixels_array) 
