#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 18:52:17 2023

@author: leicheng
"""

import numpy as np
import matplotlib.pyplot as plt

###########################  Load Centers  #########################################
path = "/xdisk/caos/leicheng/my_rawdata_0624/0624/"
filename = path + 'selected_pixels_array.npy' 
# Load the data from the npy file
data = np.load(filename, allow_pickle=True)
####################################################################################
# Define the plot sizes and grid parameters
dpi=300
img_plot_width = 640
img_plot_height = 480
radar_plot_width = 256
radar_plot_height = 256
num_regions = 40
region_width = img_plot_width // num_regions
region_height = img_plot_height // num_regions

# Create the figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6),dpi=dpi)

# Plot for img_points
ax1.set_xlim([0, img_plot_width])
ax1.set_ylim([0, img_plot_height])
ax1.scatter(data[:, 0, 0], data[:, 0, 1], color='red', label='img_points')

# Add grid lines
for i in range(num_regions):
    ax1.axvline(i * region_width, color='gray', linestyle='--')
    ax1.axhline(i * region_height, color='gray', linestyle='--')

ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Plot of img_points')

# Plot for radar_points
ax2.set_xlim([0, radar_plot_width])
ax2.set_ylim([0, radar_plot_height])
ax2.scatter(data[:, 1, 0], data[:, 1, 1], color='blue', label='radar_points')

ax2.legend()
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Plot of radar_points')

# Plot for both img_points and radar_points
ax3.set_xlim([0, img_plot_width])
ax3.set_ylim([0, img_plot_height])
ax3.scatter(data[:, 0, 0], data[:, 0, 1], color='red', label='img_points')
ax3.scatter(data[:, 1, 0], data[:, 1, 1], color='blue', label='radar_points')

# Add grid lines
for i in range(num_regions):
    ax3.axvline(i * region_width, color='gray', linestyle='--')
    ax3.axhline(i * region_height, color='gray', linestyle='--')

ax3.legend()
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Plot of img_points and radar_points')

# Display the plots
plt.show()




###################################
# Set the number of point pairs to plot
num = data.shape[0]
#num = 10   #data.shape[0]

# Create a new figure
fig = plt.figure(figsize=(8, 6),dpi=dpi)
ax = fig.add_subplot(1, 1, 1)

# Plot img_points
ax.scatter(data[:num, 0, 0], data[:num, 0, 1], color='blue', label='Image points')

# Plot radar_points
ax.scatter(data[:num, 1, 0], data[:num, 1, 1], color='red', label='Radar points')

# Define a colormap for the connecting lines
cmap = plt.cm.get_cmap('rainbow')
colors = [cmap(i / num) for i in range(num)]

# Add connecting lines between corresponding points with different colors
for i in range(num):
    img_point = data[i, 0]
    radar_point = data[i, 1]
    ax.plot([img_point[0], radar_point[0]], [img_point[1], radar_point[1]], color=colors[i])

# Set plot limits
# ax.set_xlim([0, max(img_plot_width, radar_plot_width)])
# ax.set_ylim([0, max(img_plot_height, radar_plot_height)])

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Correspondence between Radar points and Image points')

# Add legend
ax.legend()

# Display the plot
plt.show()


#################
# Create a new figure for Plot of img points
fig_img_points = plt.figure(figsize=(8, 6),dpi=dpi)
ax_img_points = fig_img_points.add_subplot(1, 1, 1)

# Plot img_points
ax_img_points.scatter(data[:, 0, 0], data[:, 0, 1], color='red', label='img_points')

# # Add grid lines
# for i in range(num_regions):
#     ax_img_points.axvline(i * region_width, color='gray', linestyle='--')
#     ax_img_points.axhline(i * region_height, color='gray', linestyle='--')

ax_img_points.legend(loc='upper left')
ax_img_points.set_xlabel('X')
ax_img_points.set_ylabel('Y')
ax_img_points.set_title('Plot of img_points')

# Display the plot
plt.show()


# Create a new figure for Plot of radar points
fig_radar_points = plt.figure(figsize=(8, 6))
ax_radar_points = fig_radar_points.add_subplot(1, 1, 1)

# Plot radar_points
ax_radar_points.scatter(data[:, 1, 0], data[:, 1, 1], color='blue', label='radar_points')

ax_radar_points.legend(loc='upper left')
ax_radar_points.set_xlabel('X')
ax_radar_points.set_ylabel('Y')
ax_radar_points.set_title('Plot of radar_points')

# Display the plot
plt.show()




