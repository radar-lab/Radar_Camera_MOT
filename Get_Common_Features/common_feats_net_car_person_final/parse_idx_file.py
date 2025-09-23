#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
:->
run the script to get the lines containing any element from the filter_list saved in the output.txt file
"""

############################ radar npy ######################################## 
# Input file path
input_file = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx.txt'
# # Output file path
# output_file = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx_2class.txt'
# # Filter list
# filter_list = ['jiahao', 'car-shuting']

# # Output file path
# output_file = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx_2class_scl.txt'
# # Filter list
# filter_list = ['shuting', 'car-lei']

# # Output file path
# output_file = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx_2class_lcs.txt'
# # Filter list
# filter_list = ['lei', 'car-shuting']

# Output file path
output_file = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx_2class_csl.txt'
# Filter list
filter_list = ['car-lei', 'car-shuting']

# Open the input file and output file
with open(input_file, 'r') as f_input, open(output_file, 'w') as f_output:
    # Read the input file line by line
    for line in f_input:
        # Split the line at the semicolon (;)
        parts = line.strip().split(';')
        if len(parts) == 2:
            # Check if the first part (before the semicolon) is in the filter list
            if parts[0] in filter_list:
                # Write the matching lines to the output file
                f_output.write(line)
                
        # # Check if the line contains any element from the filter list
        # if any(filter_word in line for filter_word in filter_list):
        #     # Write the matching lines to the output file
        #     f_output.write(line)
        
        
############################ radar heatmap ########################################        
# Input file path
input_file = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_img_idx.txt'
# Output file path
output_file = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_img_idx_2class.txt'
# Filter list
filter_list = ['hao', 'lei']

# Open the input file and output file
with open(input_file, 'r') as f_input, open(output_file, 'w') as f_output:
    # Read the input file line by line
    for line in f_input:
        # Split the line at the semicolon (;)
        parts = line.strip().split(';')
        if len(parts) == 2:
            # Check if the first part (before the semicolon) is in the filter list
            if parts[0] in filter_list:
                # Write the matching lines to the output file
                f_output.write(line)
                
        # # Check if the line contains any element from the filter list
        # if any(filter_word in line for filter_word in filter_list):
        #     # Write the matching lines to the output file
        #     f_output.write(line)