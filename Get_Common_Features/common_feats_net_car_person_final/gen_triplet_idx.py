#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
The code reads the contents from img.txt and rad.txt files, 
generates lines with the same class and lines with different classes based on the specified number of lines. 
Then, it writes the generated result into the output.txt file, where each line has 
the output_lines list contains lines in the format <img_path>;<rad_path>;<0 or 1>;<img_class>;<rad_class>,
 (1 for the same class, 0 for different classes). 
The order of the generated lines is randomized to ensure an equal number of lines for the same class and different classes.
"""

import random

num = 2000  # Specify the number of lines to generate
#filter_class=['jiahao', 'car-shuting']
#filter_class=['lei', 'car-lei']
#filter_class=['sijie', 'car-lei']
#filter_class=['hao', 'car-shuting']
filter_class=['hao', 'jiahao']
output_file = 'triple_idx_hao-jiahao.txt' #'triple_idx_2class.txt' #'triple_idx.txt'  # Generated output file path

# img_file = 'img.txt'  # img.txt file path
# rad_file = 'rad.txt'  # rad.txt file path
img_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519_camera/img_idx.txt'
rad_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx.txt'

# Read the contents of img.txt file
with open(img_idx_file, 'r') as f:
    img_lines = f.readlines()

# Read the contents of rad.txt file
with open(rad_idx_file, 'r') as f:
    rad_lines = f.readlines()

# Remove the trailing newline character from each line
img_lines = [line.strip() for line in img_lines]
rad_lines = [line.strip() for line in rad_lines]

# Create a dictionary to map image classes to their respective lines
img_class_map = {}
for img_line in img_lines:
    img_class = img_line.split(';')[0]
    if img_class in filter_class:  # Select only classes in filter_class
        if img_class not in img_class_map:
            img_class_map[img_class] = []
        img_class_map[img_class].append(img_line)

# Create a dictionary to map radar classes to their respective lines
rad_class_map = {}
for rad_line in rad_lines:
    rad_class = rad_line.split(';')[0]
    if rad_class in filter_class:  # Select only classes in filter_class
        if rad_class not in rad_class_map:
            rad_class_map[rad_class] = []
        rad_class_map[rad_class].append(rad_line)

# Determine the number of lines to generate, ensuring an equal number of lines for the same class and different class
num_same_class = num // 2
num_diff_class = num - num_same_class

# Randomly select lines with the same class
same_class_lines = []
for _ in range(num_same_class):
    img_class = random.choice(list(rad_class_map.keys()))
    img_line = random.choice(img_class_map[img_class])
    rad_line = random.choice(rad_class_map[img_class])
    same_class_lines.append((img_line, rad_line, img_class))

# Randomly select lines with different classes
diff_class_lines = []
for _ in range(num_diff_class):
    img_class = random.choice(list(img_class_map.keys()))
    rad_class = random.choice(list(rad_class_map.keys()))
    while img_class == rad_class:  # Ensure img_class and rad_class are different
        rad_class = random.choice(list(rad_class_map.keys()))
    img_line = random.choice(img_class_map[img_class])
    rad_line = random.choice(rad_class_map[rad_class])
    diff_class_lines.append((img_line, rad_line, img_class, rad_class))

# Combine lines with the same class and lines with different classes into output lines
output_lines = []
for img_line, rad_line, img_class in same_class_lines:
    img_path = img_line.split(';')[1]
    rad_path = rad_line.split(';')[1]
    output_lines.append(f"{img_path};{rad_path};1;{img_class};{img_class}")
for img_line, rad_line, img_class, rad_class in diff_class_lines:
    img_path = img_line.split(';')[1]
    rad_path = rad_line.split(';')[1]
    output_lines.append(f"{img_path};{rad_path};0;{img_class};{rad_class}")

# Shuffle the order of output lines randomly
random.shuffle(output_lines)

# Write the generated result into the output file
with open(output_file, 'w') as f:
    for line in output_lines:
        f.write(line + '\n')


################## Parse  ###################
'''
output_lines = []
# Read the contents of the output file
with open(output_file, 'r') as f:
    output_lines = f.readlines()
# Remove the trailing newline character from each line and split by ';'
output_lines = [line.strip().split(';')[:3] for line in output_lines]
# Print the parsed lines
for line in output_lines:
    print(line)
    
'''
