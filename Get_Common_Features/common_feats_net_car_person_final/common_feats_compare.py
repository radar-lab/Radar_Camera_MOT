#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""

import datetime
import os
import math
import random


import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.backend as K

from utils.dataloader import Dataset_BatchGet
from utils.utils import get_num_classes, show_config,cvt2Color, preprocess_input, resize_image,resize_radar_data,resize_with_only_padding
from nets.CSPdarknet53 import Mish

from PIL import Image
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

################ complex To real ################
def complexTo2Channels(target_array):
    """ transfer complex a + bi to [a, b]"""
    #assert target_array.dtype == np.complex64
    ### NOTE: transfer complex to (magnitude) ###
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array

def getMagnitude(target_array, power_order=2):
    """ get magnitude out of complex number """
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    """ get Log values """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array
    


def separate_real_imaginary(data):
    # Separate real and imaginary parts
    real_part = np.real(data)
    imag_part = np.imag(data)
    return real_part, imag_part

def normalize_real_imaginary(real_part, imag_part):
    # Compute the maximum absolute value for real and imaginary parts separately
    max_abs_real = np.max(np.abs(real_part))
    max_abs_imag = np.max(np.abs(imag_part))
    
    # Normalize the real and imaginary parts
    real_normalized = real_part / max_abs_real
    imag_normalized = imag_part / max_abs_imag
    
    return real_normalized, imag_normalized

    
    
def prepare_dataset(triple_idx_file= './triple_idx_lei-car-lei.txt',mode= 'test',batch_size = 4):
    # Load the data from output_lines
    output_lines = []
    with open(triple_idx_file, 'r') as f:
        output_lines = f.readlines()
    
    # Remove the trailing newline character from each line and split by ';'
    paths  = [line.strip().split(';')[:2] for line in output_lines]
    labels = [int(line.strip().split(';')[2]) for line in output_lines]
    
    val_split = 0.1 
    test_split = 0.1
    dataset_len = len(paths)
    num_val = int(dataset_len*val_split)
    num_test = int(dataset_len*test_split)
    num_train = dataset_len - num_val - num_test
    
    ## Extract the input data (img and rad paths) and the labels (0 or 1)
    # X_img = [line[0] for line in paths]
    # X_rad = [line[1] for line in paths]
    # y = [int(line[2]) for line in paths]
    if mode == 'train':
        train_path  = paths[:num_train]
        train_label = labels[:num_train]
        val_path  = paths[num_train: (num_train+num_val)]
        val_label = labels[num_train: (num_train+num_val)]
        test_path  = paths[(num_train+num_val):]
        test_label = labels[(num_train+num_val):]
        print('Train on {} samples, val on {} samples, test on {} samples, with batch size {}.'.format(num_train, num_val,num_test, batch_size))
        return train_path, train_label,val_path, val_label,test_path, test_label
    else:
        test_path      = np.array(paths)
        test_label     = np.array(labels)
        print('Test on {} samples, with batch size {}.'.format(len(test_label), batch_size))
        return test_path, test_label

def read_img_rad_from_path(paths):
    all_radar = []
    all_image = []
    for i in range(len(paths)):
        path = paths[i]
        img_path = path[0]
        rad_path = path[1]
        ######    Radar  ###############################################################
        rad_complex = np.load(rad_path, allow_pickle=True)
        all_radar.append(rad_complex)
   
        ######    Image   ###############################################################
        image = cvt2Color(Image.open(img_path))
        all_image.append(np.array(image))

    return all_image, all_radar      
##################### Normalization with Min-Max #################################
def prepare_img_rad_data(images, rad_complexs, new_size_rad=[256,256,64],new_size_img=[416,416,3],mode='Load_data'):
    '''If mode=='Load_path', rad_complexs, images arrays store the radar and img path; Else, they all store the data '''
    batch_x_real = []
    batch_x_imag = []
    batch_x_images = []
    for i in range(len(images)):
        ######    Radar  ###############################################################
        # Convert radar complex to real
        if mode=='Load_path':
            rad_path = rad_complexs[i]
            rad_complex = np.load(rad_path, allow_pickle=True)
        else:
            rad_complex = rad_complexs[i]
        # Separate real and imaginary parts
        real_part, imag_part = separate_real_imaginary(rad_complex)
        # Normalize real and imaginary parts
        real_normalized, imag_normalized = normalize_real_imaginary(real_part, imag_part)

        # Resize real and imaginary parts
        resized_real = resize_radar_data(real_normalized, new_size_rad, interpolation='nearest')
        resized_imag = resize_radar_data(imag_normalized, new_size_rad, interpolation='nearest')

        # Reshape the input data to match the expected input shape of (batch_size, height, width, channels)
        reshape_real = np.expand_dims(resized_real, axis=-1)
        reshape_imag = np.expand_dims(resized_imag, axis=-1)

        batch_x_real.append(reshape_real)
        batch_x_imag.append(reshape_imag)
        
        
        ######    Image   ###############################################################
        # convert img to array
        if mode=='Load_path':
            img_path = images[i]
            image = cvt2Color(Image.open(img_path))
        else:
            image = cvt2Color(Image.fromarray(images[i]))
        image = resize_image(image, [new_size_img[1], new_size_img[0]], letterbox_image = True)
        image = preprocess_input(np.array(image, dtype='float32'))
        
        batch_x_images.append(image)


    rad_data = [np.array(batch_x_real, dtype=np.float16), np.array(batch_x_imag, dtype=np.float16)]

    return [rad_data, np.array(batch_x_images)]    


def common_feats_find():
    img_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519_camera/img_idx.txt'

    triple_idx_file   = './triple_idx_sijie-car-lei.txt' #'./triple_idx_lei-car-lei.txt' #'./triple_idx_2class.txt' #
    input_shape_rad = [256, 256, 64]
    input_shape_img = [416, 416, 3]


    full_model_path     = "./logs/final_model.h5"
    batch_size      = 4 #24
    Init_lr         = 1e-3
    optimizer_type  = "adam"
    momentum        = 0.9
    save_dir        = 'logs'
    mode            = 'classifi'

     


    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    os.makedirs(log_dir, exist_ok=True)
    
    # Get classes list
    num_classes,classes_list = get_num_classes(img_idx_file)
    
    
    # Load the model
    model = tf.keras.models.load_model(full_model_path, custom_objects={'Mish': Mish})
    

    optimizer = {
        'adam'  : Adam(learning_rate = Init_lr, beta_1 = momentum),
        'sgd'   : SGD(learning_rate = Init_lr, momentum = momentum, nesterov=True)
    }[optimizer_type]
    

    # Compile the model
    if mode=='classifi':
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    elif mode=='siamese':
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

 
###############################  Test  #########################################

    
    test_path, test_label     = prepare_dataset(triple_idx_file, mode= 'test')
    
    test_path, test_label     = test_path[:20], test_label[:20]
    # test_dataset   = prepare_img_rad_data(images=test_path[:,0], rad_complexs=test_path[:,1], new_size_rad=[256,256,64],new_size_img=[416,416,3],mode='Load_path')
   
    images, rad_complexs = read_img_rad_from_path(test_path)
    test_dataset   = prepare_img_rad_data(images, rad_complexs, new_size_rad=[256,256,64],new_size_img=[416,416,3])
    # Predict labels for the test dataset
    predictions = model.predict(test_dataset)
    predictions = predictions.flatten().astype(int)
    
    # Get the true labels
    true_labels = test_label

    
    # Visualize some test samples with their predicted and true labels
    num_samples = 20
    sample_indices = np.random.choice(len(true_labels), size=num_samples, replace=False)
    
    for i in sample_indices:
        # Get the predicted label
        predicted_label = predictions[i]
    
        # Get the true label
        true_label = true_labels[i]
    
        # Get the corresponding image path
        image_path = test_path[i][0]
        rad_path   = test_path[i][1]
        rad_img_path = rad_path.replace('/RA_objnpy/', '/RA_objimg/').replace('.npy', '.png')
        
        # Load and display the images
        image = Image.open(image_path.rstrip('\n'))
        rad_img = Image.open(rad_img_path.rstrip('\n'))
        
        # Create a figure and axes for the current sample
        fig, axs = plt.subplots(1, 2)
        
        # Display the images in the subplot grid
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(rad_img)
        axs[1].axis('off')
        
        # Get the predicted and true labels
        predicted_class = 'same' if int(predicted_label) == 1 else 'diff'
        true_class      = 'same' if int(true_label) == 1 else 'diff'
        
        # Display the predicted and true labels
        axs[0].set_title("Image Path: {}".format(os.path.basename(image_path)))
        axs[1].set_title("RAD Image Path: {}".format(os.path.basename(rad_img_path)))
        
        # Set the overall title
        plt.suptitle("Predicted: {} | True: {}".format(predicted_class, true_class), fontsize=14)
        
        # Adjust the spacing between subplots
        plt.tight_layout()
        
        # Show the subplot grid
        plt.show()

 

    
if __name__ == "__main__":
    common_feats_find()
