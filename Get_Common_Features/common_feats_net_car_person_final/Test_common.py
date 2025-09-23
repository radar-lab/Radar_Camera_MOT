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
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler,TensorBoard,ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K
from nets.asso_net_rad import common_feat_model
from nets.asso_net_training import get_lr_scheduler, triplet_loss
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import Dataset_BatchGet
from utils.utils import get_num_classes, show_config,cvt2Color, preprocess_input, resize_image,resize_radar_data,resize_with_only_padding
from nets.CSPdarknet53 import Mish

from data_prep import concat_npy,idxFormimg, data_prep_clf
from keras.utils import plot_model
from tensorflow.python.keras.utils import np_utils
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

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
    
def load_dataset_from_txt(annotation_path,classes_list):
    with open(annotation_path) as f:
        all_path = f.readlines()
        
    paths  = []
    classes= []
    for path in all_path:
        path_split = path.split(";")
        index = classes_list.index(path_split[0])
        paths.append(path_split[1])
        classes.append(index)
    paths  = np.array(paths,dtype=object)
    classes = np.array(classes)
    return paths,classes
    
def prepare_dataset(triple_idx_file= './triple_idx_lei-car-lei.txt',mode= 'test'):
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
    
    
# Contrastive loss function
def contrastive_loss(y_true, distance, margin=1.0):
    y_true = K.cast(y_true, distance.dtype)  # Cast y_true to the same data type as distance
    loss = K.mean((1 - y_true) * K.square(distance) + y_true * K.square(K.maximum(margin - distance, 0)))
    return loss 


if __name__ == "__main__":
    train_gpu       = [0,]
    rad_dataset = '/xdisk/caos/leicheng/my_rawdata_0519/0519/' 
    img_datasets   = "/xdisk/caos/leicheng/my_rawdata_0519/0519_camera/"
    img_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519_camera/img_idx.txt'
    rad_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx.txt'
    triple_idx_file   = './triple_idx_hao-jiahao.txt' #'./triple_idx_hao-car-shuting.txt' #'./triple_idx_sijie-car-lei.txt' #'./triple_idx_lei-car-lei.txt' #'./triple_idx_jiahao-car-shuting.txt' #'./triple_idx_hao-jiahao.txt'
    input_shape_rad = [256, 256, 64]
    input_shape_img = [416, 416, 3]
    
    output_folder   = os.path.splitext(os.path.basename(triple_idx_file))[0] #.split('_', 2)[-1]
    output_path     = './test_results/' + output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_path          = "./logs/best_model.h5" #change per need
    full_model_path     = "./logs/final_model.h5"
    model_path_radClf   = './model_data/rad_classification_final_model.h5'
    model_path_imgClf   = './model_data/img_classification_final_model.h5' #'./model_data/best_model_img.h5'
    batch_size      = 16 #4
    Init_lr         = 1e-3
    Min_lr          = Init_lr * 0.01
    optimizer_type  = "adam"
    momentum        = 0.9
    lr_decay_type   = "cos"
    save_period     = 1
    save_dir        = 'logs'
    num_workers     = 1
    mode            = 'classifi'

     

    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    os.makedirs(log_dir, exist_ok=True)
    
    # Get classes list
    num_classes,classes_list = get_num_classes(img_idx_file)

    # # Create radar and image models
    # model = common_feat_model(input_shape_img,model_path_imgClf, model_path_radClf, input_shape_img, input_shape_rad)
    # if model_path != '':
    #     model.load_weights(model_path, by_name=True, skip_mismatch=True)#https://keras.io/api/models/model_saving_apis/

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
    elif mode=='contrastive_loss':
        model.compile(optimizer=optimizer,
                      loss=lambda y_true, y_pred: contrastive_loss(y_true, y_pred, margin=1.0),
                      metrics=['accuracy'])
 
###############################  Test  #########################################

    
    test_path, test_label     = prepare_dataset(triple_idx_file, mode= 'test')


    
    ## create dataset 
    test_dataset   = Dataset_BatchGet(input_shape_rad, test_path, test_label, batch_size, num_classes, rand_flag=False,new_size=input_shape_rad,new_size_img=input_shape_img)

    # Test the model
    TEST = False
    if TEST:
        test_loss, test_accuracy = model.evaluate(test_dataset)    
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)  #triple_idx_sijie-car-lei: 0.9434999823570251
    
    # Predict labels for the test dataset
    predictions = model.predict(test_dataset)
    predictions = predictions.flatten().astype(int)
    
    # Get the true labels
    true_labels = test_label
    # Print Accuracy
    accuracy = np.mean(predictions == true_labels) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")# jiahao-car-shuting:100%; hao-car-shuting:99.95%; lei-car-lei:93.80%; sijie-car-lei: 94.20%; 
    
    # Visualize some test samples with their predicted and true labels
    num_samples = 20
    sample_indices = np.random.choice(len(true_labels), size=num_samples, replace=False)
    
    for i in range(0,len(test_path)): #sample_indices:
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
        
        # Save the plot
        file_name = f'plot_{i}.png' 
        plt.savefig(os.path.join(output_path, file_name), dpi=300, bbox_inches='tight')

        
        # Show the subplot grid
        plt.show()

    

    

