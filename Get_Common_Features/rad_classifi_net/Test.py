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
from nets.asso_net_rad import rad_classification_model,build_complex_cnn
from nets.asso_net_training import get_lr_scheduler, triplet_loss
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import Dataset_BatchGet, trim_zeros
from utils.utils import get_num_classes, show_config,cvt2Color, preprocess_input, resize_image,resize_with_only_padding

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



def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
    
def load_dataset_from_txt(annotation_path,classes_list,shuffle=False):
    with open(annotation_path) as f:
        all_path = f.readlines() #all lines
    if shuffle:
        np.random.seed(500)
        np.random.shuffle(all_path)# use this combine with seed can gurante that each time same shuffle        
        
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


if __name__ == "__main__":
    # Set the hyperparameters and other settings
    train_gpu       = [0,]
    rad_dataset = '/xdisk/caos/leicheng/my_rawdata_0519/0519/' 
    img_datasets   = "/xdisk/caos/leicheng/my_rawdata_0519/0519_camera/"
    img_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519_camera/img_idx.txt'
    #rad_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx_2class.txt'
    #rad_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx_2class_scl.txt'
    rad_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx_2class_csl.txt'
    input_shape = [256, 256, 64]  # Input shape of the images
    batch_size = 4  # Number of samples per batch


    model_path      = "./model_data/best_model_rad.h5" #change per need
    darknet_weights = "./model_data/best_backbone_weights.h5"
    Init_Epoch      = 0
    Init_lr         = 1e-3
    Min_lr          = 5e-6
    optimizer_type  = "adam"
    momentum        = 0.9
    lr_decay_type   = "cos"
    save_period     = 1
    save_dir        = 'logs'
    
    #### GPU
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "TEST_" + str(time_str))
    os.makedirs(log_dir, exist_ok=True)
    

    
    # Get classes list
    num_classes,classes_list = get_num_classes(rad_idx_file)
    
    
    model = tf.keras.models.load_model('./model_data/rad_classification_final_model.h5')
    # # Create model
    # model = build_complex_cnn(input_shape, num_classes)
    # if model_path != '':
    #     model.load_weights(model_path, by_name=True, skip_mismatch=True)#https://keras.io/api/models/model_saving_apis/

    optimizer = {
        'adam'  : Adam(learning_rate = Init_lr, beta_1 = momentum),
        'sgd'   : SGD(learning_rate = Init_lr, momentum = momentum, nesterov=True)
    }[optimizer_type]
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
   
    paths,classes = load_dataset_from_txt(rad_idx_file,classes_list,shuffle=True)
    val_split = 0.1 
    test_split = 0.1
    dataset_len = len(paths)
    num_val = int(dataset_len*val_split)
    num_test = int(dataset_len*test_split)
    num_train = dataset_len - num_val - num_test
    

    
###############################  Test  #########################################

    paths      = np.array(paths)
    labels     = np.array(classes)
    
    # test_path  = paths[(num_train+num_val):]
    # test_label = labels[(num_train+num_val):]
    test_path  = paths
    test_label = labels
    
    # Read train_path and train_label from npy files
    # test_path = np.load('test_path.npy',allow_pickle=True)
    # test_label = np.load('test_label.npy',allow_pickle=True)
    # test_path = np.load('val_path.npy',allow_pickle=True)
    # test_label = np.load('val_label.npy',allow_pickle=True)
    
    print('Test on {} samples, with batch size {}.'.format(len(test_label), batch_size))
    
    
    ## create dataset 
    test_dataset = Dataset_BatchGet(input_shape, test_path, test_label, batch_size, num_classes, rand_flag=True, new_size=input_shape)
    
    #
    test_loss, test_accuracy = model.evaluate(test_dataset)
    
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    
    # Predict labels for the test dataset
    predictions = model.predict(test_dataset)
    predictions = np.argmax(predictions, axis=1)
    
    # Get the true labels
    true_labels = test_label
    #true_labels = labels[(num_train + num_val):]
    
    # Visualize some test samples with their predicted and true labels
    num_samples = 10
    sample_indices = np.random.choice(len(true_labels), size=num_samples, replace=False)
    
    for i in sample_indices:
        # Get the predicted label
        predicted_label = predictions[i]
    
        # Get the true label
        true_label = true_labels[i]
    
        # Get the corresponding image path
        image_path = test_path[i] #paths[(num_train + num_val):][i]
    
        # Load and display the image
        image = Image.open(image_path.rstrip('\n'))
        plt.imshow(image)
        plt.axis('off')
    
        # Display the predicted and true labels
        predicted_class = classes_list[predicted_label]
        true_class = classes_list[true_label]
        plt.title("Predicted: {} | True: {}".format(predicted_class, true_class))
    
        plt.show()
