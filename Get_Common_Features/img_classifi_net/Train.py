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
from nets.asso_net_rad import img_classification_model
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


# def create_model_img(input_shape):
#     input_layer = Input(shape=input_shape)
#     conv2d_branch = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
#     conv2d_branch = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv2d_branch)
#     conv2d_branch = Flatten()(conv2d_branch)
#     conv2d_branch = Dense(128, activation='relu')(conv2d_branch)
#     conv2d_output = Dense(128)(conv2d_branch)
#     model = Model(inputs=input_layer, outputs=conv2d_output)
#     return model

# def create_model_rad(input_shape):
#     input_layer = Input(shape=input_shape)
#     conv3d_branch = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(input_layer)
#     conv3d_branch = Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(conv3d_branch)
#     conv3d_branch = Flatten()(conv3d_branch)
#     conv3d_branch = Dense(128, activation='relu')(conv3d_branch)
#     conv3d_output = Dense(128)(conv3d_branch)
#     model = Model(inputs=input_layer, outputs=conv3d_output)
#     return model

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
    rad_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx.txt'
    input_shape = [416, 416, 3]  # Input shape of the images
    batch_size = 16  # Number of samples per batch
    epochs = 30 #800  # Number of training epochs


    model_path      = "./model_data/best_model_img.h5" #change per need
    darknet_weights = "./model_data/CSPdarknet53_backbone_weights.h5"
    Init_Epoch      = 0
    Init_lr         = 1e-2
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
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    os.makedirs(log_dir, exist_ok=True)
    
    
     
    #idxFormimg(idx_file_name=idx_file_name, datasets_path=datasets_path) #used to generate object_imgidx.txt
    

    
    # Get classes list
    num_classes,classes_list = get_num_classes(img_idx_file)

    # Create model
    model = img_classification_model(input_shape, num_classes)
    # if model_path != '':
    #     model.load_weights(model_path, by_name=True, skip_mismatch=True)#https://keras.io/api/models/model_saving_apis/
    # if darknet_weights != '':
    #     # Load the weights for the darknet_body from the file
    #     model.load_weights(darknet_weights, by_name=True, skip_mismatch=True)


    
    
    paths,classes = load_dataset_from_txt(img_idx_file,classes_list,shuffle=True)
    val_split = 0.1 
    test_split = 0.1
    dataset_len = len(paths)
    num_val = int(dataset_len*val_split)
    num_test = int(dataset_len*test_split)
    num_train = dataset_len - num_val - num_test
    
    epoch_step          = num_train // batch_size
    epoch_step_val      = num_val // batch_size

        
    plot_model(model,to_file='./model_data/Classif_net_img.png',show_shapes=True, show_layer_names=True,expand_nested=False)
    
    
    # Define the callbacks
    # logging         = TensorBoard(log_dir)
    # loss_history    = LossHistory(log_dir)
    checkpoint      = ModelCheckpoint(os.path.join(save_dir, "best_model_img.h5"), 
                            monitor = 'val_accuracy', save_weights_only = True, save_best_only = True, save_freq = 'epoch')#period=10        
    # checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
    #                         monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
    early_stopping  = EarlyStopping(monitor='val_accuracy', min_delta = 0, patience = 5, verbose = 1,restore_best_weights=True) 
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, epochs)
    lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
    csv_logger = CSVLogger(os.path.join(log_dir, 'training.log'))
    
    callbacks=[checkpoint, lr_scheduler, csv_logger, early_stopping]#, early_stopping

    
    print('Train on {} samples, val on {} samples, test on {} samples, with batch size {}.'.format(num_train, num_val,num_test, batch_size))


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
    
###############################  Train  #########################################

    paths      = np.array(paths)
    labels     = np.array(classes)
    
    ## create dataset 
    train_dataset   = Dataset_BatchGet(input_shape, paths[:num_train], labels[:num_train], batch_size, num_classes, rand_flag=True,new_size=[416,416,3])
    val_dataset     = Dataset_BatchGet(input_shape, paths[num_train: (num_train+num_val)], labels[num_train: (num_train+num_val)], batch_size, num_classes,rand_flag=True,new_size=[416,416,3])
    
    
    # Train the model
    history =model.fit(
            x                   = train_dataset,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataset,
            validation_steps    = epoch_step_val,
            epochs              = epochs,
            initial_epoch       = Init_Epoch,
            callbacks           = callbacks
        )
    
    # Save the trained model
    model.save(os.path.join(save_dir, 'img_classification_final_model.h5'))
    
    # Plot the training and validation loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss_plot.png'))
    plt.show()
    

