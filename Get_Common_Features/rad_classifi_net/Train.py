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
    rad_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx_2class.txt'
    input_shape = [256, 256, 64]  # Input shape of the images
    batch_size = 4  # Number of samples per batch
    epochs = 800  # Number of training epochs


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
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    os.makedirs(log_dir, exist_ok=True)
    
    
     
    #idxFormimg(idx_file_name=idx_file_name, datasets_path=datasets_path) #used to generate object_imgidx.txt
    

    
    # Get classes list
    num_classes,classes_list = get_num_classes(rad_idx_file)

    # Create model
    #model = rad_classification_model(input_shape, num_classes)
    model = build_complex_cnn(input_shape, num_classes)
    # if model_path != '':
    #     model.load_weights(model_path, by_name=True, skip_mismatch=True)#https://keras.io/api/models/model_saving_apis/
    # if darknet_weights != '':
    #     # Load the weights for the darknet_body from the file
    #     model.load_weights(darknet_weights, by_name=True, skip_mismatch=True)


    
    
    paths,classes = load_dataset_from_txt(rad_idx_file,classes_list,shuffle=True)
    val_split = 0.1 
    test_split = 0.1
    dataset_len = len(paths)
    num_val = int(dataset_len*val_split)
    num_test = int(dataset_len*test_split)
    num_train = dataset_len - num_val - num_test
    
    epoch_step          = num_train // batch_size
    epoch_step_val      = num_val // batch_size

        
    plot_model(model,to_file='./model_data/Classif_net_rad.png',show_shapes=True, show_layer_names=True,expand_nested=False)
    
    
    # Define the callbacks
    # logging         = TensorBoard(log_dir)
    # loss_history    = LossHistory(log_dir)
    checkpoint      = ModelCheckpoint(os.path.join(save_dir, "best_model_rad.h5"), 
                            monitor = 'val_loss', save_weights_only = True, save_best_only = True, save_freq = 'epoch')#period=10        
    # checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
    #                         monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 30, verbose = 1,restore_best_weights=True) 
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, epochs)
    lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
    csv_logger = CSVLogger(os.path.join(log_dir, 'training.log'))
    
    callbacks=[checkpoint, csv_logger]#, lr_scheduler, early_stopping

    
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
    
    train_path  = paths[:num_train]
    train_label = labels[:num_train]
    
    val_path  = paths[num_train: (num_train+num_val)]
    val_label = labels[num_train: (num_train+num_val)]
    
    test_path  = paths[(num_train+num_val):]
    test_label = labels[(num_train+num_val):]
    

    # Save train_path and train_label as .npy files
    np.save('train_path.npy', train_path)
    np.save('train_label.npy', train_label)
    np.save('val_path.npy', val_path)
    np.save('val_label.npy', val_label)
    np.save('test_path.npy', test_path)
    np.save('test_label.npy', test_label)
    
    # Read train_path and train_label from npy files
    train_path = np.load('train_path.npy',allow_pickle=True)
    train_label = np.load('train_label.npy',allow_pickle=True)
    val_path = np.load('val_path.npy',allow_pickle=True)
    val_label = np.load('val_label.npy',allow_pickle=True)
    test_path = np.load('test_path.npy',allow_pickle=True)
    test_label = np.load('test_label.npy',allow_pickle=True)
    

    ## create dataset 
    train_dataset   = Dataset_BatchGet(input_shape, train_path, train_label, batch_size, num_classes, rand_flag=True,new_size=input_shape)
    val_dataset     = Dataset_BatchGet(input_shape, val_path, val_label, batch_size, num_classes,rand_flag=True,new_size=input_shape)
    # train_dataset   = Dataset_BatchGet(input_shape, paths[:num_train], labels[:num_train], batch_size, num_classes, rand_flag=True,new_size=input_shape)
    # val_dataset     = Dataset_BatchGet(input_shape, paths[num_train: (num_train+num_val)], labels[num_train: (num_train+num_val)], batch_size, num_classes,rand_flag=True,new_size=input_shape)
    
    
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
    model.save(os.path.join(save_dir, 'rad_classification_final_model.h5'))
    
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
    

