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
    
def gen_batch_data(batch_size,num_classes,classes,paths,rad_classes,rad_paths, mode='train', rand_flag=True,new_size=[416,416,3],new_size_rad=[256,256,64]):
    anchors   = []
    positives = []
    negatives = []
    labels1   = []
    labels2   = []
    labels3   = []
    batch_x_real = []
    batch_x_imag = []
        
    for i in range(batch_size):
        ## construct anchor from radar
        # labels_list=list(np.unique(classes))
        # c=np.random.choice(labels_list, 1,replace=False)[0]
        if mode == 'train':
            #c               = random.randint(0, num_classes - 1)
            c               = np.random.choice([1,4], 1)[0]
        elif mode == 'val':
            c               = np.random.choice([0,5], 1)[0]
        elif mode == 'test':
            c               = 4
        selected_path   = rad_paths[rad_classes[:] == c]
        
        # choose one radar randomly
        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        # convert radar complex to real
        rad_complex = np.load(selected_path[image_indexes[0]].split()[0],allow_pickle=True)
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
 
        #anchors.append([reshape_real, reshape_imag])
        labels1.append(c)
        
  
        
        ## construct positive from img
        selected_path   = paths[classes[:] == c]
        # choose one img randomly
        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        # convert img to array
        image = cvt2Color(Image.open(selected_path[image_indexes[0]].split()[0]))
        # flip img0
        if rand()<.5 and rand_flag: 
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = resize_image(image, [new_size[1], new_size[0]], letterbox_image = True)
        image = preprocess_input(np.array(image, dtype='float32'))

        positives.append(image)
        labels2.append(c)



        ## construct negative from a different person
        different_c         = list(range(num_classes))
        different_c.pop(c)
        
        different_c_index   = np.random.choice(range(0, num_classes - 1), 1)
        current_c           = different_c[different_c_index[0]]
        
        # labels_list.remove(c)
        # current_c=np.random.choice(labels_list, 1,replace=False)[0] 
        
        selected_path       = paths[classes == current_c]
        while len(selected_path) < 1:
            different_c_index   = np.random.choice(range(0, num_classes - 1), 1)
            current_c           = different_c[different_c_index[0]]
            #current_c=np.random.choice(labels_list, 1,replace=False)[0] 
            selected_path       = paths[classes == current_c]

        # choose one img randomly
        image_indexes       = np.random.choice(range(0, len(selected_path)), 1)
        image               = cvt2Color(Image.open(selected_path[image_indexes[0]].split()[0]))
        # flip img
        if rand()<.5 and rand_flag: 
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = resize_image(image, [new_size[1], new_size[0]], letterbox_image = True)
        image = preprocess_input(np.array(image, dtype='float32'))
        
        negatives.append(image)
        labels3.append(current_c)



    #--------------------------------------------------------------#
    #Label
    #--------------------------------------------------------------#

    
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    labels3 = np.array(labels3)
    labels = np.concatenate([labels1, labels2, labels3], 0)

    labels = np_utils.to_categorical(np.array(labels), num_classes = num_classes)  

    return [np.array(batch_x_real, dtype=np.float16), np.array(batch_x_imag, dtype=np.float16)],np.array(positives),np.array(negatives), {'Embedding' : np.zeros_like(labels), 'Softmax' : labels}

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
    triple_idx_file       = './triple_idx_2class.txt'
    triple_idx_file_val   = './triple_idx_lei-car-lei.txt'
    input_shape_rad = [256, 256, 64]
    input_shape_img = [416, 416, 3]

    model_path_rad      = "./model_data/backbone_model_rad.h5" #change per need
    model_path_img      = "./model_data/backbone_model_img.h5" #change per need
    model_path_radClf   = './model_data/rad_classification_final_model.h5'
    model_path_imgClf   = './model_data/img_classification_final_model.h5' #'./model_data/best_model_img.h5'
    Init_Epoch      = 0
    epochs          = 30 #800
    batch_size      = 4 #24
    Init_lr         = 1e-3
    Min_lr          = Init_lr * 0.01
    optimizer_type  = "adam"
    momentum        = 0.9
    lr_decay_type   = "cos"
    save_period     = 1
    save_dir        = 'logs'
    num_workers     = 1
    mode            = 'classifi'

     
    #idxFormimg(idx_file_name=idx_file_name, datasets_path=datasets_path) #used to generate object_imgidx.txt
    
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

    # Create radar and image models
    model = common_feat_model(input_shape_img,model_path_imgClf, model_path_radClf, input_shape_img, input_shape_rad)
    # if model_path != '':
    #     model.load_weights(model_path_rad, by_name=True, skip_mismatch=True)#https://keras.io/api/models/model_saving_apis/


    
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
    
    epoch_step          = num_train // batch_size
    epoch_step_val      = num_val // batch_size
    
      
    plot_model(model,to_file='./model_data/Commonnet_plot.png',show_shapes=True, show_layer_names=True,expand_nested=False)

    # Define the callbacks  ##***## remember to modify  monitor = 'val_loss'
    checkpoint      = ModelCheckpoint(os.path.join(save_dir, "best_model.h5"), 
                            monitor = 'val_accuracy', save_weights_only = True, save_best_only = True, save_freq = 'epoch')#period=10        
    # checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
    #                         monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
    early_stopping  = EarlyStopping(monitor='val_accuracy', min_delta = 0, patience = 8, verbose = 1,restore_best_weights=True) 
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, epochs)
    lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
    csv_logger = CSVLogger(os.path.join(log_dir, 'training.log'))
    
    callbacks=[checkpoint, csv_logger, lr_scheduler, early_stopping]#, lr_scheduler, early_stopping
    
    print('Train on {} samples, val on {} samples, test on {} samples, with batch size {}.'.format(num_train, num_val,num_test, batch_size))


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

 
###############################  Train  #########################################
    # Extract the input data (img and rad paths) and the labels (0 or 1)
    X_img = [line[0] for line in paths]
    X_rad = [line[1] for line in paths]
    #y = [int(line[2]) for line in paths]
    
    paths      = np.array(paths)
    labels     = np.array(labels)
    
    train_path  = paths[:num_train]
    train_label = labels[:num_train]
    val_path  = paths[num_train: (num_train+num_val)]
    val_label = labels[num_train: (num_train+num_val)]
    test_path  = paths[(num_train+num_val):]
    test_label = labels[(num_train+num_val):]
    
    
    ##### only below codes are used
    train_path1, train_label1 = prepare_dataset(triple_idx_file, mode= 'test')
    train_path2, train_label2 = prepare_dataset('./triple_idx_hao-car-shuting.txt', mode= 'test')
    train_path, train_label   = np.concatenate((train_path1, train_path2)), np.concatenate((train_label1, train_label2))
    val_path, val_label       = prepare_dataset(triple_idx_file_val, mode= 'test')

    
    ## create dataset 
    train_dataset   = Dataset_BatchGet(input_shape_rad, train_path, train_label, batch_size, num_classes, rand_flag=False,new_size=input_shape_rad,new_size_img=input_shape_img)
    val_dataset     = Dataset_BatchGet(input_shape_rad, val_path, val_label, batch_size, num_classes,rand_flag=False,new_size=input_shape_rad,new_size_img=input_shape_img)

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
    model.save(os.path.join(save_dir, 'final_model.h5'))
    
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
    

