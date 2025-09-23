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
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler,TensorBoard,ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K
from nets.asso_net_rad import asso_net_rad,asso_net_img
from nets.asso_net_training import get_lr_scheduler, triplet_loss
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import AssonetDataset_rad, trim_zeros
from utils.utils import get_num_classes, show_config,cvt2Color, preprocess_input, resize_image,resize_with_only_padding

from data_prep import concat_npy,idxFormimg, data_prep_clf
from keras.utils import plot_model
from tensorflow.python.keras.utils import np_utils
from PIL import Image

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
    
# class TripletLoss(Loss):
#     def __init__(self, name='triplet_loss'):
#         super(TripletLoss, self).__init__(name=name)

#     def call(self, y_true, y_pred):
#         anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
#         distance_positive = K.sum(K.square(anchor - positive), axis=-1)
#         distance_negative = K.sum(K.square(anchor - negative), axis=-1)
#         loss = K.maximum(distance_positive - distance_negative + 0.2, 0.0)
#         return loss

class TripletLoss(Loss):
    def __init__(self, alpha=0.2, name='triplet_loss'):
        super(TripletLoss, self).__init__(name=name)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Extract anchor, positive, and negative embeddings
        batch_size = tf.shape(y_pred)[0] // 3
        anchor, positive, negative = y_pred[:batch_size], y_pred[batch_size:2 * batch_size], y_pred[-batch_size:]
        
        # Compute the distances between anchor and positive, and anchor and negative samples
        distance_positive = K.sum(K.square(anchor - positive), axis=-1)
        distance_negative = K.sum(K.square(anchor - negative), axis=-1)
        
        # Adjust the basic loss by adding the alpha value to control the gap between positive and negative samples
        basic_loss = distance_positive - distance_negative + self.alpha
        
        # Select the loss for samples where the basic loss is greater than zero
        idxs = tf.where(basic_loss > 0)
        select_loss = tf.gather_nd(basic_loss, idxs)
        
        # Compute the final loss by summing and averaging the select loss
        loss = K.sum(K.maximum(basic_loss, 0)) / tf.cast(tf.maximum(1, tf.shape(select_loss)[0]), tf.float32)
        
        return loss



class SoftmaxLoss(Loss):
    def __init__(self, name='softmax_loss'):
        super(SoftmaxLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        loss = K.mean(K.categorical_crossentropy(y_true, y_pred))
        return loss

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
        
    for i in range(batch_size):
        ## construct anchor from radar
        # labels_list=list(np.unique(classes))
        # c=np.random.choice(labels_list, 1,replace=False)[0]
        if mode == 'train':
            #c               = random.randint(0, num_classes - 1)
            c               = random.randint(0, 2)
        elif mode == 'val':
            c               = 3
        elif mode == 'test':
            c               = 4
        selected_path   = rad_paths[rad_classes[:] == c]
        
        # choose one radar randomly
        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        # convert radar complex to real
        rad_complex = np.load(selected_path[image_indexes[0]].split()[0],allow_pickle=True)
        rad_real = complexTo2Channels(rad_complex)
        resized_data = resize_with_only_padding(rad_real, new_size_rad)   
        anchors.append(resized_data)
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

    return np.array(anchors),np.array(positives),np.array(negatives), {'Embedding' : np.zeros_like(labels), 'Softmax' : labels}




if __name__ == "__main__":
    train_gpu       = [0,]
    rad_dataset = '/xdisk/caos/leicheng/my_rawdata_0519/0519/' 
    img_datasets   = "/xdisk/caos/leicheng/my_rawdata_0519/0519_camera/"
    img_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519_camera/img_idx.txt'
    rad_idx_file   = '/xdisk/caos/leicheng/my_rawdata_0519/0519/radar_npy_idx.txt'
    input_shape_rad = [256, 256, 64]
    input_shape_img = [416, 416, 3]

    model_path      = "./model_data/" #change per need
    Init_Epoch      = 0
    epochs          = 800 #800
    batch_size      = 4 #24
    Init_lr         = 1e-3
    Min_lr          = Init_lr * 0.01
    optimizer_type  = "adam"
    momentum        = 0.9
    lr_decay_type   = "cos"
    save_period     = 1
    save_dir        = 'logs'
    num_workers     = 1

     
    #idxFormimg(idx_file_name=idx_file_name, datasets_path=datasets_path) #used to generate object_imgidx.txt
    
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    
    # Get classes list
    num_classes,classes_list = get_num_classes(img_idx_file)

    # Create radar and image models
    model_rad = asso_net_rad(input_shape_rad, num_classes, mode="train")
    model_img = asso_net_img(input_shape_img, num_classes, mode="train")
    if model_path != '':
        model_rad.load_weights(model_path+'best_model_rad.h5', by_name=True, skip_mismatch=True)#https://keras.io/api/models/model_saving_apis/
        model_img.load_weights(model_path+'best_model_img.h5', by_name=True, skip_mismatch=True)

    
    # Define the loss function
    loss_fn = TripletLoss()
    softmax_loss_fn = SoftmaxLoss()
    
    img_paths,img_classes = load_dataset_from_txt(img_idx_file,classes_list)
    rad_paths,rad_classes = load_dataset_from_txt(rad_idx_file,classes_list)
    val_len   = len(rad_classes[rad_classes[:] == 3])
    test_len  = len(rad_classes[rad_classes[:] == 4])
    train_len = len(rad_classes) - val_len - test_len
    # Define batch data
    num_batches = train_len // batch_size
    val_num_batches = val_len // batch_size
    test_num_batches = test_len // batch_size
     




        
    nbs             = 64
    lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
    lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)  

    optimizer = {
        'adam'  : Adam(learning_rate = Init_lr_fit, beta_1 = momentum),
        'sgd'   : SGD(learning_rate = Init_lr_fit, momentum = momentum, nesterov=True)
    }[optimizer_type]
    



 
       
############ Evaluate on test data in batches
    test_losses = []
    for test_batch_idx in range(test_num_batches):
        # test_start_idx = test_batch_idx * batch_size
        # test_end_idx = (test_batch_idx + 1) * batch_size
    
        # test_batch_anc = x_test_anchor[test_start_idx:test_end_idx]
        # test_batch_positive = x_test_positive[test_start_idx:test_end_idx]
        # test_batch_negative = x_test_negative[test_start_idx:test_end_idx]
        test_batch_anc,test_batch_positive,test_batch_negative, test_gt_labels = gen_batch_data(batch_size,num_classes,img_classes,img_paths,rad_classes,rad_paths, mode='test', rand_flag=True,new_size=input_shape_img,new_size_rad=input_shape_rad)
        
        # Reshape the input data to match the expected input shape of (batch_size, height, width, depth, channels)
        test_batch_anc = np.expand_dims(test_batch_anc, axis=-1)
        
        test_features_anc = model_rad(test_batch_anc)
        test_features_2d_positive = model_img(np.array(test_batch_positive))
        test_features_2d_negative = model_img(np.array(test_batch_negative))
    
        test_input_features = tf.concat([test_features_anc[1], test_features_2d_positive[1], test_features_2d_negative[1]], axis=0)
        test_loss = loss_fn(None, test_input_features)
    
        test_losses.append(test_loss)
        # Print real-time loss - Batch
        print("Batch {}/{} - test_Loss: {:.4f}".format(test_batch_idx+1, test_num_batches, test_loss))
    
    # Calculate the average test loss
    avg_test_loss = np.mean(test_losses)
    print("Test Total Loss: {:.4f}".format(avg_test_loss))
    

