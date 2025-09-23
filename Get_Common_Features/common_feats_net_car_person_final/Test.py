#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lei
"""
import numpy as np
from utils.utils_metrics import evaluate, plot_roc
from tqdm import tqdm

import math
import os
import random
from tensorflow import keras
from PIL import Image
from utils.utils import cvt2Color, preprocess_input, resize_image

import tensorflow as tf
from nets.asso_net import asso_net
from nets.asso_net_training import triplet_loss
from utils.utils import get_num_classes, show_config
from data_prep import concat_npy, idxFromimg 

def Test(testdata_loader, model, png_save_path, log_interval, batch_size):
    labels, distances = [], []
    pbar = tqdm(enumerate(testdata_loader.generate()))
    for batch_idx, (data_a, data_p, label) in pbar:
        out_a, out_p = model.predict(data_a), model.predict(data_p)
        if len(out_a)>1:
            out_a, out_p = out_a[1], out_p[1]        
        dists = np.linalg.norm(out_a - out_p, axis=1)

        distances.append(dists)
        labels.append(label)
        ### print
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                (batch_idx+1) * batch_size, testdata_loader.length//2,
                100. * (batch_idx+1) / (testdata_loader.length//2//batch_size)))

    labels      = np.array([sublabel for label in labels for sublabel in label])
    distances   = np.array([subdist for dist in distances for subdist in dist])
    
    tpr, fpr, accuracy, tar, tar_std, far, best_thresholds = evaluate(distances,labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (tar, tar_std, far))
    plot_roc(fpr, tpr, figure_name = png_save_path)
    
class TestDataset(keras.utils.Sequence):
    def __init__(self, input_shape, lines, batch_size, num_classes, imgs, random):
        self.input_shape    = input_shape
        self.lines          = lines
        self.length         = len(lines)
        self.batch_size     = batch_size
        self.num_classes    = num_classes
        self.random         = random
        self.imgs           = imgs        
        

        self.paths  = []
        self.labels = []

        self.load_dataset()
        
    def __len__(self):
        return math.ceil(self.length / float(self.batch_size))
    
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def load_dataset(self):
        for path in self.lines:
            path_split = path.split(";")
            self.paths.append(int(path_split[1].split()[0]))
            self.labels.append(int(path_split[0]))
        self.paths  = np.array(self.paths,dtype=object)
        self.labels = np.array(self.labels)  
        
        
    def generate(self):
        images1 = []
        images2 = []
        issames = []
        labels  = []
        for itera in range(self.length//2):
            # construct anchor and positive from same person
            labels_list=list(np.unique(self.labels))
            c=np.random.choice(labels_list, 1,replace=False)[0]        
            #c               = random.randint(0, self.num_classes - 1)
            selected_path   = self.paths[self.labels[:] == c]
            while len(selected_path) < 2: # if this class contains less than 2 imgs, re-select
                c=np.random.choice(labels_list, 1,replace=False)[0]
                #c               = random.randint(0, self.num_classes - 1)
                selected_path   = self.paths[self.labels[:] == c]
    
            # choose two imgs randomly
            image_indexes = np.random.choice(range(0, len(selected_path)), 2)
            # convert img to array
            #image = cvt2Color(Image.open(selected_path[image_indexes[0]]))
            image = cvt2Color(Image.fromarray(self.imgs[selected_path[image_indexes[0]]].astype(np.uint8), 'RGB'))
            # flip img1
            if self.rand()<.5 and self.random:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
            image = preprocess_input(np.array(image, dtype='float32'))
            
            images1.append(image)    
            
            if self.rand()<.5 : # for constructing same imgs group
                issame = True
                #image = cvt2Color(Image.open(selected_path[image_indexes[1]]))
                image = cvt2Color(Image.fromarray(self.imgs[selected_path[image_indexes[1]]].astype(np.uint8), 'RGB'))
                # flip img2
                if self.rand()<.5 and self.random: 
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
                image = preprocess_input(np.array(image, dtype='float32'))
                labels.append([c,c])
    
                images2.append(image)
                issames.append(issame)
                
            else:  # for constructing diff imgs group
                issame = False
                # construct negative from a different person
                # different_c         = list(range(self.num_classes))
                # different_c.pop(c)
                         
                # different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
                # current_c           = different_c[different_c_index[0]]
                
                labels_list.remove(c)
                current_c=np.random.choice(labels_list, 1,replace=False)[0]      
                
                selected_path       = self.paths[self.labels == current_c]
                while len(selected_path) < 1:
                    # different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
                    # current_c           = different_c[different_c_index[0]]
                    current_c=np.random.choice(labels_list, 1,replace=False)[0] 
                    selected_path       = self.paths[self.labels == current_c]
    
                # choose one img randomly
                image_indexes       = np.random.choice(range(0, len(selected_path)), 1)
                #image               = cvt2Color(Image.open(selected_path[image_indexes[0]]))
                image               = cvt2Color(Image.fromarray(self.imgs[selected_path[image_indexes[0]]].astype(np.uint8), 'RGB'))
                # flip img
                if self.rand()<.5 and self.random: 
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
                image = preprocess_input(np.array(image, dtype='float32'))
                labels.append([c,current_c])
    
                images2.append(image)
                issames.append(issame)
            ########################## when use Test func as callback may need batch_size, but if not, we set batch_size= len(all_test_imgs)
            if len(images1) == self.batch_size:
                yield np.array(images1), np.array(images2), np.array(issames)
                images1     = []
                images2     = []
                issames     = []
                
        yield np.array(images1), np.array(images2), np.array(issames)#last batch which is less than batch_size

             

        
class Test_callback(keras.callbacks.Callback):
    def __init__(self, testdata_loader):
        self.testdata_loader    = testdata_loader

    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):        
        labels, distances = [], []
        print("Run Testing")

        for _, (data_a, data_p, label) in enumerate(self.testdata_loader.generate()):
            out_a, out_p    = self.model.predict(data_a), self.model.predict(data_p)# [1]output embedding
            if len(out_a)>1:
                out_a, out_p = out_a[1], out_p[1]
            dists           = np.linalg.norm(out_a - out_p, axis=1)
            distances.append(dists)
            labels.append(label)

        labels      = np.array([sublabel for label in labels for sublabel in label])
        distances   = np.array([subdist for dist in distances for subdist in dist])
        _, _, accuracy, _, _, _, _ = evaluate(distances,labels)
        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return        
############################################################################################################################       

if __name__ == "__main__":
    
    tra_person_folder = './tracking_data/tracking/car_person/' #'./tracking_data/tracking/person/'
    #annotation_path = "./tracking_datasets/person/person_imgidx.txt" #change per need    
    input_shape     = [160, 160, 3]
    backbone        = "mobilenet" #change per need
    model_path      = "logs/best_model.h5" #change per need
    Init_Epoch      = 0
    Epoch           = 10 #100
    batch_size      = 24 #96
    save_dir        = 'logs/test_results/'
    png_save_path   = os.path.join(save_dir, "test_accuracy.png")
    log_interval    = 1
    test_gpu        = [0,]
    
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in test_gpu)
    ngpus_per_node                      = len(test_gpu)
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    #num_classes = get_num_classes(annotation_path)
    num_classes = None
    
    # print('Loading model...')
    # model_path = os.path.expanduser(model_path)
    # #model = tf.keras.models.load_model(model_path, custom_objects={'_triplet_loss': triplet_loss}, compile=True)
    # model = tf.keras.models.load_model(model_path, custom_objects={'_triplet_loss': triplet_loss,'K':K}, compile=False)
    # print('{} model loaded.'.format(model_path))

    model = asso_net(input_shape, backbone=backbone, mode="predict")
    print('Loading weights...')
    model_path = os.path.expanduser(model_path)
    model.load_weights(model_path, by_name=True, skip_mismatch=True)#https://keras.io/api/models/model_saving_apis/
    print('{} model loaded.'.format(model_path))

    # val_split = 0.1 #0.01
    # test_split = 0.1 #0.01
    # with open(annotation_path,"r") as f:
    #     lines = f.readlines()
    # np.random.seed(500)
    # np.random.shuffle(lines)# use this combine with seed can gurante that each time same shuffle
    # np.random.seed(None)# shuffle for spliting train and valid dataset
    # num_val = int(len(lines)*val_split)
    # num_test = int(len(lines)*test_split)
    # num_train = len(lines) - num_val - num_test
    
    # test_lines = lines[-num_test:]
    
    # Reading file to list
    # with open('test_data_idx.txt',"r") as f:
    #     test_lines = f.readlines() 
        
    test_datasets_path   = "./testing_datasets/car_person/" 
    idxFromimg(idx_file_name='test_imgidx.txt', datasets_path=test_datasets_path) 
    with open(test_datasets_path+'test_imgidx.txt',"r") as f:
        test_lines = f.readlines()    
    
    if True:            
        rad_data, imgs_label, imgs=concat_npy(tra_folder=tra_person_folder)

        testdata_loader = TestDataset(input_shape, test_lines, batch_size, num_classes, imgs, random = True)
        Test(testdata_loader, model, png_save_path, log_interval, batch_size)
        # for epoch in range(Init_Epoch, Epoch):
        #     Test(testdata_loader, model, png_save_path, log_interval, batch_size)

        
        