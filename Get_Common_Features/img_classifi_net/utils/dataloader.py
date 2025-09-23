import math
import os
import random

from tensorflow import keras
import numpy as np
from tensorflow.python.keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import cv2
from PIL import Image
from .utils import cvt2Color, preprocess_input, resize_image


def trim_zeros(arr,mode='b',axis=0):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    A string with ‘f’ representing trim from front and ‘b’ to trim from back. 
    Default is ‘fb’, trim zeros from both front and back of the array.
    https://stackoverflow.com/questions/55917328/numpy-trim-zeros-in-2d-or-3d
    """
    if mode=='fb':
        if axis == -1: # both n-D
            slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
        else: # along the specified axis
            slices = tuple(slice(idx.min(), idx.max() + 1) if idx_en==axis else slice(0,np.size(arr,axis=idx_en)) for idx_en, idx in enumerate(np.nonzero(arr)))#[0:1] will keep the data structure  
            
    elif mode=='f':
        if axis == -1: # both n-D
            slices = tuple(slice(idx.min(), np.size(arr,axis=idx_en)) for idx_en, idx in enumerate(np.nonzero(arr)))
        else:  # along the specified axis
            slices = tuple(slice(idx.min(), np.size(arr,axis=idx_en)) if idx_en==axis else slice(0,np.size(arr,axis=idx_en)) for idx_en, idx in enumerate(np.nonzero(arr)))#[0:1] will keep the data structure  
            
    elif mode=='b':
        if axis == -1: # both n-D
            slices = tuple(slice(0, idx.max() + 1) for idx in np.nonzero(arr))
        else:  # along the specified axis
            slices = tuple(slice(0, idx.max() + 1) if idx_en==axis else slice(0,np.size(arr,axis=idx_en)) for idx_en, idx in enumerate(np.nonzero(arr)))#[0:1] will keep the data structure  
            
    return arr[slices]

def proj_rad2img(rad_xyz, cali_method=1):
    ## intrinsic mat
    cameraMatrix = np.array([[554.4203610089122, 0.,               299.0464166708532],
                             [  0.,              556.539219672516, 265.177086523325 ],
                             [  0.,              0.,               1.              ]])
    ############# distortion coefficents
    dist_coeffs = np.array([-0.3941065587817811, 0.1667170598953747, -0.003527054281471521, 0.001866412711427509, 0]).reshape(5,1)
    if cali_method:
        ##### my own calibration
        rotation_vector=np.array([[ 1.30927651],
                                  [-1.29183232],
                                  [ 1.09104368]])
        translation_vector=np.array([[-0.00115773],
                                     [-0.0608874 ],
                                     [-0.01496503]])
        r2imgpoints, jacobian = cv2.projectPoints(rad_xyz, rotation_vector, translation_vector, cameraMatrix, dist_coeffs)
        r2imgpoints=np.squeeze(r2imgpoints)
        r2img_x_points=r2imgpoints[:,0]
        r2img_y_points=r2imgpoints[:,1]
    else:
        ## use physical measurements
        radarpoints_a=rad_xyz.copy()
        R1 = np.eye(3)
        R1_vector = cv2.Rodrigues(R1)[0] #convert to rotatin vector
        t1 = np.array([0.,-0.045,0.])
        radarpoints_a[:,0] = -rad_xyz[:,1]
        radarpoints_a[:,1] = -rad_xyz[:,2]
        radarpoints_a[:,2] =  rad_xyz[:,0]
        r2imgpoints, jacobian = cv2.projectPoints(radarpoints_a, R1_vector, np.expand_dims(t1,axis=1), cameraMatrix, dist_coeffs)
        r2imgpoints = np.squeeze(r2imgpoints)
        r2img_x_points_pm = r2imgpoints[:,0]
        r2img_y_points_pm = r2imgpoints[:,1]
    
    return r2imgpoints


class Dataset_BatchGet(keras.utils.Sequence):
    def __init__(self, input_shape, paths, labels, batch_size, num_classes, rand_flag=True,new_size=[416,416,3]):
        self.input_shape    = input_shape        
        self.batch_size     = batch_size
        self.num_classes    = num_classes
        self.rand_flag      = rand_flag
        self.new_size       = new_size        
        self.paths          = paths
        self.labels         = labels
        self.length         = len(paths)

        
    def __len__(self):
        return math.ceil(self.length / float(self.batch_size)) #int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, index): # index's range is [0, _len_]
        batch_paths = self.paths[index * self.batch_size : (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size : (index + 1) * self.batch_size]
        
        batch_x = []
        for path in batch_paths:
            # convert img to array
            image = cvt2Color(Image.open(path.rstrip('\n')))
            # flip img
            if self.rand()<.5 and self.rand_flag: 
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.new_size[1], self.new_size[0]], letterbox_image = False)
            image = preprocess_input(np.array(image, dtype='float32'))
            batch_x.append(image)
            
        batch_y = keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)
        
        return np.array(batch_x), np.array(batch_y)


    def NormalizeData_colwise(self,data):
        #return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-16)
        # call MinMaxScaler object
        min_max_scaler = MinMaxScaler()
        # norm a numpy array column-wise
        data_norm = min_max_scaler.fit_transform(data)
        return data_norm
        
            
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def load_dataset(self):
        for path in self.lines:
            path_split = path.split(";")
            self.paths.append(int(path_split[1].split()[0]))
            self.labels.append(int(path_split[0]))
        self.paths  = np.array(self.paths,dtype=np.object)
        self.labels = np.array(self.labels)
        
    


