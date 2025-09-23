import math
import os
import random

from tensorflow import keras
import numpy as np
from tensorflow.python.keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import cv2
from PIL import Image
from .utils import cvt2Color, preprocess_input, resize_image,resize_radar_data,resize_with_only_padding

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

def normalize_data(data):
    '''  data represents the 3D radar data in the shape [width, height, depth, channels]. 
    The normalize_data function calculates the mean and standard deviation along the width, height, and depth dimensions (i.e., channels remain unchanged).
    It then subtracts the mean and divides by the standard deviation to normalize the data. '''
    # Compute mean along each channel
    mean = np.mean(data, axis=(0, 1, 2))
    # Compute standard deviation along each channel
    std = np.std(data, axis=(0, 1, 2))
    # Normalize the data
    normalized_data = (data - mean) / std
    return normalized_data

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

class Dataset_BatchGet(keras.utils.Sequence):
    def __init__(self, input_shape, paths, labels, batch_size, num_classes, rand_flag=True,new_size=[256,256,64],new_size_img=[416,416,3]):
        self.input_shape    = input_shape        
        self.batch_size     = batch_size
        self.num_classes    = num_classes
        self.rand_flag      = rand_flag
        self.new_size       = new_size
        self.new_size_img   = new_size_img        
        self.paths          = paths
        self.labels         = labels
        self.length         = len(paths)

        
    def __len__(self):
        return math.ceil(self.length / float(self.batch_size)) #int(np.ceil(len(self.paths) / self.batch_size))

    
    ##################### Normalization with Min-Max #################################
    def __getitem__(self, index):
        batch_paths = self.paths[index * self.batch_size : (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size : (index + 1) * self.batch_size]
    
        batch_x_real = []
        batch_x_imag = []
        batch_x_images = []
        for i in range(len(batch_paths)):
            path = batch_paths[i]
            img_path = path[0]
            rad_path = path[1]
            ######    Radar  ###############################################################
            # Convert radar complex to real
            rad_complex = np.load(rad_path, allow_pickle=True)
            # Separate real and imaginary parts
            real_part, imag_part = separate_real_imaginary(rad_complex)
            # Normalize real and imaginary parts
            real_normalized, imag_normalized = normalize_real_imaginary(real_part, imag_part)
    
            # Resize real and imaginary parts
            resized_real = resize_radar_data(real_normalized, self.new_size, interpolation='nearest')
            resized_imag = resize_radar_data(imag_normalized, self.new_size, interpolation='nearest')
    
            # Reshape the input data to match the expected input shape of (batch_size, height, width, channels)
            reshape_real = np.expand_dims(resized_real, axis=-1)
            reshape_imag = np.expand_dims(resized_imag, axis=-1)
    
            batch_x_real.append(reshape_real)
            batch_x_imag.append(reshape_imag)
            
            
            ######    Image   ###############################################################
            # convert img to array
            image = cvt2Color(Image.open(img_path))
            # flip img0
            if self.rand()<.5 and self.rand_flag: 
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.new_size_img[1], self.new_size_img[0]], letterbox_image = True)
            image = preprocess_input(np.array(image, dtype='float32'))
            
            batch_x_images.append(image)
    
        #batch_y = keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)
        rad_data = [np.array(batch_x_real, dtype=np.float16), np.array(batch_x_imag, dtype=np.float16)]
    
        return [rad_data, np.array(batch_x_images)], batch_labels


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
        
    


