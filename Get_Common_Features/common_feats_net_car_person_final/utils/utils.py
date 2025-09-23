import numpy as np
from PIL import Image
import tensorflow        as tf
#from skimage.transform import resize

from scipy.ndimage import zoom

def cvt2Color(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def resize_3d_data(data, target_shape):
    resized_data = []

    # Convert to PIL image
    pil_img = Image.fromarray(data)

    # Resize to the target shape
    resized_img = pil_img.resize((target_shape[1], target_shape[0]))

    # Convert the resized image back to NumPy array
    resized_img = np.array(resized_img)

    # Pad the resized image to the target depth
    depth_padding = target_shape[2] - resized_img.shape[2]
    if depth_padding > 0:
        resized_img = np.pad(resized_img, ((0, 0), (0, 0), (0, depth_padding)), mode='constant')

    resized_data.append(resized_img)

    return resized_data


def resize_with_only_padding(data, target_shape):
    # Get the original shape
    original_shape = data.shape
    # Create an array of zeros with the target shape
    resized_data = np.zeros(target_shape)
    # Calculate the starting indices for the original data within the padded array
    start_indices = tuple((np.array(target_shape) - np.array(original_shape)) // 2)
    # Copy the original data into the padded array
    resized_data[start_indices[0]:start_indices[0]+original_shape[0],
                 start_indices[1]:start_indices[1]+original_shape[1],
                 start_indices[2]:start_indices[2]+original_shape[2]] = data

    return resized_data




def resize_radar_data(data, target_shape, interpolation='nearest'):
    # Input parameters:
    # data: Original radar data, shape (range, azimuth, doppler)
    # target_shape: Target size, e.g., (target_range, target_azimuth, target_doppler)
    # interpolation: Interpolation method, options: 'nearest', 'bilinear', 'trilinear'
    #In nearest neighbor interpolation, the missing values are filled with the values of the nearest neighboring data points. This means that the original data points are simply replicated or duplicated to fill in the gaps. As a result, the data distribution remains unchanged.
    #when dealing with radar data where the statistical properties and distribution of the data are important for analysis and interpretation.
    #In the 'bilinear' case, the resizing is performed separately along each dimension using bilinear interpolation. 
    #The order=2 parameter specifies trilinear interpolation, which considers the neighboring values along all three dimensions to compute the interpolated values. Trilinear interpolation takes into account the spatial relationships between adjacent voxels in all three dimensions, resulting in a smoother and more accurate interpolation.
    # Returns:
    # resized_data: Resized radar data, shape (target_range, target_azimuth, target_doppler)
    
    # Get the original data size
    orig_range, orig_azimuth, orig_doppler = data.shape
    target_range, target_azimuth, target_doppler = target_shape
    
    # Calculate the scaling ratios
    scale_range = target_range / orig_range
    scale_azimuth = target_azimuth / orig_azimuth
    scale_doppler = target_doppler / orig_doppler
    
    # Resize based on the interpolation method
    if interpolation == 'nearest':
        # Perform nearest neighbor interpolation along range dimension
        resized_data = zoom(data, (scale_range, 1, 1), order=0)

        # Perform nearest neighbor interpolation along azimuth dimension
        resized_data = zoom(resized_data, (1, scale_azimuth, 1), order=0)

        # Perform nearest neighbor interpolation along doppler dimension
        resized_data = zoom(resized_data, (1, 1, scale_doppler), order=0)
    elif interpolation == 'bilinear':
        # Perform bilinear interpolation along range dimension
        resized_data = zoom(data, (scale_range, 1, 1), order=1)

        # Perform bilinear interpolation along azimuth dimension
        resized_data = zoom(resized_data, (1, scale_azimuth, 1), order=1)

        # Perform bilinear interpolation along doppler dimension
        resized_data = zoom(resized_data, (1, 1, scale_doppler), order=1)
    elif interpolation == 'trilinear':
        resized_data = zoom(data, (scale_range, scale_azimuth, scale_doppler), order=2)
    else:
        raise ValueError("Unsupported interpolation method. Choose from 'nearest', 'bilinear', 'trilinear'.")
    
    return resized_data

# def get_num_classes(annotation_path):
#     with open(annotation_path) as f:
#         dataset_path = f.readlines()

#     labels = []
#     for path in dataset_path:
#         path_split = path.split(";")
#         labels.append(int(path_split[0]))
#     num_classes = np.max(labels) + 1
#     return num_classes

def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(path_split[0])
    unique_classes = sorted(list(set(labels)))
    num_classes = len(unique_classes)
    return num_classes,unique_classes

def preprocess_input(image,rescale_type=0): #scale
    if rescale_type==0:
        image /= 255.0 # rescale to [0,1] 
    elif rescale_type==1:  
        image = (image/ 127.5) - 1 # rescale to [-1,1]
    
    # # Pre-trained Xception weights requires that input be scaled
    # # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # # outputs: `(inputs * scale) + offset`
    # scale_layer = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    # image = scale_layer(image)
    # image = image.numpy()
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

#-------------------------------------------------------------------------------------------------------------------------------#
#   From https://github.com/ckyrkou/Keras_FLOP_Estimator 
#   Fix lots of bugs
#-------------------------------------------------------------------------------------------------------------------------------#
def net_flops(model, table=False, print_result=True):
    if (table == True):
        print("\n")
        print('%25s | %16s | %16s | %16s | %16s | %6s | %6s' % (
            'Layer Name', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPS'))
        print('=' * 120)
        

    t_flops = 0
    factor  = 1e9

    for l in model.layers:
        try:
            o_shape, i_shape, strides, ks, filters = ('', '', ''), ('', '', ''), (1, 1), (0, 0), 0
            flops   = 0

            name    = l.name
            
            if ('InputLayer' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                

            elif ('Reshape' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]


            elif ('Padding' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]


            elif ('Flatten' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                

            elif 'Activation' in str(l):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                

            elif 'LeakyReLU' in str(l):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    flops   += i_shape[0] * i_shape[1] * i_shape[2]
                    

            elif 'MaxPooling' in str(l):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]
                    

            elif ('AveragePooling' in str(l) and 'Global' not in str(l)):
                strides = l.strides
                ks      = l.pool_size
                
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    flops   += o_shape[0] * o_shape[1] * o_shape[2]


            elif ('AveragePooling' in str(l) and 'Global' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    flops += (i_shape[0] * i_shape[1] + 1) * i_shape[2]
                

            elif ('BatchNormalization' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    temp_flops = 1
                    for i in range(len(i_shape)):
                        temp_flops *= i_shape[i]
                    temp_flops *= 2
                    
                    flops += temp_flops
                

            elif ('Dense' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                
                    temp_flops = 1
                    for i in range(len(o_shape)):
                        temp_flops *= o_shape[i]
                        
                    if (i_shape[-1] == None):
                        temp_flops = temp_flops * o_shape[-1]
                    else:
                        temp_flops = temp_flops * i_shape[-1]
                    flops += temp_flops

            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' not in str(l)):
                strides = l.strides
                ks      = l.kernel_size
                filters = l.filters
                bias    = 1 if l.use_bias else 0
                
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    if (filters == None):
                        filters = i_shape[2]
                    flops += filters * o_shape[0] * o_shape[1] * (ks[0] * ks[1] * i_shape[2] + bias)


            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' in str(l) and 'SeparableConv2D' not in str(l)):
                strides = l.strides
                ks      = l.kernel_size
                filters = l.filters
                bias    = 1 if l.use_bias else 0
            
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    if (filters == None):
                        filters = i_shape[2]
                    flops += filters * o_shape[0] * o_shape[1] * (ks[0] * ks[1] + bias)
                

            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' in str(l)):
                strides = l.strides
                ks      = l.kernel_size
                filters = l.filters
                
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]
                    
                    if (filters == None):
                        filters = i_shape[2]
                    flops += i_shape[2] * o_shape[0] * o_shape[1] * (ks[0] * ks[1] + bias) + \
                             filters * o_shape[0] * o_shape[1] * (1 * 1 * i_shape[2] + bias)

            elif 'Model' in str(l):
                flops = net_flops(l, print_result=False)
                
            t_flops += flops

            if (table == True):
                print('%25s | %16s | %16s | %16s | %16s | %6s | %5.4f' % (
                    name[:25], str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops))
                
        except:
            pass
    
    t_flops = t_flops * 2
    if print_result:
        show_flops = t_flops / factor
        print('Total GFLOPs: %.3fG' % (show_flops))
    return t_flops
