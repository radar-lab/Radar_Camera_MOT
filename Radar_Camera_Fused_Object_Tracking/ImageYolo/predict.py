import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import shutil
import os
# Set the necessary environment variables before importing GPU-related libraries
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace "0" with the index of the GPU device you want to use

def load_radimg_yolo_model(confidence = 0.5,nms_iou = 0.3):
    ############### Lei ###################################
    import os, sys
    # Inserting the directoryat the beginning of the path 
    #This ensures that Python will search for modules in this directory first when importing.
    # sys.path.insert(0, "ImageYolo/")
    # current_directory = os.getcwd()
    # print("Current Directory:", current_directory)
    #######################################################
    from yolo import YOLO
    # # Set the necessary environment variables before importing GPU-related libraries
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace "0" with the index of the GPU device you want to use
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    print("Load model begin.")
    classes_path = 'ImageYolo/model_data/radar_classes.txt'
    model_path = 'ImageYolo/model_data/best_epoch_weights.h5'
    anchors_path = 'ImageYolo/model_data/RA_anchors.txt'
    font_path = 'ImageYolo/model_data/simhei.ttf'
    input_shape = [256, 256]

    yolo = YOLO(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path, font_path =font_path,
                input_shape=input_shape, confidence=confidence, nms_iou=nms_iou)
    print("Load model done.")
    return yolo


def process_imgs_to_get_bboxes(images,confidence = 0.5,nms_iou = 0.3,yolo=None):
    if yolo is None:
        ############### Lei ###################################
        import os, sys
        # Inserting the directoryat the beginning of the path 
        #This ensures that Python will search for modules in this directory first when importing.
        # sys.path.insert(0, "ImageYolo/")
        # current_directory = os.getcwd()
        # print("Current Directory:", current_directory)
        #######################################################
        from yolo import YOLO
        
        # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
            
        print("Load model begin.")
        classes_path = 'ImageYolo/model_data/radar_classes.txt'
        model_path = 'ImageYolo/model_data/best_epoch_weights.h5'
        anchors_path = 'ImageYolo/model_data/RA_anchors.txt'
        font_path = 'ImageYolo/model_data/simhei.ttf'
        input_shape = [256, 256]
    
        yolo = YOLO(model_path=model_path, classes_path=classes_path, anchors_path=anchors_path, font_path =font_path,
                    input_shape=input_shape, confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

    from tqdm import tqdm
    #dir_save_path = "img_out/"
    # if os.path.exists(dir_save_path):
    #     shutil.rmtree(dir_save_path)
    # os.makedirs(dir_save_path)

    all_bboxes = []
    for image in tqdm(images):
        #image = Image.open(image_path)
        # Assuming you have a NumPy array named 'image_array'
        image = Image.fromarray(image)
        r_image,out_boxes = yolo.detect_image(image)
        out_boxes = out_boxes[:, [1, 0, 3, 2]] #convert to [xmin, ymin, xmax, ymax]
        all_bboxes.append(out_boxes)
        #r_image.show()
        #r_image.save(os.path.join(dir_save_path, str(np.random.randint(0, 101))+'.png'), quality=95, subsampling=0)
    
    return all_bboxes
    
if __name__ == "__main__":
    
    from yolo import YOLO
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Load model.")
    classes_path    = 'model_data/radar_classes.txt'
    model_path      = 'model_data/best_epoch_weights.h5' #'logs/best_model.h5' # 
    anchors_path    = 'model_data/RA_anchors.txt'
    input_shape     = [256, 256]
    confidence      = 0.5
    nms_iou         = 0.3  #bbox larger than nms_iou will be considered as overlapping bbox. a smaller IOU threshold can retain more detection boxes, while a larger IOU threshold can reduce the redundancy of overlapping boxes.
    
    yolo = YOLO(model_path=model_path, classes_path=classes_path,anchors_path=anchors_path,input_shape=input_shape, confidence = confidence, nms_iou = nms_iou)
    print("Load model done.")


    #mode = "predict"
    mode = "idx_file"

    crop            = False
    count           = False


    dir_save_path   = "img_out/"


    if mode == "idx_file":
        import os
        from tqdm import tqdm
        
        
        if os.path.exists(dir_save_path):
            shutil.rmtree(dir_save_path)
        os.makedirs(dir_save_path)
        
        test_file_path  = './test_paths.txt'
        # Read the file and store paths in a list
        with open(test_file_path, 'r') as f:
            paths = f.readlines()
        # Remove leading/trailing whitespaces and newlines from paths
        image_paths = [path.strip() for path in paths]
    
            
        for image_line in tqdm(image_paths):
            image_path  = image_line.split()[0]
            image       = Image.open(image_path)
            r_image,out_boxes     = yolo.detect_image(image)
            r_image.save(os.path.join(dir_save_path, os.path.basename(image_path)), quality=95, subsampling=0)

    elif mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image,out_boxes = yolo.detect_image(image, crop = crop, count = count)
                r_image.show()

