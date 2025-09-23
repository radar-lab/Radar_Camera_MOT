import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def draw_and_save_boxes(image, out_boxes, out_scores, out_classes, class_names, output_path, output_filename, thickness=3):
    """
    Draw bounding boxes and labels on an image and save it as a high-quality image.

    Parameters:
    - image_path: Path to the image file
    - out_boxes: List of bounding boxes, each box is [top, left, bottom, right]
    - out_scores: List of scores for each bounding box
    - out_classes: List of class indices for each bounding box
    - class_names: List of class names
    - colors: List of colors for each class
    - output_path: Directory to save the output image
    - output_filename: Filename to save the output image
    - thickness: Thickness of the bounding box lines
    """
    colors = [
        (255/255, 0/255, 255/255, 0.3),  # Magenta with 30% transparency
        (255/255, 0/255, 0/255, 0.3),  # Red with 30% transparency
        (0/255, 255/255, 0/255, 0.3),  # Green with 30% transparency
        (0/255, 0/255, 255/255, 0.3),  # Blue with 30% transparency
        (255/255, 255/255, 0/255, 0.3),  # Yellow with 30% transparency
        (0/255, 255/255, 255/255, 0.3),  # Cyan with 30% transparency        
        (192/255, 192/255, 192/255, 0.3),  # Silver with 30% transparency
        (128/255, 0/255, 128/255, 0.3),  # Purple with 30% transparency
        (255/255, 165/255, 0/255, 3/255),  # Orange with 30% transparency
        (0/255, 128/255, 0/255, 0.3)   # Dark Green with 30% transparency
    ]    
    # Load the image
    image = np.array(image) #np.array(Image.open(image_path))
    height, width, _ = image.shape

    # Create the figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 8))

    # Display the image
    ax.imshow(image)
    ax.axis('off')

    # Draw bounding boxes and labels
    for i, c in list(enumerate(out_classes)):
        predicted_class = class_names[int(c)]
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(height, np.floor(bottom).astype('int32'))
        right = min(width, np.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        print(label, top, left, bottom, right)

        # Draw the bounding box
        rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=thickness, edgecolor=colors[int(c)], facecolor='none')
        ax.add_patch(rect)

        # Calculate label position
        text_origin = np.array([left + 1, top - 12])

        # Draw the label background
        rect = patches.Rectangle(text_origin, 1, 12, linewidth=0, edgecolor='none', facecolor=colors[int(c)], alpha=0.5)
        ax.add_patch(rect)

        # Draw the label text
        ax.text(left + 1, top - 3, label, fontsize=12, color='white', verticalalignment='top' )

    # Adjust the layout
    plt.tight_layout()

    # Check if the output path exists, create if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the high-quality image
    full_output_path = os.path.join(output_path, output_filename)
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')

    # Show the image
    #plt.show()
    
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
    #mode = "idx_file"
    mode = "folder"

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
            r_image, out_boxes  = yolo.detect_image(image,plot=True)
            r_image.save(os.path.join(dir_save_path, os.path.basename(image_path)), quality=100, subsampling=0)
            # out_boxes, out_scores, out_classes,class_names   = yolo.detect_image(image)
            # draw_and_save_boxes(image, out_boxes, out_scores, out_classes, class_names, 
            #                     dir_save_path, os.path.basename(image_path))
    elif mode == "folder":
        import glob,re
        import os
        from tqdm import tqdm
        
        
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)
        
        
        image_paths = glob.glob("/xdisk/caos/leicheng/my_rawdata_0624/0624/jiahao-hao/RA" + "/*.png")
        image_paths = sorted(image_paths, key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else float('inf'))        
    
            
        for image_path in tqdm(image_paths):            
            image       = Image.open(image_path)
            r_image, out_boxes  = yolo.detect_image(image,plot=True)
            r_image.save(os.path.join(dir_save_path, os.path.basename(image_path)), quality=100, subsampling=0)   
    
    elif mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
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

