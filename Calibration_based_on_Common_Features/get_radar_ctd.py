############### add search path ###################################
import sys
sys.path.append("./RADYolo/") # add search path: sys.path.append("../../")
#######################################################
import time
import numpy as np
import tensorflow as tf
from yolo import YOLO
import utils.radar_loader as loader
from tqdm import tqdm
import utils.helper as helper
import glob
import xml.etree.ElementTree as ET
#######################################################################################
def crop_3d_data(data, xmin, xmax, ymin, ymax):
    cropped_data = data[ymin:ymax, xmin:xmax, :]
    return cropped_data

def xml2rad(path,rad_fr_path,savepath):
    all_files = glob.glob('{}/*xml'.format(path))
    all_files.sort(key=lambda x: int(x.split('.')[0].split('/')[-1]))
    for xml_file in tqdm(all_files):
        tree    = ET.parse(xml_file)
        height  = int(tree.findtext('./size/height'))
        width   = int(tree.findtext('./size/width'))
        if height<=0 or width<=0:
            continue
        
        index = xml_file.split('.')[0].split('/')[-1]
        rad_path = rad_fr_path+ index + '.npy'
        rad_data = np.load(rad_path,allow_pickle=True)
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin')))
            ymin = int(float(obj.findtext('bndbox/ymin')))
            xmax = int(float(obj.findtext('bndbox/xmax')))
            ymax = int(float(obj.findtext('bndbox/ymax')))
            
            #cropped_data = crop_3d_data(rad_data, xmin, xmax, ymin, ymax)
            cropped_data = crop_3d_data(rad_data, xmin, xmax, rad_data.shape[0]-ymax, rad_data.shape[0]-ymin)#array is inverse with the RA img
            # ### only for testing if the generated RA npy is correct by showing the RA image
            # ## convert radar complex to real
            # rad_complex = cropped_data
            # rad_real = complexTo2Channels(rad_complex)
            # resized_data = resize_radar_data(rad_real, target_shape=[256, 256, 64], interpolation='nearest')
            # #plt.imshow(np.sum(resized_data, axis=-1), cmap='viridis', origin='lower')
            # RA = getLog(getSumDim(getMagnitude(rad_complex, power_order=2), target_axis=-1), scalar=10, log_10=True)
            # plt.imshow(RA, cmap='viridis', origin='lower')# attention --- origin='lower' just for showing
            # plt.show()
        
        ### save x_cent,y_cent,w,h,class
        with open(savepath+ index +'.npy' , 'wb') as f:
            np.save(f, cropped_data) 
        # ### only for testing if the generated RA npy is correct by showing the RA image
        # # convert radar complex to real
        # rad_complex = np.load(savepath+ index +'.npy',allow_pickle=True)
        # rad_real = complexTo2Channels(rad_complex)
        # resized_data = resize_radar_data(rad_real, target_shape=[256, 256, 64], interpolation='nearest')
        # plt.imshow(np.sum(resized_data, axis=-1), cmap='viridis', origin='lower')
        # plt.show()
#################################################################################
### radar bin to real value
def binsTOreal(numRangeBins,range_resolution, numDopplerBins,doppler_resolution, numAngleBins):
    """
    Map range_velo_angle bins to range(m), velocity(m/s) and angle(deg)
    """
    #'range'
    rng_grid = np.arange(numRangeBins) * range_resolution

    #'velocity'
    vel_grid = np.arange(numDopplerBins) * doppler_resolution
    
    #'angle'
    # for [-90, 90], radian will be [-1, 1]
    ang_bins = np.arange(numAngleBins) - numAngleBins/2
    agl_grid = np.rad2deg(np.arcsin(2*ang_bins/numAngleBins))  # rad to deg: np.rad2deg # np.deg2rad

    return rng_grid,vel_grid,agl_grid

def RADYolo_main():
    
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
        
    yolo = YOLO()
            
    ### config param ###
    global_mean= 3.2438383
    global_std = 6.8367246
    
    numRangeBins = 256 #numADCSamples
    numDopplerBins = 64 #256 #numLoopsPerFrame
    numAngleBins = 256 #128 #64

    range_resolution   = 0.24412109375
    doppler_resolution = 0.1932591218305504
    rng_grid,vel_grid,agl_grid = binsTOreal(numRangeBins,range_resolution, numDopplerBins,doppler_resolution, numAngleBins)
    rng_grid = np.flip(rng_grid)


    
    path="/xdisk/caos/leicheng/my_rawdata"+ "/frames/*.npy"
    all_RAD_files = glob.glob(path)
    all_RAD_files.sort(key=lambda x: int(x.split('.')[0].split('/')[-1]))
    
    #all_RAD_files = all_RAD_files[30:2634]  # train_data
    start_frame = 2634
    all_RAD_files = all_RAD_files[start_frame:]  # test_data
    

    pbar = tqdm(total=len(all_RAD_files))
    model_RAD_st = []

 
    ### load GT label
    label_path = "/xdisk/caos/leicheng/my_rawdata/RA_label/RA_label.npy"
    gt_label = np.load(label_path,allow_pickle=True)        
        
    """ Load data one by one for generating evaluation images """
    print("\n Start getting XY, it might take a while... \n")
    sequence_num = -1
    interpolation=15.
    real_xy = []
    rad_ts = []
    rad_ts_start=1679618481
    for RAD_file in all_RAD_files:#all_RAD_files #[all_RAD_files[0]]
        xy_frame = []
        sequence_num += 1
        rad_ts_frame=rad_ts_start + (start_frame+sequence_num)*(1/30)
        rad_ts.append(rad_ts_frame)
        with open("/xdisk/caos/leicheng/my_rawdata/RA_label" +'/RA_ts_label.npy' , 'wb') as f:
            np.save(f, np.array(rad_ts))
        

        ### load RAD input ###
        RAD_complex = loader.readRAD(RAD_file)
   
        ### NOTE: real time visualization ###
        RA = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
                        power_order=2), target_axis=-1), scalar=10, log_10=True)

        
        RA_img = helper.norm2Image(RA)[..., :3]
   
        RAD_data = helper.complexTo2Channels(RAD_complex)


        # global_mean= 3.748522719837311
        # global_std= 6.8367246 #0.6060742654553342
        RAD_data = (RAD_data - global_mean) / \
                            global_std                                    
        data = np.expand_dims(RAD_data, axis=0)
        
        
        
      
        if data is None :
            pbar.update(1)
            continue

        model_RAD_time_start = time.time()
        
        nms_pred,r_image = yolo.detect_image(data, crop = False, count = False)

        ### draw Pred
        for i in range(len(nms_pred)):
            bbox3d = nms_pred[i, :4]
            cls = int(nms_pred[i, 5])
            a=agl_grid[round(bbox3d[0])]
            r=rng_grid[round(bbox3d[1])]
            x,y = helper.polarToCartesian(r, np.deg2rad(a))
            xy_frame.append([x, y])
            
        real_xy.append(xy_frame)
        
        
        
        model_RAD_st.append(time.time() - model_RAD_time_start)

        pbar.update(1)
    print("------", " The average inference time for RAD Boxes: ", np.mean(model_RAD_st))

    return real_xy,rad_ts
    
      
if __name__ == "__main__":    
    RADYolo_main()