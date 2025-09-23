import datetime
import os


import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler,TensorBoard,ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam

from nets.asso_net_rad import asso_net_rad
from nets.asso_net_training import get_lr_scheduler, triplet_loss
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import AssonetDataset_rad, trim_zeros
from utils.utils import get_num_classes, show_config

from data_prep import concat_npy,idxFormimg, data_prep_clf
from keras.utils import plot_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

if __name__ == "__main__":
    train_gpu       = [0,]
    tra_person_folder = '../association_net/tracking_data/tracking/car_person/' #npy files; '/home/lei/Desktop/tracking/association_net/'
    datasets_path   = "../association_net/tracking_datasets/car_person/"
    idx_file_name   = 'object_imgidx.txt'
    annotation_file = idx_file_name
    #annotation_file = datasets_path + idx_file_name #"./tracking_datasets/person/object_imgidx.txt" #change per need    
    input_shape     = [16,16,2] #[16,16,2] for projection-xy; [16,16,4] for xyzv
    #model_path      = "model_data/facenet_mobilenet.h5" #change per need
    model_path      = "./model_data/rad_classi_best_wei.h5" #change per need
    Init_Epoch      = 0
    Epoch           = 800 #800
    batch_size      = 24 #24
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
    
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # if ngpus_per_node > 1 and ngpus_per_node > len(gpus):
    #     raise ValueError("The number of GPUs specified for training is more than the GPUs on the machine")

    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))

    num_classes = get_num_classes(annotation_file)
    
    if ngpus_per_node > 1:
        with strategy.scope():
            model = asso_net_rad(input_shape, num_classes,  mode="train")
            if model_path != '':
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
    else:
        model = asso_net_rad(input_shape, num_classes, mode="train")
        if model_path != '':
            model.load_weights(model_path, by_name=True, skip_mismatch=True)#https://keras.io/api/models/model_saving_apis/


    val_split = 0.1 
    test_split = 0.1 
    with open(annotation_file,"r") as f:
        lines = f.readlines()
    np.random.seed(500)
    np.random.shuffle(lines)# use this combine with seed can gurante that each time same shuffle
    np.random.seed(None)# shuffle for spliting train and valid dataset
    num_val = int(len(lines)*val_split)
    num_test = int(len(lines)*test_split)
    #num_test = 0
    num_train = len(lines) - num_val - num_test
    # np.savetxt("test_data_idx.txt", lines[-num_test:], fmt='%s')# save test_data idx to txt
    # test_data_lines = np.loadtxt('test_data_idx.txt')
    # writing to file
    file1 = open('test_data_idx.txt', 'w')
    file1.writelines(lines[-num_test:])
    file1.close()
    # # Reading file to list
    # with open('test_data_idx.txt',"r") as f1:
    #     test_lines = f1.readlines()
        
    show_config(
        num_classes = num_classes,  model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )    

    if True:
        if batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3.")
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('dataset is too small to run')
            
        ## read Npy data    
        rad_data, imgs_label, imgs=concat_npy(tra_folder=tra_person_folder)
        ## split data
        ctr,xyzv,saervc,bbox=data_prep_clf(rad_data,imgs_label) 
        xyzv = xyzv.transpose(1,2,0) #xyzv=(4,NUM,256) --> xyzv=(NUM,256,4)
        
        ## create dataset 
        train_dataset   = AssonetDataset_rad(input_shape, lines[:num_train], batch_size, num_classes, ngpus_per_node, xyzv, random = True)
        val_dataset     = AssonetDataset_rad(input_shape, lines[num_train: (num_train+num_val)], batch_size, num_classes, ngpus_per_node, xyzv, random = False)

        optimizer = {
            'adam'  : Adam(learning_rate = Init_lr_fit, beta_1 = momentum),
            'sgd'   : SGD(learning_rate = Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]


        if ngpus_per_node > 1:
            with strategy.scope():
                model.compile(
                    loss={'Embedding' : triplet_loss(), 'Softmax' : 'categorical_crossentropy'}, 
                    optimizer = optimizer, metrics = {'Softmax' : 'categorical_accuracy'}
                )
        else:
            model.compile(
                loss={'Embedding' : triplet_loss(), 'Softmax' : 'categorical_crossentropy'}, 
                optimizer = optimizer, metrics = {'Softmax' : 'categorical_accuracy'}
            )
            
        plot_model(model,to_file='./model_data/Commonnet_plot_rad.png',show_shapes=True, show_layer_names=True,expand_nested=False)
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        logging         = TensorBoard(log_dir)
        loss_history    = LossHistory(log_dir)
        checkpoint      = ModelCheckpoint(os.path.join(save_dir, "best_model_rad.h5"), 
                                monitor = 'val_loss', save_weights_only = False, save_best_only = True, save_freq = 'epoch')        
        # checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
        #                         monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = save_period)
        early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 30, verbose = 1)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)

        callbacks       = [logging, loss_history, checkpoint, lr_scheduler]
        #callbacks       = [logging, loss_history, checkpoint, lr_scheduler,early_stopping]

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit(
            x                   = train_dataset,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataset,
            validation_steps    = epoch_step_val,
            epochs              = Epoch,
            initial_epoch       = Init_Epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = callbacks
        )
