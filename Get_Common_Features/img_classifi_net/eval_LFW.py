import tensorflow as tf

from nets.facenet import facenet
from utils.dataloader import LFWDataset
from utils.utils_metrics import test

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":

    backbone        = "mobilenet"

    input_shape     = [160, 160, 3]

    model_path      = "model_data/facenet_mobilenet.h5"

    lfw_dir_path    = "lfw"
    lfw_pairs_path  = "model_data/lfw_pair.txt"

    batch_size      = 256
    log_interval    = 1

    png_save_path   = "model_data/roc_test.png"

    test_loader     = LFWDataset(dir=lfw_dir_path,pairs_path=lfw_pairs_path, batch_size=batch_size, input_shape=input_shape)

    model           = facenet(input_shape, backbone=backbone, mode="predict")
    model.load_weights(model_path, by_name = True)

    test(test_loader, model, png_save_path, log_interval, batch_size)
