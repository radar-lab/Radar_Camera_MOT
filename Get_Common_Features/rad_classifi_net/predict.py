import tensorflow as tf
from PIL import Image

from assonet_pred import assonet_pred

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":
    model = assonet_pred()
        
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1) # img/0.png img/23763.png img/23771.png
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:') # img/1.png img/621.png img/23765.png img/23382.png img/23273.png
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        
        distance = model.detect_image(image_1,image_2)
        print(distance)
