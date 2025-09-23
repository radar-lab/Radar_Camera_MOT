import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nets.asso_net import asso_net
from utils.utils import preprocess_input, resize_image, show_config



class assonet_pred(object):
    _defaults = {
        #"model_path"        : "model_data/facenet_mobilenet.h5",
        "model_path"        : "logs/best_model.h5",
        "input_shape"       : [160, 160, 3],
        "backbone"          : "mobilenet",
        "letterbox_image"   : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        self.generate()
        
        show_config(**self._defaults)
        
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        self.model = asso_net(self.input_shape, backbone=self.backbone, mode="predict")
        
        print('Loading weights into state dict...')
        self.model.load_weights(self.model_path, by_name=True)
        print('{} model loaded.'.format(model_path))
    

    def detect_image(self, image_1, image_2):
        image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        image_2 = resize_image(image_2, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        
        photo_1 = np.expand_dims(preprocess_input(np.array(image_1, np.float32)), 0)
        photo_2 = np.expand_dims(preprocess_input(np.array(image_2, np.float32)), 0)


        output1 = self.model.predict(photo_1)
        output2 = self.model.predict(photo_2)
    

        l1 = np.linalg.norm(output1-output2, axis=1)
        # l1 = np.sum(np.square(output1 - output2), axis=-1)

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va= 'bottom',fontsize=11)
        plt.show()
        return l1

