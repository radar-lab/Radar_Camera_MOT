from functools import wraps, reduce

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Layer, ZeroPadding2D)
from tensorflow.keras.regularizers import l2

def compose(*funcs):
    '''   
    reduce(fun,seq) is used to apply a particular function passed in its argument to all of the list elements mentioned in the sequence:      
    1, At first step, first two elements of sequence are picked and the result is obtained.
    2, Next step is to apply the same function to the previously attained result and the number just succeeding the second element and the result is again stored.
    3, This process continues till no more elements are left in the container.

    '''
    if funcs: #https://www.geeksforgeeks.org/reduce-in-python/  https://www.geeksforgeeks.org/python-lambda-anonymous-functions-filter-map-reduce/
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)#Mish(BatchNormalization(DarknetConv2D(*args, **no_bias_kwargs)))
    else:
        raise ValueError('Composition of empty sequence not supported.')

class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))#x*tanh(log(exp(x) + 1))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'   
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   DarknetConv2D + BatchNormalization + Mish
#---------------------------------------------------#
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


def resblock_body(x, num_filters, num_blocks, all_narrow=True, weight_decay=5e-4):

    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2), weight_decay=weight_decay)(preconv1)

    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), weight_decay=weight_decay)(preconv1)

    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), weight_decay=weight_decay)(preconv1)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1), weight_decay=weight_decay),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3), weight_decay=weight_decay))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), weight_decay=weight_decay)(mainconv)


    route = Concatenate()([postconv, shortconv])


    return DarknetConv2D_BN_Mish(num_filters, (1,1), weight_decay=weight_decay)(route)


def darknet_body(x, weight_decay=5e-4):
    x = DarknetConv2D_BN_Mish(32, (3,3), weight_decay=weight_decay)(x)
    x = resblock_body(x, 64, 1, False, weight_decay=weight_decay)
    x = resblock_body(x, 128, 2, weight_decay=weight_decay)
    x = resblock_body(x, 256, 8, weight_decay=weight_decay)
    feat1 = x
    x = resblock_body(x, 512, 8, weight_decay=weight_decay)
    feat2 = x
    # x = resblock_body(x, 1024, 4, weight_decay=weight_decay)
    # feat3 = x
    return feat2

