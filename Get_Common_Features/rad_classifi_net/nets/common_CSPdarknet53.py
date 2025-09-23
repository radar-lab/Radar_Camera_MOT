from functools import wraps, reduce

from tensorflow.keras import backend as K
#import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (LeakyReLU, Add, Concatenate,BatchNormalization,Dropout,Activation,
                                     Flatten,Dense,Conv2D, Conv3D, Layer, ZeroPadding2D, ZeroPadding3D,MaxPooling3D,GlobalAveragePooling3D)
from tensorflow.keras.regularizers import l2
#from utils.utils import compose

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
        #return inputs * tf.math.tanh(tf.math.softplus(inputs))#x*tanh(log(exp(x) + 1))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

#------------------------------------------------------#
#   单次卷积DarknetConv2D
#   如果步长为2则自己设定padding方式。https://blog.csdn.net/weixin_40576010/article/details/88639686
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """ 2D convolutional layer in this research 
    Args:
        input_tensor        ->          input 2D tensor, [None, w, h, channels]
        filters             ->          output channels, int
        kernel_size         ->          kernel size, int
        strides             ->          strides, tuple, (strides, strides)
        padding             ->          "same" or "valid", no captical letters
        activation          ->          "leaky_relu" or "mish"
    """    
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'   
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#------------------------------------------------------#
#   单次卷积DarknetConv3D
#   如果步长为2则自己设定padding方式。https://blog.csdn.net/weixin_40576010/article/details/88639686
#------------------------------------------------------#
@wraps(Conv3D)
def DarknetConv3D(*args, **kwargs):
    """ 3D convolutional layer in this research 
    Args:
        input_tensor        ->          input 3D tensor, [None, w, h, d, channels]
        filters             ->          output channels, int
        kernel_size         ->          kernel size, int
        strides             ->          strides, tuple, (strides, strides, strides)
        padding             ->          "same" or "valid", no captical letters
        activation          ->          "leaky_relu" or "mish"
        use_bias            ->          False, batch_normalization and use_bias cannot co-exist
        padding             ->          'valid', no padding for 'strides'==(2, 2, 2), otherwise 'same'
        
    Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
           activation=None, use_bias=use_bias, kernel_initializer='glorot_uniform', 
           bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
           )(input_tensor)
    """
    ##### NOTE: batch_normalization and use_bias cannot co-exist #####
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2, 2) else 'same'   
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv3D(*args, **darknet_conv_kwargs)


#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + Mish
#---------------------------------------------------#
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv3D + BatchNormalization + Mish
#---------------------------------------------------#
def DarknetConv3D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv3D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())
#---------------------------------------------------------------------#
#   小残差块结构
#---------------------------------------------------------------------#
def resblock(x, Conv, num_filters,kernel1,kernel2, all_narrow, weight_decay=5e-4):
    # y = DarknetConv2D_BN_Mish(num_filters//2, (1,1), weight_decay=weight_decay)(x)
    # y = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3), weight_decay=weight_decay))(y)
    # x = Add()([x,y])
    y = compose(
            Conv(num_filters//2, kernel1, weight_decay=weight_decay),
            Conv(num_filters//2 if all_narrow else num_filters, kernel2, weight_decay=weight_decay))(x)
    x = Add()([x,y])

    return x
#--------------------------------------------------------------------#
#   CSPdarknet的结构块
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后建立一个大的残差边shortconv、这个大残差边绕过了很多的小残差结构
#   主干部分会对num_blocks进行循环，循环内部是残差结构。
#   对于整个CSPdarknet的结构块，就是一个大残差块+内部多个小残差块
#--------------------------------------------------------------------#
#----------------------------------------------------------------#
#   3D 残差卷积
#----------------------------------------------------------------#
def resblock_body_3D(x, num_filters, num_blocks, all_narrow=True, weight_decay=5e-4,kernel1=(1,1,1),kernel2=(3,3,3)):
    #----------------------------------------------------------------#
    #   第一个卷积： 利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的减半压缩
    #   Out_W= (img_W − kernel_size + 2Pad )/Stride+1; Out_H= (img_H − kernel_size + 2Pad )/Stride+1; Out_D = filters_num
    #   [None, 256, 256, 64, 32] --> [None, 257, 257, 65, 32] --> [None, 128, 128, 32, 64]
    #----------------------------------------------------------------#
    preconv1 = ZeroPadding3D(((1,0),(1,0),(1,0)))(x)     #preconv1.shape
    preconv1 = DarknetConv3D_BN_Mish(num_filters, kernel2, strides=(2,2,2), weight_decay=weight_decay)(preconv1)

    #--------------------------------------------------------------------#
    #   然后建立一个大的残差边shortconv、这个大残差边绕过了很多的残差结构,shortconv.shape[None, 128, 128, 32, 64]
    #--------------------------------------------------------------------#
    shortconv = DarknetConv3D_BN_Mish(num_filters//2 if all_narrow else num_filters, kernel1, weight_decay=weight_decay)(preconv1)

    #----------------------------------------------------------------#
    #   第二个卷积mainconv.shape [None, 128, 128, 32, 64]
    #----------------------------------------------------------------#
    mainconv = DarknetConv3D_BN_Mish(num_filters//2 if all_narrow else num_filters, kernel1, weight_decay=weight_decay)(preconv1)
    #----------------------------------------------------------------#
    #   构建num_blocks个小残差块：进行循环，循环内部是残差结构(小残差块),这个残差是相加Add; [None, 128, 128, 32, 64]
    #----------------------------------------------------------------#    
    for i in range(num_blocks):
        # y = compose(
        #         DarknetConv2D_BN_Mish(num_filters//2, (1,1), weight_decay=weight_decay),
        #         DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3), weight_decay=weight_decay))(mainconv)
        # mainconv = Add()([mainconv,y])
        mainconv = resblock(mainconv, DarknetConv3D_BN_Mish, num_filters,kernel1,kernel2, all_narrow, weight_decay)
    #----------------------------------------------------------------#
    #   第三个卷积 [None, 128, 128, 32, 64]
    #----------------------------------------------------------------#        
    postconv = DarknetConv3D_BN_Mish(num_filters//2 if all_narrow else num_filters, kernel1, weight_decay=weight_decay)(mainconv)

    #----------------------------------------------------------------#
    #   将大残差边再堆叠回来,这个残差是concatenate [None, 128, 128, 32, 128]
    #----------------------------------------------------------------#
    route = Concatenate()([postconv, shortconv])

    # 最后对通道数进行整合
    return DarknetConv3D_BN_Mish(num_filters, kernel1, weight_decay=weight_decay)(route)


def build_resnet(inputs):
    # First Convolutional layer with L2 regularization
    x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu',
                      kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Residual blocks
    residual_block_count = 3
    for _ in range(residual_block_count):
        residual = x
        # First Convolutional layer of residual block with L2 regularization
        x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu',
                          kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Second Convolutional layer of residual block with L2 regularization
        x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu',
                          kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Add the residual connection
        x = Add()([x, residual])
        x = Activation('relu')(x)

    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)

    # Fully-connected layers with L2 regularization
    x = Dense(units=512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    return x

def build_conv3d_model(inputs):
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu')(inputs)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(0.3)(x)

    return x


#---------------------------------------------------#
#   CSPdarknet53 的主体部分 main function
#   输入为一张256x256x64的range-azimuth-doppler
#   输出为三个有效特征层
#---------------------------------------------------#

def darknet_body_3D(x, weight_decay=5e-4):
    #x = build_resnet(x)
    x = build_conv3d_model(x)
    return x