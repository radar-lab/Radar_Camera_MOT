import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers 
from tensorflow.keras.models import Model
from nets.common_CSPdarknet53 import DarknetConv2D, DarknetConv3D, darknet_body_3D, compose
from nets.CSPdarknet53 import darknet_body
from tensorflow.keras.layers import (Input, Lambda,LeakyReLU, Add, Concatenate,BatchNormalization,Dropout,Activation,
                                     Flatten,Dense,Conv2D, Conv3D, Layer, ZeroPadding2D, ZeroPadding3D,MaxPooling3D,GlobalAveragePooling3D)
from tensorflow.keras import regularizers


def create_conv_layers(inputs,stride1,stride2,kernels):
    """Creates convolutional layers. Output shape: [W-K+2P]/S+1; for padding=same: W/S"""
    input_shape = inputs.shape[1:]
    conv = layers.Conv2D(64, (3, 3), strides=stride1, 
                         padding='same', activation='relu', input_shape=input_shape)(inputs)
    conv = layers.Conv2D(kernels, (3, 3), strides=stride2, padding='same', activation='relu')(conv)
    return conv

def define_rad_model(radar_size, n_classes=2):
    # Make convolutional layers 
    radar_input = layers.Input(shape=radar_size,name='Radar_Frame')# name='Radar_Frame'
    #radar_conv = create_conv_layers(radar_input,stride1=(1,1),stride2=(1,1),kernels=32)
    radar_conv = create_conv_layers(radar_input,stride1=(1,1),stride2=(1,1),kernels=16)
    # Create a feature vector.
    radar_conv = layers.BatchNormalization()(radar_conv)
    radar_conv = layers.Dropout(0.5)(radar_conv)
    #conv = layers.Conv2D(64, 3, padding='same', activation='relu')(radar_conv)
    maxp=layers.MaxPooling2D()(radar_conv)
    batn=layers.BatchNormalization()(maxp)
    fv = layers.Flatten()(batn)
    # Create dense layers and operate on the feature vector.
    #dense1 = layers.Dense(128, activation='relu')(fv)
    dense = layers.Dropout(0.5)(fv)
    # dense = layers.Dense(64, activation='relu')(dense)
    # dense = layers.Dropout(0.5)(dense)
    # Classifier.
    #clsi = layers.Dense(units=n_classes, activation='softmax')(dense)
    #clsi = layers.Dense(units=1, activation='sigmoid')(dense)
    clsi = layers.Dense(units=128,  name='rad_feat')(dense)#activation='linear',
    clsi = layers.BatchNormalization()(clsi)
    # Create model and compile it.
    model = Model(inputs=[radar_input], outputs=[clsi], name='RadClsNet')
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #                loss='binary_crossentropy', metrics=['accuracy'])
    return model,radar_input   

def rad_feat_model(input_shape, channels=1, weight_decay=5e-4):
    input_shape_3D = input_shape.copy()
    input_shape_3D.append(channels)   #[w, h, d, channel] = [256x256x64x1]
    radar_input   = Input(input_shape_3D,name='Radar_RAD')
    #---------------------------------------------------#   
    #   32,32,256    (None, 32, 32, 8, 256) (None, 32, 32, 8, 128) (None, 32, 32, 8, 64)
    #   16,16,512    (None, 16, 16, 4, 512) (None, 16, 16, 4, 256) (None, 16, 16, 4, 128)
    #   8, 8,1024    (None, 8, 8, 2, 1024) (None, 8, 8, 2, 512)    (None, 8, 8, 2, 256)
    #---------------------------------------------------#
    feat = darknet_body_3D(radar_input, weight_decay=weight_decay)
    fv = layers.Flatten()(feat)
    embedding = layers.Dense(units=128,  name='rad_feat')(fv)
    model = Model(inputs=[radar_input], outputs=[embedding], name='RadNet')
    return model,radar_input 

def img_feat_model(input_shape, channels=1, weight_decay=5e-4):

    img_input   = Input(input_shape,name='Image')
    
    feat = darknet_body(img_input, weight_decay=weight_decay)
    fv = layers.Flatten()(feat)
    embedding = layers.Dense(units=128,  name='img_feat')(fv)
    model = Model(inputs=[img_input], outputs=[embedding], name='ImgNet')
    return model,img_input

def asso_net_rad(input_shape, num_classes = None, mode = "train"):
    #inputs = Input(shape=input_shape)
    #model,inputs = define_rad_model(radar_size=input_shape)
    model,inputs = rad_feat_model(input_shape)

    if mode == "train":
        logits          = Dense(num_classes)(model.output)
        softmax         = Activation("softmax", name = "Softmax")(logits)
        
        normalize       = Lambda(lambda  x: tf.keras.backend.l2_normalize(x, axis=1), name="Embedding")(model.output)
        combine_model   = Model(inputs, [softmax, normalize])
        return combine_model
    elif mode == "predict":
        x = Lambda(lambda  x: tf.keras.backend.l2_normalize(x, axis=1), name="Embedding")(model.output)
        model = Model(inputs,x)
        return model
    else:
        raise ValueError('Unsupported mode - `{}`, Use train, predict.'.format(mode))

def asso_net_img(input_shape, num_classes = None, mode = "train"):
    model,inputs = img_feat_model(input_shape)

    if mode == "train":
        logits          = Dense(num_classes)(model.output)
        softmax         = Activation("softmax", name = "Softmax")(logits)
        
        normalize       = Lambda(lambda  x: tf.keras.backend.l2_normalize(x, axis=1), name="Embedding")(model.output)
        combine_model   = Model(inputs, [softmax, normalize])
        return combine_model
    elif mode == "predict":
        x = Lambda(lambda  x: tf.keras.backend.l2_normalize(x, axis=1), name="Embedding")(model.output)
        model = Model(inputs,x)
        return model
    else:
        raise ValueError('Unsupported mode - `{}`, Use train, predict.'.format(mode))
        


        
# Define the img_classification_model function
def img_classification_model(input_shape, num_classes=5, channels=1, weight_decay=5e-4):
    img_input = layers.Input(input_shape, name='Image')
    
    feat = darknet_body(img_input, weight_decay=weight_decay)
    fv = layers.Flatten()(feat)
    embedding = layers.Dense(units=128, name='img_feat')(fv)
    output = layers.Dense(units=num_classes, activation='softmax', name='classification')(embedding)
    
    model = Model(inputs=[img_input], outputs=[output], name='ImgClassificationNet')
    return model        
        
# Define the rad_classification_model function
def rad_classification_model(input_shape, num_classes=5, channels=1, weight_decay=5e-4):
    input_shape_3D = input_shape.copy()
    input_shape_3D.append(channels)   #[w, h, d, channel] = [256x256x64x1]
    radar_input   = Input(input_shape_3D,name='Radar_RAD')
    feat = darknet_body_3D(radar_input, weight_decay=weight_decay)
    fv = layers.Flatten()(feat)
    embedding = layers.Dense(units=128,  name='rad_feat')(fv)
    output = layers.Dense(units=num_classes, activation='softmax', name='Radclassification')(embedding)
    
    model = Model(inputs=[radar_input], outputs=[output], name='RADClassificationNet')
    return model 

def build_complex_cnn(input_shape, num_classes=5, channels=1, weight_decay=1e-1, dropout_rate=0.5):
    # input_shape_3D = input_shape.copy()
    # input_shape_3D.append(channels)   #[w, h, d, channel] = [256x256x64x1]
    input_shape_3D = list(input_shape) + [channels]
    # Input layers for real and imaginary parts
    input_real = Input(input_shape_3D,name='Radar_real')
    input_imag = Input(input_shape_3D,name='Radar_imag')

    # Convolutional layers for real part
    x_real = Conv3D(16, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(input_real)
    x_real = MaxPooling3D(pool_size=(2, 2, 2))(x_real)
    x_real = Conv3D(8, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x_real)
    x_real = MaxPooling3D(pool_size=(2, 2, 2))(x_real)
    x_real = Conv3D(8, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x_real)
    x_real = MaxPooling3D(pool_size=(2, 2, 2))(x_real)

    # Convolutional layers for imaginary part
    x_imag = Conv3D(16, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(input_imag)
    x_imag = MaxPooling3D(pool_size=(2, 2, 2))(x_imag)
    x_imag = Conv3D(8, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x_imag)
    x_imag = MaxPooling3D(pool_size=(2, 2, 2))(x_imag)
    x_imag = Conv3D(8, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x_imag)
    x_imag = MaxPooling3D(pool_size=(2, 2, 2))(x_imag)

    # Flatten the outputs
    x_real = Flatten()(x_real)
    x_imag = Flatten()(x_imag)

    # Concatenate real and imaginary parts
    x = Concatenate()([x_real, x_imag])

    # Fully-connected layers
    x = Dense(units=128,  name='rad_feat', activation='relu')(x)
    x = Dropout(rate=dropout_rate)(x)
    output = Dense(num_classes, activation='softmax')(x)
    

    # # Create separate models for real and imaginary parts
    # model_real = Model(inputs=input_real, outputs=output)
    # model_imag = Model(inputs=input_imag, outputs=output)

    # Combine the models into a single model
    model = Model(inputs=[input_real, input_imag], outputs=output)

    return model
        