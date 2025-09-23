import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense, Input, Lambda,Concatenate,Subtract
from tensorflow.keras.models import Model
from nets.common_CSPdarknet53 import DarknetConv2D, DarknetConv3D, darknet_body_3D, compose
from nets.CSPdarknet53 import darknet_body, Mish

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

def rad_feat_model(model_path='./model_data/rad_classification_final_model.h5'):
    # Load the complete model
    model = tf.keras.models.load_model(model_path)
    # Remove the classification layers at the end
    base_model = model.layers[-2].output
    # Create a new model with only the layers before the classification layers
    new_model = tf.keras.Model(inputs=model.input, outputs=base_model, name='ClfNet')
    # Freeze the layers in the new model to keep their weights unchanged
    for layer in new_model.layers:
        layer.trainable = True  #False#True  #False#
    return new_model 

def img_feat_model(input_shape,model_path, num_classes=5):
    # # Load the weights model
    # model = img_classification_model(input_shape, num_classes)
    # model.load_weights(model_path, by_name=True, skip_mismatch=True)#https://keras.io/api/models/model_saving_apis/
    
    # Load the complete model
    model = tf.keras.models.load_model(model_path, custom_objects={'Mish': Mish})
    # Remove the classification layers at the end
    base_model = model.layers[-2].output
    # Create a new model with only the layers before the classification layers
    new_model = tf.keras.Model(inputs=model.input, outputs=base_model, name='ClfNet')
    # Freeze the layers in the new model to keep their weights unchanged
    for layer in new_model.layers:
        layer.trainable = True  #False #True  #False#
    return new_model



def asso_net_rad(model_path, num_classes = None, mode = "train"):
    model = rad_feat_model(model_path)

    embedding       = Dense(128, name='rad_feat2')(model.output)
    normalize       = Lambda(lambda  x: tf.keras.backend.l2_normalize(x, axis=1), name="Embedding")(embedding)
    model = Model(model.input,normalize)
    return model



 

def asso_net_img(input_shape,model_path, num_classes = None, mode = "train"):
    new_model = img_feat_model(input_shape,model_path, num_classes)

    embedding       = Dense(128, name='img_feat2')(new_model.output)
    normalize       = Lambda(lambda  x: tf.keras.backend.l2_normalize(x, axis=1), name="Embedding")(embedding)
    pred_model   = Model(new_model.input, normalize)
    return pred_model

        

def common_feat_model(input_shape_img,img_model_path, rad_model_path, img_input_shape, rad_input_shape,mode='classifi'):    
    # Extract features from img input
    img_model = img_feat_model(input_shape_img,img_model_path)
    img_feat  = Dense(128)(img_model.output)
    img_feat  = Dense(128)(img_feat)

    # Extract features from rad input
    rad_model = rad_feat_model(rad_model_path)
    rad_feat  = Dense(128)(rad_model.output)
    rad_feat  = Dense(128)(rad_feat)
    
    if mode=='classifi':
        # Concatenate img and rad features
        x = Concatenate(name='concatenated_features')([img_feat, rad_feat])
        
        # Add more dense layers
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)

        # Output layer for binary classification
        output = Dense(1, activation='sigmoid')(x)
    elif mode=='siamese':
        ### Siamese networks -- Instead of using the Concatenate layer, while to subtract them as a single layer. 
        # # Subtract img_feat and rad_feat  and take the absolute value of the subtraction result
        subtracted = Subtract()([img_feat, rad_feat])
        x = Lambda(lambda x: K.abs(x))(subtracted) #abs_subtracted
        
        # Add more dense layers
        x = Dense(64, activation='relu')(x)
        #x = Dense(32, activation='relu')(x)

        # Output layer for binary classification
        output = Dense(1, activation='sigmoid')(x)
    elif mode=='contrastive_loss':
        ### Contrastive Loss -- Contrastive Loss focuses on pairs of samples and explicitly minimizes the distance between similar pairs and maximizes the distance between dissimilar pairs. 
        # Triplet Loss, on the other hand, uses triplets of samples and aims to minimize the distance between an anchor sample and a positive sample (same class) while maximizing the distance between the anchor sample and a negative sample (different class).
        # Compute Euclidean distance between features
        def euclidean_distance(vects):
            x, y = vects
            sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
            return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
        distance = Lambda(euclidean_distance)([img_feat, rad_feat])
    
        # Output layer for contrastive loss
        output = Dense(1, activation='sigmoid')(distance)
        
        # # Contrastive loss function
        # def contrastive_loss(y_true, distance, margin=1.0):
        #     loss = K.mean((1 - y_true) * K.square(distance) + y_true * K.square(K.maximum(margin - distance, 0)))
        #     return loss
        
        

        

    # Create the combined binary_classification model
    model = Model(inputs=[rad_model.input, img_model.input], outputs=output)
    
    # if mode=='contrastive Loss':
    #     model.compile(loss=lambda y_true, y_pred: contrastive_loss(y_true, y_pred, margin=1.0), optimizer='adam')

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

# Define the img_classification_model function
def img_classification_model(input_shape, num_classes=5, channels=1, weight_decay=5e-4):
    img_input = layers.Input(input_shape, name='Image')
    
    feat = darknet_body(img_input, weight_decay=weight_decay)
    fv = layers.Flatten(name='flat')(feat)
    embedding = layers.Dense(units=128, name='img_feat')(fv)
    output = layers.Dense(units=num_classes, activation='softmax', name='classification')(embedding)
    
    model = Model(inputs=[img_input], outputs=[output], name='ImgClassificationNet')
    return model       