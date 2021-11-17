import tensorflow as tf
import numpy as np
from keras.layers import Input, Add, Dense, concatenate, Activation, Conv2D, GlobalAveragePooling2D, LeakyReLU, Dropout, Permute, Flatten
from keras.models import Sequential
from keras.activations import sigmoid
from keras.preprocessing import image
from keras.models import Model, load_model
from softattention import SoftAttention
from keras.applications.vgg16 import VGG16
from vgg16_conv3 import VGG


def inception_block(input,filter):
    conv1 = Conv2D(filter, kernel_size=(3,3), padding='same')(input)
    conv2 = Conv2D(filter, kernel_size=(3,3), padding='same')(input)
    conv2_reduce = Conv2D(filter, kernel_size=(1,1), padding='same')(conv2)

    output_layer = concatenate([conv1, conv2_reduce]) 
    return output_layer

def instance_norm(input):
    with tf.compat.v1.variable_scope("instancenorm"):
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.compat.v1.get_variable("offset", [1, 1, 1, channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.compat.v1.get_variable("scale", [1, 1, 1, channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        variance_epsilon = 1e-5
        
        normalized = (((input - mean) / tf.sqrt(variance + variance_epsilon)) * scale) + offset
        return normalized

def HA_conv(input,filter,s,d):

    conv1 = Conv2D(filter, kernel_size=(3,3),strides = (s,s),padding='same',dilation_rate=(d,d), activation='sigmoid')(input)

    IN_input = instance_norm(input)
    incep_out = inception_block(input)
    #ReLU
    conv2 = Conv2D(filter,kernel_size=(3,3),strides = (s,s),padding='same',dilation_rate=(d,d),activation=LeakyReLU())(IN_input)

   # Element wise multi 
    multi_conv = tf.math.multiply(conv1, conv2)

    # Element wise add
    output = tf.math.add(multi_conv, incep_out)
    return output

def ST_conv(input,filter):
    conv_layer = Conv2D(filter,kernel_size=(3,3),padding='same',activation=LeakyReLU())(input)

    return conv_layer

def Two_stream(input):
    input_layer = ST_conv(input,64)

    layerH1 = HA_conv(input_layer,64,2,1)
    layerH2 = HA_conv(layerH1,64,2,1)
    layerH3 = HA_conv(layerH2,64,2,1)

    layerH4 = HA_conv(layerH3,128,2,1) #skip 1
    layerH5 = HA_conv(layerH4,128,2,1)
    layerH6 = HA_conv(layerH5,128,2,1) #skip 2

    layerH7 = HA_conv(layerH6,128,2,2)
    LayerH7 = Add()([layerH7, layerH6])

    layerH8 = ST_conv(layerH7,128)
    layerH8 = Add()([layerH8, layerH4])
    layerH9 = ST_conv(layerH8,64)  # out of HA


    layerS1 = ST_conv(input_layer,64)
    layerS2 = ST_conv(layerS1,64)
    layerS3 = ST_conv(layerS2,64)

    layerS4 = ST_conv(layerS3,128) #skip 1
    layerS5 = ST_conv(layerS4,128)
    layerS6 = ST_conv(layerS5,128) #skip 2

    layerS7 = ST_conv(layerS6,128)
    layerS7 = Add()([layerS7, layerS6])

    layerS8 = ST_conv(layerS7,128)
    layerS8 = Add()([layerS8, layerS4])

    layerS9 = ST_conv(layerS8,64) # out of ST

    return layerH9, layerS9



def AFS(inputs):
    gap = GlobalAveragePooling2D()(inputs)
    layer = Dense(128)(gap)
    layer = Dense(16)(layer)
    layer = Dense(128, activation='sigmoid')(layer)

    drate = 0.2
    a = SoftAttention(128,dropout_rate=drate)(layer)
    soft_out = tf.linalg.matmul(a,layer)

    outputs = tf.math.multiply(soft_out, inputs)
    return outputs
     

def FU_branch(HA, ST):
    inputs =  concatenate([HA,ST])

    layer1 = AFS(inputs)

    layer2 = ST_conv(layer1,128)
    layer3 = ST_conv(layer2, 128)
    layer4 = ST_conv(layer3,64)
    layer5 = ST_conv(layer4,64)
    
    return layer5

img_width, img_height = 96, 96   # images.shape

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))



