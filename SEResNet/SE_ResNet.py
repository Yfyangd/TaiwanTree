from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras import layers, Input, models
import tensorflow as tf

def create_se_resnet(classes, img_input, include_top, initial_conv_filters, filters,
                      depth, width, weight_decay, pooling):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)

    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    for i in range(N[0]): # range(2): 0~1
            x = resnet_block(x, filters[0], width)

    for k in range(1, len(N)): # range(1,4): 1~3
            x = resnet_block(x, filters[k], width, strides=(2, 2))

    for i in range(N[k] - 1): # range(1): 0
            x = resnet_block(x, filters[k], width)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)


    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  activation='sigmoid')(x)
    
    return x