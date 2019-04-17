# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:07:26 2019

@author: Beichen
"""

# -*- coding: utf-8 -*-

import numpy as np
#import tensorflow as tf
from keras.models import Model
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract



def DnCNN():
    
    inpt = Input(shape=(None,None,1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model



# =============================================================================
# model = DnCNN()
# from keras.utils import plot_model
# plot_model(model, show_shapes=True, to_file='DnCNN_model.png')
# =============================================================================











