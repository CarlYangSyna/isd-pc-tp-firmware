# -*- coding: utf-8 -*-
""" -----------------------------------------------------------------

                      COMPANY CONFIDENTIAL
                       INTERNAL USE ONLY

 Copyright (C) 1997 - 2015  Synaptics Incorporated.  All right reserved.

 This document contains information that is proprietary to Synaptics
 Incorporated. The holder of this document shall treat all information
 contained herein as confidential, shall use the information only for its
 intended purpose, and shall protect the information in whole or part from
 duplication, disclosure to any other party, or dissemination in any media
 without the written permission of Synaptics Incorporated.

 Synaptics Incorporated
 1251 McKay Drive
 San Jose, CA   95131
 (408) 454-5100

----------------------------------------------------------------- """

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from losses import pairwise_cross_entropy_alpha
from tensorflow.keras.optimizers import Adam

def get_dense_model_image():
    model = Sequential()
    model.add(tf.keras.Input(shape = (13, 13, 1)))
    model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(layers.Dense(48, activation = 'relu'))
    model.add(layers.Dense(24, activation = 'relu'))
    model.add(layers.Dense(16, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
        
    model.compile(loss = "binary_crossentropy",
                  optimizer = Adam(learning_rate = 1e-3), 
                  metrics = ['accuracy'])
    return model

def get_model(mode):
    alpha = np.array([[1, 1, 5], [1, 1, 5], [50, 50, 1]])
    if 'image' in mode:
        if 'dense' in mode:
            model = get_dense_model_image()
        else:
            print('CNN Model setting is incorrect, select from image_big, image_small, image_dense')
            exit(-1)
    return model