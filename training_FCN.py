# -*- coding: utf-8 -*-
""" -----------------------------------------------------------------

                      COMPANY CONFIDENTIAL
                       INTERNAL USE ONLY

 Copyright (C) 1997 - 2025  Synaptics Incorporated.  All right reserved.

 This document contains information that is proprietary to Synaptics
 Incorporated. The holder of this document shall treat all information
 contained herein as confidential, shall use the information only for its
 intended purpose, and shall protect the information in whole or part from
 duplication, disclosure to any other party, or dissemination in any media
 without the written permission of Synaptics Incorporated.

 Synaptics Incorporated
 1109 McKay Drive
 San Jose, CA   95131
 (408) 454-5100

----------------------------------------------------------------- """

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import glob
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models import get_model
from losses import CosineLearningRate

from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train(mode = 'image', enable_edge_processing = False):        
    train_datagen = ImageDataGenerator(fill_mode = 'nearest', # set mode for filling points outside the input boundaries
                                       horizontal_flip = True)
    valid_datagen = ImageDataGenerator()    
    
    x_trn = []
    y_trn = []    
    pickle_files = os.path.join("data", "*.pickle") 
    fnames = glob.glob(pickle_files)


    for fname in fnames:
        f = open(fname, 'rb')
        x, y, a, t = pickle.load(f)
        f.close()
        
        x = np.expand_dims(x, axis = -1)
        x_trn.extend(x)
        y_trn.extend(y)
            
    ind = np.asarray(range(len(x_trn)))
    np.random.shuffle(ind)    
    
    x_trn = [x_trn[i] for i in ind]
    y_trn = [y_trn[i] for i in ind]
    
    val_stride = int(100 / 20)
    y_val = np.asarray(y_trn[:: val_stride], dtype = np.int8)
    x_val = np.asarray(x_trn[:: val_stride], dtype = np.int16)
    
    del x_trn[:: val_stride]
    del y_trn[:: val_stride]
    
    y_trn = np.asarray(y_trn, dtype = np.int8)
    x_trn = np.asarray(x_trn, dtype = np.int16)
    
    train_generator = train_datagen.flow(x_trn, y_trn, batch_size = 32)
    valid_generator = valid_datagen.flow(x_val, y_val, batch_size = 32)
    
    model = get_model(mode)    
    model.summary()    
    
    adam = optimizers.Adam(learning_rate = 1e-3)
    
    model.compile(loss = "binary_crossentropy",
                  optimizer = adam,
                  metrics = ['accuracy'])
    
    cb_lr = CosineLearningRate(initial_lr = 0.001, 
                               epochs = 50, 
                               steps_per_epoch = len(train_generator))    
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                filepath = "best_model",
                                save_best_only = True, monitor = 'val_loss',
                                mode = 'min', verbose = 1)   
    history = model.fit(train_generator,                        
                        epochs = 50,
                        batch_size = 32, 
                        verbose = 1,
                        validation_data = valid_generator,
                        callbacks = [cb_lr, cb_checkpoint], 
                        class_weight = {0: 1., 1: 1., 2: 1.})
    
    print(history.history.keys())    
    plt.title("Training and validation accuracy") 
    plt.xlabel("Epoch") 
    plt.ylabel("ACC") 
    plt.plot(history.history["accuracy"], label = "Training ACC") 
    plt.plot(history.history["val_accuracy"], label = "Validation ACC") 
    plt.legend()
    
    plt.figure()
    plt.title("Training and validation loss") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss") 
    plt.plot(history.history["loss"], label = 'Training loss') 
    plt.plot(history.history["val_loss"], label = 'Validation loss') 
    plt.legend()
    plt.show()

def main():
    mode = 'image_dense'
    enable_edge_processing = True
    train(mode, enable_edge_processing)

if __name__ == '__main__':
    main()
