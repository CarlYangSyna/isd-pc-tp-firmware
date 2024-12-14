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

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import glob
import pickle
import numpy as np
import tensorflow as tf

def main():    
    x_trn = []    
    pickle_files = os.path.join("data", "*.pickle") 
    fnames = glob.glob(pickle_files)

    # Load data from pickle files as representative_dataset for full-integer quantization
    for fname in fnames:
        f = open(fname, 'rb')
        x, y, a, t = pickle.load(f)
        f.close()
        x_trn.extend(x)
    
    def representative_dataset_generator():
        count = 0
        for value in x_trn:
            value = value.astype(np.float32)

            value = np.expand_dims(value, axis = 0)
            value = np.expand_dims(value, axis = -1)
            yield [value]
            count += 1
            if count > 10000:
                break

    # Convert the model to TFLite model using full-integer quantization
    converter = tf.lite.TFLiteConverter.from_saved_model("best_model")     
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_generator
    tflite_model = converter.convert()   
    
    open("ACM_converted_full_integer_model.tflite", "wb").write(tflite_model)
    print(f"TFLite model: model size = {len(tflite_model)}")    

if __name__ == '__main__':
    main()