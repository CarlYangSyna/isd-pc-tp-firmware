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
 1251 McKay Drive
 San Jose, CA   95131
 (408) 454-5100

----------------------------------------------------------------- """

import os
import argparse
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import glob
import pickle
import numpy as np
import tensorflow as tf


def main(dataFolder, modelFolder):
    converter = tf.lite.TFLiteConverter.from_saved_model(modelFolder)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    
    tflite_model = converter.convert()

    open(os.path.join('best_model', 'ACM_converted_qat_int8_model.tflite'), "wb").write(tflite_model)
    print(f"TFLite model: model size = {len(tflite_model)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataFolder', help='Path of data folder')
    parser.add_argument('modelFolder', help='Path of best_model folder')

    args = parser.parse_args()

    main(args.dataFolder, args.modelFolder)