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

from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_generator):
        self.validation_generator = validation_generator
        super(ConfusionMatrixCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        # Get the true labels and predictions for the validation set
        val_pred = self.model.predict(self.validation_generator) # , batch_size=32)
        val_labels = self.validation_generator.annotations
        #val_data_half_size = int(val_pred.shape[0] / 2)
        # The first half of the validation consists of center touches and the second half is edge/corner touches
        val_data_half_size = int(len(val_labels) / 2)
        # Calculate the confusion matrix
        cm_1 = confusion_matrix(val_labels[:val_data_half_size].argmax(axis=1),
                                val_pred[:val_data_half_size].argmax(axis=1),
                                labels=np.array([0, 1, 2]), normalize='true')
        cm_2 = confusion_matrix(val_labels[val_data_half_size:2*val_data_half_size].argmax(axis=1),
                                val_pred[val_data_half_size:2*val_data_half_size].argmax(axis=1),
                                labels=np.array([0, 1, 2]), normalize='true')
        # Log the confusion matrix to TensorBoard
        print(f'Confusion matrix center: \n{cm_1}')
        print(f'Confusion matrix edge: \n{cm_2}')
