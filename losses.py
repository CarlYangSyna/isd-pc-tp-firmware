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

import tensorflow as tf
import numpy as np
import math


def pairwise_cross_entropy_alpha(alpha):
    def pairwise_cross_entropy(y_true, y_pred):
        alpha_repeat_batch = np.repeat(alpha[np.newaxis, :, :], 32, axis=0)
        alpha_repeat_batch_tensor = tf.constant(alpha_repeat_batch, dtype=np.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        clipped_y_pred = tf.clip_by_value(y_pred,
                                          clip_value_min=tf.constant(epsilon, dtype=tf.float32),
                                          clip_value_max=tf.constant(1, dtype=tf.float32))
        weights = tf.matmul(alpha_repeat_batch_tensor, tf.expand_dims(clipped_y_pred, -1))
        loss = tf.expand_dims(tf.cast(y_true, dtype=tf.float32), axis=-1) * tf.math.log(tf.expand_dims(clipped_y_pred, axis=-1)) \
               * weights/tf.math.log(tf.constant(2, dtype=tf.float32))
        loss = -tf.math.reduce_mean(loss)
        return loss

    return pairwise_cross_entropy


class CosineLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, epochs, steps_per_epoch):
        super(CosineLearningRate, self).__init__()
        self.initial_lr = initial_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.max_lr = initial_lr
        self.min_lr = initial_lr / 10

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.max_lr)

    def on_epoch_end(self, epoch, logs=None):
        # Optionally, you can reset the learning rate at the end of each epoch
        # to its initial value or some other value
        # self.update_lr(0)
        cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / self.epochs))
        print(f'New learning rate is {cosine_lr}')
        tf.keras.backend.set_value(self.model.optimizer.lr, cosine_lr)