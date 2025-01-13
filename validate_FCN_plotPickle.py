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
import cv2
import glob
import pickle
import numpy as np
import tensorflow as tf

# Set the flag to draw the image
draw_image = True

def main():    
    # Load the pickle file
    pickle_path = os.path.join("data") 
    pickle_number = input("Enter a number between 1 and 30 to validate the selected pickle file in data folder: ")
    pickle_file = f"S{pickle_number}_clumpBasedOnBox_13x13.pickle"
    pickle_path = os.path.join(pickle_path, pickle_file)
    with open(pickle_path, 'rb') as f:
        xs, ys, ps, ts = pickle.load(f)
    # Convert xs to float32
    xs = xs.astype(np.float32)

    # Load the model
    model_path = os.path.join("best_model")
    model = tf.keras.models.load_model(model_path)
    scores = model.predict(xs)

    count_error = 0
    count_total = len(xs)
    min_val = 32767
    max_val = -32768
    for i, (x, y) in enumerate(zip(xs, ys)):
        im = np.repeat(x * 30, 20, axis = 0)
        im = np.repeat(im, 20, axis = 1)     
        im = np.dstack((im,) * 3)    
        # Calculate the minimum and maximum pixel values in the image
        min_val = min(min_val, np.min(im))
        max_val = max(max_val, np.max(im))
        # Adjust the contrast of the image based on the global pixel value range
        im = cv2.convertScaleAbs(im, alpha=(255.0 / (max_val - min_val)), beta=(-min_val * 255.0 / (max_val - min_val)))
        
        str_type = "finger"
        font_color = (-32767, 32767, -32767)
        if scores[i] < 0.5:
            str_type = "palm"
            if y == 1:
                count_error += 1
                font_color = (-32767, -32767, 32767)
        else:
            if y == 0:
                count_error += 1
                font_color = (-32767, -32767, 32767)
            
        if draw_image:
            cv2.putText(im, f"acc = {((i+1) - count_error) * 100 / (i+1):.2f}%", 
                        (0, 15), cv2.FONT_HERSHEY_PLAIN, 1, (-32767, 32767, -32767), 2, cv2.LINE_AA)
            cv2.putText(im, f"error rate = {count_error} / {i+1} ({count_error / (i+1) * 100:.2f}%)", (0, 30), cv2.FONT_HERSHEY_PLAIN, 1, (-32767, -32767, 32767), 2, cv2.LINE_AA)
            cv2.putText(im, str_type, (0, 45), cv2.FONT_HERSHEY_PLAIN, 1, font_color, 2, cv2.LINE_AA)
            
            cv2.imshow("clump", im)
            cv2.waitKey(1)

    if draw_image:         
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Calculate accuracy and error rate
    accuracy = ((len(xs) - count_error) * 100) / len(xs)
    error_rate = (count_error / len(xs)) * 100

    print(f"Accuracy: {len(xs) - count_error} / {len(xs)} ({accuracy:.2f}%)")
    print(f"Error Rate: {count_error} / {len(xs)} ({error_rate:.2f}%)")
    
if __name__ == '__main__':
    main()
