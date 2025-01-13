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
import sys
import argparse

if sys.platform == 'linux':
  sys.path.insert(0, os.path.join(sys.path[0], '..', '..', 'packages'))

import numpy as np
import tensorflow as tf

def main():
    wb = []
    scale_bias = []
    scale_filter = []

    pre_scale = 0
    pre_zero_point = 0
    rom_address = 0x10000
    
    config = os.path.join('config')

    model = tf.lite.Interpreter(model_path = os.path.join('ACM_converted_full_integer_model.tflite'))
    model.allocate_tensors()
    tensor_details = model.get_tensor_details()

    # =========================== generate nn_config.h file ===========================
    f = open(os.path.join(config, 'nn_config.h'), "w")
    for i, detail in enumerate(tensor_details):
        layer_name = detail['name']
        layer_shape = detail['shape']
        quantization = detail['quantization']
        quantization_scale = detail['quantization_parameters']['scales']
        quantization_zero_point = detail['quantization_parameters']['zero_points']

        if quantization_scale.size == 0:
            quantization_scale = ""
        else:
            quantization_scale = quantization_scale[0]

        if quantization_zero_point.size == 0:
            quantization_zero_point = ""
        else:
            quantization_zero_point = quantization_zero_point[0]

        if "quantize" in layer_name:
            f.write(
                f"#define QUANTIZE_SCALE {1 / quantization_scale}f\n")
            f.write(
                f"#define QUANTIZE_ZERO_POINT {quantization_zero_point}\n\n")
        elif "Reshape" in layer_name:
            pre_scale, pre_zero_point = quantization
            f.write(f"#define flatten_SIZE {layer_shape[1]}\n")
            f.write(f"#define flatten_output_scale {pre_scale}f\n")
            f.write(f"#define flatten_output_zero_point {pre_zero_point}\n\n")
        elif "BiasAdd" in layer_name:
            if "ReadVariableOp" in layer_name:
                scale_bias.append(quantization)
                wb.append(model.tensor(detail['index'])())
            else:
                lname = layer_name.split('/')[1]
                if lname == "dense":
                    lname = "dense_0"

                shift = 0
                scale, zero_point = quantization
                bias_scale, bias_zero_point = scale_bias.pop()
                filter_scale, filter_zero_point = scale_filter.pop()

                output_multiplier = bias_scale / scale
                output_multiplier_shifted = output_multiplier

                while output_multiplier_shifted < 1:
                    shift -= 1
                    output_multiplier_shifted *= 2
                while output_multiplier_shifted >= 1:
                    shift += 1
                    output_multiplier_shifted /= 2
                output_multiplier_shifted *= (2 ** 31)

                f.write(f"#define {lname}_SIZE {layer_shape[1]}\n")
                f.write(f"#define {lname}_bias_scale {bias_scale}f\n")
                f.write(f"#define {lname}_bias_zero_point {bias_zero_point}\n")
                f.write(f"#define {lname}_filter_scale {filter_scale}f\n")
                f.write(f"#define {lname}_filter_zero_point {filter_zero_point}\n")

                f.write(f"#define {lname}_output_scale {scale}f\n")
                f.write(f"#define {lname}_output_zero_point {zero_point}\n\n")

                f.write(f"#define {lname}_activation_min {-128 - zero_point}\n")
                f.write(f"#define {lname}_activation_max {127 - zero_point}\n")
                f.write(f"#define {lname}_output_shift {shift:d}\n")
                f.write(f"#define {lname}_output_multiplier {output_multiplier_shifted:.0f}\n\n")
                pre_scale = scale
        elif "MatMul" in layer_name:
            scale_filter.append(quantization)
            wb.append(model.tensor(detail['index'])())
        elif layer_name == "StatefulPartitionedCall:01":
            f.write(f"#define DEQUANTIZE_SCALE {quantization_scale}f\n")
            f.write(f"#define DEQUANTIZE_ZERO_POINT {quantization_zero_point}\n\n")

    f.write(f"#define MODEL_ROM_PTR {rom_address:d}\n\n")
    f.close()

    # =========================== generate acm_nn_model.h file ===========================
    f = open(os.path.join(config, 'acm_nn_model.h'), "w")
    for i, detail in enumerate(tensor_details):
        layer_name = detail['name']

        if "BiasAdd" in layer_name:
            if "ReadVariableOp" in layer_name:
                WriteWeightAndBias(model, detail, f, True)
        elif "MatMul" in layer_name:
            WriteWeightAndBias(model, detail, f, False)
    f.close()

    # =========================== generate acm_nn_model.asm file ===========================
    f = open(os.path.join(config, 'acm_nn_model.asm'), "w")
    # f.write("org 0x10000\n")
    f.write(f"org 0x{rom_address:05X}\n")
    while len(wb):
        data = wb.pop()
        if data.ndim < 2:
            for i in range(len(data)):
                data_hex = f"word 0x{data[i] & 65535:04X}\n"
                f.write(data_hex)
        else:
            for i in range(data.shape[0]):
                decByteToHexWord(data[i], f)
    f.close()

def WriteWeightAndBias(model, detail, f, expand):
    layer_name = detail['name']
    lname = layer_name.split('/')[1]
    if lname == "dense":
        lname = "dense_0"
    data = model.tensor(detail['index'])()

    if expand:
        data = np.expand_dims(data, axis=0)
        f.write(f"const int16 {lname}_bias[] = {{\n")
    else:
        f.write(f"const int16 {lname}_weight[] = {{\n")

    l = data.shape[0]
    for j in range(l):
        s = f"{list(data[j])}\n"
        s = s.replace('[', '')
        if j == l - 1:
            s = s.replace(']', "};")
        else:
            s = s.replace(']', ',')
        f.write(s)

def decByteToHexWord(data, f):
    for i in range(0, data.shape[0], 2):
        # data_hex = data[i]
        if i + 1 >= data.shape[0]:
            data_hex = f"word 0x{data[i] & 65535:04X}\n"
        else:
            data_hex = f"word 0x{data[i + 1] & 255:02X}{data[i] & 255:02X}\n"
        f.write(data_hex)


if __name__ == '__main__':
    main()
