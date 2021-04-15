import numpy as np
import openpyxl as xl
import pandas as pd
import os
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, PReLU


def batch_create_records(filenames, v):
    a = np.load(filenames)
    print(tf.shape(a))
    a = np.swapaxes(a, 1, 3)
    a = np.swapaxes(a, 1, 2)
    print(tf.shape(a))

    inputs = np.zeros((np.size(a, 0) - 1, np.size(a, 1), np.size(a, 2), np.size(a, 3)))
    outputs = np.zeros((np.size(a, 0) - 1, np.size(a, 1), np.size(a, 2), np.size(a, 3)))

    for i in range(np.size(a, 0) - 25):
        b = np.sqrt(np.square(a[i, :, :, 1]) + np.square(a[i, :, :, 2]))
        max_v = np.amax(b)
        inputs[i, :, :, 0] = a[i, :, :, 0]
        inputs[i, :, :, 1] = a[i, :, :, 1] / max_v
        inputs[i, :, :, 2] = a[i, :, :, 2] / max_v
        outputs[i, :, :, 0] = a[i + 25, :, :, 0]
        outputs[i, :, :, 1] = a[i + 25, :, :, 1] / max_v
        outputs[i, :, :, 2] = a[i + 25, :, :, 2] / max_v

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def image_pair_example(image1, image2):
        rows = image1.shape[0]  # 66
        cols = image1.shape[1]  # 256
        depth = image1.shape[2]  # 3
        image1_raw = image1.tobytes()
        image2_raw = image2.tobytes()
        feature = {
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'image1_raw': _bytes_feature(image1_raw),
            'image2_raw': _bytes_feature(image2_raw)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    n_samples = inputs.shape[0]
    output_file = "./Square/TIME10/Val/TFRs/tfr" + str(v) + ".tfrecord"
    data_type = inputs.dtype
    print(f"Data type: {data_type}")
    print(type(data_type))
    if True:
        print("false")
        with tf.io.TFRecordWriter(output_file) as writer:
            for i in range(n_samples):
                tf_example = image_pair_example(inputs[i], outputs[i])
                writer.write(tf_example.SerializeToString())
        writer.close()


# -----ONE SHOT CODE
# batch_create_records("./NPYs/npy0.45.npy","0.45")
# exit()


# count = 0.1
# upper = 1
# increment = 0.025
# velocity_vector = count
# while count < upper:
#     count = count + increment
#     velocity_vector = np.append(velocity_vector, count)
#
# print(velocity_vector)

velocity_vector = (0.1125, 0.3125, 0.5125)
# -----LOOP CODE
j = 0
for file_name in os.listdir("./Square/Val/NPYs/"):
    batch_create_records("./Square/Val/NPYs/" + file_name, velocity_vector[j])
    j = j + 1
