import numpy as np
import openpyxl as xl
import pandas as pd
import os
import tensorflow as tf
import pickle
import glob
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, PReLU
import time


files_used_to_train = 1250
a = np.load("./NPYs/npy0.1.npy")


print(tf.shape(a))
a = np.swapaxes(a, 1, 3)
a = np.swapaxes(a, 1, 2)
print(tf.shape(a))

(inputs, outputs) = a[0:files_used_to_train-2], a[1:files_used_to_train-1]

files = tf.data.Dataset.list_files(glob.glob("./Square/TIME10/TFRs"+"*/*"))
dataset = tf.data.TFRecordDataset(files)


# parsing and decoding functions
def parse_record(record):
    name_to_features = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'depth': tf.io.FixedLenFeature([], tf.int64),
      'image1_raw': tf.io.FixedLenFeature([], tf.string),
      'image2_raw': tf.io.FixedLenFeature([], tf.string),
      }
    return tf.io.parse_single_example(record, name_to_features)


def decode_record(record):
    precord = parse_record(record)
    image1 = tf.io.decode_raw(
        precord['image1_raw'], out_type=inputs.dtype, little_endian=True, fixed_length=None, name=None
    )
    image2 = tf.io.decode_raw(
        precord['image2_raw'], out_type=inputs.dtype, little_endian=True, fixed_length=None, name=None
    )
    height = precord['height']
    width = precord['width']
    depth = precord['depth']
    image1 = tf.reshape(image1, (height, width, depth))
    image2 = tf.reshape(image2, (height, width, depth))
    return image1, image2


num_threads = 8
dataset = dataset.map(decode_record, num_parallel_calls=num_threads)
dataset = dataset.shuffle(10000)
dataset = dataset.batch(5)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


valFiles = tf.data.Dataset.list_files(glob.glob("./Square/TFRs" + "*/*"))
valdataset = tf.data.TFRecordDataset(valFiles)
valdataset = valdataset.map(decode_record, num_parallel_calls=num_threads)
valdataset = valdataset.shuffle(10000)
valdataset = valdataset.batch(5)
valdataset = valdataset.prefetch(tf.data.experimental.AUTOTUNE)

shape1 = (33, 32, 4)
shape2 = (11, 8, 32)
shape3 = (1, 1, 256)
model = tf.keras.models.Sequential()

# Layer 1
model.add(tf.keras.layers.Conv2D(
  4, (2, 8),
  strides=(2, 8),
  activation=PReLU(),
  input_shape=(66, 256, 3)
  ))

# Layer 2
model.add(tf.keras.layers.Conv2D(
  32, (3, 4),
  strides=(3, 4),
  activation='tanh',
  input_shape=shape1
  ))

# Layer 3
model.add(tf.keras.layers.Conv2D(
  256, (11, 8),
  strides=(11, 8),
  activation='tanh',
  input_shape=shape2
  ))

# Layer 4
model.add(tf.keras.layers.Conv2DTranspose(
  32, (11, 8),
  strides=(11, 8),
  activation='tanh',
  input_shape=shape3
  ))

# Layer 5
model.add(tf.keras.layers.Conv2DTranspose(
  4, (3, 4),
  strides=(3, 4),
  activation='tanh',
  input_shape=shape2
  ))

# Layer 6
model.add(tf.keras.layers.Conv2DTranspose(
  3, (2, 8),
  strides=(2, 8),
  activation=PReLU(),
  input_shape=shape1
  ))

# Compile and fit

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['accuracy'])
model.fit(dataset, epochs=200, validation_data=valdataset)
model.save('FINAL_MODEL.model')
