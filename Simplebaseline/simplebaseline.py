import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def _make_deconv_layer(num_deconv_layers):
    seq_model = tf.keras.models.Sequential()
    for i in range(num_deconv_layers):
        seq_model.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), padding='same'))
        seq_model.add(tf.keras.layers.BatchNormalization())
        seq_model.add(tf.keras.layers.ReLU())
    return seq_model

def Simplebaseline(input_shape=(256, 256, 3), num_heatmap=16):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet')(inputs)
    x = _make_deconv_layer(3)(x)
    out = tf.keras.layers.Conv2D(num_heatmap, kernel_size=(1,1), padding='same')(x)
    model = tf.keras.Model(inputs, out, name='simple_baseline')
    return model