# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : tf 2.0 CNN
#
#  DESCRIPTION   : none
#
#  RESTRICTIONS  : none
#
#  REQUIRES      : none
#
# ---------------------------------------------------------------------------------
# Copyright 2019 Pay Giesselmann, Max Planck Institute for Molecular Genetics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Pay Giesselmann
# ---------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
from tf_util import gelu, LayerNormalization




def opening1d(input, d=3):
    se = tf.reshape(tf.ones(3), (1, 3, 1))
    inner = tf.expand_dims(input, axis=-3)
    strides = [1, 1, 1, 1]
    dilations = [1, 1, 1, 1]
    inner = tf.nn.erosion2d(inner, se, strides=strides, padding='SAME', data_format='NHWC', dilations=dilations)
    inner = tf.nn.dilation2d(inner, se, strides=strides, padding='SAME', data_format='NHWC', dilations=dilations)
    inner = tf.squeeze(inner, axis=-3)
    return inner




def closing1d(input, d=3):
    se = tf.reshape(tf.ones(3), (1, 3, 1))
    inner = tf.expand_dims(input, axis=-3)
    strides = [1, 1, 1, 1]
    dilations = [1, 1, 1, 1]
    inner = tf.nn.dilation2d(inner, se, strides=strides, padding='SAME', data_format='NHWC', dilations=dilations)
    inner = tf.nn.erosion2d(inner, se, strides=strides, padding='SAME', data_format='NHWC', dilations=dilations)
    inner = tf.squeeze(inner, axis=-3)
    return inner




def smooth1d(input, d=3):
    return closing1d(opening1d(input, d=d), d=d)




def antirectifier(x):
    x -= tf.math.reduce_mean(x, axis=1, keepdims=True)
    x = tf.nn.l2_normalize(x, axis=1)
    pos = gelu(x)
    neg = gelu(-x)
    return tf.concat([pos, neg], axis=-1)




class Morphological(tf.keras.layers.Layer):
    def __init__(self, hparams={}, mode='opening', d_filter=2, trainable=True, **kwargs):
        super(Morphological, self).__init__(**kwargs)
        self.mode = mode
        self.d_filter = d_filter
        self.trainable = trainable
        self.hparams = hparams.copy()

    def get_config(self):
        config = super(Morphological, self).get_config()
        config['mode'] = self.mode
        config['d_filter'] = self.d_filter
        config['trainable'] = self.trainable
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return input_shape

    def build(self, input_shape):
        self.erosion_filters = self.add_weight("{name}_erosion".format(name=self.name),
            shape=(1, self.d_filter, input_shape[-1]),
            initializer='glorot_uniform', trainable=self.trainable)
        self.dilation_filters = self.add_weight("{name}_dilation".format(name=self.name),
            shape=(1, self.d_filter, input_shape[-1]),
            initializer='glorot_uniform', trainable=self.trainable)
        return super(Morphological, self).build(input_shape)

    def call(self, input, training=True, mask=None):
        # expand to NHWC
        inner = tf.expand_dims(input, axis=-3)
        strides = [1, 1, 1, 1]
        dilations = [1, 1, 1, 1]
        if self.mode == 'opening':
            inner = tf.nn.erosion2d(inner, self.erosion_filters, strides=strides,
                            padding='SAME', data_format='NHWC', dilations=dilations)
            inner = tf.nn.dilation2d(inner, self.dilation_filters, strides=strides,
                            padding='SAME', data_format='NHWC', dilations=dilations)
        elif self.mode == 'closing':
            inner = tf.nn.dilation2d(inner, self.dilation_filters, strides=strides,
                            padding='SAME', data_format='NHWC', dilations=dilations)
            inner = tf.nn.erosion2d(inner, self.erosion_filters, strides=strides,
                            padding='SAME', data_format='NHWC', dilations=dilations)
        else:
            raise NotImplementedError()
        inner = tf.squeeze(inner, axis=-3)
        return inner




class SignalFeatureMorph(tf.keras.Model):
    def __init__(self, hparams={}, **kwargs):
        super(SignalFeatureMorph, self).__init__(**kwargs)
        self.cnn_features = hparams.get("cnn_features") or 64
        self.kernel_size = hparams.get("cnn_kernel") or 8
        self.pool_size = hparams.get("cnn_pool_size") or 3
        self.pool_stride = hparams.get("cnn_pool_stride") or 1
        self.d_model = hparams.get('d_model') or 256
        # convolutional layer
        def feature_Conv1D(kernel_size):
            return tf.keras.layers.SeparableConv1D(self.cnn_features // 4, kernel_size,
                kernel_initializer='glorot_uniform',
                strides=1, padding='same',
                activation=None,
                data_format='channels_last')
        self.conv_1a = feature_Conv1D(self.kernel_size // 4)
        self.conv_1b = feature_Conv1D(self.kernel_size // 3)
        self.conv_1c = feature_Conv1D(self.kernel_size // 2)
        self.conv_1d = feature_Conv1D(self.kernel_size)
        self.norm_layer_1 = LayerNormalization(epsilon=1e-6)
        self.conv_2 = tf.keras.layers.Conv1D(self.d_model, self.kernel_size,
            kernel_initializer='glorot_uniform',
            strides=1, padding='same',
            activation=None,
            data_format='channels_last')
        self.conv_3 = tf.keras.layers.SeparableConv1D(self.d_model, self.kernel_size,
            kernel_initializer='glorot_uniform',
            strides=1, padding='same',
            activation=None,
            data_format='channels_last')
        # pool layer
        self.pool_1 = tf.keras.layers.MaxPool1D(pool_size=self.pool_size,
            strides=2, padding='same', data_format='channels_last')
        #self.pool_1b = tf.keras.layers.MaxPool1D(pool_size=8,
        #    strides=self.cnn_features // self.d_model, padding='same', data_format='channels_first')
        self.pool_2 = tf.keras.layers.MaxPool1D(pool_size=self.pool_size // 2,
            strides=self.pool_stride // 2, padding='same', data_format='channels_last')
        self.pool_3 = tf.keras.layers.MaxPool1D(pool_size=self.pool_size // 2,
            strides=1, padding='same', data_format='channels_last')
        self.norm_layer_2 = LayerNormalization(epsilon=1e-6)
        #self.act_layer = tf.keras.layers.Activation(tf.nn.elu)
        self.act_layer = tf.keras.layers.Lambda(gelu)

    def call(self, input, training=False):
        #smooth_1 = smooth1d(input, d=5)
        #smooth_2 = smooth1d(input, d=9)
        #smooth_3 = smooth1d(input, d=13)
        # convolutional I
        inner_1a = self.conv_1a(input)
        inner_1b = self.conv_1b(input)
        inner_1c = self.conv_1c(input)
        inner_1d = self.conv_1d(input)
        inner = self.norm_layer_1(tf.concat([inner_1a, inner_1b, inner_1c, inner_1d],axis=-1))
        inner = self.act_layer(inner)
        inner = self.pool_1(inner) # (batch_size, sig_len // 2, cnn_features)
        # convolutional II
        inner = self.conv_2(inner) # (batch_size, sig_len, d_model)
        inner = self.pool_2(inner)
        # convolutional III
        inner = self.conv_3(inner) # (batch_size, sig_len, d_model)
        inner = self.pool_3(inner)
        inner = self.norm_layer_2(inner)
        #inner = self.act_layer(inner)
        return inner




class SignalFeatureCNN(tf.keras.Model):
    def __init__(self, hparams={}, **kwargs):
        super(SignalFeatureCNN, self).__init__(**kwargs)
        self.d_model = hparams.get("d_model") or 128
        self.kernel_size = hparams.get("cnn_kernel") or 8
        self.pool_stride = hparams.get("cnn_pool_stride") or 1
        self.pool_size = hparams.get("cnn_pool_size") or 3
        kernel_initializer = 'glorot_uniform'
        self.cnn1 = tf.keras.layers.Conv1D(self.d_model, self.kernel_size,
            kernel_initializer=kernel_initializer,
            strides=1,
            padding='same',
            activation=tf.nn.elu,
            data_format='channels_last')
        self.cnn2 = tf.keras.layers.Conv1D(self.d_model, self.kernel_size,
            kernel_initializer=kernel_initializer,
            strides=1,
            padding='same',
            activation=tf.nn.elu,
            data_format='channels_last')
        self.pool = tf.keras.layers.MaxPool1D(pool_size=self.pool_size,
            strides=self.pool_stride,
            padding='same',
            data_format='channels_last')
        #self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, input, training=True):
        inner = self.cnn1(input)
        inner = self.cnn2(inner) + inner
        output = self.pool(inner)
        return output
