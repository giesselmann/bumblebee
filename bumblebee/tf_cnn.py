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




kernel_initializer = 'glorot_uniform'




class Morphological(tf.keras.layers.Layer):
    def __init__(self, hparams={}, mode='opening', d_filter=2, **kwargs):
        super(Morphological, self).__init__(**kwargs)
        self.mode = mode
        self.d_filter = d_filter
        self.hparams = hparams.copy()

    def get_config(self):
        config = super(Morphological, self).get_config()
        config['mode'] = self.mode
        config['d_filter'] = self.d_filter
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        return input_shape

    def build(self, input_shape):
        self.erosion_filters = self.add_weight("{name}_erosion".format(name=self.name),
            shape=(1, self.d_filter, input_shape[-1]),
            initializer='he_uniform', trainable=True)
        self.dilation_filters = self.add_weight("{name}_dilation".format(name=self.name),
            shape=(1, self.d_filter, input_shape[-1]),
            initializer='he_uniform', trainable=True)
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
        self.d_filter = [2, 4, 8, 16, 32]
        self.n_features = hparams.get('n_features') or 1
        self.kernel_size = hparams.get("cnn_kernel") or 8
        self.pool_size = hparams.get("cnn_pool_size") or 3
        self.pool_stride = hparams.get("cnn_pool_stride") or 1
        self.d_model = hparams.get('d_model') or 256
        self.opening_layer1 = [Morphological(hparams=hparams, mode='opening', d_filter=d, name="opening1_{}".format(d)) for d in self.d_filter]
        self.closing_layer1 = [Morphological(hparams=hparams, mode='closing', d_filter=d, name="closing1_{}".format(d)) for d in self.d_filter]
        #self.opening_layer2 = [Morphological(hparams=hparams, mode='opening', d_filter=d, name="opening2_{}".format(d)) for d in self.d_filter]
        #self.closing_layer2 = [Morphological(hparams=hparams, mode='closing', d_filter=d, name="closing2_{}".format(d)) for d in self.d_filter]
        self.conv_1 = tf.keras.layers.Conv1D(self.d_model // 4, self.kernel_size,
            kernel_initializer=kernel_initializer,
            strides=1, padding='same',
            activation=None,
            data_format='channels_last')
        self.conv_2 = tf.keras.layers.Conv1D(self.d_model // 4, self.kernel_size,
            kernel_initializer=kernel_initializer,
            strides=1, padding='same',
            activation=None,
            data_format='channels_last')
        self.conv_3 = tf.keras.layers.SeparableConv1D(self.d_model, self.kernel_size,
            kernel_initializer=kernel_initializer,
            strides=1, padding='same',
            activation=None,
            data_format='channels_last')
        self.norm0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=16,
            strides=1,
            padding='same',
            data_format='channels_last')
        self.pool2 = tf.keras.layers.MaxPool1D(pool_size=16,
            strides=1,
            padding='same',
            data_format='channels_last')
        self.pool3 = tf.keras.layers.MaxPool1D(pool_size=8,
            strides=self.pool_stride,
            padding='same',
            data_format='channels_last')

    def call(self, input, training=False):
        inner = tf.tile(input, [1, 1, self.n_features]) # (batch_size, sig_len, n_features)
        # smoothing I
        opened_shards = [layer(inner) for layer in self.opening_layer1]
        inner = tf.concat([layer(shard) for layer, shard in zip(self.closing_layer1, opened_shards)], axis=-1)
        # smoothing II
        #opened_shards = [layer(inner) for layer in self.opening_layer2]
        #inner = tf.concat([layer(shard) for layer, shard in zip(self.closing_layer2, opened_shards)], axis=-1)
        inner = self.norm0(inner)
        # convolutional I
        inner = self.norm1(self.conv_1(inner)) # (batch_size, sig_len, d_model)
        inner = self.pool1(inner) # (batch_size, sig_len, d_model)
        # convolutional II
        inner = self.norm2(self.conv_2(inner)) # (batch_size, sig_len, d_model)
        inner = self.pool2(inner) # (batch_size, sig_len, d_model)
        # convolutional III
        inner = self.norm3(self.conv_3(inner)) # (batch_size, sig_len, d_model)
        output = self.pool3(inner) # (batch_size, sig_len, d_model)
        return output




class SignalFeatureCNN(tf.keras.Model):
    def __init__(self, hparams={}, **kwargs):
        super(SignalFeatureCNN, self).__init__(**kwargs)
        self.d_model = hparams.get("d_model") or 128
        self.kernel_size = hparams.get("cnn_kernel") or 8
        self.pool_stride = hparams.get("cnn_pool_stride") or 1
        self.pool_size = hparams.get("cnn_pool_size") or 3
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
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, input):
        inner = self.cnn1(input)
        inner = self.cnn2(self.norm1(inner))
        output = self.norm2(self.pool(inner))
        return output
