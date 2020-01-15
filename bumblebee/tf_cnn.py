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




class SignalFeatureCNN(tf.keras.Model):
    def __init__(self, hparams={}, **kwargs):
        super(SignalFeatureCNN, self).__init__(**kwargs)
        self.d_model = hparams.get("d_model") or 128
        self.kernel_size = hparams.get("cnn_kernel") or 8
        self.pool_stride = hparams.get("cnn_pool_stride") or 1
        self.pool_size = hparams.get("cnn_pool_size") or 3
        self.cnn1 = tf.keras.layers.Conv1D(self.d_model, self.kernel_size,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            data_format='channels_last')
        self.cnn2 = tf.keras.layers.Conv1D(self.d_model, self.kernel_size,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            data_format='channels_last')
        self.pool = tf.keras.layers.MaxPool1D(pool_size=self.pool_size,
            strides=self.pool_stride,
            padding='same',
            data_format='channels_last')
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, input):
        cnn1 = self.cnn1(input)
        cnn2 = self.cnn2(cnn1)
        output = self.norm(self.pool(cnn1 + cnn2))
        return output
