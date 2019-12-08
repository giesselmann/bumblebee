# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : BumbleBee
#
#  DESCRIPTION   : Nanopore Basecalling
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




class Discriminator(tf.keras.Model):
    def __init__(self, hparams={}, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        d_model = 128
        d_filter = 8
        nff = 5
        pool_size = 5
        rate = 0.1
        self.cnn = tf.keras.Sequential(
                            [
                             # (batch_size, seq_len, dff)
                             tf.keras.layers.Conv1D(d_model, d_filter,
                                     padding='same',
                                     data_format='channels_last',
                                     activation=tf.nn.leaky_relu
                                     ),
                             tf.keras.layers.Dropout(rate),
                             tf.keras.layers.MaxPool1D(pool_size=pool_size,
                                     strides=1,
                                     padding='same',
                                     data_format='channels_last'),
                            ] * nff,
                        name='CNN'
                        )
        self.mlp = tf.keras.Sequential(
                            [
                             tf.keras.layers.Dense(64, activation='relu'),
                            ] * 3,
                        name='MLP'
                        )
        self.dense = tf.keras.layers.Dense(1, name='Dense')

    def call(self, inputs, training=False, mask=None):
        inner = self.cnn(inputs, training=training)
        inner = self.mlp(inner, training=training)
        output = self.dense(inner, training=training)
        return output




if __name__ == "__main__":
    model = Discriminator()
    state = tf.zeros((16, 140, 32))
    output = model(state)
    print(output.shape)
