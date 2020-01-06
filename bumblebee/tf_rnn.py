# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : tf 2.0 RNN
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




def point_wise_act_network(dff, ponder_bias_init=1.0):
    return tf.keras.Sequential(
            [
            # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(dff, activation=tf.nn.leaky_relu),
            # (batch_size, seq_len, 1)
            tf.keras.layers.Dense(1,
                use_bias=True,
                bias_initializer=tf.constant_initializer(ponder_bias_init),
                activation=tf.nn.sigmoid),
            ]
        )




def separable_conv_act_network(dff, d_filter, ponder_bias_init=1.0):
    return tf.keras.Sequential(
            [
            # (batch_size, seq_len, dff)
            tf.keras.layers.SeparableConv1D(dff, d_filter,
                    padding='causal',   # do not violate timing in decoder
                    data_format='channels_last',
                    activation=tf.nn.leaky_relu
                    ),
            # (batch_size, seq_len, 1)
            tf.keras.layers.Dense(1,
                use_bias=True,
                bias_initializer=tf.constant_initializer(ponder_bias_init),
                activation=tf.nn.sigmoid),
            ]
        )




class ACT(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(ACT, self).__init__(**kwargs)
        self.halt_epsilon = hparams.get('halt_epsilon') or 0.01
        self.ponder_bias_init = hparams.get('ponder_bias_init') or 1.0
        self.act_type = hparams.get('act_type') or 'point_wise'
        self.act_dff = hparams.get('act_dff') or 8
        self.act_conv_filter = hparams.get('act_conv_filter') or 4
        self.hparams = hparams.copy()

    def get_config(self):
        config = super(ACT, self).get_config()
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        # input_shape == (state, halting_probability, remainders, n_updates)
        assert isinstance(input_shape, list) and len(input_shape) == 4
        # output_shape == (update_weights, halting_probability, remainders, n_updates)
        return (input_shape[-1],) + input_shape[1:]

    def build(self, input_shape):
        # (batch_size, seq_len, d_model)
        if self.act_type == 'point_wise':
            self.ponder_kernel = point_wise_act_network(
                self.act_dff,
                self.ponder_bias_init)
        elif self.act_type == 'separable_convolution':
            self.ponder_kernel = separable_conv_act_network(
                self.act_dff,
                self.act_conv_filter,
                self.ponder_bias_init)
        else:
            raise NotImplementedError()
        return super(ACT, self).build(input_shape)

    def call(self, inputs, training=False, mask=None):
        assert isinstance(inputs, list) and len(inputs) == 4
        state, halting_probability, remainders, n_updates = inputs
        self.halt_threshold = tf.constant(1 - self.halt_epsilon, dtype=tf.float32)
        p = self.ponder_kernel(state) # (batch_size, seq_len, 1)
        p = tf.squeeze(p, axis=-1) # (batch_size, seq_len)
        # Mask for inputs which have not halted yet
        still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)
        if mask is not None:
            halting_probability += (1-mask) # (batch_size, seq_len) with 1.0 on valid steps
        # Mask of inputs which halted at this step
        new_halted = tf.cast(
            tf.greater(halting_probability + p * still_running, self.halt_threshold),
            tf.float32) * still_running
        # Mask of inputs which haven't halted, and didn't halt this step
        still_running = tf.cast(
            tf.less_equal(halting_probability + p * still_running, self.halt_threshold),
            tf.float32) * still_running
        # Add the halting probability for this step to the halting
        # probabilities for those input which haven't halted yet
        halting_probability += p * still_running
        # Compute remainders for the inputs which halted at this step
        remainders += new_halted * (1 - halting_probability)
        # Add the remainders to those inputs which halted at this step
        halting_probability += new_halted * remainders
        # Increment n_updates for all inputs which are still running
        n_updates += still_running + new_halted
        # Compute the weight to be applied to the new state and output
        # 0 when the input has already halted
        # p when the input hasn't halted yet
        # the remainders when it halted this step
        update_weights = tf.expand_dims(p * still_running + new_halted * remainders, -1)
        return [update_weights, halting_probability, remainders, n_updates]
