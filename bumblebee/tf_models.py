# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : models
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
from tf_cnn import SignalFeatureCNN
from tf_rnn import ACT
from tf_util import positional_encoding




def optimus_ffn(d_model, dff, d_filter, nff=4, pool_size=3, padding='same', **kwargs):
    layers = []
    for n in range(nff):
        layers.extend(
            [
            # (batch_size, seq_len, dff)
            tf.keras.layers.SeparableConv1D(dff, d_filter,
                    padding=padding,
                    data_format='channels_last',
                    activation=tf.nn.leaky_relu
                    ),
            tf.keras.layers.MaxPool1D(pool_size=pool_size,
                    strides=1,
                    padding='same',
                    data_format='channels_last'),
            ]
        )
    layers += [
                #tf.keras.layers.LayerNormalization(epsilon=1e-6),
                # (batch_size, seq_len, d_model)
                tf.keras.layers.Dense(d_model,
                        activation=None)
                ]
    return tf.keras.Sequential(layers, **kwargs)




class OptimusPrime(tf.keras.Model):
    def __init__(self, hparams={}, **kwargs):
        super(OptimusPrime, self).__init__(**kwargs)
        self.d_model = hparams.get('d_model') or 16
        self.d_output = hparams.get('d_output') or 5
        self.halt_epsilon = hparams.get('halt_epsilon') or 0.01
        self.max_iterations = hparams.get('max_iterations') or 1
        self.max_timescale = hparams.get('encoder_time_scale') or 10000
        self.time_penalty = hparams.get('time_penalty') or 0.01
        self.rate = hparams.get('rate') or 0.1
        self.feature_cnn = SignalFeatureCNN(hparams=hparams, name='SignalCNN')
        #self.pos_encoding = positional_encoding(300,
        #                                                    self.max_iterations,
        #                                                    int(self.d_model),
        #                                                    max_timescale=self.max_timescale)
        #self.ffn = optimus_ffn(self.d_model,
        #                hparams.get('dff') or 256,
        #                nff = hparams.get('nff') or 6,
        #                d_filter = hparams.get('ff_filter') or 32, name='FFN')
        self.gru1 = tf.keras.layers.GRU(self.d_model,
                            activation='tanh',
                            recurrent_activation='sigmoid',
                            recurrent_dropout=0,
                            unroll=False,
                            use_bias=True,
                            reset_after=True,
                            return_sequences=True, return_state=False)
        self.gru2 = tf.keras.layers.GRU(self.d_model,
                            activation='tanh',
                            recurrent_activation='sigmoid',
                            recurrent_dropout=0,
                            unroll=False,
                            use_bias=True,
                            reset_after=True,
                            return_sequences=True, return_state=False,
                            go_backwards=True)
        self.gru_merge = tf.keras.layers.Dense(self.d_model, activation=None, name='gru_merge')
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm')
        #self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm')
        self.dropout1 = tf.keras.layers.Dropout(self.rate, name='dropout1')
        #self.dropout2 = tf.keras.layers.Dropout(self.rate, name='dropout2')
        self.act_layer = ACT(hparams=hparams, name='act_layer')
        self.dense = tf.keras.layers.Dense(self.d_output, activation=None, name='Dense')

    def call(self, inputs, training=False, mask=None):
        input_data, input_len = inputs
        state = self.feature_cnn(input_data, training=training)
        state_shape_static = state.get_shape() # (batch_size, sig_len, d_model)
        state_slice = slice(0, 2)
        update_shape = tf.shape(state)[state_slice]
        halting_probability = tf.zeros(update_shape, name="halting_probability")
        remainders = tf.zeros(update_shape, name="remainder")
        n_updates = tf.zeros(update_shape, name="n_updates")
        step = tf.cast(0, dtype=tf.int32)

        downsample_factor = tf.cast(tf.shape(input_data)[-2] // tf.shape(state)[-2], input_len.dtype)
        downsample_len = tf.cast(tf.squeeze(input_len) // downsample_factor, tf.int32)     # (batch_size,)
        mask = tf.sequence_mask(downsample_len, tf.shape(state)[-2], dtype=tf.float32)     # (batch_size, seq_len)

        # define update and halt-condition
        def update_state(state, step, halting_probability, remainders, n_updates):
            #transformed_state = state + self.pos_encoding[:,step,:,:]
            in1 = self.gru1(state, training=training, mask=tf.cast(mask, tf.bool))
            in2 = self.gru2(state, training=training, mask=tf.cast(mask, tf.bool))
            inner = self.gru_merge(tf.concat([in1, in2], axis=-1))
            inner = self.dropout1(inner, training=training)
            out1 = self.norm1(inner + state)
            #inner = self.ffn(out1)
            #inner = self.dropout2(inner, training=training)
            #transformed_state = self.norm2(inner + out1)
            transformed_state = out1
            update_weights, halting_probability, remainders, n_updates = self.act_layer(
                    [transformed_state, halting_probability, remainders, n_updates],
                    training=training, mask=mask)
            transformed_state = ((transformed_state * update_weights) +
                                state * (1 - update_weights))
            step += 1
            return (transformed_state, step, halting_probability, remainders, n_updates)

        # While loop stops when this predicate is FALSE.
        # Ie all (probability < 1-eps AND counter < N) are false.
        def should_continue(u0, u1, halting_probability, u2, n_updates):
            del u0, u1, u2
            return tf.reduce_any(
                    tf.logical_and(
                        tf.less(halting_probability, 1.0 - self.halt_epsilon),
                        tf.less(n_updates, self.max_iterations)))

        # Do while loop iterations until predicate above is false.
        (new_state, _, _, remainders, n_updates) = tf.while_loop(
          should_continue, update_state,
          (state, step, halting_probability, remainders, n_updates),
          maximum_iterations=self.max_iterations,
          parallel_iterations=1,
          swap_memory=training,
          back_prop=training)
        output = self.dense(new_state)
        _act_loss = (n_updates + remainders) * mask      # (batch_size, sig_len)

        # ponder loss
        _lengths = tf.reduce_sum(mask, axis=-1)
        act_loss = tf.reduce_sum(_act_loss, axis=-1) / _lengths * self.time_penalty
        n_updates_mean = tf.reduce_sum(n_updates * mask, axis=-1) / _lengths
        n_updates_stdv = tf.sqrt(tf.reduce_sum(tf.square(n_updates - tf.expand_dims(n_updates_mean, axis=-1)) * mask, axis=-1) / _lengths)
        #self.add_loss(act_loss)
        tf.summary.scalar("ponder_time", tf.reduce_mean(n_updates_mean))
        tf.summary.scalar("ponder_stdv", tf.reduce_mean(n_updates_stdv))

        return output, downsample_len, act_loss




if __name__ == "__main__":
    hparams = {
                'd_model' : 256,
                'd_output' : 5,
                'cnn_kernel' : 32,
                'cnn_pool_size' : 6,
                'cnn_pool_stride' : 3,
                'dff' : 1024,
                'nff' : 3,
                'ff_filter' : 32,
                'ff_pool_size' : 3,
                'act_type' : 'point_wise',
                'act_dff' : 32,
                'ponder_bias_init' : 2.0,
                'max_iterations' : 10,
                'time_penalty' : 0.01
              }
    model = Optimus(hparams=hparams)
    input_data = tf.zeros((16, 180, 1), dtype=tf.float32)
    input_len = tf.ones((16,), dtype=tf.int32)
    output = model((input_data, input_len), training=True)
    model.summary()
    print(output.shape)
