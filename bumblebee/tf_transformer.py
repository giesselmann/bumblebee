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
import numpy as np
import matplotlib.pyplot as plt




def positional_encoding(seq_len, depth, d_model,
                        max_timescale=10000, random_shift=False):
    def get_angles(pos, j, shift=0.0):
        angle_rates = 1 / np.power(max_timescale,
                                   (2 * (j//2)) / np.float32(d_model))
        return pos * angle_rates + shift
    if random_shift:
        pos_shift = np.random.uniform(0, max_timescale, 1)
    else:
        pos_shift = 0.0
    pos_rads = get_angles(np.arange(seq_len)[np.newaxis, :, np.newaxis],
                          np.arange(d_model)[np.newaxis, np.newaxis, :], shift=pos_shift)
    depth_rads = get_angles(np.arange(depth)[:, np.newaxis, np.newaxis],
                            np.arange(d_model)[np.newaxis, np.newaxis, :])

    rads = pos_rads + depth_rads
    # apply sin to even indices in the array; 2i
    rads[:,:, 0::2] = np.sin(pos_rads[:, :, 0::2]) + np.sin(depth_rads[:, :, 0::2])
    # apply cos to odd indices in the array; 2i+1
    rads[:,:, 1::2] = np.cos(pos_rads[:, :, 1::2]) + np.cos(depth_rads[:, :, 1::2])
    pos_encoding = rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32) # (1, depth, seq_len, d_model)




def create_padding_mask(lengths, max_len):
    msk = 1 - tf.sequence_mask(lengths, max_len, dtype=tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return msk[:, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)




def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)




def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
      output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights




def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])




class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_model = hparams.get('d_model') or 256
        self.num_heads = hparams.get('num_heads') or 4
        self.memory_comp = hparams.get('memory_comp')
        self.memory_comp_pad = hparams.get('memory_comp_pad') or 'same'
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads
        self.hparams = hparams.copy()

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        # input is list of v, k, q
        assert isinstance(input_shape, list) and len(input_shape) == 3
        return input_shape[-1]

    def build(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 3
        _, _, d_model = input_shape[-1]
        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)
        if self.memory_comp:
            self.k_comp = tf.keras.layers.Convolution1D(self.d_model,
                                kernel_size=(self.memory_comp,),
                                strides=self.memory_comp,
                                padding=self.memory_comp_pad,
                                data_format='channels_last')
            self.v_comp = tf.keras.layers.Convolution1D(self.d_model,
                                kernel_size=(self.memory_comp,),
                                strides=self.memory_comp,
                                padding=self.memory_comp_pad,
                                data_format='channels_last')
        self.dense = tf.keras.layers.Dense(self.d_model)
        return super(MultiHeadAttention, self).build(input_shape)

    def split_heads(self, x, seq_len):
          """Split the last dimension into (num_heads, depth).
          Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
          """
          x = tf.reshape(x, (-1, seq_len, self.num_heads, self.depth))
          return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, vkq, **kwargs):
        v, k, q = vkq
        seq_len_q = tf.shape(q)[1]
        mask = kwargs.get('mask')
        # (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        if self.memory_comp:
            k = self.k_comp(k)
            v = self.v_comp(v)
            if mask is not None:
                mask = mask[...,::self.memory_comp]
        seq_len_k = tf.shape(k)[1]
        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, seq_len_q)
        k = self.split_heads(k, seq_len_k)
        v = self.split_heads(v, seq_len_k)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
          q, k, v, mask=mask)
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (-1, seq_len_q, self.d_model))
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        return [output, attention_weights]




class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = hparams.get('d_model') or 256
        self.dff = hparams.get('dff') or 1024
        self.rate = hparams.get('rate') or 0.1
        self.hparams = hparams.copy()
        self.hparams['memory_comp'] = hparams.get('input_memory_comp')
        self.hparams['memory_comp_pad'] = hparams.get('input_memory_comp_pad')

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.mha = MultiHeadAttention(hparams=self.hparams)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        return super(EncoderLayer, self).build(input_shape)

    def call(self, input, **kwargs):
        # attn_output == (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha([input, input, input], mask=kwargs.get('mask'))
        attn_output = self.dropout1(attn_output, training=kwargs.get('training'))
        out1 = self.layernorm1(input + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=kwargs.get('training'))
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2




class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.d_model = hparams.get('d_model') or 256
        self.dff = hparams.get('dff') or 1024
        self.rate = hparams.get('rate') or 0.1
        self.hparams = hparams.copy()

    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.hparams['memory_comp'] = self.hparams.get('target_memory_comp')
        self.hparams['memory_comp_pad'] = self.hparams.get('target_memory_comp_pad')
        self.mha1 = MultiHeadAttention(hparams=self.hparams)
        self.hparams['memory_comp'] = self.hparams.get('input_memory_comp')
        self.hparams['memory_comp_pad'] = self.hparams.get('input_memory_comp_pad')
        self.mha2 = MultiHeadAttention(hparams=self.hparams)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        self.dropout3 = tf.keras.layers.Dropout(self.rate)
        return super(DecoderLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert len(inputs) == 4 or len(inputs) == 2
        if len(inputs) == 4:
            x, enc_output, look_ahead_mask, padding_mask = inputs
        else:
            x, enc_output = inputs
            look_ahead_mask = None
            padding_mask = None
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # attn_weights_block1 == (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1([x, x, x],
                training=kwargs.get('training'), mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, kwargs.get('training'))
        out1 = self.layernorm1(attn1 + x)
        # attn_weights_block2 == (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2([enc_output, enc_output, out1],
                training=kwargs.get('training'), mask=padding_mask)
        attn2 = self.dropout2(attn2, kwargs.get('training'))
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, kwargs.get('training'))
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3




class ACT(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(ACT, self).__init__(**kwargs)
        self.halt_epsilon = hparams.get('halt_epsilon') or 0.01
        self.ponder_bias_init = hparams.get('ponder_bias_init') or 1.0
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
        self.ponder_kernel = tf.keras.layers.Dense(1,
                activation=tf.nn.sigmoid,
                use_bias=True,
                bias_initializer=tf.constant_initializer(self.ponder_bias_init))
        return super(ACT, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list) and len(inputs) == 4
        state, halting_probability, remainders, n_updates = inputs
        self.halt_threshold = tf.constant(1 - self.halt_epsilon, dtype=tf.float32)
        p = self.ponder_kernel(state) # (batch_size, seq_len, 1)
        p = tf.squeeze(p, axis=-1) # (batch_size, seq_len)
        mask = kwargs.get('mask') # (batch_size, 1, 1, seq_len) or None
        # Mask for inputs which have not halted yet
        still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)
        if mask is not None:
            mask = tf.squeeze(mask) # (batch_size, seq_len) with 0.0 on valid steps
            halting_probability += mask
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




class Encoder(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_input = hparams.get('d_input') or 4096
        self.d_model = hparams.get('d_model') or 256
        self.max_iterations = hparams.get('max_iterations') or 8
        self.max_timescale = hparams.get('encoder_time_scale') or 10000
        self.random_shift = hparams.get('random_shift') or False
        self.halt_epsilon = hparams.get('halt_epsilon') or 0.01
        self.time_penalty = hparams.get('act_time_penalty') or 0.01
        self.hparams = hparams.copy()

    def get_config(self):
        config = super(Encoder, self).get_config()
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        # (batch_size, seq_len)
        assert len(input_shape) == 2
        return input_shape + (self.d_model,)

    def build(self, input_shape):
        assert len(input_shape) == 2
        _, sequence_length = input_shape
        self.pos_encoding = positional_encoding(int(sequence_length),
                                                            self.max_iterations,
                                                            int(self.d_model),
                                                            max_timescale=self.max_timescale,
                                                            random_shift=self.random_shift)
        self.emb_layer = tf.keras.layers.Embedding(self.d_input, self.d_model, name='Encoder_Embedding')
        self.enc_layer = EncoderLayer(hparams=self.hparams)
        self.act_layer = ACT(hparams=self.hparams)
        self.time_penalty_t = tf.cast(self.time_penalty, tf.float32)
        return super(Encoder, self).build(input_shape)

    def call(self, x, **kwargs):
        state = self.emb_layer(tf.cast(x, tf.int32))
        state *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # init ACT
        state_shape_static = state.get_shape() # (batch_size, seq_len, d_model)
        state_slice = slice(0, 2)
        update_shape = tf.shape(state)[state_slice]
        halting_probability = tf.zeros(update_shape, name="halting_probability")
        remainders = tf.zeros(update_shape, name="remainder")
        n_updates = tf.zeros(update_shape, name="n_updates")
        previous_state = tf.zeros_like(state, name="previous_state")
        step = tf.cast(0, dtype=tf.int32)
        # define update and halt-condition
        def update_state(state, step, halting_probability, remainders, n_updates):
            transformed_state = state + self.pos_encoding[:,step,:,:]
            transformed_state = self.enc_layer(transformed_state,
                    training=kwargs.get('training'), mask=kwargs.get('mask'))
            update_weights, halting_probability, remainders, n_updates = self.act_layer(
                    [transformed_state, halting_probability, remainders, n_updates],
                    training=kwargs.get('training'), mask=kwargs.get('mask'))
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
          swap_memory=True,
          back_prop=kwargs.get('training'))
        act_loss = self.time_penalty_t * tf.math.reduce_mean(remainders + n_updates, axis=1)
        self.add_loss(act_loss)
        tf.summary.scalar("ponder_times_encoder", tf.reduce_mean(n_updates))
        #tf.summary.scalar("ponder_loss_encoder", tf.reduce_mean(act_loss))
        # x.shape == (batch_size, seq_len, d_model)
        return new_state




class Decoder(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.d_output = hparams.get('d_output') or 6
        self.d_model = hparams.get('d_model') or 256
        self.max_iterations = hparams.get('max_iterations') or 8
        self.num_heads = hparams.get('num_heads') or 8
        self.dff = hparams.get('dff') or 1024
        self.max_timescale = hparams.get('decoder_time_scale') or 1000
        self.random_shift = hparams.get('random_shift') or False
        self.halt_epsilon = hparams.get('halt_epsilon') or 0.01
        self.time_penalty = hparams.get('time_penalty') or 0.01
        self.rate = hparams.get('rate') or 0.1
        self.hparams = hparams.copy()

    def get_config(self):
        config = super(Decoder, self).get_config()
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        # (batch_size, seq_len)
        assert isinstance(input_shape, list) and len(input_shape) == 4
        return input_shape[0] + (self.d_model,)

    def build(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 4
        _, sequence_length = input_shape[0]
        self.pos_encoding = positional_encoding(int(sequence_length),
                                                self.max_iterations,
                                                int(self.d_model),
                                                max_timescale=self.max_timescale,
                                                random_shift=self.random_shift)
        self.emb_layer = tf.keras.layers.Embedding(self.d_output, self.d_model)
        self.dec_layer = DecoderLayer(hparams=self.hparams)
        self.act_layer = ACT(hparams=self.hparams)
        self.time_penalty_t = tf.cast(self.time_penalty, tf.float32)
        self.look_ahead_mask = create_look_ahead_mask(sequence_length)
        return super(Decoder, self).build(input_shape)

    def call(self, input, **kwargs):
        x, enc_output, input_padding_mask, target_padding_mask = input
        look_ahead_mask = tf.maximum(target_padding_mask, self.look_ahead_mask)
        seq_len = tf.shape(x)[1]
        state = self.emb_layer(tf.cast(x, tf.int32))
        state *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # init ACT
        state_shape_static = state.get_shape() # (batch_size, seq_len, d_model)
        state_slice = slice(0, 2)
        update_shape = tf.shape(state)[state_slice]
        halting_probability = tf.zeros(update_shape, name="halting_probability")
        remainders = tf.zeros(update_shape, name="remainder")
        n_updates = tf.zeros(update_shape, name="n_updates")
        previous_state = tf.zeros_like(state, name="previous_state")
        step = tf.cast(0, dtype=tf.int32)
        # define update and halt-condition
        def update_state(state, step, halting_probability, remainders, n_updates):
            transformed_state = state + self.pos_encoding[:,step,:,:]
            transformed_state = self.dec_layer([transformed_state, enc_output, look_ahead_mask, input_padding_mask], **kwargs)
            update_weights, halting_probability, remainders, n_updates = self.act_layer(
                [transformed_state, halting_probability, remainders, n_updates],
                training=kwargs.get('training'), mask=target_padding_mask)
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
          swap_memory=True,
          back_prop=kwargs.get('training'))
        act_loss = self.time_penalty_t * tf.math.reduce_mean(remainders + n_updates, axis=1)
        self.add_loss(act_loss)
        tf.summary.scalar("ponder_times_decoder", tf.reduce_mean(n_updates))
        #tf.summary.scalar("ponder_loss_decoder", tf.reduce_mean(act_loss))
        # x.shape == (batch_size, seq_len, d_model)
        return new_state




class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        self.d_output = hparams.get('d_output') or 6
        self.input_memory_comp = hparams.get('input_memory_comp')
        self.target_memory_comp = hparams.get('target_memory_comp')
        self.hparams = hparams.copy()

    def get_config(self):
        config = super(TransformerLayer, self).get_config()
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 4
        return input_shape[1] # (batch_size, tar_seq_len, d_output)

    def build(self, input_shape):
        self.encoder = Encoder(hparams=self.hparams)
        self.decoder = Decoder(hparams=self.hparams)
        self.final_layer = tf.keras.layers.Dense(self.d_output)

    def create_masks(self, input_lengths, target_lengths, input_max, target_max):
        # Encoder padding mask
        # Used in encoder self-attention
        enc_padding_mask = create_padding_mask(input_lengths, input_max)
        # Decoder input padding mask
        # This padding mask is used to mask the encoder outputs.
        # Used in the 2nd attention block in the decoder.
        dec_input_padding_mask = enc_padding_mask
        # Decoder target padding mask
        # Used in the 1st attention block in the decoder.
        # Used in the 1st attention block in the decoder with the look_ahead_mask
        dec_target_padding_mask = create_padding_mask(target_lengths, target_max)
        return enc_padding_mask, dec_input_padding_mask, dec_target_padding_mask

    def call(self, inputs, training=True, mask=None):
        input, target, input_lengths, target_lengths = inputs
        enc_padding_mask, dec_input_padding_mask, dec_target_padding_mask = self.create_masks(
                input_lengths, target_lengths, input.shape[1], target.shape[1])
        ## enc_output.shape == # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(input, training=training, mask=enc_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder([target, enc_output, dec_input_padding_mask, dec_target_padding_mask],
                training=training, mask=None)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, d_output)
        return final_output




#def Transformer(input_length, target_length, hparams={}):
#    input = tf.keras.Input(shape=(input_length,), name='inputs')
#    target = tf.keras.Input(shape=(target_length,), name='targets')
#    input_len = tf.keras.Input(shape=(1,), name='input_lengths')
#    target_len = tf.keras.Input(shape=(1,), name='target_lengths')
#    output = TransformerLayer(hparams=hparams, name='transformer')(
#            (input, target, input_len, target_len))
#    model = tf.keras.models.Model(inputs=[input, target, input_len, target_len],
#                                  outputs=output)
#    return model




class Transformer(tf.keras.Model):
    def __init__(self, hparams={}):
        super(Transformer, self).__init__()
        self.transformer_layer = TransformerLayer(hparams)

    def call(self, inputs, training=False):
        input, target, input_lengths, target_lengths = inputs
        return self.transformer_layer(inputs, training=training)




if __name__ == '__main__':
    tf.InteractiveSession()
    d_input = 4096
    d_output = 6
    d_model = 512
    sig_len = 50000
    seq_len = 5000
    max_timescale = 50
    sample_transformer = Transformer(d_input, d_model, d_output,
                                max_iterations=2, num_heads=8, dff=2048,)
    temp_input = tf.random.uniform((64, sig_len, 1))
    temp_target = tf.random.uniform((64, seq_len, 1))
    temp_input_len = tf.random.uniform((64, 1))
    temp_target_len = tf.random.uniform((64, 1))
    tf_out = sample_transformer(temp_input, temp_target, temp_input_len, temp_target_len, training=False)
    sample_transformer.summary()
