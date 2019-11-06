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




def get_angles(pos, j, max_timescale, d_model):
    angle_rates = 1 / np.power(max_timescale,
                               (2 * (j//2)) / np.float32(d_model))
    return pos * angle_rates




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
    j = np.arange(d_model)[np.newaxis, np.newaxis, :]
    pos_rads = get_angles(np.arange(seq_len)[np.newaxis, :, np.newaxis], j)
    #depth_rads = get_angles(np.arange(depth)[:, np.newaxis, np.newaxis],
    #                        np.arange(d_model)[np.newaxis, np.newaxis, :])
    depth_rads = get_angles(np.logspace(0, np.log10(max_timescale), depth)[:, np.newaxis, np.newaxis], j)
    rads = pos_rads + depth_rads
    # apply sin to even indices in the array; 2i
    rads[:, 0::2, :] = np.sin(pos_rads[:, 0::2, :]) + np.sin(depth_rads[:, :, :])
    # apply cos to odd indices in the array; 2i+1
    rads[:, 1::2, :] = np.cos(pos_rads[:, 1::2, :]) + np.cos(depth_rads[:, :, :])
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
      output
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
    return output




def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(dff,
            activation='relu'),
      # (batch_size, seq_len, d_model)
      tf.keras.layers.Dense(d_model,
            activation='sigmoid')
    ])




def separable_conv_feed_forward_network(d_model, dff, d_filter, padding='same'):
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.SeparableConv1D(dff, d_filter,
                padding=padding,
                data_format='channels_last',
                activation='relu',
                depthwise_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
                pointwise_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        # (batch_size, seq_len, d_model)
        tf.keras.layers.Dense(d_model,
                activation='sigmoid'),
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

    def call(self, vkq, training, mask):
        v, k, q = vkq
        seq_len_q = tf.shape(q)[1]
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
        scaled_attention = scaled_dot_product_attention(q, k, v, mask=mask)
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (-1, seq_len_q, self.d_model))
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        return output




class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = hparams.get('d_model') or 256
        self.dff = hparams.get('dff') or 1024
        self.dff_type = hparams.get('encoder_dff_type') or hparams.get('dff_type') or 'point_wise'
        self.dff_filter = hparams.get('encoder_dff_filter_width') or 8
        self.rate = hparams.get('rate') or 0.1
        self.hparams = hparams.copy()
        self.hparams['memory_comp'] = hparams.get('input_memory_comp')
        self.hparams['memory_comp_pad'] = 'same'

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.mha = MultiHeadAttention(hparams=self.hparams)
        if self.dff_type == 'point_wise':
            self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)
        elif self.dff_type == 'separable_convolution':
            self.ffn = separable_conv_feed_forward_network(self.d_model, self.dff, self.dff_filter)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        return super(EncoderLayer, self).build(input_shape)

    def call(self, input, training, mask):
        # attn_output == (batch_size, input_seq_len, d_model)
        attn_output = self.mha([input, input, input], training=training, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(input + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2




class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.d_model = hparams.get('d_model') or 256
        self.dff = hparams.get('dff') or 1024
        self.dff_type = hparams.get('decoder_dff_type') or hparams.get('dff_type') or 'point_wise'
        self.dff_filter = hparams.get('decoder_dff_filter_width') or 8
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
        self.hparams['memory_comp_pad'] = 'causal'
        self.mha1 = MultiHeadAttention(hparams=self.hparams)
        self.hparams['memory_comp'] = self.hparams.get('input_memory_comp')
        self.hparams['memory_comp_pad'] = 'same'
        self.mha2 = MultiHeadAttention(hparams=self.hparams)
        if self.dff_type == 'point_wise':
            self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)
        elif self.dff_type == 'separable_convolution':
            self.ffn = separable_conv_feed_forward_network(self.d_model, self.dff,
                    self.dff_filter,
                    padding='causal')
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        self.dropout3 = tf.keras.layers.Dropout(self.rate)
        return super(DecoderLayer, self).build(input_shape)

    def call(self, inputs, training, mask):
        assert len(inputs) == 4 or len(inputs) == 2
        if len(inputs) == 4:
            x, enc_output, look_ahead_mask, padding_mask = inputs
        else:
            x, enc_output = inputs
            look_ahead_mask = None
            padding_mask = None
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # attn_weights_block1 == (batch_size, target_seq_len, d_model)
        attn1 = self.mha1([x, x, x],
                training=training, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training)
        out1 = self.layernorm1(attn1 + x)
        # attn_weights_block2 == (batch_size, target_seq_len, d_model)
        attn2 = self.mha2([enc_output, enc_output, out1],
                training=training, mask=padding_mask)
        attn2 = self.dropout2(attn2, training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3




class ACT(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(ACT, self).__init__(**kwargs)
        self.halt_epsilon = hparams.get('halt_epsilon') or 0.01
        self.ponder_bias_init = hparams.get('ponder_bias_init') or 1.0
        self.act_type = hparams.get('act_type') or 'dense'
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
        if self.act_type == 'dense':
            self.ponder_kernel = tf.keras.layers.Dense(1,
                activation=tf.nn.sigmoid,
                use_bias=True,
                bias_initializer=tf.constant_initializer(self.ponder_bias_init))
        else:
            self.ponder_kernel = tf.keras.layers.SeparableConv1D(1, 4,
                    padding='causal',
                    data_format='channels_last',
                    activation=tf.nn.sigmoid)
        return super(ACT, self).build(input_shape)

    def call(self, inputs, training, mask):
        assert isinstance(inputs, list) and len(inputs) == 4
        state, halting_probability, remainders, n_updates = inputs
        self.halt_threshold = tf.constant(1 - self.halt_epsilon, dtype=tf.float32)
        p = self.ponder_kernel(state) # (batch_size, seq_len, 1)
        p = tf.squeeze(p, axis=-1) # (batch_size, seq_len)
        # Mask for inputs which have not halted yet
        still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)
        if mask is not None:
            halting_probability += tf.squeeze(mask) # (batch_size, seq_len) with 0.0 on valid steps
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
        self.d_model = hparams.get('d_model') or 256
        self.max_iterations = hparams.get('encoder_max_iterations') or 8
        self.max_timescale = hparams.get('encoder_time_scale') or 10000
        self.random_shift = hparams.get('random_shift') or False
        self.halt_epsilon = hparams.get('halt_epsilon') or 0.01
        self.time_penalty = hparams.get('encoder_time_penalty') or hparams.get('time_penalty') or 0.01
        self.rate = hparams.get('rate') or 0.1
        self.hparams = hparams.copy()
        self.hparams['act_type'] = hparams.get('encoder_act_type') or hparams.get('act_type') or 'dense'

    def get_config(self):
        config = super(Encoder, self).get_config()
        config['hparams'] = self.hparams
        return config

    def compute_output_shape(self, input_shape):
        # (batch_size, seq_len)
        assert len(input_shape) == 2
        return input_shape + (self.d_model,)

    def build(self, input_shape):
        assert len(input_shape) == 3
        _, sequence_length, d_model = input_shape
        self.pos_encoding = positional_encoding(int(sequence_length),
                                                            self.max_iterations,
                                                            int(self.d_model),
                                                            max_timescale=self.max_timescale,
                                                            random_shift=self.random_shift)
        self.dropout = tf.keras.layers.SpatialDropout1D(self.rate)
        self.enc_layer = EncoderLayer(hparams=self.hparams)
        self.act_layer = ACT(hparams=self.hparams)
        self.time_penalty_t = tf.cast(self.time_penalty, tf.float32)
        return super(Encoder, self).build(input_shape)

    def call(self, x, training, mask):
        state = x
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
            if step == tf.cast(0, dtype=tf.int32):
                transformed_state = self.dropout(transformed_state, training=training)
            transformed_state = self.enc_layer(transformed_state,
                    training=training, mask=mask)
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
          parallel_iterations=4,
          swap_memory=False,
          back_prop=training)
        act_loss = remainders + n_updates
        if mask is not None:
            _msk = tf.squeeze(1-mask)
            act_loss = tf.reduce_sum(act_loss * _msk, axis=-1, keepdims=True) # / tf.reduce_sum(_msk, axis=-1, keepdims=True)
            act_loss *= self.time_penalty_t
            n_updates_mean = tf.divide(tf.reduce_sum(n_updates * _msk, axis=-1), tf.squeeze(tf.reduce_sum(_msk, axis=-1)))
        else:
            act_loss = self.time_penalty_t * tf.math.reduce_sum(act_loss, axis=-1)
            n_updates_mean = tf.reduce_mean(n_updates, axis=-1)
        self.add_loss(act_loss)
        tf.summary.scalar("ponder_times_encoder", tf.reduce_mean(n_updates_mean))
        #tf.summary.scalar("ponder_loss_encoder", tf.reduce_sum(act_loss))
        # x.shape == (batch_size, seq_len, d_model)
        return new_state




class Decoder(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.d_output = hparams.get('d_output') or 6
        self.d_model = hparams.get('d_model') or 256
        self.max_iterations = hparams.get('decoder_max_iterations') or 8
        self.num_heads = hparams.get('num_heads') or 8
        self.max_timescale = hparams.get('decoder_time_scale') or 1000
        self.random_shift = hparams.get('random_shift') or False
        self.halt_epsilon = hparams.get('halt_epsilon') or 0.01
        self.time_penalty = hparams.get('decoder_time_penalty') or hparams.get('time_penalty') or 0.01
        self.rate = hparams.get('rate') or 0.1
        self.hparams = hparams.copy()
        self.hparams['act_type'] = hparams.get('decoder_act_type') or hparams.get('act_type') or 'dense'

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
        self.dropout = tf.keras.layers.SpatialDropout1D(self.rate)
        self.dec_layer = DecoderLayer(hparams=self.hparams)
        self.act_layer = ACT(hparams=self.hparams)
        self.time_penalty_t = tf.cast(self.time_penalty, tf.float32)
        self.look_ahead_mask = create_look_ahead_mask(sequence_length)
        return super(Decoder, self).build(input_shape)

    def call(self, input, training, mask):
        x, enc_output, input_padding_mask, target_padding_mask = input
        # look ahead and dropout masks
        look_ahead_mask = tf.maximum(target_padding_mask, self.look_ahead_mask)
        seq_len = tf.shape(x)[1]
        # dropout/flip and embedding
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
            if step == tf.cast(0, dtype=tf.int32):
                transformed_state = self.dropout(transformed_state, training=training)
            transformed_state = self.dec_layer([transformed_state, enc_output, look_ahead_mask, input_padding_mask],
                training=training, mask=mask)
            update_weights, halting_probability, remainders, n_updates = self.act_layer(
                [transformed_state, halting_probability, remainders, n_updates],
                training=training, mask=target_padding_mask)
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
          parallel_iterations=4,
          swap_memory=False,
          back_prop=training)
        act_loss = remainders + n_updates
        if target_padding_mask is not None:
            _msk = tf.squeeze(1-target_padding_mask)
            act_loss = tf.reduce_sum(act_loss * _msk, axis=-1, keepdims=True) # / tf.reduce_sum(_msk, axis=-1, keepdims=True)
            act_loss *= self.time_penalty_t
            n_updates_mean = tf.divide(tf.reduce_sum(n_updates * _msk, axis=-1), tf.squeeze(tf.reduce_sum(_msk, axis=-1)))
        else:
            act_loss = self.time_penalty_t * tf.math.reduce_sum(act_loss, axis=-1)
            n_updates_mean = tf.reduce_mean(n_updates, axis=-1)
        self.add_loss(act_loss)
        tf.summary.scalar("ponder_times_decoder", tf.reduce_mean(n_updates_mean))
        #tf.summary.scalar("ponder_loss_decoder", tf.reduce_sum(act_loss))
        # x.shape == (batch_size, seq_len, d_model)
        return new_state




class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, hparams={}, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        self.d_output = hparams.get('d_output') or 6
        self.rate = hparams.get('rate') or 0.1
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
        self.d_output_t = tf.cast(self.d_output, tf.int32)
        self.final_layer = tf.keras.layers.Dense(self.d_output)
        return super(TransformerLayer, self).build(input_shape)

    def __flip_target__(self, target):
        flp_mask = tf.random.uniform(target.shape, 0.0, 1.0, dtype=tf.float32)
        flp_mask = tf.cast(tf.less_equal(flp_mask, self.rate), tf.int32)
        flp_mask = tf.concat([tf.zeros((target.shape[0], 1), dtype=tf.int32), flp_mask[:,1:]],axis=1)  # do not flip SOS character
        flp_val = tf.random.uniform(target.shape, tf.cast(0, tf.int32), self.d_output_t - 2, dtype=tf.int32)
        flp_val *= flp_mask
        target = (target + flp_val) % self.d_output_t
        return target

    def call(self, inputs, training=True, mask=None):
        if len(inputs) == 4:    # trainig
            input, target, input_lengths, target_lengths = inputs
            input_max = input.shape[1]
            target_max = target.shape[1]
            enc_padding_mask = create_padding_mask(input_lengths, input_max)
            dec_input_padding_mask = create_padding_mask(input_lengths, input_max)
            dec_target_padding_mask = create_padding_mask(target_lengths, target_max)
            # enc_output.shape == # (batch_size, inp_seq_len, d_model)
            enc_output = self.encoder(input, training=training, mask=enc_padding_mask)
            # flip random bases in target sequence with dropout rate
            #if training:
            #    target = self.__flip_target__(target)
            # dec_output.shape == (batch_size, tar_seq_len, d_model)
            dec_output = self.decoder([target, enc_output, dec_input_padding_mask, dec_target_padding_mask],
                    training=training, mask=None)
            final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, d_output)
            return final_output
        else:   # prediction
            input, input_lengths, target_max = inputs
            input_max = input.shape[1]
            target = tf.concat([tf.ones_like(input_lengths) * self.d_output_t - 2, # init with sos token
                                tf.zeros((tf.shape(input)[0],) + (target_max-1,), dtype=input_lengths.dtype)], axis=-1)
            target_lengths = tf.ones_like(input_lengths)
            # run encoder once
            enc_padding_mask = create_padding_mask(input_lengths, input_max)
            dec_input_padding_mask = create_padding_mask(input_lengths, input_max)
            enc_output = self.encoder(input, training=False, mask=enc_padding_mask)
            target_active = tf.ones(target_lengths.shape, dtype=tf.bool)
            step = tf.cast(0, dtype=tf.int32)
            predictions = tf.zeros_like(target)
            # update decoder target with new predictions
            def update_state(step, target, u0, target_lengths, target_active):
                del u0
                dec_target_padding_mask = create_padding_mask(target_lengths, target_max)
                dec_output = self.decoder([target, enc_output, dec_input_padding_mask, dec_target_padding_mask],
                        training=False, mask=None)
                predictions = self.final_layer(dec_output)  # (batch_size, tar_seq_len, d_output)
                final_output = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)   # (batch_size, tar_seq_len)
                target = tf.concat([target[:,:step+1],
                                    final_output[:,step:-1]], axis=-1)
                target_active = tf.logical_and(target_active, tf.expand_dims(tf.less(final_output[:,step], self.d_output_t -2), -1))
                target_lengths += tf.cast(target_active, target_lengths.dtype)
                step += 1
                return (step, target, predictions, target_lengths, target_active)

            # stop when all sequences yielded eos
            def should_continue(u0, u1, u2, u3, target_active):
                del u0, u1, u2, u3
                return tf.reduce_any(target_active)

            # loop over decoder until all sequences stop with eos token or target_max reached
            # Do while loop iterations until predicate above is false.
            (_, target, predictions, target_lengths, _) = tf.while_loop(
              should_continue, update_state,
              (step, target, predictions, target_lengths, target_active),
              maximum_iterations=target_max-1,
              parallel_iterations=1,
              swap_memory=True,
              back_prop=False)
            #target = tf.concat([target[:,:-1], tf.ones_like(input_lengths) * self.d_output_t - 1], axis=-1)
            return [predictions, target_lengths]




class Transformer(tf.keras.Model):
    def __init__(self, hparams={}):
        super(Transformer, self).__init__()
        self.cnn = tf.keras.layers.Convolution1D(hparams.get('d_model'),
                            kernel_size=(hparams.get("cnn_kernel") or 8,),
                            strides=1,
                            padding='same',
                            data_format='channels_last')
        #self.dropout = tf.keras.layers.SpatialDropout1D(hparams.get('rate') or 0.1)
        self.pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid', data_format='channels_last')
        self.transformer_layer = TransformerLayer(hparams)

    def call(self, inputs, training=False, mask=None):
        if len(inputs) == 4:    # training
            input, target, input_lengths, target_lengths = inputs
            inner = self.cnn(input)
            inner = self.pool(inner)
            output = self.transformer_layer([inner, target, input_lengths, target_lengths], training=training)
            return output
        elif len(inputs) == 3:  # prediction
            input, input_lengths, target_max = inputs
            inner = self.cnn(input)
            inner = self.pool(inner)
            output = self.transformer_layer([inner, input_lengths, target_max])
            return output




if __name__ == '__main__':
    minibatch_size = 32
    d_model = 32
    sig_len = 128

    q = tf.random.uniform((minibatch_size, sig_len, d_model))
    k = tf.random.uniform((minibatch_size, sig_len, d_model))
    v = tf.random.uniform((minibatch_size, sig_len, d_model))

    z = scaled_dot_product_attention(q, k, v, None)
    z1 = scaled_local_attention(q, k, v, None)
