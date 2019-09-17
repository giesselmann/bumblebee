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




def positional_encoding(seq_len, depth, d_model, max_timescale=10000):
    def get_angles(pos, j):
        angle_rates = 1 / np.power(max_timescale,
                                   (2 * (j//2)) / np.float32(d_model))
        return pos * angle_rates
    pos_rads = get_angles(np.arange(seq_len)[np.newaxis, :, np.newaxis],
                          np.arange(d_model)[np.newaxis, np.newaxis, :])
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
    msk = tf.sequence_mask(lengths, max_len, dtype=tf.float32)
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




class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3], name='transpose_mha_split')

    def call(self, vkq, training=True, mask=None):
        v, k, q = vkq
        batch_size = tf.shape(q)[0]
        # (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
          q, k, v, mask)
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3], name='transpose_mha_out')
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        return [output, attention_weights]




def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])




class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=True, mask=None):
        # attn_output == (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha([x, x, x], mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2




class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, input, training=True, mask=None):
        x, enc_output, look_ahead_mask, padding_mask = input
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # attn_weights_block1 == (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1([x, x, x], mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        # attn_weights_block2 == (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2([enc_output, enc_output, out1], mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3




class ACT():
    def __init__(self, fn_preprocess, fn_transform,
                 max_iterations=10, halt_epsilon=0.01):
        super(ACT, self).__init__()
        self.fn_preprocess = fn_preprocess
        self.fn_transform = fn_transform
        self.max_iterations = max_iterations
        self.halt_epsilon = halt_epsilon
        self.ponder_kernel = tf.keras.layers.Dense(1,
                activation=tf.nn.sigmoid,
                use_bias=True,
                bias_initializer=tf.constant_initializer(0.1))

    def __call__(self, state, training, mask, preprocess_kwargs={}, transform_args=[], transform_kwargs={}):
        threshold = 1.0 - self.halt_epsilon
        state_shape_static = state.get_shape()
        state_slice = slice(0, 2)
        update_shape = tf.shape(state)[state_slice]
        halting_probability = tf.zeros(update_shape, name="halting_probability")
        remainders = tf.zeros(update_shape, name="remainder")
        n_updates = tf.zeros(update_shape, name="n_updates")
        previous_state = tf.zeros_like(state, name="previous_state")
        step = tf.constant(0, dtype=tf.int32)

        def ut_function(state, step, halting_probability, remainders, n_updates, previous_state):
            """implements act (position-wise halting).
            Args:
              state: 3-D Tensor: [batch_size, length, channel]
              step: indicates number of steps taken so far
              halting_probability: halting probability
              remainders: act remainders
              n_updates: act n_updates
              previous_state: previous state
            Returns:
              transformed_state: transformed state
              step: step+1
              halting_probability: halting probability
              remainders: act remainders
              n_updates: act n_updates
              new_state: new state
            """
            # add time and depth encoding
            state = self.fn_preprocess(state, step, **preprocess_kwargs)
            #with tf.variable_scope("sigmoid_activation_for_pondering", reuse=True):
            p = self.ponder_kernel(state)
            # maintain position-wise probabilities
            p = tf.squeeze(p, axis=-1)
            # Mask for inputs which have not halted yet
            still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)
            # Mask of inputs which halted at this step
            new_halted = tf.cast(
                tf.greater(halting_probability + p * still_running, threshold),
                tf.float32) * still_running
            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = tf.cast(
                tf.less_equal(halting_probability + p * still_running, threshold),
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
            update_weights = tf.expand_dims(
                p * still_running + new_halted * remainders, -1)
            # apply transformation on the state
            transformed_state = self.fn_transform(state, step, *transform_args, **transform_kwargs)
            # update running part in the weighted state and keep the rest
            new_state = ((transformed_state * update_weights) +
                         (previous_state * (1 - update_weights)))
            # Add in the weighted state
            new_state = (transformed_state * update_weights) + previous_state
            # remind TensorFlow of everything's shape
            transformed_state.set_shape(state_shape_static)
            for x in [halting_probability, remainders, n_updates]:
              x.set_shape(state_shape_static[state_slice])
            new_state.set_shape(state_shape_static)
            step += 1
            return (transformed_state, step, halting_probability, remainders, n_updates,
                    new_state)

        # While loop stops when this predicate is FALSE.
        # Ie all (probability < 1-eps AND counter < N) are false.
        def should_continue(u0, u1, halting_probability, u2, n_updates, u3):
            del u0, u1, u2, u3
            return tf.reduce_any(
                    tf.logical_and(
                        tf.less(halting_probability, threshold),
                        tf.less(n_updates, self.max_iterations)))

        # Do while loop iterations until predicate above is false.
        (_, _, _, remainder, n_updates, new_state) = tf.while_loop(
          should_continue, ut_function,
          (state, step, halting_probability, remainders, n_updates, previous_state),
          maximum_iterations=self.max_iterations + 1,
          parallel_iterations=1,
          swap_memory=training,
          back_prop=training)

        ponder_times = n_updates
        remainders = remainder
        return [new_state, ponder_times, remainders]




class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_input, d_model, max_iterations, num_heads, dff,
                 halt_epsilon=0.01, time_penalty=0.01, rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_iterations = max_iterations
        self.halt_epsilon = halt_epsilon
        self.time_penalty = time_penalty
        self.embedding = tf.keras.layers.Embedding(d_input, d_model)
        self.enc_layer = EncoderLayer(d_model, num_heads, dff, rate)
        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        _, sequence_length = input_shape
        self.pos_encoding = positional_encoding(int(sequence_length), self.max_iterations, int(self.d_model))
        def fn_preprocess(state, step, **kwargs):
            return state + self.pos_encoding[:,step,:,:]
        def fn_transform(state, step, **kwargs):
            return self.enc_layer(state, **kwargs)
        self.act = ACT(fn_preprocess, fn_transform,
                        max_iterations=self.max_iterations, halt_epsilon=self.halt_epsilon)

    def call(self, x, training=True, mask=None):
        state = self.embedding(tf.cast(x, tf.int32))
        state *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        transform_kwargs = {'training' : training, 'mask' : mask}
        new_state, ponder_times, remainders = self.act(state, training, mask, transform_kwargs=transform_kwargs)
        self.add_loss(self.time_penalty * tf.math.reduce_mean(remainders + ponder_times, axis=1))
        tf.contrib.summary.scalar("ponder_times_encoder", tf.reduce_mean(ponder_times))
        # x.shape == (batch_size, seq_len, d_model)
        return [new_state, ponder_times, remainders]




class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_output, d_model, max_iterations, num_heads, dff,
                 halt_epsilon=0.01, time_penalty=0.01, rate=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_iterations = max_iterations
        self.halt_epsilon = halt_epsilon
        self.time_penalty = time_penalty
        self.embedding = tf.keras.layers.Embedding(d_output, d_model)
        self.dec_layer = DecoderLayer(d_model, num_heads, dff, rate)
        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        _, sequence_length = input_shape[0]
        self.pos_encoding = positional_encoding(int(sequence_length), self.max_iterations, int(self.d_model))
        def fn_preprocess(x, step):
            return x + self.pos_encoding[:,step,:,:]
        def fn_transform(x, step, enc_output, look_ahead_mask, padding_mask, **kwargs):
            return self.dec_layer([x, enc_output, look_ahead_mask, padding_mask], **kwargs)
        self.act = ACT(fn_preprocess, fn_transform,
                        max_iterations=self.max_iterations, halt_epsilon=self.halt_epsilon)

    def call(self, input, training=True, mask=None):
        x, enc_output, look_ahead_mask, padding_mask = input
        seq_len = tf.shape(x)[1]
        state = self.embedding(tf.cast(x, tf.int32))
        state *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # enc_output, training, look_ahead_mask, padding_mask):
        transform_args = [enc_output, look_ahead_mask, padding_mask]
        new_state, ponder_times, remainders = self.act(state, training, padding_mask, transform_args=transform_args)
        self.add_loss(self.time_penalty * tf.math.reduce_mean(remainders + ponder_times, axis=1))
        tf.contrib.summary.scalar("ponder_times_decoder", tf.reduce_mean(ponder_times))
        # x.shape == (batch_size, target_seq_len, d_model)
        return [new_state, ponder_times, remainders]




class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_input=1 ,d_model=128, d_output=6,
                 max_iterations=6, num_heads=8, dff=2048,
                 rate=0.1, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.max_iterations = max_iterations
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

    def get_config(self):
        config = super(TransformerLayer, self).get_config()
        config['d_input'] = self.d_input
        config['d_model'] = self.d_model
        config['d_output'] = self.d_output
        config['max_iterations'] = self.max_iterations
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['rate'] = self.rate
        return config

    def build(self, input_shape):
        self.encoder = Encoder(self.d_input, self.d_model,
            self.max_iterations, self.num_heads, self.dff, rate=self.rate)
        self.decoder = Decoder(self.d_output, self.d_model,
            self.max_iterations, self.num_heads, self.dff, rate=self.rate)
        self.final_layer = tf.keras.layers.Dense(self.d_output)

    def create_masks(self, input_lengths, target_lengths, input_max, target_max):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(input_lengths, input_max)
        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(input_lengths, input_max)
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(target_max)
        dec_target_padding_mask = create_padding_mask(target_lengths, target_max)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    def call(self, x, training=True, mask=False):
        input, target, input_lengths, target_lengths = x
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
                input_lengths, target_lengths, input.shape[1], target.shape[1])
        # enc_output.shape == # (batch_size, inp_seq_len, d_model)
        enc_output, ponder_times_encoder, remainders_encoder = self.encoder(
            input, training=training, mask=enc_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, ponder_times_encoder, remainders_encoder = self.decoder(
            [target, enc_output, combined_mask, dec_padding_mask], training=training, mask=None)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, d_output)
        return final_output




class Transformer(tf.keras.Model):
    def __init__(self, d_input, d_model, d_output,
                max_input_length, max_target_length,
                max_iterations=2, num_heads=8, dff=2048,
                rate=0.1):
        super(Transformer, self).__init__()
        self.transformer_layer = TransformerLayer(d_input, d_model, d_output,
                                max_iterations, num_heads, dff, rate)

    def call(self, inputs, training):
        return self.transformer_layer(inputs, training=training)




if __name__ == '__main__':
    tf.InteractiveSession()
    d_input = 4096
    d_output = 6
    d_model = 128
    sig_len = 5000
    seq_len = 500
    max_timescale = 50
    sample_transformer = Transformer(d_input, d_model, d_output,
                                max_iterations=2, num_heads=8, dff=2048,)
    temp_input = tf.random.uniform((64, sig_len, 1))
    temp_target = tf.random.uniform((64, seq_len, 1))
    temp_input_len = tf.random.uniform((64, 1))
    temp_target_len = tf.random.uniform((64, 1))
    tf_out = sample_transformer(temp_input, temp_target, temp_input_len, temp_target_len, training=False)
    sample_transformer.summary()
