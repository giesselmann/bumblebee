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
import os, argparse, re, timeit
import time
import random
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from util import pore_model




def positional_encoding(seq_len, depth, d_model,
                        max_timescale=10000, random_shift=False):
    def get_angles(pos, j, max_timescale):
        angle_rates = 1 / np.power(max_timescale,
                                   (2 * (j//2)) / np.float32(d_model))
        return pos * angle_rates
    j = np.arange(d_model)[np.newaxis, np.newaxis, :]   # (1, 1, d_model)
    pos_rads = get_angles(np.arange(seq_len)[np.newaxis, :, np.newaxis], j, max_timescale)
    #depth_rads = get_angles(np.arange(depth)[:, np.newaxis, np.newaxis],
    #                        np.arange(d_model)[np.newaxis, np.newaxis, :])
    #depth_rads = get_angles(np.logspace(0, np.log10(max_timescale), depth)[:, np.newaxis, np.newaxis], j)
    depth_rads = get_angles(np.arange(depth)[:, np.newaxis, np.newaxis], j, depth)
    rads = pos_rads + depth_rads
    # apply sin to even indices in the array; 2i
    rads[:, 0::2, :] = np.sin(pos_rads[:, 0::2, :]) + np.sin(depth_rads[:, :, :])
    # apply cos to odd indices in the array; 2i+1
    rads[:, 1::2, :] = np.cos(pos_rads[:, 1::2, :]) + np.cos(depth_rads[:, :, :])
    pos_encoding = rads[np.newaxis, ...]
    return tf.constant(tf.cast(pos_encoding, dtype=tf.float32)) # (1, depth, seq_len, d_model)




def decode_sequence(logits, alphabet='ACGT'):
    return ''.join([alphabet[i] if i < len(alphabet) else '' for i in logits])





def flip_sequence(target, d_output, rate=0.1):
    # target.shape (batch_size, target_seq_len)
    flp_mask = tf.random.uniform(target.shape, 0.0, 1.0, dtype=tf.float32)
    flp_mask = tf.cast(tf.less(flp_mask, rate), tf.int32)
    flp_mask = tf.concat([tf.zeros((target.shape[0], 1), dtype=tf.int32), flp_mask[:,1:]],axis=1)  # do not flip SOS character
    flp_val = tf.random.uniform(target.shape, tf.cast(0, tf.int32), d_output - 2, dtype=tf.int32)
    flp_val *= flp_mask
    target = (target + flp_val) % (d_output - 2)
    return target




class WarmupLRS(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, offset=0):
        super(WarmupLRS, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.offset = offset

    def update_offset(self, update):
        self.offset += update

    def __call__(self, step):
        step = tf.maximum(tf.cast(0, step.dtype), step - tf.cast(self.offset, step.dtype))
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BumbleBee")
    parser.add_argument("records", help="TF record output prefix")
    parser.add_argument("--minibatch_size", type=int, default=100, help="TF records per file")
    parser.add_argument("--batches_train", type=int, default=10000, help="Training batches")
    parser.add_argument("--batches_val", type=int, default=1000, help="Validation batches")
    parser.add_argument("--input_length", type=int, default=1000, help="Input signal window")
    parser.add_argument("--target_length", type=int, default=100, help="Target sequence length")
    parser.add_argument("--model", help="Pore model for plots")
    args = parser.parse_args()

    # Constants
    alphabet = "ACGT"
    tf_alphabet = alphabet + '^$'
    input_max_len = args.input_length
    target_max_len = args.target_length

    if args.model:
        pm = pore_model(args.model)
    else:
        pm = None

    # tfRecord files
    record_files = [os.path.join(dirpath, f) for dirpath, _, files
                        in os.walk(args.records) for f in files if f.endswith('.tfrec')]

    val_split = int(min(1, args.batches_val / args.batches_train * len(record_files)))
    val_files = record_files[:val_split]
    train_files = record_files[val_split:]

    def encode_sequence(sequence):
        ids = {char:tf_alphabet.find(char) for char in tf_alphabet}
        ret = [ids[char] if char in ids else ids[random.choice(alphabet)] for char in '^' + sequence.numpy().decode('utf-8') + '$']
        return tf.cast(ret, tf.int32)

    @tf.function
    def tf_parse(eg):
        example = tf.io.parse_example(
            eg[tf.newaxis], {
                'sequence': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                'signal': tf.io.FixedLenFeature(shape=(), dtype=tf.string)})
        seq = tf.py_function(encode_sequence, [example['sequence'][0]], tf.int32)
        sig = tf.expand_dims(
                tf.cast(
                    tf.io.parse_tensor(example['signal'][0], tf.float16),
                    tf.float32),
                axis=-1)
        seq_len = tf.expand_dims(tf.size(seq), axis=-1) - 1
        sig_len = tf.expand_dims(tf.size(sig), axis=-1)
        return ((sig, seq[:-1], sig_len, seq_len), seq[1:])

    @tf.function
    def tf_filter(input, target):
        #input, target = eg
        return (input[2] <= tf.cast(input_max_len, tf.int32) and input[3] <= tf.cast(target_max_len + 2, tf.int32))[0]

    #ds_train = (tf.data.TFRecordDataset(filenames = train_files)
    #            .map(tf_parse, num_parallel_calls=16))
    #ds_train = ds_train.filter(tf_filter)
    #ds_train = (ds_train.prefetch(args.minibatch_size * 64)
    #            .shuffle(args.minibatch_size * 64)
    #            .padded_batch(args.minibatch_size,
    #                padded_shapes=(([input_max_len, 1], [target_max_len+2,], [1,], [1,]), [target_max_len+2,]),
    #                drop_remainder=True)
    #            .repeat())

    ds_train = tf.data.Dataset.from_tensor_slices(train_files)
    ds_train = (ds_train.interleave(lambda x:
                tf.data.TFRecordDataset(filenames=x).map(tf_parse, num_parallel_calls=4), cycle_length=4, block_length=16))
    ds_train = (ds_train.filter(tf_filter)
                .prefetch(args.minibatch_size * 32)
                .shuffle(args.minibatch_size * 64)
                .padded_batch(args.minibatch_size,
                    padded_shapes=(([input_max_len, 1], [target_max_len+2,], [1,], [1,]), [target_max_len+2,]),
                    drop_remainder=True))

    ds_train_iter = iter(ds_train)
    for batch in tqdm(ds_train_iter, desc='Training'):
        if pm:
            inputs, target = batch
            input, target, input_lengths, target_lengths = inputs
            sequence = decode_sequence(target[0][1:target_lengths[0].numpy()[0]])
            f, ax = plt.subplots(2)
            ax[0].plot(pm.generate_signal(sequence, samples=None), 'k-')
            ax[1].plot(input[0][:input_lengths[0].numpy()[0]].numpy(), 'b-')
            plt.show()
