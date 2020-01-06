# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : tf 2.0 input pipelines
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
import random
import tensorflow as tf




class tf_data_basecalling():
    def __init__(self, alphabet='ACGT',
                input_min_len = 500, input_max_len = 1000,
                target_min_len = 50, target_max_len = 100,
                use_sos=True, use_eos=True):
        self.alphabet = alphabet
        self.ext_alphabet = alphabet
        self.use_sos = use_sos
        self.use_eos = use_eos
        self.ext_alphabet += '^' if use_sos else ''
        self.ext_alphabet += '$' if use_eos else ''
        self.input_min_len = input_min_len
        self.input_max_len = input_max_len
        self.target_min_len = target_min_len
        self.target_max_len = target_max_len - (len(self.ext_alphabet) - len(alphabet))
        self.ids = {char:self.ext_alphabet.find(char) for char in self.ext_alphabet}

    def encode_sequence(self, sequence):
        sequence_decoded = sequence.numpy().decode('utf-8')
        sequence_decoded = '^' + sequence_decoded if self.use_sos else sequence_decoded
        sequence_decoded = sequence_decoded + '$' if self.use_eos else sequence_decoded
        ret = [self.ids[char] if char in self.ids else self.ids[random.choice(self.alphabet)] for char in sequence_decoded]
        return tf.cast(ret, tf.int32)

    def tf_parse(self, eg):
        example = tf.io.parse_example(
            eg[tf.newaxis], {
                'sequence': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                'signal': tf.io.FixedLenFeature(shape=(), dtype=tf.string)})
        seq = tf.py_function(self.encode_sequence, [example['sequence'][0]], tf.int32)
        sig = tf.expand_dims(
                tf.cast(
                    tf.io.parse_tensor(example['signal'][0], tf.float16),
                    tf.float32),
                axis=-1)
        seq_len = tf.cast(tf.expand_dims(tf.size(seq), axis=-1), tf.int32)
        sig_len = tf.cast(tf.expand_dims(tf.size(sig), axis=-1), tf.int32)
        return ((sig, sig_len), (seq, seq_len))

    def tf_filter(self, input, target):
        #input, target = eg
        return (input[1] < tf.cast(self.input_max_len, tf.int32) and
                input[1] >= tf.cast(self.input_min_len, tf.int32) and
                target[1] < tf.cast(self.target_max_len, tf.int32) and
                target[1] >= tf.cast(self.target_min_len, tf.int32))[0]

    def get_ds_train(self, train_files, minibatch_size=64):
        ds_train = tf.data.Dataset.from_tensor_slices(train_files)
        ds_train = (ds_train.interleave(lambda x:
                    tf.data.TFRecordDataset(filenames=x).map(self.tf_parse, num_parallel_calls=1), cycle_length=8, block_length=8))
        ds_train = (ds_train
                    .filter(self.tf_filter)
                    .prefetch(minibatch_size * 64)
                    .shuffle(minibatch_size * 512) # 1024
                    .padded_batch(minibatch_size,
                        padded_shapes=(([self.input_max_len, 1], [1,]), ([self.target_max_len,], [1,])),
                        drop_remainder=True)
                    .repeat())
        return ds_train

    def get_ds_test(self, test_files, minibatch_size=64):
        ds_test = tf.data.Dataset.from_tensor_slices(test_files)
        ds_test = (ds_test.interleave(lambda x:
                    tf.data.TFRecordDataset(filenames=x).map(self.tf_parse, num_parallel_calls=1), cycle_length=8, block_length=8))
        ds_test = (ds_test
                    .filter(self.tf_filter)
                    .prefetch(minibatch_size * 32)
                    .padded_batch(minibatch_size,
                        padded_shapes=(([self.input_max_len, 1], [1,]), ([self.target_max_len,], [1,])),
                        drop_remainder=True)
                    .repeat())
        return ds_test




if __name__ == "__main__":
    import os, argparse
    import matplotlib.pyplot as plt
    from util import pore_model
    from tf_util import decode_sequence
    parser = argparse.ArgumentParser(description="tf_data")
    parser.add_argument("records", help="TF record output prefix")
    parser.add_argument("--minibatch_size", type=int, default=64, help="TF records per file")
    parser.add_argument("--batches_train", type=int, default=10000, help="Training batches")
    parser.add_argument("--batches_val", type=int, default=1000, help="Validation batches")
    parser.add_argument("--model", help="Pore model for plots")
    args = parser.parse_args()

    if args.model:
        pm = pore_model(args.model)
    else:
        pm = None

    # tfRecord files
    record_files = [os.path.join(dirpath, f) for dirpath, _, files
                        in os.walk(args.records) for f in files if f.endswith('.tfrec')]

    val_rate = args.batches_train // args.batches_val
    val_split = int(max(1, args.batches_val / args.batches_train * len(record_files)))
    test_files = record_files[:val_split]
    train_files = record_files[val_split:]

    print("Training files {}".format(len(train_files)))
    print("Test files {}".format(len(test_files)))

    tf_data = tf_data_basecalling(alphabet='ACGT', use_sos=False, use_eos=False,
                        input_min_len=500, input_max_len=1000,
                        target_min_len=50, target_max_len=100)
    tf_data_train = tf_data.get_ds_train(train_files, minibatch_size=args.minibatch_size)
    tf_data_test = tf_data.get_ds_test(test_files, minibatch_size=args.minibatch_size)

    tf_data_train_iter = iter(tf_data_train)
    tf_data_test_iter = iter(tf_data_test)

    for batch in tf_data_train_iter:
        if pm:
            input, target = batch
            input_seq, input_len = input
            target_seq, target_len = target
            sequence = decode_sequence(target_seq[0][:target_len[0].numpy()[0]])
            f, ax = plt.subplots(2)
            ax[0].plot(pm.generate_signal(sequence, samples=None), 'k-')
            ax[1].plot(input_seq[0][:input_len[0].numpy()[0]].numpy(), 'b-')
            plt.show()
