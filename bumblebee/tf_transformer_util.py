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
import os, argparse
import random
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import pore_model




class BatchGenerator():
    def __init__(self, pore_model, batches_train=100, batches_val=10,
                 minibatch_size=32,
                 input_len=1000, target_len=100, target_alphabet='ACGT'):
        self.pm = pore_model
        self.batches_train = batches_train
        self.batches_val = batches_val
        self.minibatch_size = minibatch_size
        self.input_len = input_len
        self.target_len = target_len
        self.target_alphabet = target_alphabet + '^$'
        self.val_split = batches_train * minibatch_size
        self.num_sequences = self.val_split + batches_val * minibatch_size
        self.current_train_index = 0
        self.current_val_index = self.val_split

    @property
    def target_dim(self):
        return len(self.target_alphabet)

    def __encode_sequence__(self, sequence):
        ids = {char:self.target_alphabet.find(char) for char in self.target_alphabet}
        ret = [ids[char] for char in '^' + sequence + '$']
        return ret

    def get_sequence_signal_pair(self, index):
        raise NotImplementedError

    def get_batch(self, index, size):
        input_data = np.zeros((size, self.input_len, 1), dtype=np.float32)
        target_data = np.zeros((size, self.target_len+3), dtype=np.int32)
        input_lens = np.zeros((size, 1), dtype=np.int32)
        target_lens = np.zeros((size, 1), dtype=np.int32)
        for i in range(size):
            sequence, signal = self.get_sequence_signal_pair(index + i)
            input_data[i,:len(signal),0] = signal
            target_data[i,:len(sequence)+2] = self.__encode_sequence__(sequence)
            input_lens[i] = len(signal)
            target_lens[i] = len(sequence) + 2
        return (input_data, target_data, input_lens, target_lens)

    def __next_train__(self):
        ret = self.get_batch(self.current_train_index, self.minibatch_size)
        self.current_train_index += self.minibatch_size
        if self.current_train_index >= self.val_split:
            self.current_train_index = self.current_train_index % self.minibatch_size
        return ret

    def __next_val__(self):
        ret = self.get_batch(self.current_val_index, self.minibatch_size)
        self.current_val_index += self.minibatch_size
        if self.current_val_index >= self.num_sequences:
            self.current_val_index = self.val_split + self.current_val_index % self.minibatch_size
        return ret

    def next_train(self):
        while True:
            yield self.__next_train__()

    def next_val(self):
        while True:
            yield self.__next_val__()

    def on_epoch_begin(self):
        pass




class BatchGeneratorSim(BatchGenerator):
    def __init__(self, pore_model_file, **kwargs):
        self.pm = pore_model(pore_model_file)
        super(BatchGeneratorSim, self).__init__(self.pm, **kwargs)
        self.sequences_train = self.__gen_seqs__(self.batches_train * self.minibatch_size, self.target_len)
        self.sequences_val = self.__gen_seqs__(self.batches_val * self.minibatch_size, self.target_len)

    def __gen_seqs__(self, n, length):
        seqs = [''.join([random.choice('ACGT') for _
            in range(random.randint(length / 5, length))]) for i
                in range(n)]
        return seqs

    def get_sequence_signal_pair(self, index):
        if index < self.val_split:
            sequence = self.sequences_train[index]
        else:
            sequence = self.sequences_val[(index - self.val_split) % len(self.sequences_val)]
        sim_signal = self.pm.generate_signal(sequence, samples=None, noise=True)
        nrm_signal = self.pm.quantile_nrm(sim_signal)
        return (sequence, nrm_signal)

    def on_epoch_begin(self):
        super(BatchGeneratorSim, self).on_epoch_begin()
        random.shuffle(self.sequences_train)




class BatchGeneratorSig(BatchGenerator):
    def __init__(self, pore_model_file, event_file,
                discard_quantile=0.8, **kwargs):
        self.pm = pore_model(pore_model_file)
        super(BatchGeneratorSig, self).__init__(self.pm, **kwargs)
        self.event_file = event_file
        with h5py.File(event_file, 'r') as fp_event:
            summary = fp_event['summary']
            logp_quantile = np.quantile(summary['logp'], 1-discard_quantile)
            dist_quantile = np.quantile(summary['dist'], discard_quantile)
            summary_mask = np.logical_and(summary['logp'] > logp_quantile, summary['dist'] < dist_quantile)
            signals = [i for i, (row, mask) in tqdm(enumerate(zip(summary, summary_mask)), total=len(summary_mask)) if mask]

            print(len(signals))
            f, ax = plt.subplots(2)
            ax[0].hist(summary['logp'], bins=100, range=(-10,0))
            ax[0].axvline(logp_quantile)
            ax[1].hist(summary['dist'], bins=100, range=(0,10))
            ax[1].axvline(dist_quantile)
            plt.show()



class TransformerLRS(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(TransformerLRS, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BumbleBee")
    parser.add_argument("model", help="Pore model")
    parser.add_argument("event", help="Event table")
    args = parser.parse_args()
    batch_gen = BatchGeneratorSig(args.model, args.event)
