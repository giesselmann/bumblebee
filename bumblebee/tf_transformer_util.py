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
import os, argparse, re
import random
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import pore_model




class BatchGenerator():
    def __init__(self, batches_train=100, batches_val=10,
                 minibatch_size=32,
                 max_input_len=1000, min_target_len=None, max_target_len=100, target_alphabet='ACGT'):
        self._batches_train = batches_train
        self._batches_val = batches_val
        self.minibatch_size = minibatch_size
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.min_target_len = min_target_len or max_target_len // 2
        self.target_alphabet = target_alphabet + '^$'
        self.current_train_index = 0
        self.current_val_index = batches_train * minibatch_size

    @property
    def target_dim(self):
        return len(self.target_alphabet)

    @property
    def batches_train(self):
        return self._batches_train

    @property
    def batches_val(self):
        return self._batches_val

    @property
    def val_split(self):
        return self.batches_train * self.minibatch_size

    @property
    def num_sequences(self):
        return self.val_split + self.batches_val * self.minibatch_size

    def __encode_sequence__(self, sequence):
        ids = {char:self.target_alphabet.find(char) for char in self.target_alphabet}
        ret = [ids[char] for char in '^' + sequence + '$']
        return ret

    def get_sequence_signal_pair(self, index):
        raise NotImplementedError

    def get_batch(self, index, size):
        input_data = np.zeros((size, self.max_input_len, 1), dtype=np.float32)
        target_data = np.zeros((size, self.max_target_len+3), dtype=np.int32)
        input_lens = np.zeros((size, 1), dtype=np.int32)
        target_lens = np.zeros((size, 1), dtype=np.int32)
        for i in range(size):
            sequence, signal = self.get_sequence_signal_pair(index + i)
            input_data[i,:min(len(signal), self.max_input_len),0] = signal[:self.max_input_len]
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
        super(BatchGeneratorSim, self).__init__(**kwargs)
        self.sequences_train = self.__gen_seqs__(super(BatchGeneratorSim, self).batches_train * self.minibatch_size,
                self.min_target_len, self.max_target_len)
        self.sequences_val = self.__gen_seqs__(super(BatchGeneratorSim, self).batches_val * self.minibatch_size,
                self.min_target_len, self.max_target_len)

    def __gen_seqs__(self, n, min_length, max_length):
        seqs = [''.join([random.choice('ACGT') for _
            in range(random.randint(min_length, max_length))]) for i
                in range(n)]
        return seqs

    @property
    def batches_train(self):
        return len(self.sequences_train) // self.minibatch_size

    @property
    def batches_val(self):
        return len(self.sequences_val) // self.minibatch_size

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
                 discard_quantile=0.3, **kwargs):
        self.pm = pore_model(pore_model_file)
        super(BatchGeneratorSig, self).__init__(**kwargs)
        self.event_file = event_file
        segments = []
        n_segments_train = super(BatchGeneratorSig, self).batches_train * self.minibatch_size
        n_segments_val = super(BatchGeneratorSig, self).batches_val * self.minibatch_size
        self.event_file = h5py.File(event_file, 'r')
        batches = self.event_file['batch']
        summary = self.event_file['summary'][...]
        logp_quantile = np.quantile(summary['logp'], discard_quantile)
        dist_quantile = np.quantile(summary['dist'], 1-discard_quantile)
        score_mask = np.logical_and(summary['logp'] > logp_quantile, summary['dist'] < dist_quantile)
        lengths = np.random.randint(self.min_target_len, self.max_target_len - 5, len(batches), dtype=np.uint64)
        with tqdm(total=n_segments_train + n_segments_val) as pbar:
            for batch in np.unique(batches):
                batch_mask = np.logical_and(np.equal(batches, batch), score_mask)
                batch_summary = summary[batch_mask]
                batch_lengths = lengths[batch_mask]
                batch_seq_begin = batch_summary['seq_begin'] + 10 # skip first events
                batch_seq_end = batch_seq_begin + batch_lengths
                batch_events = self.event_file['seq'][batch,:]
                batch_raw = self.event_file['raw'][batch,:]
                batch_raw_end = np.cumsum(batch_events['length'][...])
                batch_raw_begin = batch_raw_end - batch_events['length'][...]
                assert np.all(np.less_equal(batch_seq_end, batch_summary['seq_end']))
                batch_sequences = [batch_events['sequence'][begin:end].tostring().decode('utf-8')
                    for begin, end in zip(batch_seq_begin, batch_seq_end + 6)]
                batch_raw_segments = [(begin, end)
                    for begin, end in zip(batch_raw_begin[batch_seq_begin], batch_raw_end[batch_seq_end])]
                segments.extend([(batch, _seq, _begin, _end)
                    for _seq, (_begin, _end) in zip(batch_sequences, batch_raw_segments)
                        if _end - _begin <= self.max_input_len])
                pbar.update(len(batch_sequences))
                if len(segments) >= n_segments_train + n_segments_val:
                    break
        self.segments_train = segments[:n_segments_train]
        self.segments_val = segments[n_segments_train:n_segments_train + n_segments_val]

    def __del__(self):
        self.event_file.close()

    @property
    def batches_train(self):
        return len(self.segments_train) // self.minibatch_size

    @property
    def batches_val(self):
        return len(self.segments_val) // self.minibatch_size

    def get_sequence_signal_pair(self, index):
        if index < self.val_split:
            batch, sequence, begin, end = self.segments_train[index]
        else:
            batch, sequence, begin, end = self.segments_val[(index - self.val_split) % len(self.segments_val)]
        signal = self.event_file['raw'][batch,begin:end]
        nrm_signal = (signal - self.pm.model_median) / self.pm.model_MAD
        sequence = re.sub('N', lambda x : random.choice(self.target_alphabet[:-2]), sequence)
        return (sequence, nrm_signal)

    def on_epoch_begin(self):
        super(BatchGeneratorSig, self).on_epoch_begin()
        random.shuffle(self.segments_train)




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
    batch_gen = BatchGeneratorSig(args.model, args.event, batches_train=50000, batches_val=5000, max_target_len=100)
    batch_gen.on_epoch_begin()
    for i in tqdm(range(5000*32)):
        seq, sig = batch_gen.get_sequence_signal_pair(i)
