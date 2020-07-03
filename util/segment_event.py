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
# Copyright 2020 Pay Giesselmann, Max Planck Institute for Molecular Genetics
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
import os, sys, timeit
import argparse
import re, string, random
import h5py
import edlib
import tqdm
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage
from scipy import optimize
from skimage import filters
from skimage.morphology import opening, closing, dilation, erosion, rectangle

from matplotlib import pyplot as plt



class event_align():
    def __init__(self, model_file, fast5_file, fasta_file,
                    max_end_gap=0.1,
                    window=3, n_hist_bins=128,
                    edge_threshold=.4,
                    algn_alphabet=8, algn_alphabet_ext=2):  # 12/3 > 8/2
        self.max_end_gap = max_end_gap
        self.n_hist_bins = n_hist_bins
        self.edge_threshold = edge_threshold
        self.morph_filter = rectangle(1, window)
        self.edge_filter = np.array([-2, -1, 1, 2])
        self.alphabet = string.ascii_uppercase[:12]
        self.equalities = []
        for expansion in range(1, algn_alphabet_ext):
            self.equalities += [(self.alphabet[i], self.alphabet[i+expansion])
                    for i in range(len(self.alphabet) - expansion)]
        self.pore_model = pd.read_csv(model_file,
                sep='\t',
                header=None,
                names=['kmer', 'level_mean', 'level_stdv', 'weight']).set_index('kmer')
        self.fast5_file = fast5_file
        self.fasta_file = fasta_file
        self.seq_dict = None
        self.f5_records = None
        self.hist_cdf = None

    def mad_norm(self, x, clip=None):
        x_norm = (x - np.median(x)) / np.mean(np.absolute(x - np.mean(x)))
        if clip is not None:
            x_norm = np.clip(x_norm, -clip, clip)
        return x_norm

    def min_max_norm(self, x):
        q_min, q_max = np.quantile(x, (0.1, 0.9))
        m_min = np.median(x[x < q_min])
        m_max = np.median(x[x > q_max])
        range = (m_max - m_min) / 2
        offset = m_min + range
        x_norm = np.clip((x - offset) / range, -5, 5)
        return x_norm

    def morph(self, x):
        morph_signal = np.clip(x * 20 + 127, 0, 255).astype(np.dtype('uint8')).reshape((1, len(x)))
        morph_signal = opening(morph_signal, self.morph_filter)
        morph_signal = closing(morph_signal, self.morph_filter)[0].astype(np.dtype('float'))
        return self.min_max_norm(morph_signal)

    def hist_equal(self, x, train=False):
        if train:
            signal_histogram, bins = np.histogram(self.morph(x), self.n_hist_bins,
                    density=True, range=(-5, 5))
            cdf = signal_histogram.cumsum() # cumulative distribution function
            self.__hist_cdf__ = cdf / cdf[-1] # normalize
            self.__hist_bins__ = bins
        x_norm = np.interp(x, self.__hist_bins__[:-1], self.__hist_cdf__)
        x_norm -= 0.5
        return x_norm

    def skeletonize(self, x, max_iterations=10):
        str1 = np.array([0, 1, 1])
        str2 = np.array([1, 1, 0])
        for i in range(max_iterations):
            x1 = np.logical_and(x, ~ndimage.binary_hit_or_miss(x, structure1=str1))
            x2 = np.logical_and(x1, ~ndimage.binary_hit_or_miss(x1, structure1=str2))
            if np.sum(np.logical_xor(x2, x)) == 0:
                return x2
            x = x2
        return x2

    def split(self, x, w=20, max_iterations=10):
        str1 = np.array([1] + [0] * w)
        for i in range(max_iterations):
            x1 = np.logical_or(x, ndimage.binary_hit_or_miss(x, structure1=str1))
            if np.sum(np.logical_xor(x1, x)) == 0:
                return x1
            x = x1
        return x1

    def sig2char(self, x):
        ords = sorted([ord(x) for x in self.alphabet])
        quantiles = np.quantile(x, np.linspace(0,1,len(ords)))
        inds = np.digitize(x, quantiles).astype(np.int) - 1
        return ''.join([chr(ords[x]) for x in inds])

    #context
    def __enter__(self):
        # read all sequences into memory
        def seq_iter(fasta):
            with open(fasta, 'r') as fp:
                while True:
                    try:
                        name = next(fp).strip()
                        assert name[0] == '>'
                        name = name[1:]
                        seq = ''.join([c if c in 'ACTG' else random.choice('ACTG') for c in next(fp).strip().upper()])
                    except StopIteration:
                        return
                    yield name, seq
        self.seq_dict = {name:seq for name, seq in seq_iter(self.fasta_file)}
        # open fast5 and init histogram normalization on random subset
        self.f5_records = h5py.File(self.fast5_file, 'r')
        grps = random.sample(list(self.f5_records.values()),  min(len(self.f5_records), 50))
        raw = np.concatenate([self.min_max_norm(grp['Raw/Signal'][...])
                    for grp in grps]).astype(np.float32)
        _= self.hist_equal(raw, train=True)
        return self

    def __exit__(self, type, value, traceback):
        del self.seq_dict
        self.seq_dict = None
        self.f5_records.close()
        self.f5_records = None
        self.hist_cdf = None

    def __len__(self):
        if self.f5_records is not None:
            return len(self.f5_records)
        else:
            return 0

    #generator
    def events(self, min_sequence_length=200, max_event_count=20000):
        #assert self.seq_dict is not None
        assert self.f5_records is not None
        for grp in self.f5_records.values():
            read_signal = grp['Raw/Signal'][...]
            read_ID = grp['Raw'].attrs['read_id'].decode('utf-8')
            if read_ID in self.seq_dict and len(self.seq_dict[read_ID]) > min_sequence_length:
                sequence = self.seq_dict[read_ID]
            else:
                yield read_ID, False, None, None
                continue
            try:
                df = pd.DataFrame({'nrm_signal': self.hist_equal(self.min_max_norm(read_signal.astype(np.float32)))})
                t0 = timeit.default_timer()
                morph_signal = self.morph(df.nrm_signal.values)
                edge_signal = ndimage.filters.convolve1d(morph_signal, self.edge_filter)
                rising_edges = edge_signal > self.edge_threshold
                falling_edges = edge_signal < -self.edge_threshold
                rising_event = self.skeletonize(rising_edges)
                falling_event = self.skeletonize(falling_edges)
                event = np.logical_or(rising_event, falling_event)
                raw_events = np.sum(event)
                event = self.split(event)
                final_events = np.sum(event)
                t1 = timeit.default_timer()
                if np.sum(event) == 0 or np.sum(event) > max_event_count:
                    print("Failed event detection", file=sys.stderr)
                    #f, ax = plt.subplots(2, sharex=True)
                    #ax[0].plot(morph_signal, 'b')
                    #ax[0].vlines(np.nonzero(event), ymin=-1, ymax=1)
                    #ax[1].plot(edge_signal, 'b')
                    #ax[1].set_title("Events: {}".format(np.sum(event)))
                    #plt.show()
                    yield read_ID, False, None, None
                    continue
                df['event_id'] = np.cumsum(event)
                df_event = df.reset_index().groupby(by=['event_id']).agg(
                    event_median=('nrm_signal', 'median'),
                    event_mean=('nrm_signal', 'mean'),
                    event_std=('nrm_signal', 'std'),
                    #event_mad=('nrm_signal', 'mad'),
                    event_first=('nrm_signal', 'first'),
                    event_last=('nrm_signal', 'last'),
                    event_len=('nrm_signal', 'count')
                )
                t2 = timeit.default_timer()
                df_event['event_len'] = self.mad_norm(df_event['event_len'])
                sim_signal = np.array([self.pore_model.loc[sequence[i:i+6]].level_mean
                                            for i in range(len(sequence) - 5)])
                sim_signal = self.hist_equal(self.min_max_norm(sim_signal))
                sim_chars = self.sig2char(sim_signal)
                sig_chars = self.sig2char(df_event.event_median.values)
                algn = edlib.align(sim_chars, sig_chars,
                        mode='HW',
                        task='path',
                        additionalEqualities=self.equalities)
                ops = [(int(op[:-1]), op[-1]) for op in re.findall('(\d*\D)',algn['cigar'])]
                begin, end = algn['locations'][0]
                begin = begin or 0
                end = end or len(sig_chars)
                t3 = timeit.default_timer()
                #print("Times: edges: {:3f}, events: {:3f}, alignment: {:3f}".format(t1-t0, t2-t1, t3-t2), file=sys.stderr)
                #print("Raw: {}, Final: {} Seq: {}; Begin: {} - End: {} / {}".format(
                #        raw_events, final_events, len(sim_chars), begin, end, len(sig_chars)), file=sys.stderr)
                if (len(sig_chars) - (end - begin)) > (self.max_end_gap * len(sig_chars)):
                    status = False
                else:
                    status = True
                # len of alignment, step on matches/mismatches
                sim_idx = np.cumsum(np.array([True if op in '=XI' else False for n_ops, op in ops for _ in range(n_ops)])) - 1
                sim_msk = np.array([True if op in '=XD' else False for n_ops, op in ops for _ in range(n_ops)])
                df_event['event_offset'] = np.ones(df_event.shape[0], dtype=np.int32) * -1
                df_event.loc[begin:end, ('event_offset')] = sim_idx[sim_msk]
                df_event = df_event[df_event.event_offset != -1]
                sequence = sequence[df_event.event_offset.min():df_event.event_offset.max()+5]
                yield read_ID, status, sequence, df_event
            except Exception as e:
                print(e, file=sys.stderr)
                yield read_ID, False, None, None
                continue




class tf_record_writer():
    def __init__(self, record_file):
        self.record_file = record_file

    def __enter__(self):
        self.writer = tf.io.TFRecordWriter(self.record_file)
        return self

    def __exit__(self, type, value, traceback):
        self.writer.flush()
        self.writer.close()

    def __serialize_example__(self, sequence, event_table):
        sequence_b = tf.train.BytesList(value=[sequence.encode("ASCII")])
        int_features = {'event_len', 'event_offset'}
        feature = {col:tf.train.Feature(bytes_list=
                        tf.train.BytesList(value=[tf.io.serialize_tensor(event_table[col].values.astype(np.float16)).numpy()]))
                        if col not in int_features else
                   tf.train.Feature(bytes_list=
                        tf.train.BytesList(value=[tf.io.serialize_tensor(event_table[col].values.astype(np.int32)).numpy()]))
                    for col in event_table.columns}
        feature['sequence'] = tf.train.Feature(bytes_list=sequence_b)
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def write(self, sequence, event_table):
        example = self.__serialize_example__(sequence, event_table)
        self.writer.write(example)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BumbleBee Training Data v2")
    parser.add_argument("model", help="Pore model file")
    parser.add_argument("fast5", help="Raw nanopore fast5")
    parser.add_argument("fasta", help="Reference sequences")
    parser.add_argument("tfrec", help="Output tensorflow records file")
    parser.add_argument('--min_sequence_length', type=int, default=100, help="Minimum sequence length")
    parser.add_argument("--max_event_count", type=int, default=40000, help="Maximum signal length")
    parser.add_argument("--max_end_gap", type=float, default=0.1, help="Maximum signal overlap after alignment in percent")
    args = parser.parse_args()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    with event_align(args.model, args.fast5, args.fasta, max_end_gap=args.max_end_gap) as eva,\
         tf_record_writer(args.tfrec) as writer:
        print("Start event alignment for {} reads".format(len(eva)))
        passed = 0
        for read_ID, status, sequence, event_table in eva.events(args.min_sequence_length, args.max_event_count):
            #print(read_ID, status, file=sys.stderr)
            if status:
                writer.write(sequence, event_table)
                passed += 1
        print("Finished event alignment, {} reads passed".format(passed))
