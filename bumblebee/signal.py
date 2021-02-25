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
# Copyright 2021 Pay Giesselmann, Max Planck Institute for Molecular Genetics
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
import os, re, string
import random
import itertools
import edlib
import parasail
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import opening, closing, rectangle

from bumblebee.alignment import reverse_complement




class PoreModel():
    def __init__(self, file=None, rnd=None, norm=False, k=6):
        if file and os.path.isfile(file):
            value_iter = (line.split('\t', 3)[:2] for line in open(file, 'r').read().split('\n') if line)
            self.pm = {kmer:float(value) for kmer, value in value_iter}
        elif rnd:
            kmer = [''.join(c) for c in itertools.product('ACGT', repeat=k)]
            value = np.random.uniform(-rnd, rnd, len(kmer))
            self.pm = {k:v for k, v in zip(kmer, value)}
        else:
            self.pm = dict()
        if norm:
            values = list(self.pm.values())
            v_min = min(values)
            v_max = max(values)
            self.pm = {key:(value - v_min)/(v_max - v_min) * 2.2 - 1.1 for key, value in self.pm.items()}

    def __getattr__(self, name):
        return getattr(self.pm, name)

    def __getitem__(self, key):
        return self.pm.get(key) or 0.0

    def to_tsv(self, name, sep='\t'):
        with open(name, 'w') as fp:
            print('\n'.join(['\t'.join((key, str(value))) for key, value in self.pm.items()]), file=fp)




class ReadNormalizer():
    def __init__(self, norm_q=2.5e-2, bins=100, clip=1.2):
        assert clip >= 1.0
        self.norm_q = norm_q
        self.bins = bins
        self.clip = clip
        pm_hist, _ = np.histogram(np.random.normal(0, clip/3, 1000), bins, range=(-clip, clip), density=True)
        cdf = pm_hist.cumsum()
        self.cdf = cdf / cdf[-1] * 2 - 1

    def norm(self, x):
        q0, q1 = np.quantile(x, [self.norm_q, 1-self.norm_q])
        iqr = q1 - q0
        x_clip = np.clip(x, q0 - (self.clip-1)/2* iqr, q1 + (self.clip-1)/2* iqr)
        x_norm = (x_clip - q0) / iqr * 2
        # center to zero, quantile in range [-1, 1]
        return x_norm - 1

    def equalize(self, x):
        hist, bins = np.histogram(x, self.bins, range=(-self.clip, self.clip), density=True)
        x_norm = np.interp(x, bins[:-1], self.cdf)
        return x_norm




class Read():
    def __init__(self, fast5Record, normalizer, morph_events=False):
        self.fast5Record = fast5Record
        self.normalizer = normalizer
        self.norm_signal = normalizer.norm(fast5Record.raw)
        #self.eq_signal = normalizer.equalize(self.norm_signal)
        self.eq_signal = self.norm_signal
        self.morph_signal = self.__morph__(self.eq_signal)
        self.morph_events = morph_events

    def __morph__(self, x, w=3):
        flt = rectangle(1, w)
        morph_signal = np.clip(x * 100 + 127, 0, 255).astype(np.uint8).reshape((1, len(x)))
        morph_signal = opening(morph_signal, flt)
        morph_signal = closing(morph_signal, flt)[0].astype(np.float32)
        return (morph_signal - 127) / 100

    def __edges__(self, x, threshold=0.3):
        # f = np.array([-3, -1, 1, 3])
        f = np.array([0, 3, -3])
        x_f = ndimage.filters.convolve1d(self.morph_signal, f)
        x_f_rising = x_f > threshold
        x_f_falling = x_f < -threshold
        df_edge = pd.DataFrame({'edge_filter': x_f, 'rising_edges':x_f_rising, 'falling_edges':x_f_falling})
        # rising events on max filter response
        v0 = (df_edge.rising_edges.shift() != df_edge.rising_edges).cumsum()
        v1 = df_edge.groupby(v0, sort=False).edge_filter.transform('max')
        df_edge['rising_event'] = np.logical_and(df_edge['edge_filter'] == v1, df_edge.rising_edges)
        # falling event on min filter response
        v0 = (df_edge.falling_edges.shift() != df_edge.falling_edges).cumsum()
        v1 = df_edge.groupby(v0, sort=False).edge_filter.transform('min')
        df_edge['falling_event'] = np.logical_and(df_edge['edge_filter'] == v1, df_edge.falling_edges)
        df_edge['events'] = np.logical_or(df_edge['rising_event'], df_edge['falling_event'])
        df_edge['event_id'] = df_edge.events.cumsum()
        df = pd.DataFrame({'event_id':df_edge.event_id, 'signal':x})
        return df

    def __event_compression__(self, df):
        df_event = df.groupby(by='event_id', as_index=False).agg(
            event_min=('signal', 'min'),
            event_mean=('signal', 'mean'),
            event_median=('signal', 'median'),
            event_std=('signal', 'std'),
            event_max=('signal', 'max'),
            event_len=('signal', 'count')
        )
        return df_event

    def __sig2char__(self, x, alphabet):
        #ords = sorted([ord(x) for x in alphabet])
        #quantiles = np.quantile(x, np.linspace(0,1,len(ords)))
        quantiles = np.linspace(-self.normalizer.clip, self.normalizer.clip, len(alphabet))
        inds = np.digitize(x, quantiles).astype(np.int32) - 1
        #return ''.join([chr(ords[x]) for x in inds])
        return ''.join([alphabet[x] for x in inds])

    def __matrix__(self, alphabet):
        # parasail add * character to alphabet
        n = len(alphabet) + 1
        # init score matrix
        match_score = len(alphabet) // 8
        min_score = -match_score * 2
        scale = np.eye(n, k=0, dtype=int) * match_score
        for k in np.arange(1, len(alphabet)):
            scale += np.eye(n, k=k, dtype=int) * max((match_score - k), min_score)
            scale += np.eye(n, k=-k, dtype=int) * max((match_score - k), min_score)
        # write to score matrix
        matrix = parasail.matrix_create(alphabet, 0, 0)
        for i, row in enumerate(scale):
            for j, cell in enumerate(row):
                matrix[i,j] = cell
        return matrix

    def __event_align__(self, ref_signal, read_signal, alphabet_size=32):
        assert alphabet_size <= len(string.ascii_letters)
        alphabet = string.ascii_letters[:alphabet_size]
        ref_chars = self.__sig2char__(ref_signal, alphabet)
        read_chars = self.__sig2char__(read_signal, alphabet)
        gap_open = 2
        gap_extension = 6
        # query is genomic sequence span, reference is all read events
        result = parasail.sg_dx_trace_striped_32(ref_chars, read_chars, gap_open, gap_extension, self.__matrix__(alphabet))
        ref_idx = np.cumsum([c != '-' for c in result.traceback.query])
        ref_msk = np.array([c != '-' for c in result.traceback.ref])
        sim_pos = ref_idx[ref_msk] - 1
        return result.score / result.len_query, sim_pos
        # equalities = []
        # for expansion in range(1, 4):
        #     equalities += [(alphabet[i], alphabet[i+expansion]) for i in range(len(alphabet) - expansion)]
        # algn = edlib.align(ref_chars, read_chars,
        #     mode='HW',
        #     task='path',
        #     additionalEqualities=equalities)
        # ops = [(int(op[:-1]), op[-1]) for op in re.findall('(\d*\D)',algn['cigar'])]
        # begin, end = algn['locations'][0]
        # begin = begin or 0
        # end = end or len(read_chars)
        # # len of alignment, step on matches/mismatches
        # ref_idx = np.cumsum(np.array([True if op in '=XI' else False for n_ops, op in ops for _ in range(n_ops)])) - 1
        # sim_msk = np.array([True if op in '=XD' else False for n_ops, op in ops for _ in range(n_ops)])
        # ref_pos = np.ones(read_signal.shape, dtype=np.int32) * -1
        # ref_pos[begin:end+1] = ref_idx[sim_msk]
        # return algn['editDistance'] / len(ref_chars), ref_pos

    def edges(self):
        return self.__edges__(self.morph_signal if self.morph_events else self.norm_signal)

    def events(self):
        return self.__event_compression__(self.edges())

    def event_alignment(self, ref_span, pore_model, alphabet_size=32):
        df_events = self.events()
        read_seq = ref_span.seq if not ref_span.is_reverse else reverse_complement(ref_span.seq)
        read_seq_acgt = re.sub('N', lambda x: random.choice('ACGT'), read_seq)
        ref_signal = np.array([pore_model[read_seq_acgt[i:i+6]] for i in range(len(read_seq) - 5)])
        dist, ref_pos = self.__event_align__(ref_signal, df_events.event_median, alphabet_size)
        df_events['sequence_offset'] = ref_pos
        df_events = df_events[(df_events.sequence_offset != -1) & (df_events.sequence_offset != df_events.sequence_offset.max())]
        df_events['kmer'] = np.array([read_seq[i:i+6] for i in df_events.sequence_offset.astype(np.int32)])
        return dist, df_events
