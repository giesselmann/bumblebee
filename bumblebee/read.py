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
import parasail
import pkg_resources
import numpy as np
import pandas as pd
from scipy import ndimage

from bumblebee.poremodel import PoreModel



class Pattern():
    def __init__(self, pattern='CG', extension=6):
        pattern_template = r'(?<=[ACGT]{{{ext}}}){pattern}(?=[ACGT]{{{ext}}})'
        self._pattern_string = pattern_template.format(ext=extension, pattern=pattern)
        self._ext = extension

    @property
    def pattern_string(self):
        return self._pattern_string

    @property
    def extension(self):
        return self._ext




class ReadNormalizer():
    def __init__(self, norm_q=2.5e-2, bins=100, clip=1.2, edge_threshold=0.3):
        assert clip >= 1.0
        self.norm_q = norm_q
        self.bins = bins
        self.clip = clip
        self.edge_filter = np.array([0, 3, -3])
        self.edge_threshold = edge_threshold
        pm_hist, _ = np.histogram(np.random.normal(0, clip/3, 1000), bins, range=(-clip, clip), density=True)
        cdf = pm_hist.cumsum()
        self.cdf = cdf / cdf[-1] * 2 - 1

    def __morph__(self, x_norm, w=3):
        x_morph = ndimage.grey_opening(x_norm, size=w)
        x_morph = ndimage.grey_closing(x_morph, size=w)
        return x_morph

    def __edges__(self, x):
        x_f = ndimage.filters.convolve1d(self.__morph__(x), self.edge_filter)
        x_f_rising = x_f > self.edge_threshold
        x_f_falling = x_f < -self.edge_threshold
        df_edge = pd.DataFrame({'edge_filter': x_f, 'rising_edges':x_f_rising, 'falling_edges':x_f_falling})
        # rising events on max filter response
        v0 = (df_edge.rising_edges.shift() != df_edge.rising_edges).cumsum()
        v1 = df_edge.groupby(v0, sort=False).edge_filter.transform('max')
        df_edge['rising_event'] = np.logical_and(df_edge['edge_filter'] == v1, df_edge.rising_edges)
        # falling event on min filter response
        v0 = (df_edge.falling_edges.shift() != df_edge.falling_edges).cumsum()
        v1 = df_edge.groupby(v0, sort=False).edge_filter.transform('min')
        df_edge['falling_event'] = np.logical_and(df_edge['edge_filter'] == v1, df_edge.falling_edges)
        # combine and enumerate events
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
            event_ticks=('signal', 'count')
        )
        return df_event

    def norm(self, x):
        q0, q1 = np.quantile(x, [self.norm_q, 1-self.norm_q])
        iqr = q1 - q0
        x_clip = np.clip(x, q0 - (self.clip-1) * iqr/2, q1 + (self.clip-1) * iqr/2)
        # project to [0, 1] range
        x_norm = (x_clip - q0) / iqr
        # center to [-1, 1] range
        x_norm = x_norm * 2 - 1
        return x_norm

    def equalize(self, x_norm):
        hist, bins = np.histogram(x_norm, self.bins, range=(-self.clip, self.clip), density=True)
        x_eq = np.interp(x_norm, bins[:-1], self.cdf)
        return x_eq

    def events(self, read):
        df_edges = self.__edges__(self.norm(read.raw_signal))
        df_events = self.__event_compression__(df_edges)
        return df_events




class ReadAligner():
    def __init__(self, normalizer,
                 alphabet_size=32,
                 max_event_len=40):
        self.normalizer = normalizer
        self.alphabet_size = alphabet_size
        self.max_event_len = max_event_len
        assert alphabet_size <= len(string.ascii_letters)
        self.alphabet = string.ascii_letters[:alphabet_size]
        self.pm = PoreModel(pkg_resources.resource_filename('bumblebee',
            'data/r9.4_450bps.model'))
        self.__init_matrix__(self.alphabet)

    def __init_matrix__(self, alphabet):
        # parasail adds * character to alphabet
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
        self.matrix = matrix

    def __sig2char__(self, x, alphabet):
        quantiles = np.linspace(-self.normalizer.clip, self.normalizer.clip, len(self.alphabet))
        inds = np.digitize(x, quantiles).astype(np.int32) - 1
        return ''.join([self.alphabet[x] for x in inds])

    def __event_align__(self, ref_signal, read_signal):
        ref_chars = self.__sig2char__(ref_signal, self.alphabet)
        read_chars = self.__sig2char__(read_signal, self.alphabet)
        gap_open = 2
        gap_extension = 6
        # query is genomic sequence span, reference is all read events
        result = parasail.sg_dx_trace_striped_32(ref_chars, read_chars, gap_open, gap_extension, self.matrix)
        ref_idx = np.cumsum([c != '-' for c in result.traceback.query])
        ref_msk = np.array([c != '-' for c in result.traceback.ref])
        sim_pos = ref_idx[ref_msk] - 1
        return result.score / result.len_query, sim_pos

    def event_alignments(self, read):
        df_events = self.normalizer.events(read)
        ref_seq = read.ref_span.seq
        ref_signal = self.pm.signal(ref_seq)
        score, ref_pos = self.__event_align__(ref_signal, df_events.event_median)
        df_events['sequence_offset'] = ref_pos
        df_events = df_events[(df_events.sequence_offset != -1) & (df_events.sequence_offset != df_events.sequence_offset.max())]
        kmer = [ref_seq[i:i+self.pm.k] for i in
            df_events.sequence_offset.astype(np.int32)]
        df_events['kmer'] = [self.pm.idx(k) for k in kmer]
        # clip and normalize event lengths
        df_events['event_length'] = np.clip(df_events.event_ticks,
            0, self.max_event_len) / self.max_event_len
        df_events.set_index('sequence_offset', inplace=True, drop=False)
        return score, df_events




class Read():
    def __init__(self, read_signal, ref_span=None):
        self.name = read_signal.name
        self.raw_signal = read_signal.raw
        self.ref_span = ref_span

    def events(self, read_normalizer):
        df_events = read_normalizer.events(self)
        return df_events

    def event_alignments(self, read_aligner):
        assert self.ref_span is not None
        score, df_events = read_aligner.event_alignments(self)
        return score, df_events

    def feature_sites(self, df_events, pattern, k):
        assert self.ref_span is not None
        # mapped part of read sequence
        valid_offset = df_events.sequence_offset.min()
        valid_sequence = self.ref_span.seq[valid_offset : df_events.sequence_offset.max() + 1]
        ref_span_len = len(self.ref_span.seq)
        # iterate over pattern positions and write features
        for match in re.finditer(pattern.pattern_string, valid_sequence):
            match_begin = match.start()
            match_end = match.end()
            feature_begin = match_begin - pattern.extension + valid_offset
            df_feature = df_events.loc[feature_begin : match_end + valid_offset + pattern.extension - k]
            # reference position on template strand
            if not self.ref_span.is_reverse:
                template_begin = self.ref_span.pos + match_begin + valid_offset
            else:
                template_begin = self.ref_span.pos + ref_span_len - match_end - valid_offset
            yield template_begin, feature_begin, df_feature
