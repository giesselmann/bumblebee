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
import random
import tqdm
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt

from bumblebee.fast5 import Fast5Index
from bumblebee.alignment import AlignmentIndex
from bumblebee.signal import PoreModel, Read, ReadNormalizer




def main(args):
    # load draft pore model or generate random distributed
    if args.draft_model:
        pm = PoreModel(args.draft_model, norm=True)
    else:
        # init random uniform model
        pm = PoreModel(rnd=True)
    # create bam and fast5 iterator
    f5_idx = Fast5Index(args.fast5)
    algn_idx = AlignmentIndex(args.bam)
    # init normalizer
    norm = ReadNormalizer()
    # keep inital model
    pm_origin = pm.copy()
    # compare sum of absolute model differences
    def model_diff(a, b, func=np.mean):
        intersect_keys = set(a.keys()).intersection(b.keys())
        return func(np.abs([a[k] - b[k] for k in intersect_keys]))
    # new model from event table
    def derive_model(draft_model, ref_span, read, alphabet_size=32):
        score, df_events = read.event_alignment(ref_span, draft_model, alphabet_size)
        match_ratio = np.sum(np.diff(df_events.sequence_offset) == 1) / df_events.shape[0]
        df_model = df_events.groupby('kmer').agg(level_mean=('event_median', 'mean'))
        # drop 'Ns'
        df_model = df_model[df_model.index.isin(pm_origin.keys())]
        derived_model = PoreModel()
        derived_model.update(df_model.itertuples())
        ## debug plot
        #f, ax = plt.subplots(1, figsize=(20,5))
        #event_model_mean = np.array([draft_model[k] for k in df_events.kmer])
        #ax.step(df_events.event_id, event_model_mean, 'b-', alpha=0.8, label='reference')
        #ax.step(df_events.event_id, df_events.event_median, 'r-', alpha=0.8, label='read')
        #ax.legend()
        #ax.set_title("Score: {:.4f}".format(score))
        #plt.show()
        return score, match_ratio, derived_model
    lr = args.lr
    step = 0
    dist_buffer = deque()
    diff_buffer = deque()
    ratio_buffer = deque()
    pm_origin_diff = 0
    eps_break_count = 0
    ref_span_cache = []
    for i in range(args.epochs):
        with tqdm.tqdm(desc='Epoch {}'.format(i), postfix='') as pbar:
            ref_span_iter = algn_idx.records() if i == 0 else (r for r in ref_span_cache)
            for ref_span in ref_span_iter:
                if len(ref_span.seq) < 500 or len(ref_span.seq) > args.max_seq_length:
                    pbar.update(1)
                    continue
                if i == 0:
                    ref_span_cache.append(ref_span)
                # event alignment
                read = Read(f5_idx[ref_span.qname], norm, morph_events=True)
                algn_score, match_ratio, pm_derived = derive_model(pm, ref_span, read)
                # update and compare
                pm_diff = model_diff(pm_derived, pm, func=np.mean)
                if algn_score > args.threshold:
                    pm.update({key:(value * lr + pm[key] * (1-lr)) for key, value in pm_derived.items()})
                pm_origin_diff_ = model_diff(pm, pm_origin, func=np.sum)
                step += 1
                lr = args.lr * (1. / (1. + args.decay * step / 10))
                # running buffer of differences
                dist_buffer.append(algn_score)
                diff_buffer.append(pm_diff)
                ratio_buffer.append(match_ratio)
                if len(dist_buffer) > 200:
                    dist_buffer.popleft()
                    diff_buffer.popleft()
                    ratio_buffer.popleft()
                pbar.update(1)
                pbar.set_postfix_str("Score: {:.4f} Kmer.diff: {:.4f} Matches: {:.4f} Pm.diff: {:.4f}".format(np.mean(dist_buffer), np.mean(diff_buffer), np.mean(ratio_buffer), pm_origin_diff_))
                # stop iteration after no changes for 100 reads
                if abs(pm_origin_diff - pm_origin_diff_) < args.eps:
                    eps_break_count += 1
                    if eps_break_count > 100:
                        break
                else:
                    eps_break_count = 0
                pm_origin_diff = pm_origin_diff_
        if eps_break_count > 100:
            break
        random.shuffle(ref_span_cache)
        # save checkpoint model
        pm.to_tsv(args.output_model + '.e{}'.format(i))
    #pm_origin['derived'] = pm.loc[pm_origin.index.values].level_mean.values
    pm.to_tsv(args.output_model)




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False)
    parser.add_argument("output_model", type=str)
    parser.add_argument("fast5", type=str)
    parser.add_argument("bam", type=str)
    parser.add_argument("--draft_model", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--decay", default=0.001, type=float)
    parser.add_argument("--eps", default=0.0001, type=float)
    parser.add_argument("--threshold", default=1.6, type=float)
    parser.add_argument("--max_seq_length", default=2000, type=int)
    return parser
