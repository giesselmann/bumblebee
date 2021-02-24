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
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import deque
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt

from bumblebee.fast5 import Fast5Index
from bumblebee.alignment import AlignmentIndex
from bumblebee.signal import Read, ReadNormalizer




def main(args):
    # load draft pore model or generate random distributed
    if args.draft_model:
        pm = pd.read_csv(args.draft_model, sep='\t', header=None, names=['kmer', 'level_mean'], usecols=[0,1]).set_index('kmer')
        pm.level_mean = (pm.level_mean - pm.level_mean.min()) / (pm.level_mean.max() - pm.level_mean.min()) * 2 - 1
    else:
        # init random uniform model
        kmer = [''.join(c) for c in itertools.product('ACGT', repeat=6)]
        pm = pd.DataFrame({'kmer':kmer, 'level_mean':np.random.uniform(-0.1, 0.1, 4096)}).set_index('kmer')
    # create bam and fast5 iterator
    f5_idx = Fast5Index(args.fast5)
    algn_idx = AlignmentIndex(args.bam)
    # init normalizer
    norm = ReadNormalizer()
    # keep inital model
    pm_origin = pm.copy()
    def derive_model(draft_model, ref_span, read, alphabet_size=16):
        dist, df_events = read.event_alignment(ref_span, draft_model, alphabet_size)
        df_model = df_events.groupby('kmer').agg(level_mean=('event_median', 'mean'))
        ## debug plot
        #f, ax = plt.subplots(1, figsize=(10,5))
        #ax.step(df_events.event_id, df_events.event_median, 'r-', alpha=0.8)
        #event_model_mean = df_model.loc[df_events.kmer, 'level_mean']
        #ax.step(df_events.event_id, event_model_mean, 'b-', alpha=0.8)
        #ax.set_title("Dist: {:.4f}".format(dist))
        #plt.show()
        return dist, df_model
    lr = args.lr
    step = 0
    dist_buffer = deque()
    diff_buffer = deque()
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
                read = Read(f5_idx[ref_span.qname], norm, morph_events=True)
                algn_dist, pm_derived = derive_model(pm, ref_span, read)
                pm_diff = np.mean(np.abs(pm.loc[pm_derived.index, 'level_mean'] - pm_derived.level_mean.values))
                pm.loc[pm_derived.index, 'level_mean'] = (pm_derived.level_mean.values * lr) + (pm.loc[pm_derived.index, 'level_mean'] * (1-lr))
                pm_origin_diff_ = np.sum(np.abs(pm.level_mean - pm_origin.level_mean))
                step += 1
                lr *= (1. / (1. + args.decay * step / 10))
                dist_buffer.append(algn_dist)
                diff_buffer.append(pm_diff)
                if len(dist_buffer) > 200:
                    dist_buffer.popleft()
                    diff_buffer.popleft()
                pbar.update(1)
                pbar.set_postfix_str("Dist: {:.4f} Diff: {:.4f} Origin: {:.4f}".format(np.mean(dist_buffer), np.mean(diff_buffer), pm_origin_diff_))
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
        pm.to_csv(args.output_model + '.e{}'.format(i), sep='\t')
    #pm_origin['derived'] = pm.loc[pm_origin.index.values].level_mean.values
    pm.to_csv(args.output_model, sep='\t')




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
    parser.add_argument("--max_seq_length", default=2000, type=int)
    return parser
