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
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt

from bumblebee.fast5 import Fast5Index
from bumblebee.alignment import AlignmentIndex
from bumblebee.poremodel import PoreModel
from bumblebee.signal import Read, ReadNormalizer




def main(args):
    # load draft pore model or generate random distributed
    if args.draft_model:
        pm = PoreModel(args.draft_model, norm=True)
    else:
        # init random uniform model
        pm = PoreModel(rnd=True, k=6)
    # create bam and fast5 iterator
    f5_idx = Fast5Index(args.fast5)
    algn_idx = AlignmentIndex(args.bam)
    # init normalizer
    norm = ReadNormalizer()
    # new model from event table
    def derive_model(draft_model, ref_span, read, alphabet_size=32):
        score, df_events = read.event_alignment(ref_span, draft_model, alphabet_size)
        match_ratio = np.sum(np.diff(df_events.sequence_offset) == 1) / df_events.shape[0]
        df_model = df_events.groupby('kmer').agg(level_mean=('event_median', 'median'))
        # drop 'Ns'
        df_model = df_model[df_model.index.isin(draft_model.keys())]
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
    kmer_collector = defaultdict(lambda: [])
    kmer_counter = defaultdict(int)
    num_kmers = 4**pm.k
    coverage_sum = 0
    with tqdm.tqdm(desc='Event alignment', total=num_kmers*args.min_kmer_coverage, dynamic_ncols=True) as pbar:
        for ref_span in algn_idx.records():
            if len(ref_span.seq) < 500 or len(ref_span.seq) > args.max_seq_length:
                continue
            # event alignment
            read = Read(f5_idx[ref_span.qname], norm, morph_events=True)
            algn_score, match_ratio, pm_derived = derive_model(pm, ref_span, read)
            if algn_score < args.min_score:
                for kmer, value in pm_derived.items():
                    kmer_collector[kmer].append(value)
                    kmer_counter[kmer] += 1
            # check progress
            kmer_coverage = [min(value, args.min_kmer_coverage) for value in kmer_counter.values()]
            kmer_complete = sum([x == args.min_kmer_coverage for x in kmer_coverage])
            if kmer_complete == num_kmers:
                break
            coverage_sum_ = sum(kmer_coverage)
            pbar.update(coverage_sum_ - coverage_sum)
            coverage_sum = coverage_sum_
    pm_derived = PoreModel()
    pm_derived.update({kmer:stats.mode(values)[0][0] for kmer, values in kmer_collector.items()})
    pm_derived.to_tsv(args.output_model)




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False)
    parser.add_argument("output_model", type=str)
    parser.add_argument("fast5", type=str)
    parser.add_argument("bam", type=str)
    parser.add_argument("--draft_model", type=str)
    parser.add_argument("--min_score", default=1.6, type=float)
    parser.add_argument("--min_kmer_coverage", default=1000, type=int)
    parser.add_argument("--max_seq_length", default=5000, type=int)
    return parser
