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
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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
        pm = pd.DataFrame({'kmer':kmer, 'level_mean':np.random.uniform(-0.5, 0.5, 4096)}).set_index('kmer')
    # create bam and fast5 iterator
    f5_idx = Fast5Index(args.fast5)
    algn_idx = AlignmentIndex(args.bam)
    # init normalizer
    norm = ReadNormalizer()
    # sample some ref spans
    ref_spans = [span for span in tqdm.tqdm(algn_idx.records(), desc='Loading alignments') if len(span.seq) > args.min_seq_length and len(span.seq) < args.max_seq_length]
    random.seed(42)
    ref_spans = random.sample(ref_spans, min(args.random_sample, len(ref_spans)))
    print(len(ref_spans))
    reads = [Read(f5_idx[ref_span.qname], norm) for ref_span in ref_spans]
    # keep inital model
    pm_origin = pm.copy()
    alphabet_sizes = np.repeat([8, 10, 12, 14, 16], args.max_iterations // 5)
    # iterate until convergence or max_iterations
    def derive_model(draft_model, ref_spans, reads, alphabet_size=16):
        dists, events = zip(*[read.event_alignment(ref_span, draft_model, alphabet_size) for ref_span, read in tqdm.tqdm(zip(ref_spans, reads), desc='Aligning')])
        df_events = pd.concat(events)
        df_model = df_events.groupby('kmer').agg(level_mean=('event_median', 'mean'))
        return np.mean(dists), df_model
    for i, alphabet_size in enumerate(alphabet_sizes):
        algn_dist, pm_derived = derive_model(pm, ref_spans, reads, alphabet_size=alphabet_size)
        model_dist = np.mean(np.absolute(pm.loc[pm_derived.index, 'level_mean'] - pm_derived.level_mean))
        pm.loc[pm_derived.index, 'level_mean'] =  pm_derived.level_mean.values
        print("Step {}: Alignment: {:.4f} Model: {:.4}".format(i, algn_dist, model_dist))
        if model_dist < args.eps:
            break
    #pm_origin['derived'] = pm.loc[pm_origin.index.values].level_mean.values
    pm.to_csv(args.output_model, sep='\t')




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("output_model", type=str)
    parser.add_argument("fast5", type=str)
    parser.add_argument("bam", type=str)
    parser.add_argument("--draft_model", type=str)
    parser.add_argument("--random_sample", default=200, type=int)
    parser.add_argument("--eps", default=0.001, type=float)
    parser.add_argument("--max_iterations", default=20, type=int)
    parser.add_argument("--min_seq_length", default=1000, type=int)
    parser.add_argument("--max_seq_length", default=5000, type=int)
    return parser
