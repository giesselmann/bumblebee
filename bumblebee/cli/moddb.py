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
import os, re
import tqdm
import pkg_resources
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bumblebee.poremodel import PoreModel
from bumblebee.db import ModDatabase
from bumblebee.fast5 import Fast5Index
from bumblebee.alignment import AlignmentIndex
from bumblebee.poremodel import PoreModel
from bumblebee.signal import Read, ReadNormalizer




def main(args):
    # end debug
    pore_model = PoreModel(pkg_resources.resource_filename('bumblebee', 'data/r9.4_450bps.model'))
    # enumerate kmers, reserve 0 for kmers with 'N'
    kmer_idx = {kmer:i+1 for i, kmer in enumerate(sorted(list(pore_model.keys())))}
    # open/init database
    db = ModDatabase(args.db)
    # create bam and fast5 iterator
    f5_idx = Fast5Index(args.fast5)
    algn_idx = AlignmentIndex(args.bam)
    # read signal normalization
    norm = ReadNormalizer()
    pattern = r'(?<=[ACGT]{{{ext}}}){pattern}(?=[ACGT]{{{ext}}})'.format(ext=args.pattern_extension, pattern=args.pattern)
    with tqdm.tqdm(desc='Event align', dynamic_ncols=True, total=len(f5_idx)) as pbar:
        for i, ref_span in enumerate(algn_idx.records()):
            if len(ref_span.seq) < args.min_seq_length or len(ref_span.seq) > args.max_seq_length:
                continue
            read = Read(f5_idx[ref_span.qname], norm)
            score, df_events = read.event_alignment(ref_span, pore_model)
            # event_id  event_min  event_mean  event_median  event_std  event_max  event_len  sequence_offset    kmer
            if score < args.min_score:
                continue
            df_events['kmer'] = df_events.kmer.apply(lambda x: kmer_idx.get(x) or 0)
            # clip and normalize event lengths
            df_events['event_length'] = np.clip(df_events.event_len, 0, 40) / 40
            # mapped part of read sequence
            valid_offset = df_events.sequence_offset.min()
            valid_sequence = ref_span.seq[valid_offset : df_events.sequence_offset.max() + 1]
            ref_span_len = len(ref_span.seq)
            df_events.set_index('sequence_offset', inplace=True)
            # write read_record
            db_read_id = db.insert_read(ref_span, score=score)
            # iterate over pattern positions and write features
            for match in re.finditer(pattern, valid_sequence):
                match_begin = match.start()
                match_end = match.end()
                feature_begin = match_begin - args.pattern_extension + valid_offset
                df_feature = df_events.loc[feature_begin : match_end + valid_offset + args.pattern_extension - pore_model.k]
                # reference position on template strand
                if not ref_span.is_reverse:
                    feature_template_pos = ref_span.pos + match_begin + valid_offset
                else:
                    feature_template_pos = ref_span.pos + ref_span_len - match_end - valid_offset
                db_site_id = db.insert_site(db_read_id, args.mod_id, feature_template_pos)
                db.insert_features(db_site_id, df_feature, feature_begin)
            pbar.update(1)
            db.commit()
    pbar.close()




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False)
    parser.add_argument("db", type=str)
    parser.add_argument("fast5", type=str)
    parser.add_argument("bam", type=str)
    parser.add_argument("--mod_id", default=0, type=int)
    parser.add_argument("--pattern", default='CG', type=str)
    parser.add_argument("--pattern_extension", default=6, type=int)
    parser.add_argument("--min_seq_length", default=500, type=int)
    parser.add_argument("--max_seq_length", default=10000, type=int)
    parser.add_argument("--min_score", default=0.0, type=float)
    return parser
