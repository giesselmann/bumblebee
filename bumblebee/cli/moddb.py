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
import logging
import tqdm
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bumblebee.db import ModDatabase
from bumblebee.ref import Reference
from bumblebee.alignment import reverse_complement
from bumblebee.read import Pattern, Read, ReadNormalizer, ReadAligner
from bumblebee.multiprocessing import StateFunction, SourceProcess, WorkerProcess, SinkProcess
from bumblebee.worker import ReadSource, EventAligner


log = logging.getLogger(__name__)


class SiteExtractor(StateFunction):
    def __init__(self, pattern='CG', extension=7):
        self.pattern = Pattern(pattern, extension)
        read_normalizer = ReadNormalizer()
        self.read_aligner = ReadAligner(read_normalizer)

    def call(self, read, df_events, score):
        try:
            template_begin, feature_begin, df_feature = zip(
                *read.feature_sites(df_events, self.pattern, self.read_aligner.pm.k))
            return (read, template_begin, feature_begin, df_feature, score)
        except ValueError:  # raised if no features found in read
            return None




class RecordWriter(StateFunction):
    def __init__(self, db, mod_id, pattern='CG', extension=7):
        super(StateFunction).__init__()
        self.db = ModDatabase(db)
        self.mod_id = mod_id
        self.pattern = Pattern(pattern, extension)
        read_normalizer = ReadNormalizer()
        self.read_aligner = ReadAligner(read_normalizer)
        self.site_counter = 0
        self.pbar = tqdm.tqdm(desc='Inserting mappings')

    def __del__(self):
        self.pbar.close()
        log.info("Wrote {} sites to database.".format(self.site_counter))

    def call(self, read, template_begin, feature_begin, df_feature, score):
        # write read_record
        db_read_id = self.db.insert_read(read.ref_span, score=score)
        # write features
        for tb, fb, df in zip(template_begin, feature_begin, df_feature):
            db_site_id = self.db.insert_site(db_read_id, self.mod_id, tb)
            self.db.insert_features(db_site_id, df, fb)
            self.site_counter += 1
        self.db.commit()
        self.pbar.update(1)




def update_contexts(context_dict,
                    name, contig, strand,
                    pattern, extension):
    count = 0
    for match in re.finditer(pattern, contig):
        match_begin = match.start()
        context = contig[match_begin - extension : match.end() + extension]
        if strand == 0:
            context_dict[context].append((name, strand, match_begin))
        else:
            context_dict[reverse_complement(context)].append((name, strand, match_begin))
        count += 1
    return count




def main(args):
    if args.command == 'insert':
        src = SourceProcess(ReadSource,
            args=(args.fast5, args.bam, args.ref),
            kwargs={'filter_secondary': args.filter_secondary,
                    'filter_supplementary': args.filter_supplementary,
                    'min_seq_length':args.min_seq_length,
                    'max_seq_length':args.max_seq_length},
            queue_len=64)
        aligner = WorkerProcess(src.output_queue, EventAligner,
            args=(),
            kwargs={'min_score':args.min_score},
            num_worker=args.t,
            queue_len=64)
        ext = WorkerProcess(aligner.output_queue, SiteExtractor,
            args=(),
            kwargs={'pattern':args.pattern,
                    'extension':args.extension},
            num_worker=6)
        sink = SinkProcess(ext.output_queue, RecordWriter,
            args=(args.db, args.mod_id),
            kwargs={'pattern':args.pattern,
                    'extension':args.extension})
        sink.join()
        ext.join()
        aligner.join()
        src.join()
    else:   # index
        db = ModDatabase(args.db)
        db.reset_split()
        ref = Reference(args.ref)
        fwd_pattern = Pattern(args.pattern, args.extension)
        rev_pattern = Pattern(reverse_complement(args.pattern), args.extension)
        fwd_count = 0
        rev_count = 0
        context_dict = defaultdict(list)
        for name, contig in ref.contigs():
            fwd_c = update_contexts(context_dict, name, contig, 0,
                fwd_pattern.pattern_string, args.extension)
            fwd_count += fwd_c
            rev_c = update_contexts(context_dict, name, contig, 1,
                rev_pattern.pattern_string, args.extension)
            rev_count += rev_c
            log.info("Processed contig {}: {} forward and {} reverse sites".format(
                name, fwd_c, rev_c))
        del ref
        # sort by number of occurences
        context_positions = [(context, positions) for context, positions in context_dict.items()]
        context_positions.sort(key = lambda x: len(x[1]), reverse=True)
        # get validation samples from all context frequencies
        val_ids = set(np.linspace(0, len(context_positions), int(args.split * len(context_positions)), endpoint=False, dtype=int))
        for i, (context, positions) in tqdm.tqdm(enumerate(context_positions), desc='Writing', total=len(context_positions)):
            if i in val_ids:
                for p in positions:
                    db.insert_filter(*p, table='eval')
            else:
                for p in positions:
                    db.insert_filter(*p, table='train')
        db.commit()
        del db
        db = ModDatabase(args.db, require_index=True)
        db.commit()
        log.info("Finished indexing of {} forward and {} reverse sites in total".format(
            fwd_count, rev_count))




def argparser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False)
    subparsers = parser.add_subparsers(title='subcommands', dest='command',
        metavar='', required=True)

    common = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False)
    common.add_argument("db", type=str, help='Training database')
    common.add_argument("ref", type=str, metavar='str', help='Reference file')
    common.add_argument("--pattern", default='CG', type=str, metavar='',
        help='Sequence pattern for modification detection (default: %(default)s)')
    common.add_argument("--extension", default=7, type=int, metavar='int',
        help='Sequence context extension around pattern matches (default: %(default)s)')

    p_insert = subparsers.add_parser('insert', help='Insert new data', parents=[common])
    p_insert.add_argument("fast5", type=str, help='Raw signal input (file/directory)')
    p_insert.add_argument("bam", type=str, help='Alignment input (file/directory)')
    p_insert.add_argument("--mod_id", default=0, type=int, metavar='int',
        help='Modification ID of input data (default: %(default)s)')
    p_insert.add_argument("--filter_secondary", action='store_true')
    p_insert.add_argument("--filter_supplementary", action='store_true')
    p_insert.add_argument("--min_seq_length", default=500, type=int, metavar='int',
        help='Minimum sequence length (default: %(default)s)')
    p_insert.add_argument("--max_seq_length", default=20000, type=int, metavar='int',
        help='Maximum sequence length (default: %(default)s)')
    p_insert.add_argument("--min_score", default=0.0, type=float, metavar='float',
        help='Min. alignment score (default: %(default)s)')
    p_insert.add_argument("-t", default=1, type=int, metavar='int',
        help='Worker (default: %(default)s)')

    p_index = subparsers.add_parser('index', help='Index database', parents=[common], add_help=True)
    p_index.add_argument("--split", default=0.1, type=float, metavar='float',
        help='Ratio for eval/train split (default: %(default)s)')

    return parser
