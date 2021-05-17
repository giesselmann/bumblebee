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
import time
import logging
import tqdm
import pkg_resources
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bumblebee.poremodel import PoreModel
from bumblebee.db import ModDatabase
from bumblebee.fast5 import Fast5Index
from bumblebee.alignment import AlignmentIndex
from bumblebee.poremodel import PoreModel
from bumblebee.read import Pattern, Read, ReadNormalizer, ReadAligner
from bumblebee.multiprocessing import StateFunction, SourceProcess, WorkerProcess, SinkProcess


log = logging.getLogger(__name__)


# parse reads from fast5 and bam input
class ReadSource(StateFunction):
    def __init__(self, fast5, bam,
                 min_seq_length=500, max_seq_length=10000):
        super(StateFunction).__init__()
        self.f5_idx = Fast5Index(fast5)
        self.algn_idx = AlignmentIndex(bam)
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.read_counter = 0

    def __del__(self):
        log.info("Loaded {} reads from disk.".format(self.read_counter))

    def call(self):
        for i, ref_span in enumerate(self.algn_idx.records()):
            if (len(ref_span.seq) < self.min_seq_length or
                len(ref_span.seq) > self.max_seq_length):
                continue
            read_signal = self.f5_idx[ref_span.qname]
            read = Read(read_signal, ref_span=ref_span)
            yield (read, )
            self.read_counter += 1




# align read signal to reference sequence
class EventAligner(StateFunction):
    def __init__(self, min_score=0.0):
        super(StateFunction).__init__()
        self.min_score = min_score
        read_normalizer = ReadNormalizer()
        self.read_aligner = ReadAligner(read_normalizer)

    def call(self, read):
        score, df_events = read.event_alignments(self.read_aligner)
        log.debug("Aligned read {} with score {}".format(read.name, score))
        if score < self.min_score:
            return None
        else:
            return read, df_events, score




class RecordWriter(StateFunction):
    def __init__(self, db, mod_id, pattern='CG', extension=6):
        super(StateFunction).__init__()
        self.db = ModDatabase(db)
        self.mod_id = mod_id
        self.pattern = Pattern(pattern, extension)
        read_normalizer = ReadNormalizer()
        self.read_aligner = ReadAligner(read_normalizer)
        self.site_counter = 0

    def __del__(self):
        log.info("Wrote {} sites to database.".format(self.site_counter))

    def call(self, read, df_events, score):
        # write read_record
        db_read_id = self.db.insert_read(read.ref_span, score=score)
        # write features
        for template_begin, feature_begin, df_feature in read.feature_sites(df_events,
            self.pattern, self.read_aligner.pm.k):
            db_site_id = self.db.insert_site(db_read_id, self.mod_id, template_begin)
            self.db.insert_features(db_site_id, df_feature, feature_begin)
            self.site_counter += 1
        self.db.commit()




def main(args):
    src = SourceProcess(ReadSource,
        args=(args.fast5, args.bam),
        kwargs={'min_seq_length':args.min_seq_length,
                'max_seq_length':args.max_seq_length})
    worker = WorkerProcess(src.output_queue, EventAligner,
        args=(),
        kwargs={'min_score':args.min_score},
        num_worker=args.threads)
    sink = SinkProcess(worker.output_queue, RecordWriter,
        args=(args.db, args.mod_id),
        kwargs={'pattern':args.pattern,
                'extension':args.extension})
    sink.join()
    worker.join()
    src.join()




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False)
    parser.add_argument("db", type=str)
    parser.add_argument("fast5", type=str)
    parser.add_argument("bam", type=str)
    parser.add_argument("--mod_id", default=0, type=int)
    parser.add_argument("--pattern", default='CG', type=str)
    parser.add_argument("--extension", default=7, type=int)
    parser.add_argument("--min_seq_length", default=2000, type=int)
    parser.add_argument("--max_seq_length", default=10000, type=int)
    parser.add_argument("--min_score", default=0.0, type=float)
    parser.add_argument("-t", "--threads", default=1, type=int)
    return parser
