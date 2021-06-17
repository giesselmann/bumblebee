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
import logging

from bumblebee.fast5 import Fast5Index
from bumblebee.ref import Reference
from bumblebee.alignment import AlignmentIndex
from bumblebee.read import Read, ReadNormalizer, ReadAligner
from bumblebee.multiprocessing import StateFunction


log = logging.getLogger(__name__)


# parse reads from fast5 and bam input
class ReadSource(StateFunction):
    def __init__(self, fast5, bam, ref,
                 filter_secondary=False, filter_supplementary=False,
                 min_seq_length=500, max_seq_length=10000,
                 lazy_index=True):
        super(StateFunction).__init__()
        self.mapping_counter = 0
        self.f5_idx = Fast5Index(fast5, lazy_index=lazy_index)
        self.algn_idx = AlignmentIndex(bam, ref,
            filter_secondary=filter_secondary,
            filter_supplementary=filter_supplementary)
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length

    def __del__(self):
        log.info("Loaded {} mappings from disk.".format(self.mapping_counter))

    def call(self):
        for i, ref_span in enumerate(self.algn_idx.records()):
            if (len(ref_span.seq) < self.min_seq_length or
                len(ref_span.seq) > self.max_seq_length):
                log.debug("Droping read {} with length {}".format(
                    ref_span.qname, len(ref_span.seq)))
                continue
            try:
                read_signal = self.f5_idx[ref_span.qname]
            except KeyError:
                log.debug("Read {} not in fast5 files".format(ref_span.qname))
                continue
            read = Read(read_signal, ref_span=ref_span)
            yield (read, )
            self.mapping_counter += 1




# align read signal to reference sequence
class EventAligner(StateFunction):
    def __init__(self, min_score=0.0):
        super(StateFunction).__init__()
        self.min_score = min_score
        read_normalizer = ReadNormalizer()
        self.read_aligner = ReadAligner(read_normalizer)

    def call(self, read):
        score, df_events = read.event_alignments(self.read_aligner)
        if score < self.min_score or df_events.shape[0] == 0:
            log.debug("Droping read {} with alignment score {:.3f}".format(
                read.name, score))
            return None
        else:
            #log.debug("Aligned read {} with score {}".format(read.name, score))
            return read, df_events, score
