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
import sys, os, re
import logging
import bamnostic as bs
import numpy as np
from collections import namedtuple

from bumblebee.ref import Reference


log = logging.getLogger(__name__)


ReferenceSpan = namedtuple('ReferenceSpan', ['qname', 'rname', 'pos', 'seq', 'is_reverse'])


# decode cigar into list of edits
def decode_cigar(cigar):
    ops = [(int(op[:-1]), op[-1]) for op in re.findall('(\d*\D)',cigar)]
    return ops


# bool mask of cigar operations
def cigar_ops_mask(cigar, include='M=X', exclude='DN'):
    flatten = lambda l: [item for sublist in l for item in sublist]
    dec_cigar = decode_cigar(cigar)
    return np.array(flatten([[True]*l if op in include
                                            else [False]*l if op in exclude
                                            else [] for l, op in dec_cigar]))


# decode MD tag
def get_ref_from_md(seq, cigar, md):
    flatten = lambda l: [item for sublist in l for item in sublist]
    ops = [m[0] for m in re.findall(r'(([0-9]+)|([A-Z]|\^[A-Z]+))', md)]
    ref_mask = np.array(flatten([[True] * int(x) if x.isdigit() else [False] * len(x.strip('^')) for x in ops]))
    seq_mask = np.array(flatten([[True] * int(x) if x.isdigit() else [False] if not '^' in x else [] for x in ops]))
    ref_seq = np.fromiter(''.join(['-' * int(x) if x.isdigit() else x.strip('^') for x in ops]).encode('ASCII'), dtype=np.uint8)
    seq_masked = np.frombuffer(seq.encode('ASCII'), dtype=np.uint8)[cigar_ops_mask(cigar, include='M=X', exclude='SI')]
    ref_seq[ref_mask] = seq_masked[seq_mask]
    return ref_seq.tostring().decode('utf-8').upper()


def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join(complement.get(base, base) for base in reversed(seq))




class AlignmentStream():
    mapping = namedtuple('mapping', [
        'qname', 'flag', 'rname', 'pos',
        'mapq', 'cigar', 'rnext', 'pnext',
        'tlen', 'seq', 'qual', 'tags',
        'is_unmapped', 'is_secondary', 'is_supplementary', 'is_reverse'])

    def __init__(self):
        self.stdin = open(0)

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.stdin)
        fields = line.split('\t')
        flag = int(fields[1])
        pos = int(fields[3])
        def parse_md(raw):
            key, type, value = raw.split(':')
            return key, (type, value)
        return AlignmentStream.mapping(
                qname = fields[0],
                flag = flag,
                rname = fields[2],
                pos = pos - 1,      # sam is 1-based
                mapq = int(fields[4]),
                cigar = fields[5],
                rnext = fields[6],
                pnext = int(fields[7]),
                tlen = int(fields[8]),
                seq = fields[9],
                qual = fields[10],
                tags = dict([parse_md(field) for field in fields[11:]]),
                is_unmapped = flag & 0x4,
                is_reverse = flag & 0x10,
                is_secondary = flag & 0x100,
                is_supplementary = flag & 0x800
                )



class AlignmentIndex():
    def __init__(self, input, ref_file,
            filter_secondary=False, filter_supplementary=False):
        self.ref = Reference(ref_file)
        self.filter_secondary = filter_secondary
        self.filter_supplementary = filter_supplementary
        self.stdin = False
        if os.path.isfile(input):
            self.batch_files = [input]
        elif os.path.isdir(input):
            self.batch_files = [os.path.join(dirpath, f)
                for dirpath, _, files in os.walk(input)
                    for f in files if f.endswith('.bam')]
        elif input == '-' or input == 'stdin':
            self.stdin = True
        else:
            log.error("Alignment input {} is not a file, directory or stdin.".format(input))
            raise FileNotFoundError(input)

    def __parse_sam_mapping__(self, record):
        if record.seq != '*' and 'MD' in record.tags:
            ref_span = get_ref_from_md(record.seq.upper(), record.cigar, record.tags['MD'][1])
        else:
            ref_len = np.sum(cigar_ops_mask(record.cigar,
                include='MDN=X', exclude=''))
            ref_span = self.ref[record.rname][record.pos:record.pos + ref_len]
        if record.is_reverse:
            ref_span = reverse_complement(ref_span)
        return ReferenceSpan(qname=record.qname,
                            rname=record.rname,
                            pos=record.pos,
                            seq=ref_span,
                            is_reverse=record.is_reverse)

    def __parse_bam_mapping__(self, bam, record):
        rname = bam.get_reference_name(record.refID)
        if record.seq != '*' and 'MD' in record.tags:
            ref_span = get_ref_from_md(record.seq.upper(), record.cigarstring, record.tags['MD'][1])
        else:
            ref_len = np.sum(cigar_ops_mask(record.cigarstring,
                include='MDN=X', exclude=''))
            ref_span = self.ref[rname][record.pos:record.pos + ref_len]
        if record.is_reverse:
            ref_span = reverse_complement(ref_span)
        return ReferenceSpan(qname=record.query_name,
                            rname=rname,
                            pos=record.pos,
                            seq=ref_span,
                            is_reverse=record.is_reverse)

    # generator interface for fast access
    def records(self):
        if self.stdin:
            stream = AlignmentStream()
            for record in (b for b in stream if not b.is_unmapped):
                if not ((record.is_secondary and self.filter_secondary) or
                    (record.is_supplementary and self.filter_supplementary)):
                    ref_span = self.__parse_sam_mapping__(record)
                    yield ref_span
        else:
            for f in self.batch_files:
                with bs.AlignmentFile(f, 'rb') as bam:
                    for record in (b for b in bam if not b.is_unmapped):
                        if not ((record.is_secondary and self.filter_secondary) or
                            (record.is_supplementary and self.filter_supplementary)):
                            ref_span = self.__parse_bam_mapping__(bam, record)
                            yield ref_span
