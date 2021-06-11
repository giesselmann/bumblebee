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
import os, re, logging
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




class AlignmentIndex():
    def __init__(self, input, ref,
            filter_secondary=False, filter_supplementary=False):
        self.ref = Reference(ref)
        self.filter_secondary = filter_secondary
        self.filter_supplementary = filter_supplementary
        if os.path.isfile(input):
            self.batch_files = [input]
        elif os.path.isdir(input):
            self.batch_files = [os.path.join(dirpath, f)
                for dirpath, _, files in os.walk(input)
                    for f in files if f.endswith('.bam')]
        elif input == '-' or input == 'stdin':
            pass
            # TODO implement sam parser
        else:
            log.error("Alignment input {} is not a file or directory.".format(input))
            raise FileNotFoundError(input)

    def __parse_sam_mapping__(self, mapping):
        pass

    def __parse_bam_mapping__(self, bam, mapping):
        rname = bam.get_reference_name(mapping.refID)
        if mapping.seq != '*' and 'MD' in mapping.tags:
            ref_span = get_ref_from_md(mapping.seq.upper(), mapping.cigarstring, mapping.tags['MD'][1])
        else:
            ref_len = np.sum(cigar_ops_mask(mapping.cigarstring,
                include='MDN=X', exclude=''))
            ref_span = self.ref[rname][mapping.pos:mapping.pos + ref_len]
        if mapping.is_reverse:
            ref_span = reverse_complement(ref_span)
        return ReferenceSpan(qname=mapping.query_name,
                            rname=rname,
                            pos=mapping.pos,
                            seq=ref_span,
                            is_reverse=mapping.is_reverse)

    # generator interface for fast access
    def records(self):
        for f in self.batch_files:
            with bs.AlignmentFile(f, 'rb') as bam:
                for mapping in (b for b in bam if not b.is_unmapped):
                    if not ((mapping.is_secondary and self.filter_secondary) or
                        (mapping.is_supplementary and self.filter_supplementary)):
                        ref_span = self.__parse_bam_mapping__(bam, mapping)
                        yield ref_span
