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
import bamnostic as bs
import numpy as np
from collections import namedtuple


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
def decode_md(seq, cigar, md):
    flatten = lambda l: [item for sublist in l for item in sublist]
    ops = [m[0] for m in re.findall(r'(([0-9]+)|([A-Z]|\^[A-Z]+))', md)]
    ref_mask = np.array(flatten([[True] * int(x) if x.isdigit() else [False] * len(x.strip('^')) for x in ops]))
    seq_mask = np.array(flatten([[True] * int(x) if x.isdigit() else [False] if not '^' in x else [] for x in ops]))
    ref_seq = np.fromiter(''.join(['-' * int(x) if x.isdigit() else x.strip('^') for x in ops]).encode('ASCII'), dtype=np.uint8)
    seq_masked = np.frombuffer(seq.encode('ASCII'), dtype=np.uint8)[cigar_ops_mask(cigar, include='M=X', exclude='SI')]
    ref_seq[ref_mask] = seq_masked[seq_mask]
    return ref_seq.tostring().decode('utf-8')


def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join(complement.get(base, base) for base in reversed(seq))




class AlignmentIndex():
    def __init__(self, input):
        if os.path.isfile(input):
            self.batch_files = [input]
        elif os.path.isdir(input):
            self.batch_files = [os.path.join(dirpath, f) for dirpath, _, files in os.walk(input) for f in files if f.endswith('.bam')]

    # generator interface for fast access
    def records(self):
        for f in self.batch_files:
            with bs.AlignmentFile(f, 'rb') as bam:
                for mapping in (b for b in bam if not (b.is_unmapped or b.is_secondary or b.is_supplementary)):
                    # get seq for secondary mappings
                    if mapping.is_secondary or mapping.is_supplementary:
                        # TODO implement ref lookup from fasta
                        seq = ''
                    else:
                        seq = mapping.seq
                    try:
                        ref_span = decode_md(seq, mapping.cigarstring, mapping.tags['MD'][1])
                    except KeyError:
                        # TODO implement ref lookup from fasta
                        raise
                    if mapping.is_reverse:
                        ref_span = reverse_complement(ref_span)
                    yield ReferenceSpan(qname=mapping.query_name, rname=bam.get_reference_name(mapping.refID), pos=mapping.pos, seq=ref_span, is_reverse=mapping.is_reverse)
