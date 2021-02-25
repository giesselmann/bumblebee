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
import os
import random
import itertools
import numpy as np




class PoreModel():
    def __init__(self, file=None, rnd=None, norm=False, k=6):
        if file and os.path.isfile(file):
            value_iter = (line.split('\t', 3)[:2] for line in open(file, 'r').read().split('\n') if line)
            self.pm = {kmer:float(value) for kmer, value in value_iter}
            self.k = len(next(iter(self.pm)))
        elif rnd:
            kmer = [''.join(c) for c in itertools.product('ACGT', repeat=k)]
            value = np.random.uniform(-rnd, rnd, len(kmer))
            self.pm = {k:v for k, v in zip(kmer, value)}
            self.k = k
        else:
            self.pm = dict()
            self.k = k
        if norm:
            values = list(self.pm.values())
            v_min = min(values)
            v_max = max(values)
            self.pm = {key:(value - v_min)/(v_max - v_min) * 2.2 - 1.1 for key, value in self.pm.items()}

    def __getattr__(self, name):
        return getattr(self.pm, name)

    def __getitem__(self, key):
        return self.pm.get(key) or 0.0

    def to_tsv(self, name, sep='\t'):
        keys = sorted(list(self.pm.keys()))
        with open(name, 'w') as fp:
            print('\n'.join(['\t'.join((key, str(self.pm[key]))) for key in keys]), file=fp)
