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
import re




# load reference in fasta format
class Reference:
    def __init__(self, ref_file):
        self._contigs = {}
        self.ref_file = ref_file
        if not os.path.isfile(ref_file):
            raise FileNotFoundError("{} is not a file.".format(ref_file))
        self.is_loaded = False

    def __getattr__(self, name):
        return getattr(self._contigs, name)

    def __fastaIter__(self):
        with open(self.ref_file, 'r') as fp:
            line = next(fp).strip()
            while True:
                if line is not None and len(line) > 0:
                    if line[0] == '>':      # fasta
                        name = line[1:].split(' ')[0]
                        sequence = next(fp).strip()
                        try:
                            line = next(fp).strip()
                        except StopIteration:
                            yield name, sequence.upper()
                            return
                        while line is not None and len(line) > 0 and line[0] != '>':
                            sequence += line
                            try:
                                line = next(fp).strip()
                            except StopIteration:
                                yield name, sequence.upper()
                                return
                        yield name, sequence.upper()
                if line is None:
                    raise StopIteration()

    def __load__(self):
        self._contigs = {name:seq for name, seq in self.__fastaIter__()}
        self.is_loaded = True

    def __getitem__(self, key):
        if not self.is_loaded:
            self.__load__()
        return self._contigs[key]

    def contigs(self):
        if not self.is_loaded:
            self.__load__()
        for key, value in self._contigs.items():
            yield key, value
