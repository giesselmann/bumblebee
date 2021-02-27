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
import h5py
import numpy as np
from collections import namedtuple


Fast5Record = namedtuple('Fast5Record', ['name', 'raw'])


class Fast5Index():
    def __init__(self, input, index=None):
        if os.path.isfile(input):
            self.batch_files = [input]
        elif os.path.isdir(input):
            self.batch_files = [os.path.join(dirpath, f) for dirpath, _, files in os.walk(input) for f in files if f.endswith('.fast5')]
        else:
            raise FileNotFoundError("Input {} is not an existing file or directory".format(input))
        self.index = None

    def grp2id(grp):
        return re.sub(r'^read_', '', grp)

    def id2grp(id):
        return 'read_' + id

    def __len__(self):
        if self.index is None:
            self.__index__()
        return len(self.index)

    def __index__(self):
        self.index = {}
        for f in self.batch_files:
            with h5py.File(f, 'r') as fp:
                read_ids = [Fast5Index.grp2id(grp) for grp in fp.keys()]
                self.index.update(((read_id,f) for read_id in read_ids))

    # random access by read ID
    def __getitem__(self, key):
        if self.index is None:
            self.__index__()
        if key in self.index:
            with h5py.File(self.index[key], 'r') as fp:
                name = key
                raw = fp['{}/Raw/Signal'.format(Fast5Index.id2grp(name))][...].astype(np.float32)
                return Fast5Record(name=name, raw=raw)
        else:
            raise KeyError("Key {} not in fast5 files".format(key))

    # list of read IDs
    def names(self):
        if self.index is None:
            self.__index__()
        return list(self.index.keys())

    # generator interface for fast access
    def records(self):
        for f in self.batch_files:
            with h5py.File(f, 'r') as fp:
                for key, grp in fp.items():
                    if not isinstance(grp, h5py.Group):
                        continue
                    name = Fast5Index.grp2id(key)
                    raw = grp['Raw/Signal'][...].astype(np.float32)
                    yield Fast5Record(name=name, raw=raw)
