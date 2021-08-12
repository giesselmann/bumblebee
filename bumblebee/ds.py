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
import time
import logging
import random
import torch
import numpy as np
import multiprocessing as mp

from bumblebee.worker import ReadSource
from bumblebee.read import ReadNormalizer, ReadAligner
from bumblebee.db import ModDatabase


log = logging.getLogger(__name__)


class SeqDataset(torch.utils.data.IterableDataset):
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        log.info("Dataset init")
        self = worker_info.dataset
        read_normalizer = ReadNormalizer()
        self.read_aligner = ReadAligner(read_normalizer)

    def read_source_process(fast5, bam, ref, con):
        read_source = ReadSource(fast5, bam, ref,
            lazy_index=False)
        while True:
            # wait for epoch start signal
            cmd = con.recv()
            if cmd == 'shutdown':
                log.info("Stopping read source")
                return
            if cmd == 'get':
                log.info("Read source empty")
                con.send(StopIteration)
            elif cmd == 'start':
                log.info("Starting new epoch")
                for read in read_source():
                    cmd = con.recv()
                    if cmd == 'get':
                        con.send(*read)
                    elif cmd == 'stop':
                        break
                    elif cmd == 'shutdown':
                        return

    def __init__(self, fast5, bam, ref, cache_dir):
        super(SeqDataset).__init__()
        self.rs_con, self.w_con = mp.Pipe()
        self.con_lock = mp.Lock()
        self.read_source = mp.Process(target=SeqDataset.read_source_process,
            args=(fast5, bam, ref, self.rs_con))
        self.read_source.start()
        read_normalizer = ReadNormalizer()
        self.read_aligner = ReadAligner(read_normalizer)

    def __del__(self):
        worker_info = torch.utils.data.get_worker_info()
        # stop read source from main process
        if worker_info is None:
            with self.con_lock:
                self.w_con.send("shutdown")
            self.read_source.join()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or worker_info.id == 0:
            with self.con_lock:
                self.w_con.send("stop")
                self.w_con.send("start")
        self.current_read = None
        self.current_score = 0.0
        self.current_events = None
        self.current_offset = 0
        return self

    def __next__(self):
        while True:
            if self.current_read is None:
                with self.con_lock:
                    self.w_con.send("get")
                    read = self.w_con.recv()
                if read is StopIteration:
                    raise StopIteration
                else:
                    self.current_read = read
                    self.current_score, self.current_events = read.event_alignments(self.read_aligner)
                    self.current_offset = self.current_events.sequence_offset.min()
        mock = np.zeros((100, 6), dtype=np.float32)
        return mock




class ModDataset(torch.utils.data.Dataset):
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        self = worker_info.dataset  # the dataset copy in this worker process
        if hasattr(self, 'init_db'):
            self.init_db()
        else:
            self.dataset.init_db()

    def __init__(self, db_file, mod_ids,
                 train=True, balance=True,
                 max_features=32, min_score=1.0,
                 min_weight=1, max_weight=None,
                 config={}):
        self.db_file = db_file
        self.max_features = max_features
        db = ModDatabase(db_file, require_index=True, require_split=True)
        # read feature IDs
        log.info("Loading site IDs.")
        feature_ids = {mod_id:db.get_feature_ids(mod_id,
            max_features=max_features,
            train=train,
            min_score=min_score,
            min_weight=min_weight, max_weight=max_weight) for mod_id in mod_ids}
        feature_count = [len(fids) for fids in feature_ids.values()]
        # truncate to smallest class
        if balance:
            min_feature_count = min(feature_count)
            self.total = min_feature_count * len(mod_ids)
            if self.total == 0:
                log.error("Found 0 balanced sites ({}) for mod-IDs {}".format(
                    ','.join([str(c) for c in feature_count]),
                    ','.join([str(m) for m in mod_ids])))
                exit(0)
        else:
            min_feature_count = max(feature_count)
            self.total = sum(feature_count)
        log.info("Found {} {} sites".format(self.total, 'train' if train else 'eval'))
        self.features = mp.Array('Q', self.total, lock=False)
        self.weights = mp.Array('Q', self.total, lock=False)
        fp_rate = config.get('fp_rate')
        if fp_rate is None:
            # copy feature rowid into shared memory
            it = (id for value in feature_ids.values() for id in value[:min_feature_count])
            for i, id_weight in enumerate(it):
                self.features[i] = id_weight[0]
                self.weights[i] = id_weight[1]
            random.shuffle(self.features)
            self.fp_labels = False
        else:
            # copy feature rowid and sample random false labels
            # with given rate
            self.labels = mp.Array('Q', self.total, lock=False)
            replacements = {label:[x for x in mod_ids if x!=label]
                for label in mod_ids}
            it = ((id, key) for key, value in feature_ids.items() for id in value[:min_feature_count])
            self.fp_labels = True
            for i, (id_weight, label) in enumerate(it):
                self.features[i] = id_weight[0]
                self.weights[i] = id_weight[1]
                self.labels[i] = random.choice(replacements[label]) if np.random.rand() < fp_rate[label] else label
        # init if running in main process
        if not torch.utils.data.get_worker_info():
            self.init_db()

    def init_db(self):
        self.db = ModDatabase(self.db_file)

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        label, length, kmers, offsets, features = self.db.get_feature(
                self.features[index])
        weight = float(self.weights[index])
        if self.fp_labels:
            label = self.labels[index]
        kmers_padd = np.zeros(self.max_features, dtype=np.int64)
        offsets_padd = np.zeros(self.max_features, dtype=np.int64)
        features_padd = np.zeros((self.max_features, 6), dtype=np.float32)
        kmers_padd[:length] = kmers
        offsets_padd[:length] = [offset + 1 for offset in offsets]
        features_padd[:length, :] = features
        return (label, weight,
                {'lengths': length,
                 'kmers': kmers_padd,
                 'offsets': offsets_padd,
                 'features': features_padd})
