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
import random
import torch
import multiprocessing
import numpy as np

from bumblebee.db import ModDatabase


log = logging.getLogger(__name__)


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
                 max_features=32, min_score=1.0):
        self.db_file = db_file
        self.max_features = max_features
        db = ModDatabase(db_file, require_index=True, require_split=True)
        # read feature IDs
        log.info("Loading site IDs.")
        feature_ids = {mod_id:db.get_feature_ids(mod_id,
            max_features=max_features,
            train=train,
            min_score=min_score) for mod_id in mod_ids}
        feature_count = [len(feature_ids) for feature_ids in feature_ids.values()]
        # truncate to smallest class
        if balance:
            min_feature_count = min(feature_count)
            self.total = min_feature_count * len(mod_ids)
            if self.total == 0:
                log.error("Found 0 balanced sites for mod-IDs {}".format(
                    ','.join(mod_ids)))
                exit(0)
        else:
            min_feature_count = max(feature_count)
            self.total = sum(feature_count)
        log.info("Found {} {} sites".format(self.total, 'train' if train else 'eval'))
        self.features = multiprocessing.Array('Q', self.total, lock=False)
        # copy feature rowid into shared memory
        it = (id for value in feature_ids.values() for id in value[:min_feature_count])
        for i, rowid in enumerate(it):
            self.features[i] = rowid
        random.shuffle(self.features)
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
        kmers_padd = np.zeros(self.max_features, dtype=np.int64)
        offsets_padd = np.zeros(self.max_features, dtype=np.int64)
        features_padd = np.zeros((self.max_features, 6), dtype=np.float32)
        kmers_padd[:length] = kmers
        offsets_padd[:length] = [offset + 1 for offset in offsets]
        features_padd[:length, :] = features
        return (label, {'lengths': length,
                        'kmers': kmers_padd,
                        'offsets': offsets_padd,
                        'features': features_padd})





# dataset yielding batches of (class, lengths, kmers, features)
class BatchedModDataset(torch.utils.data.Dataset):
    def __init__(self, db, mod_ids,
                train=True, balance=True,
                batch_size=32, max_features=32,
                min_score=1.0):
        self.db = db
        self.mod_ids = mod_ids
        self.train = train
        self.batch_size = batch_size
        self.max_features = max_features
        # init batch ids
        print("Preparing {} dataset:".format('training' if train else 'evaluation'))
        print("\tRead feature ids...")
        self.feature_ids = {mod_id:db.get_feature_ids(mod_id, max_features=max_features, train=train, min_score=min_score) for mod_id in mod_ids}
        # balance dataset
        if balance:
            min_feature_count = min([len(feature_ids) for feature_ids in self.feature_ids.values()])
            self.feature_ids = {mod_id:feature_ids[:min_feature_count] for mod_id, feature_ids in self.feature_ids.items()}
        self.feature_ids = [x for feature_ids in self.feature_ids.values() for x in feature_ids]
        random.shuffle(self.feature_ids)
        print("\tWrite batch ids...")
        self.feature_ids = self.feature_ids[:len(self.feature_ids)-len(self.feature_ids)%batch_size]
        self.db.set_feature_batch(self.__feature_batch_iter__(), train=train)

    def __feature_batch_iter__(self):
        for i, feature_id in enumerate(self.feature_ids):
            yield feature_id, i // self.batch_size

    def __len__(self):
        return len(self.feature_ids) // self.batch_size

    def __getitem__(self, index):
        labels, lengths, kmers, features = self.db.get_feature_batch(index, train=self.train)
        # zero padd kmers and features
        kmers_padd = np.zeros((self.batch_size, self.max_features), dtype=np.int64)
        features_padd = np.zeros((self.batch_size, self.max_features, 6), dtype=np.float32)
        for i, l in enumerate(lengths):
            kmers_padd[i, 0:l] = kmers[i]
            features_padd[i, 0:l, :] = np.array(features[i], dtype=np.float32)
        return (np.array(labels), {'lengths': np.array(lengths),
                                    'kmers': kmers_padd,
                                    'features': features_padd})

    def shuffle(self):
        random.shuffle(self.feature_ids)
        self.db.set_feature_batch(self.__feature_batch_iter__(), train=self.train)
