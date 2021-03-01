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
import random
import torch
import numpy as np



# dataset yielding batches of (class, lengths, kmers, features)
class ModDataset(torch.utils.data.Dataset):
    def __init__(self, db, mod_ids, balance=True, batch_size=32, max_features=32):
        self.db = db
        self.mod_ids = mod_ids
        self.batch_size = batch_size
        self.max_features = max_features
        # init batch ids
        self.feature_ids = {mod_id:db.get_feature_ids(mod_id, max_features) for mod_id in mod_ids}
        # balance dataset
        if balance:
            min_feature_count = min([len(feature_ids) for feature_ids in self.feature_ids.values()])
            self.feature_ids = {mod_id:feature_ids[:min_feature_count] for mod_id, feature_ids in self.feature_ids.items()}
        self.feature_ids = [x for feature_ids in self.feature_ids.values() for x in feature_ids]
        random.shuffle(self.feature_ids)
        self.feature_ids = self.feature_ids[:len(self.feature_ids)-len(self.feature_ids)%batch_size]
        self.db.set_feature_batch(self.__feature_batch_iter__())

    def __feature_batch_iter__(self):
        for i, feature_id in enumerate(self.feature_ids):
            yield feature_id, i // self.batch_size

    def __len__(self):
        return len(self.feature_ids) // self.batch_size

    def __getitem__(self, index):
        labels, lengths, kmers, features = self.db.get_batch(index)
        # zero padd kmers and features
        kmers_padd = np.zeros((self.batch_size, self.max_features), dtype=np.int64)
        features_padd = np.zeros((self.batch_size, self.max_features, 6), dtype=np.float32)
        for i, l in enumerate(lengths):
            kmers_padd[i, 0:l] = kmers[i]
            features_padd[i, 0:l, :] = np.array(features[i], dtype=np.float32)
        return (np.array(labels), {'lengths': np.array(lengths),
                                    'kmers': kmers_padd,
                                    'features': features_padd})
