# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : Semantic Scholar S2 corpus MinHash
#
#  DESCRIPTION   : none
#
#  RESTRICTIONS  : none
#
#  REQUIRES      : none
#
# ---------------------------------------------------------------------------------
# Copyright 2019 Pay Giesselmann, Max Planck Institute for Molecular Genetics
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
import glob, gzip, json, re
import argparse, signal, pickle
import multiprocessing
import numpy as np
from tqdm import tqdm
from datasketch import MinHash




def hash_record(record):
    m = MinHash(num_perm=256, seed=42)
    token = ' '.join([record['title'], record['paperAbstract']]).lower().split()
    _ = [m.update(t.encode('utf-8')) for t in token]
    return (record['id'], m.hashvalues)




if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signal, frame: exit(0))
    parser = argparse.ArgumentParser(description='Hash json records')
    parser.add_argument('input', help='Input file')
    parser.add_argument('output', help='Output file')
    parser.add_argument('--t', type=int, default=1, help='Parallel worker')
    args = parser.parse_args()

    with gzip.open(args.input, 'rb') as fp:
        data = [json.loads(line) for line in fp.read().decode('utf-8').split('\n') if line != '']

    pool = multiprocessing.Pool(processes=args.t)
    hashes = [hash for hash in tqdm(pool.imap_unordered(hash_record, data), desc='hashing', ncols=75, total=len(data))]
    with gzip.open(args.output, 'w') as fp:
        fp.write(pickle.dumps(hashes))
