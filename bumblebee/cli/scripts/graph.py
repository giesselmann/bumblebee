# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : Semantic Scholar S2 corpus graph construction
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
import os, sys, glob, argparse
import re
import gzip, json
from collections import defaultdict
from tqdm import tqdm




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph from S2 dataset')
    parser.add_argument('input', help='Input prefix')
    parser.add_argument('output', help='Output prefix')
    parser.add_argument('--t', type=int, default=1, help='Parallel worker')
    args = parser.parse_args()

    # load incoming citations 'inCitations'
    batches = glob.glob(os.path.join(args.input, '*.gz'))
    print('Found {} batches.'.format(len(batches)))

    # dict with key paper and value set of incoming citations
    edges = defaultdict(set)

    for batch in tqdm(batches, desc='batch'):
        with gzip.open(batch, 'rb') as fp:
            data = fp.read()
        records = [json.loads(record.decode())
            for record in re.split(rb'\n(?=\{.*?\})', data)];
        [edges[record['id']].update(set(record['inCitations']))
            for record in tqdm(records, desc='Parsing', leave=False) if len(record['inCitations'])];

    print("Loaded {} cited nodes.".format(len(edges)))

    # enumerate edges
    nodes = set(edges.keys())
    [nodes.update(s) for s in edges.values()];

    print("Loaded {} total nodes".format(len(nodes)))

    node_enum = {id:num for num, id in enumerate(nodes)}

    # write mapping, graph and summary
    print("writing nodemap")
    with gzip.open(args.output + '.nodemap.gz', 'w') as fp:
        fp.write('\n'.join(['\t'.join((key, str(value)))
            for key, value in node_enum.items()]).encode())
    print("Writing edgelist")
    with open(args.output + '.edgelist', 'w') as fp:
        print('\n'.join(['\t'.join((str(node_enum[source]), str(node_enum[target])))
            for target, sources in edges.items() for source in sources]), file=fp)
    with open(args.output + '.summary', 'w') as fp:
        print("{} nodes with incoming citations".format(len(edges)), file=fp)
        print("{} nodes total".format(len(nodes)), file=fp)
        print("{} citation edges".format(sum([len(s) for s in edges.values()])), file=fp)
