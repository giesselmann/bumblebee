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
import logging
import yaml
import torch
import pkg_resources as pkg
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import bumblebee.modnn
from bumblebee.read import Pattern, ReadNormalizer, ReadAligner
from bumblebee.multiprocessing import StateFunction
from bumblebee.multiprocessing import SourceProcess, WorkerProcess, SinkProcess
from bumblebee.worker import ReadSource, EventAligner


log = logging.getLogger(__name__)



class ModCaller(StateFunction):
    def __init__(self, pattern='CG', extension=7, max_features=40):
        super(StateFunction).__init__()
        self.pattern = Pattern(pattern, extension)
        read_normalizer = ReadNormalizer()
        self.read_aligner = ReadAligner(read_normalizer)
        # model and max_features
        self.max_features = max_features

    def call(self, read, df_events, score):
        # split read in batches
        template_pos = []
        lengths = []
        kmers = []
        features = []
        for pos, _, df_feature in read.feature_sites(df_events,
            self.pattern, self.read_aligner.pm.k):
            if df_feature.shape[0] <= self.max_features:
                template_pos.append(pos)
                lengths.append(df_feature.shape[0])
                kmers.append(list(df_feature.kmer))
                features.append([(row.event_min, row.event_mean,
                    row.event_median, row.event_std,
                    row.event_max, row.event_length)
                        for row in df_feature.itertuples()])
        # predict modification

        return (read, [])




class RecordWriter(StateFunction):
    def __init__(self):
        super(StateFunction).__init__()
        self.read_counter = 0

    def __del__(self):
        log.info("Processed {} reads.".format(self.read_counter))

    def call(self, read, predictions):
        # write features
        self.read_counter += 1





def main(args):
    # init torch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.device) if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    log.info("Using device {}".format(device))
    # load config
    pkg_config = pkg.resource_filename('bumblebee', 'config/{}.yaml'.format(args.config))
    pkg_model = pkg.resource_filename('bumblebee', 'models/{}.pt'.format(args.config))
    # config and weights as file or name in repository
    pkg_config = args.config if os.path.isfile(args.config) else pkg_config
    pkg_model = args.model if os.path.isfile(args.model) else pkg_model
    if os.path.isfile(pkg_config):
        # load config from repository
        log.info("Loading config file {}".format(pkg_config))
        with open(pkg_config, 'r') as fp:
            config = yaml.safe_load(fp)
    else:
        log.error("Could not load model config {}".format(args.config))
        exit(-1)
    if os.path.isfile(pkg_model):
        # load model weights from repository
        log.info("Loading model weights file {}".format(pkg_model))
        state_dict = torch.load(pkg_model)
    else:
        log.error("Could not find weights file")
        exit(-1)

    # init model
    try:
        model = getattr(bumblebee.modnn, config['model'])(config['max_features'], config['params'])
    except Exception as e:
        log.error("Coud not find model definition for {}:\n{}".format(config['model'], e))
        exit(-1)
    if 'n_averaged' in state_dict:
        # swa model
        model.load_state_dict({re.sub('module.', '', key):value
            for key, value in state_dict.items() if 'module.' in key})
    else:
        model.load_state_dict(state_dict)
    log.info("Loaded model")

    # init worker pipeline
    src = SourceProcess(ReadSource,
        args=(args.fast5, args.bam),
        kwargs={'min_seq_length':args.min_seq_length,
                'max_seq_length':args.max_seq_length})
    aligner = WorkerProcess(src.output_queue, EventAligner,
        args=(),
        kwargs={'min_score':args.min_score},
        num_worker=1)
    caller = WorkerProcess(aligner.output_queue, ModCaller,
        args=(),
        kwargs={'pattern':config['pattern'],
                'extension':config['extension'],
                'max_features':config['max_features']},
        num_worker=1)
    sink = SinkProcess(caller.output_queue, RecordWriter,
        args=(),
        kwargs={})
    sink.join()
    caller.join()
    aligner.join()
    src.join()




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("config", type=str)
    parser.add_argument("fast5", type=str, help='Raw signal input (file/directory)')
    parser.add_argument("bam", type=str, help='Alignment input (file/directory)')
    parser.add_argument("--model", default='', type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--min_seq_length", default=2000, type=int, metavar='int',
        help='Minimum sequence length (default: %(default)s)')
    parser.add_argument("--max_seq_length", default=10000, type=int, metavar='int',
        help='Maximum sequence length (default: %(default)s)')
    parser.add_argument("--min_score", default=1.0, type=float, metavar='float',
        help='Min. alignment score (default: %(default)s)')
    return parser
