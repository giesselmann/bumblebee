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
import tqdm
import math
import torch
import numpy as np
import pkg_resources as pkg
import multiprocessing as mp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import bumblebee.modnn
from bumblebee.read import Pattern, ReadNormalizer, ReadAligner
from bumblebee.multiprocessing import StateFunction
from bumblebee.multiprocessing import SourceProcess, WorkerProcess, SinkProcess
from bumblebee.worker import ReadSource, EventAligner


log = logging.getLogger(__name__)


class ModCaller(StateFunction):
    def __init__(self, config, model, device):
        super(StateFunction).__init__()
        # config
        self.pattern = Pattern(config['pattern'], config['extension'])
        read_normalizer = ReadNormalizer()
        self.read_aligner = ReadAligner(read_normalizer)
        self.max_features = config['max_features']
        self.model = model
        self.device = device
        self.batch_size = 32

    def __padded_tensor__(self, length, kmers, features):
        kmers_padd = np.zeros(self.max_features, dtype=np.int64)
        features_padd = np.zeros((self.max_features, 6), dtype=np.float32)
        kmers_padd[:length] = kmers
        features_padd[:length, :] = features
        return (torch.tensor(length),
                torch.tensor(kmers_padd),
                torch.tensor(features_padd))

    def __predict__(self, inputs):
        # list of tensor tuples to tuple of stacked tensors
        lengths, kmers, features = [torch.stack(value)
            for value in zip(*inputs)]
        # run forward
        with torch.no_grad():
            kmers = kmers.to(self.device)
            features = features.to(self.device)
            prediction, _, _ = self.model(lengths, kmers, features)
        return [tuple(x) for x in prediction.detach().cpu().numpy()]

    def call(self, read, df_events, score):
        # split read in batches
        #log.debug("Calling read {}".format(read.name))
        template_pos = []
        inputs = []
        predictions = []
        for pos, _, df_feature in read.feature_sites(df_events,
                self.pattern, self.read_aligner.pm.k):
            if df_feature.shape[0] <= self.max_features and df_feature.shape[0] > 1:
                length, kmers, features = self.__padded_tensor__(
                    df_feature.shape[0], df_feature.kmer,
                    df_feature[['event_min', 'event_mean', 'event_median',
                                'event_std', 'event_max', 'event_length']])
                template_pos.append(pos)
                inputs.append((length, kmers, features))
            if len(inputs) >= self.batch_size:
                predictions.extend(self.__predict__(inputs[:self.batch_size]))
                del inputs[:self.batch_size]
        # run remaining sites
        if len(inputs):
            predictions.extend(self.__predict__(inputs[:self.batch_size]))
        return (read, template_pos, predictions)




class RecordWriter(StateFunction):
    def __init__(self):
        super(StateFunction).__init__()
        self.read_counter = 0

    def __del__(self):
        log.info("Processed {} reads.".format(self.read_counter))

    def call(self, read, template_pos, predictions):
        # write features
        self.read_counter += 1
        #log.debug("writing {} with {} sites".format(read.name, len(predictions)))
        strand = '-' if read.ref_span.is_reverse else '+'
        chr = read.ref_span.rname
        for pos, (p0, p1) in zip(template_pos, predictions):
            value = '1' if p1 > p0 else '0'
            print('\t'.join([chr, str(pos), str(pos+2),
                             read.name, value, strand, str(p1-p0)]))




def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.device) if use_cuda else "cpu")
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
    # load weights
    if 'n_averaged' in state_dict:
        # swa model
        model.load_state_dict({re.sub('module.', '', key):value
            for key, value in state_dict.items() if 'module.' in key})
    else:
        model.load_state_dict(state_dict)
    log.info("Loaded model")

    model.to(device)
    model.eval()

    # init worker pipeline
    src = SourceProcess(ReadSource,
        args=(args.fast5, args.bam),
        kwargs={'min_seq_length':args.min_seq_length,
                'max_seq_length':args.max_seq_length})
    aligner = WorkerProcess(src.output_queue, EventAligner,
        args=(),
        kwargs={'min_score':args.min_score},
        num_worker=4)
    aligner_queue = aligner.output_queue
    caller = ModCaller(config, model, device)
    writer_queue =mp.Queue(32)
    sink = SinkProcess(writer_queue, RecordWriter,
        args=(),
        kwargs={})
    # predict in main Process using CUDA
    pid = '(PID: {})'.format(os.getpid())
    with tqdm.tqdm(desc='Processing') as pbar:
        while True:
            try:
                obj = aligner_queue.get(block=True, timeout=1)
            except queue.Empty:
                obj = None
            if obj is StopIteration:
                log.debug("Received StopIteration in MainProcess {}".format(pid))
                break
            elif obj is not None:
                try:
                    res = caller(*obj)
                except Exception as ex:
                    log.error("Exception in worker (Proceeding with remaining jobs):\n {}".format(str(ex)))
                    continue
                if res is not None:
                    writer_queue.put(res)
                    pbar.update(1)
            else:
                continue
    writer_queue.put(StopIteration)
    writer_queue.close()
    # wait for completion
    sink.join()
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
    parser.add_argument("--min_seq_length", default=500, type=int, metavar='int',
        help='Minimum sequence length (default: %(default)s)')
    parser.add_argument("--max_seq_length", default=50000, type=int, metavar='int',
        help='Maximum sequence length (default: %(default)s)')
    parser.add_argument("--min_score", default=1.0, type=float, metavar='float',
        help='Min. signal alignment score (default: %(default)s)')
    return parser
