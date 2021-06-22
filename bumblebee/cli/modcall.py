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
import time
import logging
import yaml
import tqdm
import math
import torch
import queue
import numpy as np
import pkg_resources as pkg
import multiprocessing as mp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import bumblebee.modnn
from bumblebee.read import Pattern, ReadNormalizer, ReadAligner
from bumblebee.multiprocessing import StateFunction, StateIterator
from bumblebee.multiprocessing import SourceProcess, WorkerProcess, SinkProcess
from bumblebee.worker import ReadSource, EventAligner


log = logging.getLogger(__name__)


# Extract target sites from each aligned reead
class SiteExtractor(StateIterator):
    def __init__(self, config):
        super(StateIterator).__init__()
        self.pattern = Pattern(config['pattern'], config['extension'])
        read_normalizer = ReadNormalizer()
        self.read_aligner = ReadAligner(read_normalizer)
        self.max_features = config['max_features']

    def call(self, read, df_events, score):
        for pos, feature_begin, df_feature in read.feature_sites(df_events,
                self.pattern, self.read_aligner.pm.k):
            if df_feature.shape[0] <= self.max_features and df_feature.shape[0] > 1:
                yield (read.ref_span,
                       pos,
                       df_feature.shape[0],
                       df_feature.kmer,
                       df_feature.index.values - feature_begin,
                       df_feature[['event_min', 'event_mean', 'event_median',
                                   'event_std', 'event_max', 'event_length']])




# predict modification for each read and target site
class ModCaller(StateIterator):
    def __init__(self, config, model, device):
        super(StateFunction).__init__()
        # config
        self.max_features = config['max_features']
        self.model = model
        self.device = device
        self.batch_size = 64
        self.ref_spans = []
        self.inputs = []

    def __padded_tensor__(self, length, kmers, offsets, features):
        kmers_padd = np.zeros(self.max_features, dtype=np.int64)
        offsets_padd = np.zeros(self.max_features, dtype=np.int64)
        features_padd = np.zeros((self.max_features, 6), dtype=np.float32)
        kmers_padd[:length] = kmers
        offsets_padd[:length] = [offset + 1 for offset in offsets]
        features_padd[:length, :] = features
        return (torch.tensor(length),
                torch.tensor(kmers_padd),
                torch.tensor(offsets_padd),
                torch.tensor(features_padd))

    def __predict__(self, inputs):
        # list of tensor tuples to tuple of stacked tensors
        lengths, kmers, offsets, features = [torch.stack(value)
            for value in zip(*inputs)]
        # run forward
        with torch.no_grad():
            kmers = kmers.to(self.device)
            offsets = offsets.to(self.device)
            features = features.to(self.device)
            prediction, _, _ = self.model(lengths, kmers, offsets, features)
            # prediction is (batch_size, num_classes)
            prediction = torch.nn.functional.softmax(prediction, dim=1)
        return [tuple(x) for x in prediction.detach().cpu().numpy()]

    def __process_batch__(self):
        ref_spans, positions = zip(*self.ref_spans[:self.batch_size])
        inputs = [self.__padded_tensor__(*input)
            for input in self.inputs[:self.batch_size]]
        predictions = self.__predict__(inputs)
        del self.ref_spans[:self.batch_size]
        del self.inputs[:self.batch_size]
        for ref_span, position, prediction in zip(ref_spans,
            positions, predictions):
            yield ref_span, position, prediction

    def call(self, ref_span, position, length, kmer, offsets, features):
        # split read in batches
        #log.debug("Calling read {}".format(read.name))
        self.ref_spans.append((ref_span, position))
        self.inputs.append((length, kmer, offsets, features))
        if len(self.inputs) >= self.batch_size:
            for x in self.__process_batch__():
                yield x

    def close(self):
        if len(self.inputs) > 0:
            for x in self.__process_batch__():
                yield x



class RecordWriter(StateFunction):
    def __init__(self, config):
        super(StateFunction).__init__()
        self.site_counter = 0
        self.length = len(config['pattern'])

    def __del__(self):
        log.info("Processed {} sites.".format(self.site_counter))

    def call(self, ref_span, template_pos, predictions):
        # write features
        self.site_counter += 1
        #log.debug("writing {} with {} sites".format(ref_span.qname, len(predictions)))
        strand = '-' if ref_span.is_reverse else '+'
        chr = ref_span.rname
        value = str(np.argmax(predictions))
        print('\t'.join([
            chr,
            str(template_pos),
            str(template_pos+self.length),
            ref_span.qname,
            value,
            strand] +
            [str(p) for p in predictions]))




def main(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.device)
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
        args=(args.fast5, args.bam, args.ref),
        kwargs={'min_seq_length':args.min_seq_length,
                'max_seq_length':args.max_seq_length})
    aligner = WorkerProcess(src.output_queue, EventAligner,
        args=(),
        kwargs={'min_score':args.min_score},
        num_worker=args.nproc)
    extractor = WorkerProcess(aligner.output_queue, SiteExtractor,
        args=(config,),
        kwargs={},
        num_worker=1)
    extractor_queue = extractor.output_queue
    caller = ModCaller(config, model, device)
    writer_queue =mp.Queue(32)
    writer = SinkProcess(writer_queue, RecordWriter,
        args=(config,),
        kwargs={})
    # predict in main Process using CUDA
    pid = '(PID: {})'.format(os.getpid())
    with tqdm.tqdm(desc='Processing', unit='alignments') as pbar:
        while True:
            try:
                obj = extractor_queue.get(block=True, timeout=1)
            except queue.Empty:
                obj = None
            if obj is StopIteration:
                log.debug("Received StopIteration in MainProcess {}".format(pid))
                break
            elif obj is not None:
                try:
                    for res in caller(*obj):
                        writer_queue.put(res)
                        pbar.update(1)
                except Exception as ex:
                    log.error("Exception in MainProcess (Proceeding with remaining jobs):\n {}".format(str(ex)))
                    continue
            else:
                continue
    # process remaining samples
    for res in caller.close():
        writer_queue.put(res)
    writer_queue.put(StopIteration)
    writer_queue.close()
    writer_queue.join_thread()
    # wait for completion
    log.debug("Waiting for reader to complete")
    src.join()
    log.debug("Waiting for aligner to complete")
    aligner.join()
    log.debug("Waiting for writer to complete")
    writer.join()






def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("config", type=str)
    parser.add_argument("fast5", type=str, help='Raw signal input (file/directory)')
    parser.add_argument("bam", type=str, help='Alignment input (file/directory)')
    parser.add_argument("ref", type=str, help='Alignment reference file')
    parser.add_argument("--model", default='', type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--nproc", default=4, type=int)
    parser.add_argument("--min_seq_length", default=500, type=int, metavar='int',
        help='Minimum sequence length (default: %(default)s)')
    parser.add_argument("--max_seq_length", default=50000, type=int, metavar='int',
        help='Maximum sequence length (default: %(default)s)')
    parser.add_argument("--min_score", default=1.0, type=float, metavar='float',
        help='Min. signal alignment score (default: %(default)s)')
    return parser
