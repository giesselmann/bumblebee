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
import signal
import numpy as np
import pkg_resources as pkg
import multiprocessing as mp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import bumblebee.modnn
from bumblebee.read import Pattern, ReadNormalizer, ReadAligner
from bumblebee.multiprocessing import StateFunction, StateIterator
from bumblebee.multiprocessing import SourceProcess, WorkerProcess, SinkProcess
from bumblebee.worker import ReadSource, EventAligner, SiteExtractor


log = logging.getLogger(__name__)




# predict modification for each read and target site
class ModCaller(StateFunction):
    def __init__(self, config, model, device):
        super(StateFunction).__init__()
        # config
        self.max_features = config['max_features']
        self.model = model
        self.device = device
        self.batch_size = 256
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
        #predictions = [(0, 1) for _ in range(len(inputs))]
        del self.ref_spans[:self.batch_size]
        del self.inputs[:self.batch_size]
        return ref_spans, positions, predictions

    def call(self, ref_span, position, length, kmer, offsets, features):
        # split read in batches
        #log.debug("Calling read {}".format(read.name))
        self.ref_spans.extend(zip(ref_span, position))
        self.inputs.extend(zip(length, kmer, offsets, features))
        if len(self.inputs) >= self.batch_size:
            return self.__process_batch__()

    def last(self):
        while len(self.inputs) > 0:
            return self.__process_batch__()




class RecordWriter(StateFunction):
    def __init__(self, config):
        super(StateFunction).__init__()
        self.site_counter = 0
        self.length = len(config['pattern'])

    def __del__(self):
        log.info("Processed {} sites.".format(self.site_counter))

    def call(self, ref_spans, template_pos, predictions):
        # write features
        for ref_span, pos, pred in zip(ref_spans, template_pos, predictions):
            strand = '-' if ref_span.is_reverse else '+'
            chr = ref_span.rname
            value = str(np.argmax(pred))
            print('\t'.join([
                chr,
                str(pos),
                str(pos+self.length),
                ref_span.qname,
                value,
                strand] +
                [str(p) for p in pred]))
            self.site_counter += 1




def main(args):
    #mp.set_start_method('spawn')
    use_cuda = torch.cuda.is_available() and args.device is not None
    if use_cuda:
        torch.cuda.set_device(args.device)
    else:
        torch.set_num_threads(args.threads)
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
        state_dict = torch.load(pkg_model, map_location=device)
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

    model.to(device)
    model.eval()
    log.info("Loaded model")

    # save default signal handlers
    SIGINT_default_handler = signal.getsignal(signal.SIGINT)
    SIGTERM_default_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    
    # init worker pipeline
    src = SourceProcess(ReadSource,
        args=(args.fast5, args.bam, args.ref),
        kwargs={'min_seq_length':args.min_seq_length,
                'max_seq_length':args.max_seq_length,
                'pbar': True})
    aligner = WorkerProcess(src.output_queue, SiteExtractor,
        args=(),
        kwargs={'min_score':args.min_score,
                'config':config},
        num_worker=args.nproc)
    extractor_queue = aligner.output_queue
    caller = ModCaller(config, model, device)
    writer_queue =mp.Queue(64)
    writer = SinkProcess(writer_queue, RecordWriter,
        args=(config,),
        kwargs={})

    # restore default signal handler
    def signal_handler(signal, frame):
        log.debug("Received interrupt, shutting down.")
        src.terminate()
        aligner.terminate()
        writer.terminate()
        exit(-1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # predict in main Process using CUDA
    # main loop
    while True:
        try:
            try:
                obj = extractor_queue.get(block=True, timeout=1)
            except queue.Empty:
                obj = None
            if obj is StopIteration:
                # reader process is done
                log.debug("Received StopIteration in MainProcess")
                break
            elif obj is not None:
                res = caller(*obj)
                writer_queue.put(res)
            else:
                continue
        except Exception as ex:
            log.error("Exception in MainProcess (Proceeding with remaining jobs):\n {}".format(str(ex)))
            continue

    # process remaining samples
    res = caller.last()
    if res is not None:
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
    parser.add_argument("--model", default='', type=str, help='Modification model')
    parser.add_argument("--nproc", default=4, type=int, help='Signal alignment processes')
    parser.add_argument("--device", default=None, type=int, help='CUDA device if available')
    parser.add_argument("--threads", default=16, type=int, help='Threads if running on CPU')
    parser.add_argument("--min_seq_length", default=500, type=int, metavar='int',
        help='Minimum sequence length (default: %(default)s)')
    parser.add_argument("--max_seq_length", default=None, type=int, metavar='int',
        help='Maximum sequence length (default: %(default)s)')
    parser.add_argument("--min_score", default=1.0, type=float, metavar='float',
        help='Min. signal alignment score (default: %(default)s)')
    return parser
