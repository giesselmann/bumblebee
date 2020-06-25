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
# Copyright 2020 Pay Giesselmann, Max Planck Institute for Molecular Genetics
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
import os, sys, argparse
import string, re
import h5py
import edlib
import itertools
import numpy as np
import pomegranate as pg
import tensorflow as tf
from multiprocessing import Process, Queue
from signal import signal, SIGPIPE, SIG_DFL
from collections import deque
from skimage.morphology import opening, closing, rectangle
from matplotlib import pyplot as plt
from tqdm import tqdm




# sam alignment parser
class sam():
    def __init__(self):
        pass

    def __reverse_complement__(seq):
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N' : 'N'}
        return "".join(complement.get(base, base) for base in reversed(seq))

    def __parse_value__(value, type):
        if type in {'FLAG', 'POS', 'MAPQ', 'PNEXT', 'TLEN', 'i'}:
            return int(value)
        elif type in {'f'}:
            return float(value)
        else:
            return value

    def __parse_tag__(raw_tag):
        tag_name, tag_type, tag_value = raw_tag.split(':')
        return tag_name, sam.__parse_value__(tag_value, tag_type)

    # decode cigar into list of edits
    def __decodeCigar__(cigar):
        ops = [(int(op[:-1]), op[-1]) for op in re.findall('(\d*\D)',cigar)]
        return ops

    # return length of recognized operations in decoded cigar
    def __opsLength__(ops, recOps='MIS=X'):
        n = [op[0] for op in ops if op[1] in recOps]
        return sum(n)

    # bool mask of cigar operations
    def __cigar_ops_mask__(cigar, include='M=X', exclude='DN'):
        flatten = lambda l: [item for sublist in l for item in sublist]
        dec_cigar = sam.__decodeCigar__(cigar)
        return np.array(flatten([[True]*l if op in include
                                                else [False]*l if op in exclude
                                                else [] for l, op in dec_cigar]))

    # decode MD tag
    def __decode_md__(seq, cigar, md):
        flatten = lambda l: [item for sublist in l for item in sublist]
        ops = [m[0] for m in re.findall(r'(([0-9]+)|([A-Z]|\^[A-Z]+))', md)]
        ref_mask = np.array(flatten([[True] * int(x) if x.isdigit() else [False] * len(x.strip('^')) for x in ops]))
        seq_mask = np.array(flatten([[True] * int(x) if x.isdigit() else [False] if not '^' in x else [] for x in ops]))
        ref_seq = np.fromiter(''.join(['-' * int(x) if x.isdigit() else x.strip('^') for x in ops]).encode('ASCII'), dtype=np.uint8)
        seq_masked = np.frombuffer(seq.encode('ASCII'), dtype=np.uint8)[sam.__cigar_ops_mask__(cigar, include='M=X', exclude='SI')]
        ref_seq[ref_mask] = seq_masked[seq_mask]
        return ref_seq.tostring().decode('utf-8')

    def parse_sam_line(sam_line):
        sam_fields_raw = sam_line.strip().split('\t')
        sam_fields_names = ['QNAME', 'FLAG', 'RNAME', 'POS', 'MAPQ', 'CIGAR', 'RNEXT', 'PNEXT', 'TLEN', 'SEQ', 'QUAL']
        sam_fields = {name:sam.__parse_value__(value, name) for name, value in zip(sam_fields_names, sam_fields_raw)}
        sam_fields.update({sam.__parse_tag__(tag) for tag in sam_fields_raw[11:]})
        return sam_fields

    def get_ref_sequence(sam_record):
        if 'MD' in sam_record and sam_record['SEQ'] != '*':
            seq = sam.__decode_md__(sam_record['SEQ'], sam_record['CIGAR'], sam_record['MD'])
            if sam_record['FLAG'] & 0x16:
                seq = sam.__reverse_complement__(seq)
            return seq
        else:
            return ''

    def get_seq_quality(sam_record):
        if sam_record['QUAL'] != '*':
            return np.mean([ord(c) for c in sam_record['QUAL']])
        else:
            return 0




# filte out values with score below quantile
class quantile_queue():
    def __init__(self, quantile=(0.0, 1.0), buffer_size=128):
        self.quantile = quantile if isinstance(quantile, tuple) else (quantile, 1.0)
        self.score_buffer = np.full((buffer_size,), 0, dtype=np.float64)
        self.value_buffer = deque()

    def __len__(self):
        return len(self.value_buffer)

    def push(self, score, value):
        self.score_buffer = np.roll(self.score_buffer, 1)
        self.score_buffer[0] = score
        self.value_buffer.appendleft((score, value))

    def pop(self):
        q = np.quantile(self.score_buffer, self.quantile)
        while len(self.value_buffer):
            score, value = self.value_buffer.pop()
            if score >= q[0] and score <= q[1]:
                return score, value
        return None, None




# basic simulation
class pore_model():
    def __init__(self, model_file):
        def model_iter(iterable):
            for line in iterable:
                yield line.strip().split('\t')[:3]
        with open(model_file, 'r') as fp:
            model_dict = {x[0]:(float(x[1]), float(x[2])) for x in model_iter(fp)}
        self.kmer = len(next(iter(model_dict.keys())))
        self.model_median = np.median([x[0] for x in model_dict.values()])
        self.model_MAD = np.mean(np.absolute(np.subtract([x[0] for x in model_dict.values()], self.model_median)))
        self.model_values = np.array([x[0] for x in model_dict.values()])
        min_state = min(model_dict.values(), key=lambda x:x[0])
        max_state = max(model_dict.values(), key=lambda x:x[0])
        self.model_min = min_state[0] - 6 * min_state[1]
        self.model_max = max_state[0] + 6 * max_state[1]
        self.model_dict = model_dict
        self.base_q = np.quantile(self.model_values, np.linspace(0,1,8))

    def generate_signal(self, sequence, samples=10, noise=False):
        signal = []
        level_means = np.array([self.model_dict[kmer][0] if not 'N' in kmer else self.model_median for kmer in
            [sequence[i:i+self.kmer] for i in range(len(sequence)-self.kmer + 1)]])
        if samples and not noise:
            sig = np.repeat(level_means, samples)
        elif not noise:
            sig = np.repeat(level_means, np.random.uniform(6, 10, len(level_means)).astype(int))
        else:
            level_stdvs = np.array([self.model_dict[kmer][1] if not 'N' in kmer else 2.0 for kmer in
                [sequence[i:i+self.kmer] for i in range(len(sequence)-self.kmer + 1)]])
            level_samples = np.random.uniform(6, 10, len(level_means)).astype(int)
            level_means = np.repeat(level_means, level_samples)
            level_stdvs = np.repeat(level_stdvs, level_samples)
            sig = np.random.normal(level_means, 3 * level_stdvs)
        return sig

    def quantile_nrm(self, signal_raw):
        raw_q = np.quantile(signal_raw, np.linspace(0,1,8))
        p = np.poly1d(np.polyfit(raw_q, self.base_q, 3))
        return p(signal_raw)




# filter, normalization and binarization
class signal_processing():
    def median_MAD(signal):
        median = np.median(signal)
        MAD = np.mean(np.absolute(np.subtract(signal, median)))
        return (median, MAD)

    def flt(raw_signal, width=3):
        flt_signal = raw_signal
        raw_median, raw_MAD = signal_processing.median_MAD(raw_signal)
        morph_signal = (flt_signal - raw_median) / raw_MAD
        morph_signal = np.clip(morph_signal * 24 + 127, 0, 255).astype(np.dtype('uint8')).reshape((1, len(morph_signal)))
        flt = rectangle(1, width)
        morph_signal = opening(morph_signal, flt)
        morph_signal = closing(morph_signal, flt)[0].astype(np.dtype('float'))
        return ((morph_signal - 127) / 24) * raw_MAD + raw_median

    def sig2char(raw_signal, alphabet=string.ascii_uppercase[:5]):
        ords = sorted([ord(x) for x in alphabet])
        quantiles = np.quantile(raw_signal, np.linspace(0,1,len(ords)))
        inds = np.digitize(raw_signal, quantiles).astype(np.int) - 1
        return ''.join([chr(ords[x]) for x in inds])




# semi-global signal alignment
class signal_alignment():
    def __init__(self, pm, samples=12, alphabet=string.ascii_uppercase[:5], expand=0):
        self.expand = expand
        self.alphabet = alphabet
        self.samples = samples
        self.pm = pm

    def __pos__(self, querry, sequence):
        alphabet = sorted([x for x in set([x for x in sequence + querry])])
        equalities = []
        assert self.expand < len(alphabet)
        for expansion in range(1, self.expand):
            equalities += [(alphabet[i], alphabet[i+expansion])
                for i in range(len(alphabet) - expansion)]
        algn = edlib.align(querry, sequence, mode='HW', task='locations',
            additionalEqualities=equalities)
        begin, end = algn['locations'][0]
        dist = algn['editDistance'] / (end - begin)
        return dist, int(begin), int(end)

    def pos(self, querry, nrm_signal):
        flt_signal = signal_processing.flt(nrm_signal)
        flt_char = signal_processing.sig2char(flt_signal, alphabet=self.alphabet)
        sim_signal = self.pm.generate_signal(querry, samples=self.samples, noise=False)
        sim_char = signal_processing.sig2char(sim_signal, alphabet=self.alphabet)
        dist, begin, end = self.__pos__(sim_char, flt_char)
        return dist, begin, end




# profile HMM
class profileHMM(pg.HiddenMarkovModel):
    def __init__(self, sequence,
                 pm_base, transition_probs={}, state_prefix='',
                 no_silent=False,
                 std_scale=1.0, std_offset=0.0
                 ):
        super().__init__()
        self.pm_base = pm_base
        self.sequence = sequence
        self.state_prefix = state_prefix
        self.no_silent = no_silent
        self.std_scale = std_scale
        self.std_offset = std_offset
        self.transition_probs = {'match_loop': .75,     # .75
                                 'match_match': .15,    # .15          sum to 1
                                 'match_insert': .09,   # .09
                                 'match_delete': .01,   # .01

                                 'insert_loop' : .15,   # .15
                                 'insert_match_0': .40, # .40          sum to 1
                                 'insert_match_1': .40, # .40
                                 'insert_delete': .05,  # .05

                                 'delete_delete': .005, # .005
                                 'delete_insert': .05,  # .05          sum to 1
                                 'delete_match': .945   # .945
                                 }
        for key, value in transition_probs.items():
            self.transition_probs[key] = value
        self.__init_model__()

    def __init_model__(self):
        self.match_states, insertion_states, deletetion_states = self.__extract_states__(self.sequence)
        self.insertion_states = insertion_states
        self.deletion_states = deletetion_states
        self.s1 = pg.State(None, name=self.state_prefix+'s1')
        self.s2 = pg.State(None, name=self.state_prefix+'s2')
        self.e1 = pg.State(None, name=self.state_prefix+'e1')
        self.e2 = pg.State(None, name=self.state_prefix+'e2')
        self.__connect_states__()

    def __extract_states__(self, sequence):
        match_states = []
        insertion_states = []
        deletion_states = []
        digits = np.ceil(np.log10(len(sequence) - self.pm_base.kmer + 1)).astype(np.int)
        for idx, kmer in enumerate([sequence[i:i+self.pm_base.kmer] for i in range(len(sequence) - self.pm_base.kmer + 1)]):
            state_name = self.state_prefix + str(idx).rjust(digits,'0')
            state_mean, state_std = self.pm_base.model_dict[kmer] if kmer in self.pm_base.model_dict else (self.pm_base.model_median, self.pm_base.model_MAD)
            match_states.append(pg.State(pg.NormalDistribution(state_mean, state_std * self.std_scale + self.std_offset), name=state_name + 'm'))
            if not self.no_silent:
                deletion_states.append(pg.State(None, name=state_name + 'd'))
            insertion_states.append(pg.State(pg.UniformDistribution(self.pm_base.model_min, self.pm_base.model_max),
                                    name=state_name + 'i'))
        return match_states, insertion_states, deletion_states

    def __connect_states__(self):
        self.add_states(self.match_states)
        self.add_states(self.insertion_states)
        if not self.no_silent:
            self.add_states(self.deletion_states)
        self.add_states([self.s1, self.s2, self.e1, self.e2])
        # matches
        for i, state in enumerate(self.match_states):
            self.add_transition(state, state, self.transition_probs['match_loop'], group='match_loop')
            if i < len(self.match_states) - 1:
                self.add_transition(state, self.match_states[i + 1], self.transition_probs['match_match'], group='match_match')
        # insertions
        for i, state in enumerate(self.insertion_states):
            self.add_transition(state, state, self.transition_probs['insert_loop'], group='insert_loop')
            self.add_transition(self.match_states[i], state, self.transition_probs['match_insert'], group='match_insert')
            self.add_transition(state, self.match_states[i], self.transition_probs['insert_match_1'], group='insert_match_1')
            if i < len(self.deletion_states) - 1 and not self.no_silent:
                self.add_transition(state, self.deletion_states[i+1], self.transition_probs['insert_delete'], group='insert_delete')
            if i < len(self.match_states) - 1:
                self.add_transition(state, self.match_states[i+1], self.transition_probs['insert_match_0'], group='insert_match_0')
        # deletions
        if not self.no_silent:
            for i, state in enumerate(self.deletion_states):
                self.add_transition(state, self.insertion_states[i], self.transition_probs['delete_insert'], group='delete_insert')
                if i > 0:
                    self.add_transition(self.match_states[i-1], state, self.transition_probs['match_delete'], group='match_delete')
                if i < len(self.match_states) - 1:
                    self.add_transition(state, self.match_states[i+1], self.transition_probs['delete_match'], group='delete_match')
                if i < len(self.deletion_states) - 1:
                    self.add_transition(state, self.deletion_states[i+1], self.transition_probs['delete_delete'], group='delete_delete')
            self.add_transition(self.s1, self.deletion_states[0], 1)
            self.add_transition(self.s2, self.match_states[0], 1)
            self.add_transition(self.deletion_states[-1], self.e1, self.transition_probs['delete_delete'])
            self.add_transition(self.deletion_states[-1], self.e2, self.transition_probs['delete_match'])
        else:
            for i, state in enumerate(self.match_states):
                if i < len(self.match_states) - 2:
                    self.add_transition(state, self.match_states[i+2], self.transition_probs['match_delete'], group='match_delete')
            self.add_transition(self.s1, self.insertion_states[0], 1)
            self.add_transition(self.s2, self.match_states[0], 1)
        self.add_transition(self.insertion_states[-1], self.e1, self.transition_probs['insert_delete'], group='insert_delete')
        self.add_transition(self.insertion_states[-1], self.e2, self.transition_probs['insert_match_0'], group='insert_match_0')
        self.add_transition(self.match_states[-1], self.e2, self.transition_probs['match_match'])
        self.add_transition(self.match_states[-1], self.e1, self.transition_probs['match_delete'])

    def bake(self, *args, **kwargs):
        self.add_transition(self.start, self.s1, .5)
        self.add_transition(self.start, self.s2, .5)
        self.add_transition(self.e1, self.end, 1)
        self.add_transition(self.e2, self.end, 1)
        super().bake(*args, **kwargs)

    def viterbi(self, sequence, **kwargs):
        p, path = super().viterbi(np.clip(sequence, self.pm_base.model_min, self.pm_base.model_max), **kwargs)
        if path is not None:
            path = [x[1].name for x in path if x[0] < self.silent_start]
            p_nrm = p / len(path)
            return p_nrm, path
        else:
            return 0, []




# Factory worker
# Parse SAM input into read_ID and reference sequence span
class sam_parser():
    def __init__(self,
                 drop_quantile=0.3,
                 buffer_size=128,
                 min_sequence_length=1000,
                 max_sequence_length=50000):
        self.qqueue = quantile_queue(quantile=drop_quantile,
                                     buffer_size=buffer_size)
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        # fill quantile buffer
        parsed_records = 0
        sys.stdin = open(0)
        while parsed_records < buffer_size:
            try:
                sam_line = next(sys.stdin)
                parsed_records = self.__parse_sam_line__(sam_line)
            except StopIteration:
                return

    def __parse_sam_line__(self, sam_line):
        sam_record = sam.parse_sam_line(sam_line)
        #print(sam_record['QNAME'])
        ref_seq = sam.get_ref_sequence(sam_record)
        seq_qual = sam.get_seq_quality(sam_record)
        if seq_qual > 0 and len(ref_seq) < self.max_sequence_length:
            map_qual = sam_record['AS'] / len(ref_seq) if 'AS' in sam_record else seq_qual
            self.qqueue.push(map_qual, [sam_record['QNAME'], ref_seq])
        return len(self.qqueue)

    def __iter__(self):
        return self

    def __next__(self):
        score, value = self.qqueue.pop()
        while not score:
            sam_line = next(sys.stdin)
            self.__parse_sam_line__(sam_line)
            score, value = self.qqueue.pop()
        # tuples of (ID, ref_seq)
        return [(value, {}),]




# slice signal using semi-global signal alignment
class signal_slicer():
    def __init__(self, model_file, fast5_file,
                 slice_width=500,
                 drop_quantile=(0.0, 0.9),
                 buffer_size=128,):
        self.fast5_file = fast5_file
        self.pm = pore_model(model_file)
        self.slice_width = int(slice_width)
        self.signal_slice_width = int(slice_width * 10)
        self.qqueue = quantile_queue(quantile=drop_quantile,
                                     buffer_size=buffer_size)
        self.sa = signal_alignment(self.pm)

    def __call__(self, ID, ref_seq):
        if len(ref_seq) < self.slice_width:
            return []
        with h5py.File(self.fast5_file, 'r') as fp:
            if 'read_' + ID in fp:
                raw_signal = fp['read_' + ID]['Raw/Signal'][...].astype(np.float32)
            else:
                return []
        nrm_signal = self.pm.quantile_nrm(raw_signal)
        nrm_signal_begin = 0
        nrm_signal_end = 2 * self.signal_slice_width
        for slice_begin, slice_end in zip(range(0,len(ref_seq)-self.slice_width, self.slice_width),
                                          range(self.slice_width, len(ref_seq), self.slice_width)):
            ref_seq_slice = ref_seq[slice_begin:slice_end]
            nrm_signal_slice = nrm_signal[nrm_signal_begin:nrm_signal_end]
            dist, begin, end = self.sa.pos(ref_seq_slice, nrm_signal_slice)
            self.qqueue.push(dist, (ref_seq_slice, nrm_signal_slice[begin:end].copy()))
            nrm_signal_begin += end - self.signal_slice_width // 2
            nrm_signal_end = nrm_signal_begin + 2 * self.signal_slice_width
        slices = []
        while len(self.qqueue):
            score, value = self.qqueue.pop()
            if score is not None:
                slices.append((value, {}))
                #f, ax = plt.subplots(2)
                #ax[0].plot(value[1])
                #ax[1].plot(self.pm.generate_signal(value[0]))
                #plt.show()
        # tuples of (ref_seq, nrm_signal)
        return slices




# precise signal alignment using profile HMMs
class signal_aligner():
    def __init__(self, model_file,
                 drop_quantile=(0.0, 0.9),
                 buffer_size=128,
                 min_sequence_length=50,
                 max_sequence_length=250,
                 max_signal_length=3000):
        self.pm = pore_model(model_file)
        self.qqueue = quantile_queue(quantile=drop_quantile,
                                     buffer_size=buffer_size)
        def uniform_iter(m1, m2):
            while True:
                yield np.random.uniform(m1, m2)
        self.sample_length_iter = uniform_iter(min_sequence_length, max_sequence_length)
        self.max_signal_length = max_signal_length

    def __call__(self, ref_seq, nrm_signal):
        pHMM = profileHMM(ref_seq, self.pm, std_scale=2.0)
        pHMM.bake(merge='None')
        hmm_dist, path = pHMM.viterbi(nrm_signal)
        event_means = self.pm.generate_signal(ref_seq, samples=1)
        event_length = np.zeros(len(ref_seq), dtype=np.int32)
        for n, states in itertools.groupby(path, key=lambda x : int(x[:-1])):
            event_length[n] = len(list(states))
        events = np.repeat(event_means, event_length[:len(event_means)])
        event_ids = np.repeat(np.arange(len(event_means), dtype=np.int32), event_length[:len(event_means)])
        event_dist = np.sum(np.abs(events - nrm_signal)) / len(events)
        self.qqueue.push(event_dist, (ref_seq, nrm_signal, event_ids))
        #f, ax = plt.subplots(2, sharex=True)
        #ax[0].plot(events, 'k-')
        #ax[0].plot(nrm_signal, 'b.')
        #ax[0].set_title('Score: {}, distance: {}'.format(hmm_dist, event_dist))
        #ax[1].plot(event_ids)
        #plt.show()
        samples = []
        while len(self.qqueue):
            score, value = self.qqueue.pop()
            if score:
                ref_seq, nrm_signal, event_ids = value
                sample_begin = int(10)
                sample_end = int(sample_begin + next(self.sample_length_iter))
                while sample_end < np.max(event_ids) - 10:
                    sample_sig_begin = np.argmin(np.abs(event_ids - sample_begin))
                    sample_sig_end = np.argmin(np.abs(event_ids - sample_end))
                    sample_sig = nrm_signal[sample_sig_begin:sample_sig_end].copy()
                    sample_sig = (sample_sig - self.pm.model_median) / self.pm.model_MAD
                    sample_seq = ref_seq[sample_begin:sample_end+5]
                    #if len(sample_sig) <= self.max_signal_length:
                    samples.append(((sample_seq, sample_sig), {}))
                    #sample_sim_signal = self.pm.generate_signal(sample_seq)
                    #f, ax = plt.subplots(2)
                    #ax[0].plot(sample_sim_signal)
                    #ax[1].plot(sample_sig)
                    #plt.show()
                    sample_begin = sample_end
                    sample_end = int(sample_begin + next(self.sample_length_iter))
        return samples




# TF record writer
class tf_record_writer():
    def __init__(self, output_file):
        self.writer = tf.io.TFRecordWriter(output_file)
        self.pbar = tqdm(desc="TF records")

    def __del__(self):
        self.writer.flush()
        self.writer.close()
        self.pbar.close()

    def __serialize_example__(self, sequence, signal):
        sequence_b = tf.train.BytesList(value=[sequence.encode("ASCII")])
        signal_b = tf.train.BytesList(value=[tf.io.serialize_tensor(signal.astype(np.float16)).numpy()])
        feature = {
            'sequence' :  tf.train.Feature(bytes_list=sequence_b),
            'signal' : tf.train.Feature(bytes_list=signal_b)
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def __call__(self, sequence, signal):
        example = self.__serialize_example__(sequence, signal)
        self.writer.write(example)
        self.pbar.update(n=1)




# Multiprocess Factory
class mc_factory():
    def __init__(self, job_list):
        self.input_queues = [None] + [Queue(128) for job in job_list[1:]]
        self.output_queues = self.input_queues[1:] + [None]
        self.worker = []
        for job, input_q, output_q in zip(job_list, self.input_queues, self.output_queues):
            t, job_class, args, kwargs = job
            _worker = []
            for i in range(t):
                p = Process(target=self.__worker__, args=(job_class, args, kwargs),
                            kwargs={'input_queue' : input_q, 'output_queue' : output_q})
                _worker.append(p)
                _worker[-1].start()
            self.worker.append(tuple(_worker))

    def __worker__(self, job_class, job_args, job_kwargs, input_queue=None, output_queue=None):
        job_worker = job_class(*job_args, **job_kwargs)
        while True:
            # worker job with input queue
            if input_queue:
                inputs = input_queue.get()
                if inputs is None:
                    break
                for input in inputs:
                    input_args, input_kwargs = input
                    try:
                        job_result = job_worker(*input_args, **input_kwargs)
                        # forward result to next stage
                        if output_queue:
                            output_queue.put(job_result)
                    except Exception:
                        continue
            # generator job
            else:
                try:
                    job_result = next(job_worker)
                    if output_queue:
                        output_queue.put(job_result)
                except StopIteration:
                    break
                except Exception:
                    continue

    def join(self):
        for worker, input_q in zip(self.worker, self.input_queues):
            if not input_q:
                for w in worker:
                    w.join()
            else:
                for _ in range(len(worker)):
                    input_q.put(None)
                for w in worker:
                    w.join()




# main
if __name__ == '__main__':
    signal(SIGPIPE,SIG_DFL)
    parser = argparse.ArgumentParser(description="BumbleBee Training Data")
    parser.add_argument("model", help="Pore model file")
    parser.add_argument("fast5", help="Raw nanopore fast5")
    parser.add_argument("tfrec", help="Output TF record file")
    parser.add_argument("--slicer", type=int, default=1, help="Signal slice worker")
    parser.add_argument("--mapper", type=int, default=1, help="Signal alignment worker")
    parser.add_argument("--max_read_length", type=int, default=50000, help="Maximum read length for input")
    parser.add_argument("--min_seq_length", type=int, default=100, help="Minimum sequence length in training samples")
    parser.add_argument("--max_seq_length", type=int, default=500, help="Maximum sequence length in training samples")
    args = parser.parse_args()

    job_list = [
        (1, sam_parser, [],
            {'drop_quantile':0.1,
             'min_sequence_length':1000,
             'max_sequence_length':args.max_read_length}),
        (args.slicer, signal_slicer, [args.model, args.fast5],
            {'slice_width':args.max_seq_length + 50,
             'buffer_size':64}),
        (args.mapper, signal_aligner, [args.model],
            {'min_sequence_length':args.min_seq_length,
             'max_sequence_length':args.max_seq_length,
             'max_signal_length':args.max_seq_length*15,
             'buffer_size':128}),
        (1, tf_record_writer, [args.tfrec], {})
    ]

    factory = mc_factory(job_list)
    factory.join()
