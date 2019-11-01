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
import os, sys, glob
import argparse
import re, itertools, string
import h5py
import edlib
import numpy as np
import scipy.signal as sp
import pomegranate as pg
from tqdm import tqdm
from skimage.morphology import opening, closing, rectangle
from multiprocessing import Process, Queue
#from matplotlib import pyplot as plt



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

    def quantile_nrm(self, signal_raw, q=30):
        base_q = np.quantile(self.model_values, np.linspace(0,1,q))
        raw_q = np.quantile(signal_raw, np.linspace(0,1,q))
        p = np.poly1d(np.polyfit(raw_q, base_q, 3))
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




# semi-global signal alignment
class signal_alignment():
    def __init__(self, model_file, samples=12, alphabet=string.ascii_uppercase[:5], expand=0):
        self.expand = expand
        self.alphabet = alphabet
        self.samples = samples
        self.pm = pore_model(model_file)

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
        return dist, begin, end

    def pos(self, querry, nrm_signal):
        flt_signal = signal_processing.flt(nrm_signal)
        flt_char = signal_processing.sig2char(flt_signal, alphabet=self.alphabet)
        sim_signal = self.pm.generate_signal(querry, samples=self.samples, noise=False)
        sim_char = signal_processing.sig2char(sim_signal, alphabet=self.alphabet)
        dist, begin, end = self.__pos__(sim_char, flt_char)
        return dist, begin, end

    def map(self, querry, nrm_signal):
        algn_dist, algn_begin, algn_end = self.pos(querry, nrm_signal)
        if algn_end - algn_begin > (len(querry) * 15):
            return None
        pHMM = profileHMM(querry, self.pm, std_scale=2.0)
        pHMM.bake(merge='None')
        hmm_dist, path = pHMM.viterbi(nrm_signal[algn_begin:algn_end])
        event_means = self.pm.generate_signal(querry, samples=1)
        event_length = np.zeros(len(querry), dtype=np.int32)
        for n, states in itertools.groupby(path, key=lambda x : int(x[:-1])):
            event_length[n] = len(list(states))
        events = np.repeat(event_means, event_length[:len(event_means)])
        event_ids = np.repeat(np.arange(len(event_means), dtype=np.int32), event_length[:len(event_means)])
        event_dist = np.sum(np.abs(events - nrm_signal[algn_begin:algn_end])) / len(events)
        #f, ax = plt.subplots(2, sharex=True)
        #ax[0].plot(events, 'k-')
        #ax[0].plot(nrm_signal[algn_begin:algn_end], 'b.')
        #ax[0].set_title('Score: {}, distance: {}'.format(hmm_dist, event_dist))
        #ax[1].plot(event_ids)
        #plt.show()
        return hmm_dist, event_dist, event_length, nrm_signal[algn_begin:algn_end]

    def slice(self, id, sequence, signal, slice_width=500):
        nrm_signal = self.pm.quantile_nrm(signal)
        slices = []
        for i, (slice_begin, slice_end) in enumerate(zip(range(0,len(sequence)-slice_width, slice_width),
                                          range(slice_width, len(sequence), slice_width))):
            slice_sequence = sequence[slice_begin:slice_end]
            mapping = self.map(slice_sequence, nrm_signal)
            if not mapping:
                continue
            hmm_dist, event_dist, event_lengths, event_values = mapping
            slices.append((id + '_' + str(i).rjust(4,'0'),
                           slice_sequence,
                           event_lengths, event_values,
                           hmm_dist, event_dist))
        return slices




class alignment_file():
    summary_dtype = np.dtype([('ID', 'S41'),
                              ('seq_begin', 'u8'),
                              ('seq_end', 'u8'),
                              ('raw_begin', 'u8'),
                              ('raw_end', 'u8'),
                              ('logp', 'f4'),
                              ('dist', 'f4')])
    event_dtype = np.dtype([('sequence', 'S1'),
                            ('length', 'u4')])

    def write_batch(batch_file, slices):
        summary_table = np.empty(shape=len(slices), dtype=alignment_file.summary_dtype)
        ids, sequences, event_lengths, event_values, hmm_dists, event_dists = zip(*slices)
        summary_table['ID'] = list(ids)
        seq_len = np.array([len(s) for s in sequences], dtype=np.uint64)
        seq_off = np.hstack([np.zeros(1), np.cumsum(seq_len)[:-1]])
        raw_len = np.array([len(s) for s in event_values], dtype=np.uint64)
        raw_off = np.hstack([np.zeros(1), np.cumsum(raw_len)[:-1]])
        summary_table['seq_begin'] = seq_off
        summary_table['seq_end'] = seq_off + seq_len
        summary_table['raw_begin'] = raw_off
        summary_table['raw_end'] = raw_off + raw_len
        summary_table['logp'] = hmm_dists
        summary_table['dist'] = event_dists
        # write to disk
        with h5py.File(batch_file, 'w') as fp_h5:
            sequence_dst = fp_h5.create_dataset('seq', (np.sum(seq_len),), dtype=alignment_file.event_dtype)
            raw_dst = fp_h5.create_dataset('raw', (np.sum(raw_len),), compression="gzip", dtype=np.float32)
            sequence_dst['sequence'] = np.array(list(''.join(sequences)), dtype='S1')
            sequence_dst['length'] = np.hstack(event_lengths)
            raw_dst[:] = np.hstack(event_values)
            summary_dst = fp_h5.create_dataset("summary", data=summary_table, shuffle=True)

    def merge_batches(batch_dir, virtual_file, recursive=False):
        input_files = []
        for d in batch_dir:
            if recursive:
                input_files.extend([os.path.join(dirpath, f)
                    for dirpath, _, files in os.walk(d) for f in files if f.endswith('.hdf5')])
            else:
                input_files.extend(glob.glob(os.path.join(d, '*.hdf5')))
        if len(input_files) == 0:
            return
        summary_sources = []
        seq_sources = []
        raw_sources = []
        input_files_relative = [os.path.relpath(input_file, os.path.commonpath([virtual_file, input_file])) for input_file in input_files]
        for input_file in input_files:
            input_file_relative = os.path.relpath(input_file, os.path.commonpath([virtual_file, input_file]))
            with h5py.File(input_file, 'r') as fp_h5:
                summary_sources.append(h5py.VirtualSource(input_file_relative, 'summary', shape=fp_h5['summary'].shape))
                summary_dtype = fp_h5['summary'].dtype
                seq_sources.append(h5py.VirtualSource(input_file_relative, 'seq', shape=fp_h5['seq'].shape))
                seq_dtype = fp_h5['seq'].dtype
                raw_sources.append(h5py.VirtualSource(input_file_relative, 'raw', shape=fp_h5['raw'].shape))
                raw_dtype = fp_h5['raw'].dtype

        summary_layout = h5py.VirtualLayout(shape=(np.sum([x.shape[0] for x in summary_sources]), ), dtype=summary_dtype)
        seq_layout = h5py.VirtualLayout(shape=(len(seq_sources), np.max([x.shape[0] for x in seq_sources])), dtype=seq_dtype)
        raw_layout = h5py.VirtualLayout(shape=(len(raw_sources), np.max([x.shape[0] for x in raw_sources])), dtype=raw_dtype)
        summary_offset = 0
        for i, (summary_src, seq_src, raw_src) in enumerate(zip(summary_sources, seq_sources, raw_sources)):
            summary_layout[summary_offset:summary_offset+summary_src.shape[0]] = summary_src
            summary_offset += summary_src.shape[0]
            seq_layout[i,:seq_src.shape[0]] = seq_src
            raw_layout[i,:raw_src.shape[0]] = raw_src
        with h5py.File(virtual_file, 'w') as fp_h5:
            fp_h5.create_virtual_dataset('summary', summary_layout)
            fp_h5.create_virtual_dataset('seq', seq_layout, fillvalue=np.array([('N', 0)], dtype=alignment_file.event_dtype))
            fp_h5.create_virtual_dataset('raw', raw_layout, fillvalue=0.0)
            fp_h5.create_dataset('batch', data=np.repeat(np.arange(len(summary_sources)), [x.shape[0] for x in summary_sources]))
            fp_h5.create_dataset('batch_files', data='\n'.join(input_files_relative))




def alignment_worker(model_file, raw_file, seq_queue, aln_queue):
    sa = signal_alignment(model_file)
    with h5py.File(raw_file, 'r') as fp_f5:
        while True:
            input = seq_queue.get()
            if not input:
                print("[DEBUG] worker terminating.", file=sys.stderr)
                break
            ID, sequence = input
            if 'read_' + ID in fp_f5:
                raw_signal = fp_f5['read_' + ID]['Raw/Signal'][...].astype(np.float)
                slices = sa.slice(ID, sequence, raw_signal)
                if slices:
                    aln_queue.put(slices)
                else:
                    print("[DEBUG] received empty slice.", file=sys.stderr)
            else:
                print("[DEBUG] read {} not in input file".format(ID), file=sys.stderr)
    aln_queue.close()



def alignment_writer(aln_queue, output_file):
    slices = []
    while True:
        input = aln_queue.get()
        if not input:
            break
        slices.extend(input)
    if slices:
        alignment_file.write_batch(output_file, slices)




class main():
    def __init__(self):
        parser = argparse.ArgumentParser(
        description='Bumblebee segmentation: a nanopore raw signal something',
        usage='''segmentation.py <command> [<args>]
Available commands are:
   align      Signal alignment of bulk-fast5 batches
   merge      Merge alignment batches in h5 virtual dataset
''')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command', file=sys.stderr)
            parser.print_help(file=sys.stderr)
            exit(1)
        getattr(self, args.command)(sys.argv[2:])

    def filter(self, argv):
        parser = argparse.ArgumentParser(description="BumbleBee")
        parser.add_argument("model", help="Pore model")
        args = parser.parse_args(argv)
        for sam_line in sys.stdin:
            sam_fields = sam_line.strip().split('\t')
            sam_tags = {f.split(':')[0]:f.split(':')[2] for f in sam_fields[11:]}
            sam_read_quality = sam_fields[10]

            ops = [(int(op[:-1]), op[-1]) for op in re.findall('(\d*\D)',cigar)]

            
            sam_algn_quality = int(sam_tags.get("AS") or '0') / np.sum([])


    def align(self, argv):
        parser = argparse.ArgumentParser(description="BumbleBee")
        parser.add_argument("model", help="Pore model")
        parser.add_argument("sequences", help="Reference spans in fasta format")
        parser.add_argument("fast5", help="Raw signals in bulk-fast5 format")
        parser.add_argument("output", help="Signal alignment output in hdf5 format")
        parser.add_argument("--t", type=int, default=1, help="Threads")
        args = parser.parse_args(argv)
        # start worker and writer
        seq_queue = Queue(args.t * 2)
        aln_queue = Queue()
        worker = []
        for i in range(args.t):
            worker.append(Process(target=alignment_worker,
                                    args=(args.model, args.fast5, seq_queue, aln_queue,)))
            worker[-1].start()
        writer = Process(target=alignment_writer, args=(aln_queue, args.output,))
        writer.start()
        def fa_parser(iterable):
            fa_iter = iter(iterable)
            while True:
                try:
                    name = next(fa_iter).strip()
                    ID = name[1:].split()[0]
                    sequence = next(fa_iter).upper()
                    yield ID, sequence
                except StopIteration:
                    return
        with open(args.sequences, 'r') as fp_fa:
            for ID, sequence in tqdm(fa_parser(fp_fa), desc='Processing:', ncols=0, unit='read'):
                seq_queue.put((ID, sequence))
        for _ in range(args.t):
            seq_queue.put(None)
        for w in worker:
            w.join()
        aln_queue.put(None)
        writer.join()

    def merge(self, argv):
        parser = argparse.ArgumentParser(description="BumbleBee")
        parser.add_argument("output", help="Output file")
        parser.add_argument("input", nargs='+', help="Batch directory")
        parser.add_argument("--recursive", action='store_true', help="Scan batch directory recursively")
        args = parser.parse_args(argv)
        alignment_file.merge_batches(args.input, args.output, recursive=args.recursive)

    def convert(self, argv):
        parser = argparse.ArgumentParser(description="BumbleBee")
        parser.add_argument("input", help="Input file")
        parser.add_argument("output", help="Output file")
        args = parser.parse_args(argv)
        with h5py.File(args.output, 'w') as fp_out, h5py.File(args.input, 'r') as fp_in:
            fp_out.create_dataset("summary", data=fp_in['summary'][...], shuffle=True)
            fp_out.create_dataset("seq", data=fp_in['seq'][...], shuffle=True)
            fp_out.create_dataset("raw", data=fp_in['raw'][...].astype(np.float16), compression='gzip', chunks=(2048,))




# main
if __name__ == "__main__":
    main()
