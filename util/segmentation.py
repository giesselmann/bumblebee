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
# Copyright [2019] [Pay Giesselmann, Max Planck Institute for Molecular Genetics]
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
import os
import argparse
import h5py
import edlib
import numpy as np
import scipy.signal as sp
import pomegranate as pg




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
    def __init__(self):
        pass

    def median_MAD(signal):
        median = np.median(signal)
        MAD = np.mean(np.absolute(np.subtract(signal, median)))
        return (median, MAD)

    def quantile_nrm(signal_base, signal_raw):
        base_q = np.quantile(signal_base, np.linspace(0,1,30))
        raw_q = np.quantile(signal_raw, np.linspace(0,1,30))
        p = np.poly1d(np.polyfit(raw_q, base_q, 3))
        return p(signal_raw)

    def flt(raw_signal):
        flt_signal = raw_signal
        raw_median, raw_MAD = signal_processing.median_MAD(raw_signal)
        morph_signal = (flt_signal - raw_median) / raw_MAD
        morph_signal = np.clip(morph_signal * 24 + 127, 0, 255).astype(np.dtype('uint8')).reshape((1, len(morph_signal)))
        flt1 = rectangle(1, 3)
        flt2 = rectangle(1, 3)
        morph_signal = opening(morph_signal, flt1)
        morph_signal = closing(morph_signal, flt2)[0].astype(np.dtype('float'))
        return ((morph_signal - 127) / 24) * raw_MAD + raw_median

    def sig2char(raw_signal, alphabet=string.ascii_uppercase[:5]):
        ords = sorted([ord(x) for x in alphabet])
        quantiles = np.quantile(raw_signal, np.linspace(0,1,len(ords)))
        inds = np.digitize(raw_signal, quantiles).astype(np.int) - 1
        return ''.join([chr(ords[x]) for x in inds])




# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BumbleBee")
    parser.add_argument("model", help="pore model")
    parser.add_argument("sequences", help="Reference span in fasta format")
    parser.add_argument("fast5", help="Raw signal in bulk-fast5 format")
    parser.add_argument("output", help="Signal alignment output in hdf5 format")
    parser.add_argument("--t", type=int, default=1, help="Threads")
    args = parser.parse_args()
