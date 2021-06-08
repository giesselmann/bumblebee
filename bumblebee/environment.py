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
import logging
import random
import numpy as np
#import matplotlib.pyplot as plt
import pkg_resources as pkg
from collections import namedtuple

from bumblebee.poremodel import PoreModel
from bumblebee.worker import ReadSource, EventAligner


log = logging.getLogger(__name__)


EnvReadEventsEpisode = namedtuple('EnvReadEventsEpisode',
    ['read', 'events'])



class EnvReadEvents():
    def __init__(self, fast5, bam,
            state_length=50,
            alphabet='ACGT',
            config={}):
        self.state_length = state_length
        self.min_seq_length = config.get('min_seq_length') or 500
        self.max_seq_length =  config.get('max_seq_length') or 10000
        self.max_events = config.get('max_events')
        self.alphabet = alphabet
        self.encode = {c:i+1 for i, c in enumerate('^$' + self.alphabet)}
        self.num_actions = len(self.encode) + 1
        min_score = config.get('min_score') or 1.0
        self.clip = config.get('clip') or 15
        self.read_source = ReadSource(fast5, bam,
            min_seq_length=self.min_seq_length,
            max_seq_length=self.max_seq_length)
        self.event_aligner = EventAligner(
            min_score=min_score)
        self.event_columns = ['event_min', 'event_mean', 'event_median',
                              'event_std', 'event_max', 'event_length']
        self.pm = PoreModel(pkg.resource_filename('bumblebee',
            'data/r9.4_450bps.model'))

    def episodes(self):
        for read in self.read_source():
            algn = self.event_aligner(*read)
            if algn is not None:
                read, df_events, _ = algn
                yield EnvReadEventsEpisode(read, df_events)

    def __encode_sequence__(self, sequence):
        return np.array([self.encode.get(c) or 0 for c in sequence], dtype=np.int64)

    def __get_state__(self):
        # numpy slice [begin, end)
        seq_slice = np.s_[self.seq_step_idx:
            self.seq_step_idx+self.state_length]
        # pandas slice [begin, end]
        ev_slice = np.s_[self.event_step_idx:
            self.event_step_idx+self.state_length-1]
        events = self.episode_events.loc[ev_slice,
            self.event_columns].to_numpy().astype(np.float32)
        sequence_token = self.predicted_seq[seq_slice]
        state = (events, sequence_token)
        return state

    def reset(self, episode):
        self.seq_step_idx = 0
        self.event_step_idx = 0
        self.episode_events = episode.events.reset_index(drop=True)
        if self.max_events:
            self.episode_events = self.episode_events[:self.max_events]
        self.target_seq = self.__encode_sequence__(episode.read.ref_span.seq[
            self.episode_events.sequence_offset.min():
            self.episode_events.sequence_offset.max() + self.pm.k] + '$')
        self.predicted_seq = np.zeros(len(self.target_seq) + self.state_length,
            dtype=np.int64)
        self.predicted_seq[:self.state_length] = self.__encode_sequence__(('^' * self.state_length))
        self.target_seq_len = len(self.target_seq)
        self.episode_ev_len = len(self.episode_events)
        self.matches = 0
        self.shifts = 0
        self.false_stops = 0
        return self.__get_state__()

    def step(self, action):
        # actions are SHIFT, START, STOP, Tokens[...]
        done = False
        if action == 0:
            # SHIFT
            reward = np.clip((self.seq_step_idx - self.event_step_idx)/10, -2, 2)
            self.event_step_idx = min(
                self.event_step_idx + 10,
                self.episode_ev_len - self.state_length)
            self.shifts += 1
        else:
            # PREDICT
            if action == self.target_seq[self.seq_step_idx]:
                reward = 1.0
                self.matches += 1
                if action == 2:
                    # True stop token
                    reward += 5
                    done = True
            else:
                reward = -1.0
                if action == 2:
                    # False stop token
                    reward -= 1
                    self.false_stops += 1
            self.predicted_seq[self.seq_step_idx+self.state_length] = action
            self.seq_step_idx = min(self.seq_step_idx + 1, self.target_seq_len)
        next_state = self.__get_state__()
        done = self.seq_step_idx == self.target_seq_len or done
        info = {
            'matches': self.matches / self.target_seq_len,
            'shifts': self.shifts / self.episode_ev_len,
            'false_stops': self.false_stops / self.target_seq_len
            }
        return (next_state,
                reward,
                done,
                info)
