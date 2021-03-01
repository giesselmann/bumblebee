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
import torch
from torch.autograd import Variable




class BaseModLSTM_v1(torch.nn.Module):
    def __init__(self,
            feature_dim=6,
            k=6, embedding_dim=32, padding_idx=0,
            lstm_dim=64, lstm_layer=1,
            num_classes=2):
        super(ModCall_v1, self).__init__()
        self.lstm_dim = lstm_dim
        self.lstm_layer = lstm_layer
        self.kmer_embedding = torch.nn.Embedding(
            num_embeddings=4**k+1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim + feature_dim,
            hidden_size=lstm_dim,
            num_layers=lstm_layer,
            batch_first=True,
            bidirectional=True
        )
        self.dense = torch.nn.Linear(2*lstm_dim, num_classes)

    def forward(self, lengths, kmers, features):
        # embed kmers
        # (batch_size, seq_len, embedding_dim)
        inner = self.kmer_embedding(kmers)
        # (batch_size, seq_len, embedding_dim+k)
        inner = torch.cat([inner, features], dim=-1)
        # pack inputs
        inner = torch.nn.utils.rnn.pack_padded_sequence(inner, lengths, batch_first=True, enforce_sorted=False)
        # run LSTM
        inner, _  = self.lstm(inner)
        # unpack output
        inner, _ = torch.nn.utils.rnn.pad_packed_sequence(inner, batch_first=True)
        inner_forward = inner[range(len(inner)), lengths - 1, :self.lstm_dim]
        inner_reverse = inner[:, 0, self.lstm_dim:]
        # (batch_size, 2*lstm_dim)
        inner_reduced = torch.cat((inner_forward, inner_reverse), 1)
        # get class label
        # (batch_size, num_classes)
        inner = self.dense(inner_reduced)
        out = torch.nn.functional.softmax(inner, dim=1)
        return out




class SelfAttentionEncoder(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, input):
        pass
