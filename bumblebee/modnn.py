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
import math
import torch




class BaseModLSTM_v1(torch.nn.Module):
    def __init__(self,
            feature_dim=6,
            k=6, embedding_dim=32, padding_idx=0,
            lstm_dim=64, lstm_layer=1,
            num_classes=2):
        super(BaseModLSTM_v1, self).__init__()
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




class BaseModEncoder_v1(torch.nn.Module):
    def __init__(self,
            num_features=6, num_kmers=4**6, num_classes=2,
            embedding_dim=32, padding_idx=0,
            conv_dim=64, conv_kernel=3,
            num_heads=8
            ):
        super(BaseModEncoder_v1, self).__init__()
        self.d_model = embedding_dim + conv_dim
        self.kmer_embedding = torch.nn.Embedding(
            num_embeddings=num_kmers+1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        # N,C,L
        self.conv = torch.nn.Conv1d(num_features, conv_dim, conv_kernel,
            stride=1,
            padding=1)
        # L,N,E
        self.enc = torch.nn.TransformerEncoderLayer(self.d_model, num_heads, self.d_model*4)
        self.lstm_dim = 16
        self.lstm = torch.nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.lstm_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.dense = torch.nn.Linear(2*self.lstm_dim, num_classes)

    def forward(self, lengths, kmers, features):
        batch_size, max_len, n_features = features.size()
        # (batch_size, max_len)
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        mask = mask.cuda(features.get_device()) if features.is_cuda else mask
        # kmer embedding (batch_size, max_len, embedding_dim)
        emb = self.kmer_embedding(kmers)
        # features are (batch_size, max_len, n_features)
        conv = self.conv(features.permute(0, 2, 1)).permute(0, 2, 1)
        inner = torch.cat([emb, conv], dim=-1) * math.sqrt(self.d_model)
        # transformer encoder needs (max_len, batch_size, d_model)
        inner = self.enc(inner.permute(1, 0, 2), src_key_padding_mask = mask).permute(1, 0, 2)
        # LSTM classification
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
