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
            rnn_type='LSTM', rnn_dim=64, rnn_layer=1,
            dropout=0.1,
            num_classes=2):
        super(BaseModLSTM_v1, self).__init__()
        self.rnn_dim = rnn_dim
        self.rnn_layer = rnn_layer
        self.kmer_embedding = torch.nn.Embedding(
            num_embeddings=4**k+1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.rnn = getattr(torch.nn, rnn_type)(
            input_size=embedding_dim + feature_dim,
            hidden_size=rnn_dim,
            num_layers=rnn_layer,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dense = torch.nn.Linear(2*rnn_dim, num_classes)

    def forward(self, lengths, kmers, features):
        # embed kmers
        # (batch_size, seq_len, embedding_dim)
        inner = self.kmer_embedding(kmers)
        # (batch_size, seq_len, embedding_dim+k)
        inner = torch.cat([inner, features], dim=-1)
        # pack inputs
        inner = torch.nn.utils.rnn.pack_padded_sequence(inner, lengths, batch_first=True, enforce_sorted=False)
        # run LSTM
        inner, _  = self.rnn(inner)
        # unpack output
        inner, _ = torch.nn.utils.rnn.pad_packed_sequence(inner, batch_first=True)
        inner_forward = inner[range(len(inner)), lengths - 1, :self.rnn_dim]
        inner_reverse = inner[:, 0, self.rnn_dim:]
        # (batch_size, 2*rnn_dim)
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
            d_model=128, num_heads=8, num_layer=1
            ):
        super(BaseModEncoder_v1, self).__init__()
        self.d_model = d_model
        self.kmer_embedding = torch.nn.Embedding(
            num_embeddings=num_kmers+1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.linear1 = torch.nn.Linear(num_features+embedding_dim, d_model)
        self.act1 = torch.nn.GELU()
        # encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model,
                        nhead=num_heads,
                        dim_feedforward=d_model*4,
                        activation='gelu')
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                        num_layers=num_layer)
        #self.act2 = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(d_model, num_classes)

    def forward(self, lengths, kmers, features):
        batch_size, max_len, n_features = features.size()
        # (batch_size, max_len)
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        mask = mask.cuda(features.get_device()) if features.is_cuda else mask
        lengths = lengths.cuda(features.get_device()) if features.is_cuda else lengths
        # kmer embedding (batch_size, max_len, embedding_dim)
        emb = self.kmer_embedding(kmers)
        # concat signal features and sequence embeddings
        inner = torch.cat([emb, features], dim=-1)
        # generate features as
        # (batch_size, max_len, d_model)
        inner = self.linear1(inner)
        inner = self.act1(inner)
        # transformer encoder needs (max_len, batch_size, d_model)
        inner = self.transformer_encoder(inner.permute(1, 0, 2),
                        src_key_padding_mask = mask).permute(1, 0, 2)
        # get class label
        # (batch_size, max_len, num_classes)
        inner = self.linear2(inner)
        #inner = self.act2(inner)
        inner = torch.mul(inner, ~mask[:,:,None])
        # melt to
        # (batch_size, num_classes)
        inner = torch.sum(inner, dim=1) / lengths[:,None]
        out = torch.nn.functional.softmax(inner, dim=1)
        return out
