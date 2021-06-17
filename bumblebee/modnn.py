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

from bumblebee.nn import BiDirLSTM
from bumblebee.nn import PositionalEncoding
from bumblebee.nn import ResidualNetwork, ConvolutionalNetwork
from bumblebee.nn import TransformerACTEncoder




class BaseModLSTM_v1(torch.nn.Module):
    def __init__(self, max_features, config={}):
        super(BaseModLSTM_v1, self).__init__()
        num_features = config.get("num_features") or 6
        feature_window = config.get("feature_window") or 20
        num_kmers = config.get('num_kmers') or 4096
        embedding_dim = config.get('embedding_dim') or 32
        padding_idx = config.get("padding_idx") or 0
        dropout = config.get("dropout") or 0.1
        input_nn_dims = config.get("input_nn_dims") or [64, 128]
        rnn_type = config.get("rnn_type") or "LSTM"
        d_model = config.get("d_model") or 128
        num_layer = config.get("num_layer") or 3
        num_classes = config.get("num_classes") or 2
        self.d_model = d_model
        self.num_layer = num_layer
        self.kmer_embedding = torch.nn.Embedding(
                num_embeddings=num_kmers + 1,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx)
        self.input_nn = ResidualNetwork(num_features,# + embedding_dim,
                d_model - embedding_dim,
                input_nn_dims,
                dropout=dropout)
        self.offset_embedding = torch.nn.Embedding(
                num_embeddings=feature_window,
                embedding_dim=d_model,
                padding_idx=padding_idx)
        self.rnn = BiDirLSTM(d_model, num_layer,
                dropout=dropout,
                rnn_type=rnn_type)
        self.linear = torch.nn.Linear(d_model, num_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, lengths, kmers, offsets, features):
        # kmer embedding (batch_size, max_len, embedding_dim)
        emb = self.kmer_embedding(kmers)
        # generate features as
        # (batch_size, max_len, d_model - embedding_dim)
        inner = self.input_nn(features)
        # concat signal features and sequence embeddings
        inner = torch.cat([emb, inner], dim=-1)
        # positional encoding
        inner = inner + self.offset_embedding(offsets)
        # run LSTM
        inner = self.rnn(inner, lengths)
        # get class label
        # (batch_size, num_classes)
        out = self.linear(inner)
        return out, None, {}




class BaseModLSTM_v2(torch.nn.Module):
    def __init__(self, max_features, config={}):
        super(BaseModLSTM_v2, self).__init__()
        num_features = config.get("num_features") or 6
        feature_window = config.get("feature_window") or 20
        num_kmers = config.get('num_kmers') or 4096
        embedding_dim = config.get('embedding_dim') or 32
        padding_idx = config.get("padding_idx") or 0
        dropout = config.get("dropout") or 0.1
        input_nn_dims = config.get("input_nn_dims") or [64, 128]
        rnn_type = config.get("rnn_type") or "LSTM"
        d_model = config.get("d_model") or 128
        num_layer = config.get("num_layer") or 3
        output_nn_dims = config.get('output_nn_dims') or [128, 64]
        num_classes = config.get("num_classes") or 2
        self.d_model = d_model
        self.num_layer = num_layer
        self.kmer_embedding = torch.nn.Embedding(
                num_embeddings=num_kmers + 1,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx)
        self.input_nn = ResidualNetwork(num_features,# + embedding_dim,
                d_model - embedding_dim,
                input_nn_dims,
                dropout=dropout)
        self.offset_embedding = torch.nn.Embedding(
                num_embeddings=feature_window,
                embedding_dim=d_model,
                padding_idx=padding_idx)
        self.rnn = BiDirLSTM(d_model, num_layer,
                dropout=dropout,
                rnn_type=rnn_type)
        self.output_nn = ResidualNetwork(d_model,
                num_classes,
                output_nn_dims,
                dropout=dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, lengths, kmers, offsets, features):
        # kmer embedding (batch_size, max_len, embedding_dim)
        emb = self.kmer_embedding(kmers)
        # generate features as
        # (batch_size, max_len, d_model - embedding_dim)
        inner = self.input_nn(features)
        # concat signal features and sequence embeddings
        inner = torch.cat([emb, inner], dim=-1)
        # positional encoding
        inner = inner + self.offset_embedding(offsets)
        # run LSTM
        inner = self.rnn(inner, lengths)
        # get class label
        # (batch_size, num_classes)
        out = self.output_nn(inner)
        return out, None, {}




class BaseModEncoder(torch.nn.Module):
    def __init__(self, max_features, config={}):
        super(BaseModEncoder, self).__init__()
        # default config
        num_features = config.get("num_features") or 6
        num_kmers = config.get('num_kmers') or 4096
        embedding_dim = config.get('embedding_dim') or 32
        padding_idx = config.get("padding_idx") or 0
        dropout = config.get("dropout") or 0.1
        input_nn_dims = config.get("input_nn_dims") or [64, 128, 256]
        d_model = config.get("d_model") or 512
        num_heads = config.get("num_heads") or 4
        num_layer = config.get("num_layer") or 3
        output_nn_dims = config.get("output_nn_dims") or [512, 256, 128, 64]
        num_classes = config.get("num_classes") or 2
        # layer
        self.kmer_embedding = torch.nn.Embedding(
                num_embeddings=num_kmers + 1,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx)
        self.input_nn = ResidualNetwork(num_features,# + embedding_dim,
                d_model - embedding_dim,
                input_nn_dims,
                dropout=dropout)
        self.pos_encoder = PositionalEncoding(d_model,
                dropout=dropout,
                max_len=max_features)
        # encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model*4,
                activation='relu',
                dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(
                self.encoder_layer,
                num_layers=num_layer)
        self.output_nn = ResidualNetwork(d_model,
                num_classes,
                output_nn_dims,
                dropout=dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, lengths, kmers, offsets, features):
        batch_size, max_len, n_features = features.size()
        # (batch_size, max_len)
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        mask = mask.cuda(features.get_device()) if features.is_cuda else mask
        lengths = lengths.cuda(features.get_device()) if features.is_cuda else lengths
        # kmer embedding (batch_size, max_len, embedding_dim)
        emb = self.kmer_embedding(kmers)
        # generate features as
        # (batch_size, max_len, d_model - embedding_dim)
        inner = self.input_nn(features)
        # concat signal features and sequence embeddings
        inner = torch.cat([emb, inner], dim=-1)
        # positional encoding
        inner = self.pos_encoder(inner)
        # transformer encoder needs (max_len, batch_size, d_model)
        inner = self.transformer_encoder(inner.permute(1, 0, 2),
                        src_key_padding_mask = mask).permute(1, 0, 2)
        # get class label
        # (batch_size, max_len, num_classes)
        inner = self.output_nn(inner)
        # melt to
        # (batch_size, num_classes)
        inner = torch.mul(inner, ~mask[:,:,None])
        out = torch.sum(inner, dim=1) / lengths[:,None]
        return out, None, {}




class BaseModEncoder_v2(torch.nn.Module):
    def __init__(self, max_features, config={}):
        super(BaseModEncoder_v2, self).__init__()
        # default config
        num_features = config.get("num_features") or 6
        feature_window = config.get("feature_window") or 20
        num_kmers = config.get('num_kmers') or 4096
        embedding_dim = config.get('embedding_dim') or 32
        padding_idx = config.get("padding_idx") or 0
        dropout = config.get("dropout") or 0.1
        input_nn_dims = config.get("input_nn_dims") or [64, 128, 256]
        d_model = config.get("d_model") or 512
        num_heads = config.get("num_heads") or 4
        num_layer = config.get("num_layer") or 3
        output_nn_dims = config.get("output_nn_dims") or [512, 256, 128, 64]
        num_classes = config.get("num_classes") or 2
        # layer
        self.kmer_embedding = torch.nn.Embedding(
                num_embeddings=num_kmers + 1,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx)
        self.input_nn = ResidualNetwork(num_features,# + embedding_dim,
                d_model - embedding_dim,
                input_nn_dims,
                dropout=dropout)
        self.offset_embedding = torch.nn.Embedding(
                num_embeddings=feature_window,
                embedding_dim=d_model,
                padding_idx=padding_idx)
        # encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model*4,
                activation='relu',
                dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(
                self.encoder_layer,
                num_layers=num_layer)
        self.output_nn = ResidualNetwork(d_model,
                num_classes,
                output_nn_dims,
                dropout=dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, lengths, kmers, offsets, features):
        batch_size, max_len, n_features = features.size()
        # (batch_size, max_len)
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        mask = mask.cuda(features.get_device()) if features.is_cuda else mask
        lengths = lengths.cuda(features.get_device()) if features.is_cuda else lengths
        # kmer embedding (batch_size, max_len, embedding_dim)
        emb = self.kmer_embedding(kmers)
        # generate features as
        # (batch_size, max_len, d_model - embedding_dim)
        inner = self.input_nn(features)
        # concat signal features and sequence embeddings
        inner = torch.cat([emb, inner], dim=-1)
        # positional encoding
        inner = inner + self.offset_embedding(offsets)
        # transformer encoder needs (max_len, batch_size, d_model)
        inner = self.transformer_encoder(inner.permute(1, 0, 2),
                        src_key_padding_mask = mask).permute(1, 0, 2)
        # get class label
        # (batch_size, max_len, num_classes)
        inner = self.output_nn(inner)
        # melt to
        # (batch_size, num_classes)
        inner = torch.mul(inner, ~mask[:,:,None])
        out = torch.sum(inner, dim=1) / lengths[:,None]
        return out, None, {}




class BaseModACTEncoder(torch.nn.Module):
    def __init__(self, max_features,
            num_features=6, num_kmers=4**6, num_classes=2,
            embedding_dim=32, padding_idx=0,
            d_model=512, num_heads=4, max_depth=3,
            clone=True, time_penalty=0.05
            ):
        super(BaseModACTEncoder, self).__init__()
        self.d_model = d_model
        self.kmer_embedding = torch.nn.Embedding(
            num_embeddings=num_kmers+1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.input_nn = ResidualNetwork(num_features + embedding_dim,
            d_model,
            [64, 128, 256])
        # encoder
        self.encoder = TransformerACTEncoder(d_model,
            max_len=max_features,
            num_heads=num_heads,
            max_depth=max_depth,
            clone=clone,
            time_penalty=time_penalty)
        self.output_nn = ResidualNetwork(d_model,
            num_classes,
            [512, 256, 128])

    def forward(self, lengths, kmers, offsets, features):
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
        inner = self.input_nn(inner)
        # encoder (batch_size, max_len, d_model)
        inner, act_loss, ponder_time, remainder = self.encoder(inner, mask)
        # get class label
        # (batch_size, max_len, num_classes)
        inner = self.output_nn(inner)
        inner = torch.mul(inner, ~mask[:,:,None])
        # melt to
        # (batch_size, num_classes)
        inner = torch.sum(inner, dim=1) / lengths[:,None]
        out = torch.nn.functional.softmax(inner, dim=1)
        return out, act_loss, {'ponder_time': ponder_time,
                               'remainder': remainder}




class BaseModTransformer(torch.nn.Module):
    def __init__(self, max_features, config={}):
        super(BaseModTransformer, self).__init__()
        # default config
        num_features = config.get("num_features") or 6
        feature_window = config.get("feature_window") or 20
        num_kmers = config.get('num_kmers') or 4096
        padding_idx = config.get("padding_idx") or 0
        dropout = config.get("dropout") or 0.1
        input_nn_dims = config.get("input_nn_dims") or [64, 128, 256]
        d_model = config.get("d_model") or 512
        num_heads = config.get("num_heads") or 4
        num_layer = config.get("num_layer") or 3
        output_nn_dims = config.get("output_nn_dims") or [512, 256, 128, 64]
        num_classes = config.get("num_classes") or 2
        # layer
        self.kmer_embedding = torch.nn.Embedding(
                num_embeddings=num_kmers + 1,
                embedding_dim=d_model,
                padding_idx=padding_idx)
        self.input_nn = ResidualNetwork(num_features,
                d_model,
                input_nn_dims,
                dropout=dropout)
        self.offset_embedding = torch.nn.Embedding(
                num_embeddings=feature_window,
                embedding_dim=d_model,
                padding_idx=padding_idx)
        # encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model*4,
                activation='gelu',
                dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(
                self.encoder_layer,
                num_layers=num_layer,
                norm=torch.nn.LayerNorm(d_model))
        self.decoder_layer = torch.nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model*4,
                activation='gelu',
                dropout=dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(
                self.decoder_layer,
                num_layers=num_layer,
                norm=torch.nn.LayerNorm(d_model))
        self.output_nn = ResidualNetwork(d_model,
                num_classes,
                output_nn_dims,
                dropout=dropout)
        self._reset_parameters()

    def forward(self, lengths, kmers, offsets, features):
        batch_size, max_len, n_features = features.size()
        # (batch_size, max_len)
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        mask = mask.cuda(features.get_device()) if features.is_cuda else mask
        lengths = lengths.cuda(features.get_device()) if features.is_cuda else lengths
        # kmer embedding (batch_size, max_len, d_model)
        target = self.kmer_embedding(kmers)
        # positional encoding
        target = (target + self.offset_embedding(offsets)).permute(1, 0, 2)
        # generate features as
        # (batch_size, max_len, d_model)
        inner = self.input_nn(features)
        # positional encoding
        inner = (inner + self.offset_embedding(offsets)).permute(1, 0, 2)
        # transformer encoder needs (max_len, batch_size, d_model)
        memory = self.transformer_encoder(inner,
                        src_key_padding_mask=mask)
        # decoder
        inner = self.transformer_decoder(target, memory,
                    tgt_key_padding_mask=mask,
                    memory_key_padding_mask=mask).permute(1, 0, 2)
        # get class label
        # (batch_size, max_len, num_classes)
        inner = self.output_nn(inner)
        # melt to
        # (batch_size, num_classes)
        inner = torch.mul(inner, ~mask[:,:,None])
        out = torch.sum(inner, dim=1) / lengths[:,None]
        return out, None, {}

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
