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

from bumblebee.nn import PositionalEncoding
from bumblebee.nn import ResidualNetwork


class BaseEncoder(torch.nn.Module):
    def __init__(self, state_length, alphabet, config={}):
        super(BaseEncoder, self).__init__()
        self.state_length = state_length
        d_model = config.get('d_model') or 256
        embedding_dim = config.get('embedding_dim') or 32
        num_actions = len(alphabet) + 3
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_actions,
            padding_idx=0,
            embedding_dim=embedding_dim
        )
        self.input_nn = ResidualNetwork(
            config.get('num_features') or 9,
            d_model - embedding_dim,
            config.get('input_nn_layer') or [64, 128])
        self.pos_encoder = PositionalEncoding(
            d_model,
            dropout=config.get('dropout') or 0.1,
            max_len=state_length)
        # encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.get('num_heads') or 4,
            dim_feedforward=d_model*4,
            activation='gelu')
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=config.get('num_layer') or 1)
        self.output_nn = ResidualNetwork(
            d_model, num_actions,
            config.get('output_nn_layer') or [128, 64])

    def forward(self, events, sequence, mask=None):
        # sequence embedding
        emb = self.embedding(sequence)
        # event feature network
        # (batch_size, state_length, d_model - d_emb)
        inner = self.input_nn(events)
        # concat emb and signal features
        inner = torch.cat([emb, inner], dim=-1)
        # positional encoding
        inner = self.pos_encoder(inner)
        # transformer encoder needs (state_length, batch_size, d_model)
        # Encoder is (C, N, L)
        inner = self.transformer_encoder(inner.permute(1, 0, 2),
            mask=mask).permute(1, 0, 2)
        # (batch_size, state_length, num_actions)
        inner = self.output_nn(inner)
        action = torch.mean(inner, dim=1)
        return action




class BaseTransformer(torch.nn.Module):
    def __init__(self, state_length, alphabet, config={}):
        super(BaseTransformer, self).__init__()
        self.state_length = state_length
        d_model = config.get('d_model') or 256
        #embedding_dim = config.get('embedding_dim') or 32
        num_actions = len(alphabet) + 3
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_actions,
            padding_idx=0,
            embedding_dim=d_model
        )
        self.input_nn = ResidualNetwork(
            config.get('num_features') or 9,
            d_model,
            config.get('input_nn_layer') or [64, 128])
        self.pos_encoder = PositionalEncoding(
            d_model,
            dropout=config.get('dropout') or 0.1,
            max_len=state_length)
        # encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.get('num_heads') or 4,
            dim_feedforward=d_model*4,
            activation='relu')
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=config.get('num_layer') or 1,
            norm=torch.nn.LayerNorm(d_model))
        self.decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.get('num_heads') or 4,
            dim_feedforward=d_model*4,
            activation='relu')
        self.transformer_decoder = torch.nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=config.get('num_layer') or 1,
            norm=torch.nn.LayerNorm(d_model))
        self.output_nn = ResidualNetwork(
            d_model, num_actions,
            config.get('output_nn_layer') or [128, 64])
        self._reset_parameters()

    def forward(self, events, sequence, mask=None):
        # sequence embedding
        tgt = self.embedding(sequence)
        tgt = self.pos_encoder(tgt).permute(1, 0, 2)
        # event feature network
        # (batch_size, state_length, d_model)
        inner = self.input_nn(events)
        # positional encoding
        inner = self.pos_encoder(inner)
        # transformer encoder needs (state_length, batch_size, d_model)
        # Encoder is (C, N, L)
        memory = self.transformer_encoder(inner.permute(1, 0, 2),
            mask=mask)
        inner = self.transformer_decoder(tgt, memory,
            tgt_mask=mask).permute(1, 0, 2)
        # (batch_size, state_length, num_actions)
        inner = self.output_nn(inner)
        #action = torch.mean(inner, dim=1)
        action = torch.squeeze(inner[:,-1,:], dim=1)
        return action

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
