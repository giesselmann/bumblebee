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




# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)




class ResidualNetwork(torch.nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super(ResidualNetwork, self).__init__()
        dims = [input_dim]+hidden_layer_dims
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        self.res_fcs = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else torch.nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = torch.nn.ModuleList([torch.nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = torch.nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)
        return self.fc_out(fea)




class BiDirLSTM(torch.nn.Module):
    def __init__(self, d_model, num_layer, dropout=0.1, rnn_type='LSTM'):
        super(BiDirLSTM, self).__init__()
        self.d_model = d_model
        self.rnn = getattr(torch.nn, rnn_type)(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layer,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.linear = torch.nn.Linear(2*d_model, d_model)

    def forward(self, input, lengths):
        # pack inputs
        inner = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        # run LSTM
        inner, _  = self.rnn(inner)
        # unpack output
        inner, _ = torch.nn.utils.rnn.pad_packed_sequence(inner, batch_first=True)
        inner_forward = inner[range(len(inner)), lengths - 1, :self.d_model]
        inner_reverse = inner[:, 0, self.d_model:]
        # (batch_size, 2*d_model)
        inner_reduced = torch.cat((inner_forward, inner_reverse), 1)
        out = self.linear(inner_reduced)
        return out




class BaseModLSTM_v1(torch.nn.Module):
    def __init__(self,
            num_features=6,
            k=6, embedding_dim=32, padding_idx=0,
            rnn_type='LSTM', d_model=64, num_layer=1,
            dropout=0.1,
            num_classes=2):
        super(BaseModLSTM_v1, self).__init__()
        self.d_model = d_model
        self.num_layer = num_layer
        self.kmer_embedding = torch.nn.Embedding(
            num_embeddings=4**k+1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.rnn = getattr(torch.nn, rnn_type)(
            input_size=embedding_dim + num_features,
            hidden_size=d_model,
            num_layers=num_layer,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dense = torch.nn.Linear(2*d_model, num_classes)

    def forward(self, lengths, kmers, features):
        # embed kmers
        # (batch_size, seq_len, embedding_dim)
        inner = self.kmer_embedding(kmers)
        # (batch_size, seq_len, embedding_dim + num_features)
        inner = torch.cat([inner, features], dim=-1)
        # pack inputs
        inner = torch.nn.utils.rnn.pack_padded_sequence(inner, lengths, batch_first=True, enforce_sorted=False)
        # run LSTM
        inner, _  = self.rnn(inner)
        # unpack output
        inner, _ = torch.nn.utils.rnn.pad_packed_sequence(inner, batch_first=True)
        inner_forward = inner[range(len(inner)), lengths - 1, :self.d_model]
        inner_reverse = inner[:, 0, self.d_model:]
        # (batch_size, 2*d_model)
        inner_reduced = torch.cat((inner_forward, inner_reverse), 1)
        # get class label
        # (batch_size, num_classes)
        inner = self.dense(inner_reduced)
        out = torch.nn.functional.softmax(inner, dim=1)
        return out




class BaseModLSTM_v2(torch.nn.Module):
    def __init__(self,
            num_features=6, num_kmers=4**6,
            embedding_dim=32, padding_idx=0,
            rnn_type='LSTM', d_model=64,
            dropout=0.1,
            num_classes=2):
        super(BaseModLSTM_v2, self).__init__()
        self.d_model = d_model
        self.kmer_embedding = torch.nn.Embedding(
            num_embeddings=num_kmers+1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.linear1 = torch.nn.Linear(num_features+embedding_dim, d_model)
        self.act1 = torch.nn.GELU()
        self.rnn1 = BiDirLSTM(d_model, 1, dropout=dropout, rnn_type=rnn_type)
        self.rnn2 = BiDirLSTM(d_model, 2, dropout=dropout, rnn_type=rnn_type)
        self.rnn3 = BiDirLSTM(d_model, 3, dropout=dropout, rnn_type=rnn_type)
        self.rnn4 = BiDirLSTM(d_model, 4, dropout=dropout, rnn_type=rnn_type)
        self.linear2 = torch.nn.Linear(d_model*4, num_classes)

    def forward(self, lengths, kmers, features):
        # embed kmers
        # (batch_size, seq_len, embedding_dim)
        emb = self.kmer_embedding(kmers)
        # (batch_size, seq_len, d_model)
        inner = self.linear1(torch.cat([emb, features], dim=-1))
        inner = self.act1(inner)
        r1 = self.rnn1(inner, lengths)
        r2 = self.rnn2(inner, lengths)
        r3 = self.rnn3(inner, lengths)
        r4 = self.rnn4(inner, lengths)
        # concat and reduce to num_classes
        inner = self.linear2(torch.cat([r1, r2, r3, r4], dim=-1))
        out = torch.nn.functional.softmax(inner, dim=1)
        return out




class BaseModEncoder_v1(torch.nn.Module):
    def __init__(self,
            num_features=6, num_kmers=4**6, num_classes=2,
            embedding_dim=32, padding_idx=0,
            d_model=512, num_heads=4, num_layer=3
            ):
        super(BaseModEncoder_v1, self).__init__()
        self.d_model = d_model
        self.kmer_embedding = torch.nn.Embedding(
            num_embeddings=num_kmers+1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        #self.input_nn = torch.nn.Linear(num_features + embedding_dim, d_model)
        self.input_nn = ResidualNetwork(num_features + embedding_dim,
            d_model, [128, 256])
        self.pos_encoder = PositionalEncoding(d_model,
            dropout=0.1,
            max_len=64)
        # encoder
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model,
                        nhead=num_heads,
                        dim_feedforward=d_model*4,
                        activation='gelu')
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                        num_layers=num_layer)
        self.output_nn = ResidualNetwork(d_model, num_classes, [512, 256, 128])

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
        inner = self.input_nn(inner)
        # positional encoding
        inner = self.pos_encoder(inner)
        # transformer encoder needs (max_len, batch_size, d_model)
        inner = self.transformer_encoder(inner.permute(1, 0, 2),
                        src_key_padding_mask = mask).permute(1, 0, 2)
        # get class label
        # (batch_size, max_len, num_classes)
        inner = self.output_nn(inner)
        inner = torch.mul(inner, ~mask[:,:,None])
        # melt to
        # (batch_size, num_classes)
        inner = torch.sum(inner, dim=1) / lengths[:,None]
        out = torch.nn.functional.softmax(inner, dim=1)
        return out
