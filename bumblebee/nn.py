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




class PositionalDepthEncoding(torch.nn.Module):
    def __init__(self, d_model, depth, max_len=64):
        super(PositionalDepthEncoding, self).__init__()
        pe = torch.zeros(depth, max_len, d_model)
        # (1, max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float)[None, :, None]
        # (depth, 1, 1)
        depth = torch.arange(0, depth, dtype=torch.float)[:, None, None]
        # (d_model // 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # (max_len, d_model // 2) +
        pe[:, :, 0::2] = torch.sin(position * div_term) + torch.sin(depth * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term) + torch.cos(depth * div_term)
        # (1, depth, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, step):
        x = x + self.pe[:, step, :x.size(1), :]
        return x




class ResidualNetwork(torch.nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
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




class AdaptiveComputeTime(torch.nn.Module):
    def __init__(self, d_model, eps=0.01):
        super(AdaptiveComputeTime, self).__init__()
        self.eps = eps
        self.halt_threshold = 1 - eps
        self.ponder_nn = ResidualNetwork(d_model, 1, [128, 32])
        self.ponder_act = torch.nn.Sigmoid()
        with torch.no_grad():
            torch.nn.init.ones_(self.ponder_nn.fc_out.bias)

    def forward(self, state, halting_prob, remainders, n_updates, mask=None):
        # (batch_size, max_len, d_model)
        p = self.ponder_nn(state)
        # (batch_size, max_len)
        p = self.ponder_act(p).squeeze(-1)
        # Mask for inputs which have not halted yet
        still_running = halting_prob < 1.0
        # length mask as (batch_size, max_len)
        if mask is not None:
            halting_prob += mask
        # Mask of inputs which halted at this step
        new_halted = ((halting_prob + p * still_running) > self.halt_threshold) * still_running
        # Mask of inputs which haven't halted, and didn't halt this step
        still_running *= (halting_prob + p * still_running) <= self.halt_threshold
         # Add the halting probability for this step to the halting
        # probabilities for those input which haven't halted yet
        halting_prob += p * still_running
        # Compute remainders for the inputs which halted at this step
        remainders += new_halted * (1 - halting_prob)
        # Add the remainders to those inputs which halted at this step
        halting_prob += new_halted * remainders
        # Increment n_updates for all inputs which are still running
        n_updates += still_running + new_halted
        # Compute the weight to be applied to the new state and output
        # 0 when the input has already halted
        # p when the input hasn't halted yet
        # the remainders when it halted this step
        update_weights = torch.unsqueeze(p * still_running + new_halted * remainders, -1)
        return update_weights, halting_prob, remainders, n_updates




class TransformerACTEncoder(torch.nn.Module):
    def __init__(self, d_model, max_len=64,
                num_heads=4, max_depth=4,
                time_penalty=0.05, eps=0.01):
        super(TransformerACTEncoder, self).__init__()
        self.max_depth = max_depth
        self.time_penalty = time_penalty
        self.halt_threshold = 1 - eps
        self.pos_encoder = PositionalDepthEncoding(d_model, max_depth,
            max_len=max_len)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model*4,
            activation='gelu')
        self.act_layer = AdaptiveComputeTime(d_model, eps=eps)

    def forward(self, state, mask):
        # init ACT buffers
        batch_size, max_len, d_model = state.size()
        halting_prob = torch.zeros((batch_size, max_len), device=state.device)
        remainders = torch.zeros_like(halting_prob)
        n_updates = torch.zeros_like(halting_prob)
        #state = state * math.sqrt(d_model)
        # run encoder for max depth with ACT update weights
        for step in range(self.max_depth):
            state = self.pos_encoder(state, step)
            transformed_state = self.encoder_layer(state.permute(1, 0, 2), src_key_padding_mask=mask).permute(1, 0, 2)
            update_weights, halting_prob, remainders, n_updates = self.act_layer(transformed_state, halting_prob, remainders, n_updates, mask=mask)
            transformed_state = (transformed_state * update_weights) + state * (1-update_weights)
            if torch.all(halting_prob > self.halt_threshold):
                break
        # compute ACT loss
        lengths = torch.sum(~mask, dim=-1, keepdim=False)
        n_updates *= ~mask
        remainders *= ~mask
        act_loss = torch.sum(n_updates + remainders, dim=-1) / lengths * self.time_penalty
        n_updates_mean = torch.sum(n_updates, dim=-1) / lengths
        remainders_mean = torch.sum(remainders, dim=-1) / lengths
        return transformed_state, act_loss, n_updates_mean, remainders_mean
