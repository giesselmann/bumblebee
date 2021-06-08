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
import copy
import random
import torch
import numpy as np
from torchinfo import summary
from collections import deque


log = logging.getLogger(__name__)




class QLearnWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.online = net
        self.target = copy.deepcopy(self.online)
        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, *input, model='online', **input_args):
        if model == "online":
            return self.online(*input, **input_args)
        elif model == "target":
            return self.target(*input, **input_args)




class CacheElement():
    def __init__(self, state, next_state, action, reward, done):
        self.state = state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done




def states2tensor(states, device='cpu'):
    state_type = type(states[0])
    state_values = zip(*states)
    it = [torch.tensor(state_value, device=device)
            for state_value in state_values]
    if state_type is tuple:
        return tuple(it)
    else:
        return state_type(*it)




class Agent():
    def __init__(self, net, device, lr=1e-3, alphabet='ACGT', config={}):
        self.net = QLearnWrapper(net).to(device)
        self.device = device
        # act
        self.num_actions = len(alphabet) + 3
        self.alphabet_size = len(alphabet)
        # pretrain
        self.pretrain_rate = config.get('pretrain_rate') or 0.0
        self.pretrain_rate_decay = config.get('pretrain_rate_decay') or 0.9999
        # explore
        self.exploration_rate = config.get('exploration_rate') or 1.0
        self.exploration_rate_decay =  config.get('exploration_rate_decay') or 0.99999
        self.exploration_rate_min = config.get('exploration_rate_min') or 0.05
        self.curr_step = 0
        self.total_step = 0
        # cache & recall
        self.memory = deque(maxlen=config.get('memory_size') or 10000)
        self.batch_size = config.get('batch_size') or 32
        # optimizer
        self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        # learn
        self.burnin = config.get('burnin') or 1e4
        self.pre_train = config.get('pre_train') or 0
        self.learn_every = config.get('learn_every') or 1
        self.sync_every = config.get('sync_every') or 2e3
        self.gamma = config.get('gamma') or 0.8
        # mask for attention-like models
        self.mask = (torch.triu(
                        torch.ones(net.state_length, net.state_length),
                    diagonal=1) == 1).to(device)

    def __info__(self):
        return {'exploration_rate': self.exploration_rate,
                'pretrain_rate': self.pretrain_rate}

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict(),
                'net': self.net.state_dict(),
                'exploration_rate': self.exploration_rate,
                'pretrain_rate': self.pretrain_rate,
                'total_step': self.total_step}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.net.load_state_dict(state_dict['net'])
        self.exploration_rate = state_dict['exploration_rate']
        self.pretrain_rate = state_dict['pretrain_rate']
        self.total_step = state_dict['total_step']
        log.debug("Loaded state dict.")

    def act(self, state, next_action=None):
        # SUMMARY
        if self.curr_step == 0:
            state_t = states2tensor([state], device=self.device)
            summary(self.net.target,
                    input_data=state_t,
                    device=self.device,
                    depth=1)
        # EXPLORE
        # 0 : SHIFT
        # 1 : SEQ START
        # 2 : SEQ STOP
        # 3 : ALPHABET
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.choice(self.num_actions,
                p=[0.01, 0, 0.01] +
                [(1-0.02)/self.alphabet_size]*self.alphabet_size)
        # PRETRAIN
        elif np.random.rand() < self.pretrain_rate and next_action is not None:
            action_idx = next_action
        # EXPLOIT
        else:
            state = states2tensor([state], device=self.device)
            with torch.no_grad():
                action_values = self.net(*state,
                        mask=self.mask,
                        model="online")
                action_idx = torch.argmax(action_values, axis=1).item()
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        # decrease pretrain rate
        self.pretrain_rate *= self.pretrain_rate_decay
        # increment step
        self.curr_step += 1
        self.total_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        # cache is on CPU as numpy/scalars
        self.memory.append((state, next_state, action, reward, done))
        #self.memory.append(CacheElement(state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        #p = np.linspace(0, 1, len(self.memory), endpoint=True)
        #batch = [(s.state, s.next_state, s.action, s.reward) for s in
        #            np.random.choice(self.memory, self.batch_size,
        #            replace=False, p=p/np.sum(p))]
        state, next_state, action, reward, done = zip(*batch)
        state = states2tensor(state, device=self.device)
        next_state = states2tensor(next_state, device=self.device)
        action = torch.tensor(action, device=self.device)
        reward = torch.tensor(reward, device=self.device)
        done = torch.tensor(done, device=self.device)
        return state, next_state, action, reward, done

    def td_estimate(self, state, action):
        current_Q = self.net(*state, mask=self.mask, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(*next_state, mask=self.mask, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(*next_state, mask=self.mask, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.online.parameters(), 2.5)
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % (self.sync_every) == 0:
            self.sync_Q_target()
        if self.curr_step < self.burnin:
            return None, None, self.__info__()
        if self.curr_step % self.learn_every != 0:
            return None, None, self.__info__()
        # Sample from memory
        state, next_state, action, reward, done = self.recall()
        # Get TD Estimate
        td_est = self.td_estimate(state, action)
        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)
        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss, self.__info__())
