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
import random
import tqdm
import torch
import itertools
import collections
import numpy as np
from torchinfo import summary
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bumblebee.db import ModDatabase
from bumblebee.ds import ModDataset
from bumblebee.util import running_average
from bumblebee.modnn import BaseModLSTM_v1




def main(args):
    # init torch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    # open db and index if necessary, datasets could use multiprocessing
    db = ModDatabase(args.db, require_index=True, require_split=True)
    db.reset_batches()
    # init dataset and dataloader
    ds_train = ModDataset(db, args.mod_ids,
                batch_size=args.batch_size,
                max_features=args.max_features,
                min_score=args.min_score)
    ds_eval = ModDataset(db, args.mod_ids,
                train=False,
                batch_size=args.batch_size,
                max_features=args.max_features,
                min_score=args.min_score)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=None, shuffle=False)
    dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=None, shuffle=False)
    print("Loaded {} train and {} evaluation batches.".format(len(dl_train), len(dl_eval)))
    eval_rate = np.ceil(len(dl_train) / len(dl_eval)).astype(int)
    # init model
    model = BaseModLSTM_v1()
    _, _batch = next(iter(dl_eval))
    summary(model, input_data=[_batch['lengths'], _batch['kmers'], _batch['features']], device="cpu")
    model.to(device)
    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # running loss and accuracy
    train_loss = running_average(max_len=500)
    train_acc = running_average(max_len=500)
    eval_loss = running_average(max_len=50)
    eval_acc = running_average(max_len=50)

    # train step
    def train_step(labels, batch):
        labels = labels.to(device)
        lengths = batch['lengths']
        kmers = batch['kmers'].to(device)
        features = batch['features'].to(device)
        # zero gradients
        optimizer.zero_grad()
        # forward pass
        logits = model(lengths, kmers, features)
        prediction = torch.argmax(logits, dim=1)
        accuracy = torch.sum(prediction == labels).item() / args.batch_size
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        return loss.item(), accuracy

    # eval step
    def eval_step(labels, batch):
        with torch.no_grad():
            labels = labels.to(device)
            lengths = batch['lengths']
            kmers = batch['kmers'].to(device)
            features = batch['features'].to(device)
            # forward pass
            logits = model(lengths, kmers, features)
            prediction = torch.argmax(logits, dim=1)
            accuracy = torch.sum(prediction == labels).item() / args.batch_size
            loss = criterion(logits, labels)
            return loss.item(), accuracy

    # training loop
    for epoch in range(args.epochs):
        dl_eval_iter = iter(dl_eval)
        with tqdm.tqdm(desc='Epoch {}'.format(epoch), total=len(dl_train)) as pbar:
            for step, (labels, batch) in enumerate(dl_train):
                # copy data to device
                _train_loss, _train_acc = train_step(labels, batch)
                train_loss.append(_train_loss)
                train_acc.append(_train_acc)
                # evaluate
                if step % eval_rate == 0:
                    labels, batch = next(dl_eval_iter)
                    _eval_loss, _eval_acc = eval_step(labels, batch)
                    eval_loss.append(_eval_loss)
                    eval_acc.append(_eval_acc)
                pbar.update(1)
                pbar.set_postfix_str("Train: {:.3f} / {:.3f} Eval: {:.3f} / {:.3f}".format(
                    train_loss.mean(),
                    train_acc.mean(),
                    eval_loss.mean(),
                    eval_acc.mean()))





def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("db", type=str)
    parser.add_argument("--mod_ids", nargs='+', required=True, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_features", default=32, type=int)
    parser.add_argument("--min_score", default=1.0, type=float)
    return parser
