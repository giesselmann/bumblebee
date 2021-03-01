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
from bumblebee.modmodel import ModCall_v1




def main(args):
    # init torch
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    # open db and index if necessary, datasets could use multiprocessing
    print("Start indexing:")
    db = ModDatabase(args.db, require_index=True)
    print("Completed indexing.")
    exit(0)
    # init dataset and dataloader
    ds = ModDataset(db, args.mod_ids, batch_size=args.batch_size, max_features=args.max_features)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False)
    # init model
    model = ModCall_v1()
    _, _batch = next(iter(dl))
    summary(model, input_data=[_batch['lengths'], _batch['kmers'], _batch['features']], device="cpu")
    model.to(device)
    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # running loss and accuracy
    train_loss = collections.deque()
    train_acc = collections.deque()
    # training loop
    for epoch in range(args.epochs):
        with tqdm.tqdm(desc='Epoch {}'.format(epoch), total=len(dl)) as pbar:
            for labels, batch in dl:
                # copy data to device
                labels = labels.to(device)
                lengths = batch['lengths']
                kmers = batch['kmers'].to(device)
                features = batch['features'].to(device)
                # zero gradients
                optimizer.zero_grad()
                logits = model(lengths, kmers, features)
                prediction = torch.argmax(logits, dim=1)
                # backpropagation
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                train_acc.append(torch.sum(prediction == labels).item() / args.batch_size)
                if len(train_loss) > 500:
                    train_loss.popleft()
                    train_acc.popleft()
                pbar.update(1)
                pbar.set_postfix_str("Loss {:.3f} Acc {:.3f}".format(
                    np.mean(train_loss),
                    np.mean(train_acc)))




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
    parser.add_argument("--split", default=0.05, type=float)
    return parser
