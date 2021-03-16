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
import os
import random
import tqdm
import torch
import itertools
import collections
import numpy as np
from datetime import datetime
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bumblebee.db import ModDatabase
from bumblebee.ds import ModDataset
from bumblebee.optimizer import Lookahead
from bumblebee.util import running_average, parse_kwargs, WarmupScheduler
import bumblebee.modnn




def main(args):
    # init torch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.device) if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # summary writer
    summary_dir = os.path.join(args.prefix, args.model, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(summary_dir, exist_ok=True)
    writer = SummaryWriter(summary_dir)

    # init dataset and dataloader
    ds_train = ModDataset(args.db, args.mod_ids,
                max_features=args.max_features,
                min_score=args.min_score)
    ds_eval = ModDataset(args.db, args.mod_ids,
                train=False,
                max_features=args.max_features,
                min_score=args.min_score)
    dl_train = torch.utils.data.DataLoader(ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4, worker_init_fn=ModDataset.worker_init_fn,
            #prefetch_factor=20 * args.batch_size,
            pin_memory=True,
            drop_last=True)
    dl_eval = torch.utils.data.DataLoader(ds_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1, worker_init_fn=ModDataset.worker_init_fn,
            #prefetch_factor=20 * args.batch_size,
            pin_memory=True,
            drop_last=True)
    eval_rate = np.ceil(len(dl_train) / len(dl_eval)).astype(int)
    print("Loaded {} train and {} evaluation batches.".format(len(dl_train), len(dl_eval)))

    # init model
    model_args = dict((parse_kwargs(arg) for arg in args.kwargs))
    model = getattr(bumblebee.modnn, args.model)(**model_args)
    _, _batch = next(iter(dl_eval))
    summary(model, input_data=[_batch['lengths'], _batch['kmers'], _batch['features']], device="cpu", depth=4)
    model.to(device)
    avg_fn = lambda avg_mdl, mdl, step: 0.5 * avg_mdl + 0.5 * mdl
    swa_model = torch.optim.swa_utils.AveragedModel(model, device=device, avg_fn=avg_fn)
    swa_model.eval()
    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
    lookahead = Lookahead(optimizer, k=5, alpha=0.5) # Initialize Lookahead
    lr_scheduler = WarmupScheduler(optimizer, model.d_model, warmup_steps=8000)
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
        for _ in range(args.echo + 1):
            lookahead.zero_grad()
            # forward pass
            logits, model_loss, metrics = model(lengths, kmers, features)
            prediction = torch.argmax(logits, dim=1)
            accuracy = torch.sum(prediction == labels).item() / args.batch_size
            loss = criterion(logits, labels)
            if model_loss is not None:
                loss += model_loss
            loss = torch.mean(loss)
            loss.backward()
            lookahead.step()
        return loss.item(), accuracy, metrics

    # eval step
    def eval_step(labels, batch):
        with torch.no_grad():
            labels = labels.to(device)
            lengths = batch['lengths']
            kmers = batch['kmers'].to(device)
            features = batch['features'].to(device)
            # forward pass
            logits, model_loss, metrics = swa_model(lengths, kmers, features)
            prediction = torch.argmax(logits, dim=1)
            accuracy = torch.sum(prediction == labels).item() / args.batch_size
            loss = criterion(logits, labels)
            if model_loss is not None:
                loss += model_loss
            loss = torch.mean(loss)
            return loss.item(), accuracy, metrics

    # training loop
    for epoch in range(args.epochs):
        dl_eval_iter = iter(dl_eval)
        with tqdm.tqdm(desc='Epoch {}'.format(epoch), total=len(dl_train)) as pbar:
            for step, (labels, batch) in enumerate(dl_train):
                step_total = epoch * len(dl_train) + step
                # train step
                _train_loss, _train_acc, metrics = train_step(labels, batch)
                train_loss.append(_train_loss)
                train_acc.append(_train_acc)
                writer.add_scalar('training/loss', _train_loss, step_total)
                writer.add_scalar('training/accuracy', _train_acc, step_total)
                writer.add_scalar("learning rate", optimizer.param_groups[0]['lr'], step_total)
                for key, tensor in metrics.items():
                    writer.add_scalar('training/{}'.format(key), torch.mean(tensor).item(), step_total)
                # swa
                swa_model.update_parameters(model)
                # eval step
                if step % eval_rate == 0:
                    labels, batch = next(dl_eval_iter)
                    _eval_loss, _eval_acc, kwout = eval_step(labels, batch)
                    eval_loss.append(_eval_loss)
                    eval_acc.append(_eval_acc)
                    writer.add_scalar('validation/loss', _eval_loss, step_total)
                    writer.add_scalar('validation/accuracy', _eval_acc, step_total)
                # learning rate
                lr_scheduler.step()
                # progress
                pbar.update(1)
                pbar.set_postfix_str("Train: {:.3f} / {:.3f} Eval: {:.3f} / {:.3f}".format(
                    train_loss.mean(),
                    train_acc.mean(),
                    eval_loss.mean(),
                    eval_acc.mean()))
            torch.save(model.state_dict(),
                os.path.join(summary_dir, 'weights_{}.pt'.format(epoch)))
            torch.save(swa_model.state_dict(),
                os.path.join(summary_dir, 'weights_swa_{}.pt'.format(epoch)))

    # close & cleanup
    writer.close()




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("db", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--prefix", default='.', type=str)
    parser.add_argument("--mod_ids", nargs='+', required=True, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--echo", default=0, type=int)
    parser.add_argument("--max_features", default=40, type=int)
    parser.add_argument("--min_score", default=1.0, type=float)
    parser.add_argument("--kwargs", nargs='*', default='')
    return parser
