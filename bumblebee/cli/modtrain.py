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
import yaml
import logging
import random
import tqdm
import torch
import itertools
import collections
import pkg_resources as pkg
import numpy as np
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bumblebee.db import ModDatabase
from bumblebee.ds import ModDataset
from bumblebee.optimizer import Lookahead
from bumblebee.util import running_average, WarmupScheduler
import bumblebee.modnn


log = logging.getLogger(__name__)


def main(args):
    # init torch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.device) if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    log.info("Using device {}".format(device))

    # output directory
    output_dir = args.prefix
    os.makedirs(output_dir, exist_ok=True)
    weights_dir = os.path.join(args.prefix, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # load model config
    run_config = os.path.join(output_dir, 'config.yaml')
    pkg_config = pkg.resource_filename('bumblebee', 'config/{}.yaml'.format(args.config))
    if os.path.isfile(run_config):
        # resume training with existing config
        with open(run_config, 'r') as fp:
            config = yaml.safe_load(fp)
        log.info("Loaded config file {}".format(run_config))
    elif os.path.isfile(args.config):
        # config is provided as file
        with open(args.config, 'r') as fp:
            config = yaml.safe_load(fp)

        log.info("Loaded config file {}".format(args.config))
    elif os.path.isfile(pkg_config):
        # config is found in installation path
        with open(pkg_config, 'r') as fp:
            config = yaml.safe_load(fp)
        log.info("Loaded config file {}".format(pkg_config))
    else:
        log.error("Could not find config file for {}".format(args.config))
        exit(-1)

    # init dataset and dataloader
    log.info("Loading training dataset")
    ds_train = ModDataset(args.db, args.mod_ids,
                max_features=args.max_features,
                min_score=args.min_score)
    if args.train_fraction < 1.0:
        ds_train = torch.utils.data.Subset(ds_train,
            np.arange(int(args.train_fraction * len(ds_train))))
    dl_train = torch.utils.data.DataLoader(ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4, worker_init_fn=ModDataset.worker_init_fn,
            prefetch_factor=args.batch_size,
            pin_memory=True,
            drop_last=True)
    log.info("Loading evaluation dataset")
    ds_eval = ModDataset(args.db, args.mod_ids,
                train=False,
                max_features=args.max_features,
                min_score=args.min_score)
    if args.train_fraction < 1.0:
        ds_eval = torch.utils.data.Subset(ds_eval,
            np.arange(int(args.train_fraction * len(ds_eval))))
    dl_eval = torch.utils.data.DataLoader(ds_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1, worker_init_fn=ModDataset.worker_init_fn,
            prefetch_factor=args.batch_size,
            pin_memory=True,
            drop_last=True)
    eval_rate = np.ceil(len(dl_train) / len(dl_eval)).astype(int)
    log.info("Loaded {} train and {} evaluation batches.".format(
        len(dl_train), len(dl_eval)))

    # copy config into output directory
    with open(run_config, 'w') as fp:
        yaml.dump(config, fp)

    # init model
    try:
        model = getattr(bumblebee.modnn, config['model'])(args.max_features, config['params'])
    except Exception as e:
        log.error("Coud not find model definition for {}:\n{}".format(config['model'], e))
        exit(-1)

    # model summary
    _, _batch = next(iter(dl_eval))
    summary(model,
        input_data=[_batch['lengths'],
                    _batch['kmers'],
                    _batch['offsets'],
                    _batch['features']],
        device="cpu", depth=1)
    model.to(device)
    def avg_fn(avg_mdl, mdl, step):
        scale = min(0.99, step/1e5)
        return scale * avg_mdl + (1-scale) * mdl
    swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=avg_fn, device=device)
    swa_model.eval()

    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
    optimizer = Lookahead(optimizer, k=5, alpha=0.5) # Initialize Lookahead
    if args.lr_schedule == 'warmup':
        lr_scheduler = WarmupScheduler(optimizer, config['params']['d_model'],
        warmup_steps=4000)
    elif args.lr_schedule == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
            max_lr=args.lr,
            base_lr=args.lr/100,
            step_size_up=4000,
            step_size_down=1000,
            cycle_momentum=False)
    # load checkpoint
    chkpt_file = os.path.join(args.prefix, 'latest.chkpt')
    out_file = os.path.join(args.prefix, 'final.chkpt')
    if os.path.isfile(chkpt_file):
        checkpoint = torch.load(chkpt_file)
        step_total = checkpoint['step_total']
        last_epoch = checkpoint['last_epoch']
        model.load_state_dict(checkpoint['model'])
        swa_model.load_state_dict(checkpoint['swa_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.lr_schedule:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        log.info("Loaded latest checkpoint: Epoch {} at step {}".format(
            last_epoch+1, step_total))
    else:
        step_total = 0
        last_epoch = 0

    # running loss and accuracy
    train_loss = running_average(max_len=500)
    train_acc = running_average(max_len=500)
    eval_loss = running_average(max_len=50)
    eval_acc = running_average(max_len=50)

    # summary writer
    writer = SummaryWriter(output_dir, purge_step=step_total, max_queue=50)

    # train step
    def train_step(labels, batch):
        labels = labels.to(device)
        lengths = batch['lengths']
        kmers = batch['kmers'].to(device)
        offsets = batch['offsets'].to(device)
        features = batch['features'].to(device)
        # zero gradients
        for _ in range(args.batch_echo + 1):
            optimizer.zero_grad()
            # forward pass
            logits, model_loss, metrics = model(lengths, kmers, offsets, features)
            prediction = torch.argmax(logits, dim=1)
            accuracy = torch.sum(prediction == labels).item() / args.batch_size
            loss = criterion(logits, labels)
            if model_loss is not None:
                loss += model_loss
            loss = torch.mean(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        return loss.item(), accuracy, metrics

    # eval step
    def eval_step(labels, batch, swa=False):
        with torch.no_grad():
            labels = labels.to(device)
            lengths = batch['lengths']
            kmers = batch['kmers'].to(device)
            offsets = batch['offsets'].to(device)
            features = batch['features'].to(device)
            # forward pass
            if swa:
                logits, model_loss, metrics = swa_model(lengths, kmers, offsets, features)
            else:
                model.eval()
                logits, model_loss, metrics = model(lengths, kmers, offsets, features)
                model.train()
            prediction = torch.argmax(logits, dim=1)
            accuracy = torch.sum(prediction == labels).item() / args.batch_size
            loss = criterion(logits, labels)
            if model_loss is not None:
                loss += model_loss
            loss = torch.mean(loss)
            return loss.item(), accuracy, metrics

    def save(fout, swa=False):
        if not swa:
            torch.save(model.state_dict(),
                os.path.join(weights_dir, 'weights_{}.pt'.format(step_total)))
        else:
            torch.save(swa_model.state_dict(),
                os.path.join(weights_dir, 'weights_swa_{}.pt'.format(step_total)))
        torch.save({
            "step_total": step_total,
            "last_epoch": epoch,
            "model": model.state_dict(),
            "swa_model": swa_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if args.lr_schedule else None
            }, fout)

    # compute epochs if target steps are given
    if args.epochs:
        max_steps = args.epochs * len(dl_train)
        max_epochs = args.epochs
        swa_start_step = args.swa_start * len(dl_train)
    else:
        max_steps = args.steps
        max_epochs = np.ceil(args.steps / len(dl_train)).astype(int)
        swa_start_step = args.swa_start
    # training loop
    for epoch in range(last_epoch + 1, max_epochs + 1):
        dl_eval_iter = iter(dl_eval)
        with tqdm.tqdm(desc='Epoch {}'.format(epoch), total=len(dl_train)) as pbar:
            for step, (labels, batch) in enumerate(dl_train):
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
                if step_total > swa_start_step:
                    swa_model.update_parameters(model)
                # learning rate
                if args.lr_schedule:
                    lr_scheduler.step()
                # eval step
                if step % eval_rate == 0:
                    labels, batch = next(dl_eval_iter)
                    _eval_loss, _eval_acc, kwout = eval_step(labels, batch,
                        swa=step_total > swa_start_step)
                    eval_loss.append(_eval_loss)
                    eval_acc.append(_eval_acc)
                    writer.add_scalar('validation/loss', _eval_loss, step_total)
                    writer.add_scalar('validation/accuracy', _eval_acc, step_total)
                # raw stats output
                if args.stats:
                    with open(os.path.join(output_dir, 'stats.tsv'), 'a') as fp:
                        print('\t'.join([
                            str(step_total),
                            str(train_loss.mean()),
                            str(train_acc.mean()),
                            str(eval_loss.mean()),
                            str(eval_acc.mean())]), file=fp)
                # progress
                pbar.update(1)
                pbar.set_postfix_str("Train: {:.3f} / {:.3f} Eval: {:.3f} / {:.3f}".format(
                    train_loss.mean(),
                    train_acc.mean(),
                    eval_loss.mean(),
                    eval_acc.mean()))
                step_total += 1
                # break if max steps reached
                if step_total >= max_steps:
                    # stop epoch
                    break
            if step_total >= max_steps:
                # stop training
                break
            # save epoch to resume training later
            save(chkpt_file, swa=step_total > swa_start_step)
    # final save
    save(out_file, swa=step_total > swa_start_step)
    # close & cleanup
    writer.close()




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("db", type=str)
    parser.add_argument("config", type=str)
    parser.add_argument("--mod_ids", nargs='+', required=True, type=int)
    parser.add_argument("--prefix", default='.', type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--min_score", default=1.0, type=float)
    parser.add_argument("--max_features", default=40, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--batch_echo", default=0, type=int)
    parser.add_argument("--train_fraction", default=1.0, type=float)
    parser.add_argument("--lr", default=1.0, type=float)
    parser.add_argument("--lr_schedule", default=None,
        choices=['warmup', 'cyclic'])
    parser.add_argument("--clip_grad_norm", default=1.5, type=float)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--epochs", type=int)
    group.add_argument('--steps', type=int)
    parser.add_argument("--swa_start", default=0, type=int)
    parser.add_argument("--stats", action='store_true')

    return parser
