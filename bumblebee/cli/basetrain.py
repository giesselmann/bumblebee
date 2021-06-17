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
import os, re
import time
import logging
import yaml
import tqdm
import math
import torch
import queue
import numpy as np
import pkg_resources as pkg
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.tensorboard import SummaryWriter

import bumblebee.basenn
from bumblebee.ds import SeqDataset
from bumblebee.environment import EnvReadEvents
from bumblebee.agent import Agent


log = logging.getLogger(__name__)


def _init_device(device):
    # init torch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device) if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    log.info("Using device {}".format(device))
    return device


def _load_config(name, output_dir):
    # load config
    run_config = os.path.join(output_dir, 'config.yaml')
    pkg_config = pkg.resource_filename('bumblebee',
        'config/{}.yaml'.format(name))
    pkg_config = name if os.path.isfile(name) else pkg_config
    if os.path.isfile(run_config):
        # resume training with existing config
        with open(run_config, 'r') as fp:
            config = yaml.safe_load(fp)
        log.info("Loaded config file {}".format(run_config))
    elif os.path.isfile(pkg_config):
        # load config from repository / file
        log.info("Loading config file {}".format(pkg_config))
        with open(pkg_config, 'r') as fp:
            config = yaml.safe_load(fp)
    else:
        log.error("Could not load model config {}".format(name))
        exit(-1)

    # copy config into output directory
    if not os.path.isfile(run_config):
        with open(run_config, 'w') as fp:
            yaml.dump(config, fp)

    return config




def main_sup(args):
    # init torch
    device = _init_device(args.device)

    # output directory
    output_dir = args.prefix
    os.makedirs(output_dir, exist_ok=True)
    weights_dir = os.path.join(args.prefix, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    config = _load_config(args.config, output_dir)

    # datasets and loader
    ds = SeqDataset(args.fast5, args.bam, args.ref, output_dir)
    #ds_shuf = torch.utils.data.BufferedShuffleDataset(ds, 1024)
    dl = torch.utils.data.DataLoader(ds,
            batch_size=1,
            shuffle=False,
            num_workers=8, worker_init_fn=SeqDataset.worker_init_fn,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
            drop_last=True)
    for i, batch in tqdm.tqdm(enumerate(dl), desc='Epoch'):
        pass




def main_rf(args):
    # init torch
    device = _init_device(args.device)

    # output directory
    output_dir = args.prefix
    os.makedirs(output_dir, exist_ok=True)
    weights_dir = os.path.join(args.prefix, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    config = _load_config(args.config, output_dir)

    env = EnvReadEvents(args.fast5, args.bam,
            state_length=config['state_length'],
            alphabet=config['alphabet'],
            config=config['env'])

    # init model
    try:
        model = getattr(bumblebee.basenn, config['model'])(
            state_length=config['state_length'],
            alphabet=config['alphabet'],
            config=config['params'])
    except Exception as e:
        log.error("Coud not find model definition for {}:\n{}".format(
            config['model'], e))
        exit(-1)

    # RF agent
    agent = Agent(model, device,
        lr=args.lr, alphabet=config['alphabet'],
        config=config['agent'])
    step = 0

    # resume if checkpoint available
    chkpt_file = os.path.join(args.prefix, 'latest.chkpt')
    if os.path.isfile(chkpt_file):
        checkpoint = torch.load(chkpt_file)
        agent.load_state_dict(checkpoint['agent'])
        env.load_state_dict(checkpoint['env'])
        step = checkpoint['step']
        log.info("Resume training at step {}.".format(step))

    # summary writer
    writer = SummaryWriter(args.prefix, purge_step=step, max_queue=100)

    # train it!
    for epoch in range(args.epochs):
        for episode in tqdm.tqdm(env.episodes(), desc='Reads'):
            state, action = env.reset(episode)
            done = False
            while not done:
                action = agent.act(state, action)
                next_state, reward, done, next_action, env_info = env.step(action)
                agent.cache(state, next_state, action, reward, done)
                q, loss, agent_info = agent.learn()
                state = next_state
                action = next_action
                if loss is not None:
                    writer.add_scalar('agent/loss', loss, step)
                    writer.add_scalar('agent/q', q, step)
                for key, value in agent_info.items():
                    writer.add_scalar('agent/{}'.format(key), value, step)
                step += 1
                # save checkpoint
                if step % args.save_every == 0:
                    torch.save({
                            'agent': agent.state_dict(),
                            'env': env.state_dict(),
                            'step': step},
                        chkpt_file)
            for key, value in env_info.items():
                writer.add_scalar("env/{}".format(key), value, step)




def main(args):
    if args.mode == 'supervised':
        main_sup(args)
    else:
        main_rf(args)




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("mode", choices=['supervised', 'reinforcement'])
    parser.add_argument("config", type=str)
    parser.add_argument("fast5", type=str)
    parser.add_argument("bam", type=str)
    parser.add_argument("ref", type=str)
    parser.add_argument("--prefix", default='.', type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--min_score", default=1.0, type=float)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--clip_grad_norm", default=2.5, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_every", type=int, default=1000)
    return parser
