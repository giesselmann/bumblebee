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
import textwrap
from argparse import RawDescriptionHelpFormatter, ArgumentParser
from bumblebee.cli import pm
from bumblebee.cli import basecall, basedb, basetrain
from bumblebee.cli import modcall, moddb, modtrain

modules = ['basecall', 'modcall', 'basedb', 'moddb', 'basetrain', 'modtrain', 'pm']

__version__ = '0.1.0'




def log_level(string):
    levels = {'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    return levels[string]



def setup_logger(level=logging.INFO):
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[handler])
    logger = logging.getLogger(__name__)
    logger.debug("Logging initialized.")




def main():
    parser = ArgumentParser(
        'bumblebee',
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument('-v', '--version',
        action='version', version='%(prog)s {}'.format(__version__))
    parser.add_argument('--log', type=log_level, metavar='level', default='debug',
        choices=['error', 'warning', 'info', 'debug'], help='Log level')

    subparsers = parser.add_subparsers(
        title='subcommands',
        dest='command',
        required=True,
        metavar='',
        description=textwrap.dedent('''\
        Nanopore signal analysis software
            basecall        Basecalling from fast5 input
            modcall         Modificatin detection from fast5 and bam

            basetrain       Train new base caller model
            modtrain        Train new modification caller model

            basedb          Setup database for base caller training
            moddb           Setup database for modification caller training
            '''),
    )

    for module in modules:
        mod = globals()[module]
        p = subparsers.add_parser(module, parents=[mod.argparser()])
        p.set_defaults(func=mod.main)

    args = parser.parse_args()
    setup_logger(args.log)
    args.func(args)
