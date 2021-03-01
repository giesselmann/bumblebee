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
import re
import tqdm
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bumblebee.ref import reference
from bumblebee.alignment import reverse_complement
from bumblebee.db import ModDatabase, pattern_template


def main(args):
    pattern = pattern_template.format(ext=args.pattern_extension, pattern=args.pattern)
    # load database and reset existing split
    db = ModDatabase(args.db, require_index=True)
    db.reset_split()
    # load reference and iterate over contigs
    ref = reference(args.ref)
    is_palindrome = args.pattern == reverse_complement(args.pattern)
    template_count = 0
    reverse_count = 0
    context_dict = defaultdict(list)
    for name, contig in ref.contigs():
        print("Processing {}".format(name))
        # matches on template strand
        for match in re.finditer(pattern, contig):
            match_begin = match.start()
            match_end = match.end()
            context = contig[match_begin - args.pattern_extension : match_end + args.pattern_extension]
            context_dict[context].append((name, 0, match_begin))
            template_count += 1
            if is_palindrome:
                context_dict[reverse_complement(context)].append((name, 1, match_begin))
                reverse_count += 1
        if is_palindrome:
            continue
        # matches on reverse strand
        contig = reverse_complement(contig)
        for match in re.finditer(pattern, contig):
            match_begin = match.start()
            match_end = match.end()
            context = contig[match_begin - args.pattern_extension : match_end + args.pattern_extension]
            context_dict[context].append((name, 1, len(contig) - match_end))
            reverse_count += 1
    print("Found {} template and {} reverse strand matches.".format(template_count, reverse_count))
    print("A total of {} contexts are split into train and eval section.".format(len(context_dict)))
    # free some memory
    del ref
    # sort by number of occurences
    context_positions = [(context, positions) for context, positions in context_dict.items()]
    context_positions.sort(key = lambda x: len(x[1]), reverse=True)
    # get validation samples from all context frequencies
    val_ids = set(np.linspace(0, len(context_positions), int(args.split * len(context_positions)), endpoint=False, dtype=int))
    for i, (context, positions) in tqdm.tqdm(enumerate(context_positions), desc='Writing', total=len(context_positions)):
        if i in val_ids:
            for p in positions:
                db.insert_filter(*p, table='eval')
        else:
            for p in positions:
                db.insert_filter(*p, table='train')
    db.commit()




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False)
    parser.add_argument("db", type=str)
    parser.add_argument("ref", type=str)
    parser.add_argument("--split", default=0.1, type=float)
    parser.add_argument("--pattern", default='CG', type=str)
    parser.add_argument("--pattern_extension", default=6, type=int)
    return parser
