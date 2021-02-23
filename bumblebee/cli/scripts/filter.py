# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : Semantic Scholar S2 corpus language and encoding filter
#
#  DESCRIPTION   : none
#
#  RESTRICTIONS  : none
#
#  REQUIRES      : none
#
# ---------------------------------------------------------------------------------
# Copyright 2019 Pay Giesselmann, Max Planck Institute for Molecular Genetics
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
import sys, json, re, string
import argparse, gzip
import signal
import nltk
import multiprocessing
from tqdm import tqdm




english_vocab = set(w.lower() for w in nltk.corpus.words.words())


def is_ascii(s):
    return all(ord(c) < 128 and ord(c) > 31 for c in s)


def is_punct(s):
    return any(c in string.punctuation for c in s)


def tidy_str(s):
    # replace newline, whitespace and tokenize
    tokens = re.sub('\s+', ' ', s).split()
    # strip punctuation
    tokens = [token.rstrip(string.punctuation) for token in tokens]
    # tokenize and remove non ASCII tokens
    tokens = [token for token in tokens if token and is_ascii(token) and not is_punct(token)]
    return tokens


def vocab_overlap(tokens, vocabulary=set()):
    token_set = set(token.lower() for token in tokens)
    return len(vocabulary.intersection(token_set)) / (len(token_set) + 1)


def parse_records(record):
    txt_keys = {'title', 'paperAbstract'}
    txt_min_overlap = 0.6
    record = {key:(tidy_str(value) if key in txt_keys else value) for key, value in record.items()}
    txt_overlap = vocab_overlap([w for k in txt_keys for w in record[k]], vocabulary=english_vocab)
    record = {key:(' '.join(value) if key in txt_keys else value) for key, value in record.items()}
    if txt_overlap >= txt_min_overlap:
        return record
    else:
        return None




if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signal, frame: exit(0))
    parser = argparse.ArgumentParser(description='Filter json records')
    parser.add_argument('input', help='Input file')
    parser.add_argument('output', help='Output file')
    parser.add_argument('--t', type=int, default=1, help='Parallel worker')
    args = parser.parse_args()

    required_keys = {'id', 'title', 'year', 'paperAbstract', 'journalName', 'inCitations', 'outCitations'}
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    with gzip.open(args.input, 'rb') as fp:
        data = fp.read()

    records_str = [record.decode() for record in tqdm(re.split(rb'\n(?=\{.*?\})', data), desc='decoding', ncols=75)]
    records = [json.loads(record) for record in tqdm(records_str, desc='parsing', ncols=75)]
    records_filtered = [record for record in tqdm(records, desc='filtering', ncols=75) if all(k in record for k in required_keys)]
    print("Loaded {} records, {} with required keys".format(len(records), len(records_filtered)))

    pool = multiprocessing.Pool(processes=args.t)
    records_parsed = [record for record in tqdm(pool.imap_unordered(parse_records, records_filtered), desc='cleaning', ncols=75, total=len(records_filtered))]
    print("Parsed {} records.".format(len(records_parsed)))
    with gzip.open(args.output, 'wb') as fp:
        l = [fp.write((json.dumps({key:value for key, value in record.items() if key in required_keys}, sort_keys=True) + '\n').encode())
                    for record in tqdm(records_parsed, desc='writing', ncols=75) if record]
    print("Wrote {} records to {}".format(len(l), args.output))
