# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : Semantic Scholar S2 corpus meta data extraction
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
import os, sys, argparse
import sys, json, re
import signal
import nltk
import tensorflow as tf
from tqdm import tqdm
from collections import Counter
#from sklearn.feature_extraction import text




def parse(text):
    return re.sub(' +', ' ', text.replace('\n', ' ').replace('\t', ' '))




def is_ascii(s):
    return all(ord(c) < 128 and ord(c) > 31 for c in s)




def tokenize(text):
    text = parse(text)
    tokens = [word for phrase in text.lower().split() for word in re.split('(\W)', phrase) if word and is_ascii(word)]
    return tokens




def serialize(title, abstract, journal, year):
    title_b = tf.train.BytesList(value=[title.encode("ASCII")])
    abstract_b = tf.train.BytesList(value=[abstract.encode("ASCII")])
    journal_b = tf.train.BytesList(value=[journal.encode("ASCII")])
    year_b = tf.train.BytesList(value=[year.encode("ASCII")])
    feature = {
        'title' : tf.train.Feature(bytes_list=title_b),
        'abstract' : tf.train.Feature(bytes_list=abstract_b),
        'journal' : tf.train.Feature(bytes_list=journal_b),
        'year' : tf.train.Feature(bytes_list=year_b),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signal, frame: exit(0))
    parser = argparse.ArgumentParser(description='TF records from scholar dataset')
    parser.add_argument('prefix', help='Output file prefix')
    args = parser.parse_args()
    required_keys = {'id', 'year', 'journalName', 'title', 'paperAbstract'}

    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    vocabulary = Counter()

    record_writer = tf.io.TFRecordWriter(args.prefix + '.tfrec')

    #for line in tqdm(sys.stdin):
    for line in sys.stdin:
        fields = json.loads(line)
        if (all(k in fields for k in required_keys) and
            all(fields[k] for k in required_keys)):
            abstract = tokenize(fields['paperAbstract'])
            if not len(english_vocab.intersection(set([t for t in abstract if t.isalpha()]))) > 0.5 * len(abstract):
                continue
            journal = parse(fields['journalName'])
            if not is_ascii(journal):
                continue
            title = tokenize(fields['title'])
            year = str(fields['year']) if str(fields['year']).isnumeric() else ''
            vocabulary.update(title + abstract)
            title = " ".join(title)
            abstract = " ".join(abstract)
            record = serialize(title, abstract, journal, year)
            record_writer.write(record)
    record_writer.flush()
    record_writer.close()
    #with open(args.prefix + ".tsv", 'w') as fp:
    #    print('\n'.join(['\t'.join([word, str(count)]) for word, count in vocabulary.most_common()]), file=fp)
