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
import os, sys
import logging
import sqlite3
import itertools
import numpy as np


log = logging.getLogger(__name__)


# init tables
def __init_reads_table__(cursor):
    sql_cmd = """
    CREATE TABLE reads (
        rowid INTEGER PRIMARY KEY,
        readid TEXT NOT NULL,
        chr TEXT NOT NULL,
        pos INT4,
        strand INT1,
        score FLOAT
    );"""
    cursor.execute(sql_cmd)


def __index_reads_table__(cursor):
    cursor.execute("CREATE INDEX IF NOT EXISTS reads_chr_idx ON reads(chr, strand)")
    cursor.execute("CREATE INDEX IF NOT EXISTS reads_score_idx ON reads(score)")


def __init_sites_table__(cursor):
    sql_cmd = """
    CREATE TABLE sites (
        rowid INTEGER PRIMARY KEY,
        readid INTEGER NOT NULL,
        class INT1,
        batch INT4 DEFAULT 0,
        pos INT4,
        count INT2,
        FOREIGN KEY (readid) REFERENCES reads (rowid) ON DELETE CASCADE ON UPDATE NO ACTION
    );"""
    cursor.execute(sql_cmd)


def __index_sites_table__(cursor):
    cursor.execute("CREATE INDEX IF NOT EXISTS sites_readid_idx ON sites(readid)")
    cursor.execute("CREATE INDEX IF NOT EXISTS sites_class_idx ON sites(class)")
    cursor.execute("CREATE INDEX IF NOT EXISTS sites_batch_idx ON sites(batch)")
    cursor.execute("CREATE INDEX IF NOT EXISTS sites_pos_idx ON sites(pos)")


def __init_features_table__(cursor):
    sql_cmd = """
    CREATE TABLE features (
        rowid INTEGER PRIMARY KEY,
        siteid INTEGER NOT NULL,
        enum INT1,
        offset INT1,
        min FLOAT,
        mean FLOAT,
        median FLOAT,
        std FLOAT,
        max FLOAT,
        length FLOAT,
        kmer INT2,
        FOREIGN KEY (siteid) REFERENCES sites (rowid) ON DELETE CASCADE ON UPDATE NO ACTION
    );"""
    cursor.execute(sql_cmd)


def __index_features_table__(cursor):
    cursor.execute("CREATE INDEX IF NOT EXISTS features_siteid_idx ON features(siteid)")


def __init_filter_tables__(cursor):
    sql_cmd = """
        CREATE TABLE IF NOT EXISTS {} (
        rowid INTEGER PRIMARY KEY,
        chr TEXT NOT NULL,
        strand INT1,
        pos INT4,
        weight INT4
        )
    """
    cursor.execute(sql_cmd.format('train'))
    cursor.execute(sql_cmd.format('eval'))


def __index_filter_tables__(cursor):
    cursor.execute("CREATE INDEX IF NOT EXISTS train_idx ON train(chr, strand, pos);")
    cursor.execute("CREATE INDEX IF NOT EXISTS eval_idx ON eval(chr, strand, pos);")




# init database
def init_db(db_file, type='base'):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()
    __init_reads_table__(cursor)
    if type == 'base':
        raise NotImplementedError("Database init for {} not implemented.".format(type))
    elif type == 'mod':
        __init_sites_table__(cursor)
        __init_features_table__(cursor)
        __init_filter_tables__(cursor)
    else:
        raise NotImplementedError
    connection.commit()
    connection.close()




# Feature database for basecaller training
class BaseDatabase():
    def __init__(self, db_file):
        if not os.path.isfile(db_file):
            init_db(db_file, type='base')




# Feature database for modification caller training
class ModDatabase():
    def __init__(self, db_file, require_index=False,
                require_split=False):
        if not os.path.isfile(db_file):
            init_db(db_file, type='mod')
        self.connection = sqlite3.connect(db_file)
        self.cursor = self.connection.cursor()
        self.cursor.execute("pragma journal_mode = MEMORY;")
        self.cursor.execute("pragma synchronous = OFF;")
        self.cursor.execute("pragma cache_size = 10000;")
        self.cursor.execute("pragma threads = 2;")
        self.cursor.execute("SELECT MAX(rowid) FROM reads;")
        self.next_read_rowid = (next(self.cursor)[0] or 0) + 1
        self.cursor.execute("SELECT MAX(rowid) FROM sites;")
        self.next_site_rowid = (next(self.cursor)[0] or 0) + 1
        if require_index:
            log.info("Creating database indices if not existing")
            __index_reads_table__(self.cursor)
            __index_sites_table__(self.cursor)
            __index_features_table__(self.cursor)
            __index_filter_tables__(self.cursor)
        # require filter tables for train/val split
        if require_split:
            pass
            # TODO check split table not empty
        # in case new indices where created
        self.connection.commit()

    def __del__(self):
        self.connection.commit()
        self.connection.close()

    def commit(self):
        self.connection.commit()

    def insert_read(self, ref_span, score=0.0):
        self.cursor.execute("INSERT INTO reads (rowid, readid, chr, pos, strand, score) VALUES ({rowid}, {readid}, {chr}, {pos}, {strand}, {score});".format(
            rowid=self.next_read_rowid,
            readid='"{}"'.format(ref_span.qname),
            chr='"{}"'.format(ref_span.rname),
            pos=ref_span.pos,
            strand='{}'.format(1 if ref_span.is_reverse else 0),
            score=score
        ))
        self.next_read_rowid += 1
        return self.next_read_rowid - 1

    def insert_site(self, read_id, mod_id, pos, batch=0):
        self.cursor.execute("INSERT INTO sites (rowid, readid, class, pos) VALUES ({rowid}, {readid}, {mod_id}, {pos});".format(
            rowid=self.next_site_rowid,
            readid=read_id,
            mod_id=mod_id,
            pos=pos
        ))
        self.next_site_rowid += 1
        return self.next_site_rowid - 1

    def insert_features(self, site_id, df, feature_begin):
        # update counts in sites table
        self.cursor.execute("UPDATE sites SET count = {} WHERE rowid = {};".format(df.shape[0], site_id))
        # insert rows into features table
        sql_cmd = """
            INSERT INTO features
            (siteid, enum, offset, min, mean, median, std, max, length, kmer)
            VALUES
            ({siteid}, {enum}, {offset}, {min}, {mean}, {median}, {std}, {max}, {length}, {kmer});
        """
        for i, row in enumerate(df.itertuples()):
            try:
                self.cursor.execute(sql_cmd.format(
                    siteid=site_id,
                    enum=i,
                    offset=row.Index - feature_begin,
                    min=row.event_min,
                    mean=row.event_mean,
                    median=row.event_median,
                    std=row.event_std,
                    max=row.event_max,
                    length=row.event_length,
                    kmer=row.kmer
                ))
            except sqlite3.OperationalError:
                print(row)

    def reset_split(self):
        __init_filter_tables__(self.cursor)
        # delete existing rows
        self.cursor.execute("DROP TABLE train;")
        self.cursor.execute("DROP TABLE eval;")
        self.connection.commit()

    def insert_filter(self, chr, strand, pos, weight=1, table='train'):
        self.cursor.execute("INSERT INTO {table} (chr, strand, pos, weight) VALUES ('{chr}', {strand}, {pos}, {weight});".format(
            table=table, chr=chr,
            strand=strand, pos=pos,
            weight=weight))

    def get_feature_ids(self, mod_id, min_score=1.0, max_features=32, min_weight=1, max_weight=None, train=True):
        self.cursor.execute("SELECT sites.rowid FROM reads JOIN sites ON reads.rowid = sites.readid JOIN {table} ON reads.chr = {table}.chr AND reads.strand = {table}.strand AND sites.pos = {table}.pos WHERE sites.class = {mod_id} AND reads.score >= {min_score} AND sites.count <= {max_features} AND sites.count > 0 AND {table}.weight >= {min_weight} AND {table}.weight <= {max_weight};".format(
            table='train' if train else 'eval',
            mod_id=mod_id,
            min_score=min_score,
            max_features=max_features,
            min_weight=min_weight,
            max_weight=max_weight or sys.maxsize - 1
        ))
        return [x[0] for x in self.cursor]

    # get single feature
    def get_feature(self, feature_id):
        self.cursor.execute("SELECT class, kmer, offset, min, mean, median, std, max, length FROM sites JOIN features ON sites.rowid = features.siteid WHERE sites.rowid = {} ORDER BY enum;".format(feature_id))
        try:
            mod_ids, kmers, offsets, features = zip(*[(x[0], x[1], x[2], x[3:])
                for x in self.cursor])
            return mod_ids[0], len(kmers), kmers, offsets, features
        except:
            print(feature_id)
            raise

    # unused rows are set to zero
    def reset_batches(self):
        self.cursor.execute("UPDATE sites SET batch = 0;")
        self.connection.commit()

    # assign each site to batch id
    def set_feature_batch(self, feature_batches, train=True):
        for siteid, batch in feature_batches:
            self.cursor.execute("UPDATE sites SET batch = {} WHERE rowid = {};".format(batch+1 if train else -batch-1, siteid))
        self.connection.commit()

    # get batch of labels, lengths, kmers, features
    def get_feature_batch(self, batch_id, train=True):
        self.cursor.execute("SELECT siteid, class, kmer, min, mean, median, std, max, length FROM sites JOIN features ON sites.rowid = features.siteid WHERE batch = {} ORDER BY siteid, enum;".format(batch_id+1 if train else -batch_id-1))
        def pack_feature(iterable):
            for siteid, grp in itertools.groupby(iterable, key=lambda x : x[0]):
                mod_ids, kmers, features = zip(*[(x[1], x[2], x[3:]) for x in grp])
                yield mod_ids[0], len(kmers), kmers, features
        return zip(*[feature for feature in pack_feature(self.cursor)])
