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
import asyncio
import sqlite3
import numpy as np




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


def __init_sites_table__(cursor):
    sql_cmd = """
    CREATE TABLE sites (
        rowid INTEGER PRIMARY KEY,
        readid INTEGER NOT NULL,
        class INT1,
        batch INT4 DEFAULT 0,
        pos INT4,
        FOREIGN KEY (readid) REFERENCES reads (rowid) ON DELETE CASCADE ON UPDATE NO ACTION
    );"""
    cursor.execute(sql_cmd)


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




# init database
def init_db(db_file, type='basecall'):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()
    __init_reads_table__(cursor)
    if type == 'base':
        raise NotImplementedError("Database init for {} not implemented.".format(type))
    elif type == 'mod':
        __init_sites_table__(cursor)
        __init_features_table__(cursor)
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
    def __init__(self, db_file):
        if not os.path.isfile(db_file):
            init_db(db_file, type='mod')
        self.connection = sqlite3.connect(db_file)
        self.cursor = self.connection.cursor()
        self.cursor.execute("pragma journal_mode = MEMORY;")
        self.cursor.execute("pragma synchronous = OFF;")
        self.cursor.execute("""pragma cache_size = 100000;""")
        self.cursor.execute("SELECT MAX(rowid) FROM reads;")
        self.next_read_rowid = (next(self.cursor)[0] or 0) + 1
        self.cursor.execute("SELECT MAX(rowid) FROM sites;")
        self.next_site_rowid = (next(self.cursor)[0] or 0) + 1

    def __del__(self):
        self.connection.commit()
        self.connection.close()

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
                    std=row.event_std if not np.isnan(row.event_std) else 0,
                    max=row.event_max,
                    length=row.event_length,
                    kmer=row.kmer
                ))
            except sqlite3.OperationalError:
                print(row)

    def commit(self):
        self.connection.commit()
