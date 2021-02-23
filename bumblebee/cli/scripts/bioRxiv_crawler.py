# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : Fetch title and abstract from bioRxiv
#
#  DESCRIPTION   : none
#
#  RESTRICTIONS  : none
#
#  REQUIRES      : none
#
# ---------------------------------------------------------------------------------
# Copyright 2019-2020 Pay Giesselmann, Max Planck Institute for Molecular Genetics
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
import re, json
import timeit, time
import subprocess
from tqdm import tqdm
from tbselenium.tbdriver import TorBrowserDriver
from threading import Thread
from queue import Queue




bioRxiv_url = 'https://www.biorxiv.org/content/early/recent?page={}'




class TorSession(object):
    def __init__(self, tbb_path, port=9050):
        tor_exec = args.tbb_path + '/Browser/TorBrowser/Tor/tor'
        tor_p = subprocess.Popen([tor_exec, 'SocksPort', str(port), 'DataDirectory', "~/.tor_{}".format(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tor_p_stdout = iter(tor_p.stdout)
        stdout = next(tor_p_stdout)
        while not b'100%' in stdout:
            try:
                stdout = next(tor_p_stdout)
            except StopIteration:
                print("Failed to start TOR on SOCKS {}".format(port), file=sys.stderr)
                tor_p.terminate()
                self.tor_p = None
                return
        print("Started tor on SOCKS {}".format(port), file=sys.stderr)
        self.tor_p = tor_p

    def __enter__(self):
        return self.tor_p

    def __exit__(self, type, value, traceback):
        if self.tor_p:
            self.tor_p.terminate()




def get_pages(driver):
    page_url = bioRxiv_url.format(0)
    driver.get(page_url)
    total = driver.find_element_by_class_name('pager-last.last.odd')
    per_page = 0
    lists = driver.find_elements_by_class_name("highwire-article-citation-list")
    for l in lists:
        articles = l.find_elements_by_class_name("clearfix")
        per_page += len(articles)
    return int(total.text), per_page




def get_articles(driver, page):
    articles = []
    page_url = bioRxiv_url.format(page - 1) # URLs are zero-based
    driver.get(page_url)
    article_lists = driver.find_elements_by_class_name("highwire-article-citation-list")
    for l in article_lists:
        year = re.search('[0-9]{4}', l.find_element_by_class_name('highwire-list-title').text).group(0)
        for article in l.find_elements_by_class_name("clearfix"):
            article_url = article.find_element_by_class_name('highwire-cite-linked-title').get_attribute('href')
            articles.append((article_url, year))
    return articles




def get_article(driver, url):
    driver.get(url)
    title = driver.find_element_by_class_name('highwire-cite-title').text
    authors = driver.find_element_by_class_name('highwire-cite-authors').text
    authors = [re.sub('View ORCID Profile\n', '', x.strip()) for x in authors.split(',')]
    doi = driver.find_element_by_class_name('highwire-cite-metadata-doi.highwire-cite-metadata').text
    doi = re.sub('doi: ', '', doi)
    date = driver.find_element_by_class_name('panel-pane.pane-custom.pane-1').text
    year = re.search('[0-9]{4}', date).group(0)
    abstract = driver.find_element_by_class_name('abstract').text
    abstract = re.sub(' +', ' ', ' '.join(abstract.split('\n')[1:]))
    return {'id' : url,
            'title' : title,
            'year' : year,
            'paperAbstract' : abstract,
            'authors' : authors,
            'doi' : doi,
            'journalName' : 'bioRxiv'}




def get_article_worker(url_queue, record_queue, tbb_path, port=9050):
    while True:
        try:
            with TorSession(tbb_path, port) as ts:
                with TorBrowserDriver(tbb_path, socks_port=port) as driver:
                     while True:
                         url_year = url_queue.get()
                         if url_year is None:
                             print("Worker thread received None, terminating.", file=sys.stderr)
                             return
                         url, year = url_year
                         try:
                             article = get_article(driver, url)
                         except Exception as e:
                             if 'Unable to locate element' in str(e):
                                 url_queue.task_done()
                                 continue
                             else:
                                 raise
                         #article['year'] = year
                         #article = url
                         record_queue.put(article)
                         url_queue.task_done()
        except Exception as e:
            print("Exception in reader thread: " + str(e), file=sys.stderr)
            # ignore errors in tor, restart in next iteration
            continue




def write_record_worker(record_queue, record_file, status_file, total_articles=0):
    with tqdm(total=total_articles, desc='Reading') as pbar:
        while True:
            record = record_queue.get()
            if record is None:
                print("Writer thread received None, terminating.", file=sys.stderr)
                break
            try:
                with open(record_file, 'a') as fp, open(status_file, 'a') as fp2:
                    print(json.dumps(record, sort_keys=True), file=fp)
                    print('\t'.join([record['id'], record['year']]), file=fp2)
                    fp.flush()
                    fp2.flush()
                    pbar.update(1)
            except Exception as e:
                print('Exception in writer thread: '+ str(e), file=sys.stderr)
                continue




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BioRxiv meta data crawler')
    parser.add_argument('tbb_path', help='Path to Tor browser')
    parser.add_argument('output', help='Path to output')
    parser.add_argument('--ports', nargs="+", type=int, default=[9050], help='Socks port(s) for Tor')
    args = parser.parse_args()
    urls = dict()
    url_file = os.path.realpath(args.output) + '.urls'
    status_file = os.path.realpath(args.output) + '.status'
    record_file = os.path.realpath(args.output)

    # Stage 1: Parse URLs from articles
    if os.path.isfile(url_file):
        with open(url_file, 'r') as fp:
            urls.update([tuple(line.split('\t')) for line in fp.read().split('\n') if line != ''])

    unique_urls = set([re.sub('v[0-9]+$', '', url) for url in urls.keys()])
    with TorSession(args.tbb_path, args.ports[0]) as ts, \
         TorBrowserDriver(args.tbb_path, socks_port=args.ports[0]) as driver:
          total_pages, per_page = get_pages(driver)

    pages_visited = len(unique_urls) // per_page
    pages_remaining = total_pages - pages_visited or 1 # always visit page 0 for updates
    print("Stage 1: {} pages of {} already fetched.".format(
                pages_visited, total_pages))

    try:
        with TorSession(args.tbb_path, args.ports[0]) as ts, \
             TorBrowserDriver(args.tbb_path, socks_port=args.ports[0]) as driver:
             for current_page in range(pages_remaining + 1, 0, -1):
                articles = get_articles(driver, current_page)
                urls.update([article for article in articles])
    except:
        pass

    with open(url_file, 'w') as fp:
        print('\n'.join(['\t'.join(article_url_year) for article_url_year in urls.items()]), file=fp)

    print("Stage 1: {} pages with {} articles fetched.".format(total_pages, len(urls)))

    # Stage 2: Lookup individual article URLs
    status = set()
    if os.path.isfile(status_file):
        with open(status_file, 'r') as fp:
            status.update({tuple(line.split('\t')) for line in fp.read().split('\n') if line != ''})

    down_urls = set(urls.items()).difference(status)
    print("Stage 2: {} of {} already downloaded, {} remaining.".format(len(status), len(urls), len(down_urls)))

    # start worker
    worker = []
    url_queue = Queue(maxsize=len(args.ports))
    record_queue = Queue(maxsize=10)
    for p in args.ports:
        w = Thread(target=get_article_worker,
                args=[url_queue, record_queue, args.tbb_path],
                kwargs={'port':p})
        w.start()
        worker.append(w)
    writer = Thread(target=write_record_worker,
                args=[record_queue, record_file, status_file],
                kwargs={'total_articles' : len(down_urls)})
    writer.start()

    def safe_exit():
        print("Starting safe shutdown.")
        url_queue.join()
        print("Remaining jobs completed.")
        for p in args.ports:
            url_queue.put(None)
        print("Joining worker.")
        for w in worker:
            w.join()
        record_queue.put(None)
        print("Joining writer.")
        writer.join()

    try:
        for url in down_urls:
            url_queue.put(url)
        safe_exit()
    except KeyboardInterrupt:
        safe_exit()
    exit(0)
