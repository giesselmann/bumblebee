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
import time
import logging
import queue
import multiprocessing as mp


log = logging.getLogger(__name__)


class StateFunction():
    def __init__(self):
        pass

    def __call__(self, *args):
        return self.call(*args)

    def call(self):
        raise NotImplementedError




class SourceProcess():
    def __init__(self, src_type, args=(), kwargs={}, queue_len=32):
        self.q = mp.Queue(queue_len)
        self.e = mp.Event()
        def fn(e, q, src_type, *args, **kwargs):
            pid = '(PID: {})'.format(os.getpid())
            log.debug("Started SourceProcess {}".format(pid))
            src = src_type(*args, **kwargs)
            for obj in src():
                q.put(obj)
                if e.is_set():
                    log.debug("Received StopEvent in SourceProcess {}".format(pid))
                    break
            q.put(StopIteration)
            q.close()
            q.join_thread()
            log.debug("Terminating SourceProcess {}".format(pid))
        self.p = mp.Process(target=fn,
            args=(self.e, self.q, src_type) + args,
            kwargs=kwargs)
        self.p.start()

    @property
    def output_queue(self):
        return self.q

    def terminate(self):
        log.debug("Sending StopEvent to SourceProcess")
        self.e.set()
        self.p.join()

    def kill(self):
        self.p.kill()

    def join(self, timeout=None):
        self.p.join(timeout)




class WorkerProcess():
    def __init__(self, input_queue, worker_type, args=(), kwargs={}, queue_len=32, num_worker=1):
        self.q_in = input_queue
        self.q_out = mp.Queue(queue_len)
        self.e = mp.Event()
        self.barrier = mp.Barrier(num_worker)
        def fn(e, barrier, q_in, q_out, worker_type, *args, **kwargs):
            pid = '(PID: {})'.format(os.getpid())
            log.debug("Started Worker {}".format(pid))
            worker = worker_type(*args, **kwargs)
            while not e.is_set():
                try:
                    obj = q_in.get(block=True, timeout=1)
                except queue.Empty:
                    obj = None
                if obj is StopIteration:
                    log.debug("Received StopIteration in WorkerProcess {}".format(pid))
                    e.set()
                    break
                elif obj is not None:
                    try:
                        res = worker(*obj)
                    except Exception as ex:
                        log.error("Exception in worker (Proceeding with remaining jobs):\n {}".format(str(ex)))
                        continue
                    if res is not None:
                        q_out.put(res)
                else:
                    continue
            i = barrier.wait()
            while not q_out.empty():
                time.sleep(0.1)
            if i == 0:
                q_out.put(StopIteration)
            q_out.close()
            q_out.join_thread()
            log.debug("Terminating WorkerProcess {}".format(pid))
        self.p = []
        for _ in range(num_worker):
            self.p.append(
                mp.Process(target=fn,
                    args=(self.e, self.barrier, self.q_in, self.q_out, worker_type) + args,
                    kwargs=kwargs))
            self.p[-1].start()

    @property
    def output_queue(self):
        return self.q_out

    def terminate(self):
        log.debug("Sending StopEvent to WorkerProcess")
        self.e.set()
        self.q_in.put(StopIteration)
        for p in self.p:
            p.join()

    def kill(self):
        for p in self.p:
            p.kill()

    def join(self, timeout=None):
        for p in self.p:
            p.join(timeout)




class SinkProcess():
    def __init__(self, input_queue, sink_type, args=(), kwargs={}):
        self.e = mp.Event()
        self.q_in = input_queue
        def fn(e, q_in, sink_type, *args, **kwargs):
            pid = '(PID: {})'.format(os.getpid())
            log.debug("Started SinkProcess {}".format(pid))
            sink = sink_type(*args, **kwargs)
            while not e.is_set():
                try:
                    obj = q_in.get(block=True, timeout=1)
                except queue.Empty:
                    obj = None
                if obj is StopIteration:
                    log.debug("Received StopIteration in SinkProcess {}".format(pid))
                    e.set()
                    break
                elif obj is not None:
                    try:
                        sink(*obj)
                    except Exception as ex:
                        log.error("Exception in sink (Proceeding with remaining jobs):\n{}".format(str(ex)))
                else:
                    continue
            log.debug("Terminating SinkProcess {}".format(pid))
        self.p = mp.Process(target=fn,
            args=(self.e, self.q_in, sink_type) + args,
            kwargs=kwargs)
        self.p.start()

    def terminate(self):
        log.debug("Sending StopEvent to Sink")
        self.e.set()
        self.q_in.put(StopIteration)
        self.p.join()

    def kill(self):
        self.p.kill()

    def join(self, timeout=None):
        self.p.join(timeout)
