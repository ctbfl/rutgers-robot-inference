"""Opt-in bounded request capture; never changes policy inputs or outputs.

Set RURI_DIAGNOSTIC_DIR before starting a server. Full MessagePack records
include images for offline replay. Once the byte budget is reached, keep only
small JSON summaries. Queue saturation or recording errors never block serving.
"""

import copy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from ruri.common.zmq import pack


def summarize(record):
    request, response = record['request'], record['response']
    out = {k: record[k] for k in ('sequence', 'received_at', 'elapsed_ms')}
    out['prompt'] = response.get('prompt', request.get('prompt'))
    out['response_info'] = {k: v for k, v in response.items() if k != 'action_chunk'}
    out['rtc_context'] = {k: v for k, v in request.items()
                          if k.startswith('context.') and not isinstance(v, np.ndarray)}
    state = np.asarray(request.get('observation.state', []))
    actions = np.asarray(response.get('action_chunk', []))
    out['state'] = state.tolist()
    out['state_finite'] = bool(np.isfinite(state).all())
    out['action_shape'] = list(actions.shape)
    out['actions_finite'] = bool(np.isfinite(actions).all())
    if actions.ndim == 2 and len(actions):
        start = int(response.get('rtc.execute_from', 0))
        out['action_min'] = actions.min(axis=0).tolist()
        out['action_max'] = actions.max(axis=0).tolist()
        out['first_action'] = actions[0].tolist()
        if 0 <= start < len(actions):
            out['first_executable_action'] = actions[start].tolist()
            if state.shape == actions.shape[1:]:
                out['first_executable_minus_state'] = (actions[start] - state).tolist()
                out['max_abs_action_minus_state'] = np.abs(actions - state).max(axis=0).tolist()
        if len(actions) > 1:
            out['max_abs_adjacent_delta'] = np.abs(np.diff(actions, axis=0)).max(axis=0).tolist()
        previous = request.get('context.rtc.prev_chunk_left_over')
        if previous is not None:
            previous = np.asarray(previous)
            out['previous_shape'] = list(previous.shape)
            if previous.ndim == 2 and previous.shape[1:] == actions.shape[1:]:
                n = min(start, len(previous), len(actions))
                if n > 0:
                    out['prefix_max_abs_error'] = float(np.abs(actions[:n] - previous[:n]).max())
                if 0 < start < len(actions):
                    out['prefix_boundary_delta'] = (actions[start] - actions[start-1]).tolist()
    out['images'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                     for k, v in request.items()
                     if k.startswith('observation.images.') and isinstance(v, np.ndarray)}
    return out


class RequestRecorder:
    def __init__(self, directory, metadata, max_bytes=4 * 1024**3):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.max_bytes = max_bytes
        self.written_bytes = 0
        self.sequence = 0
        self.dropped = 0
        self.pending = queue.Queue(maxsize=4)
        self.stopping = threading.Event()
        (self.directory / 'metadata.msgpack').write_bytes(pack(metadata))
        self.thread = threading.Thread(target=self._write_loop, daemon=True,
                                       name='policy-request-recorder')
        self.thread.start()
        logging.info('Request capture enabled: %s (raw budget %.2f GiB)',
                     self.directory, max_bytes / 1024**3)

    @classmethod
    def from_environment(cls, metadata):
        directory = os.environ.get('RURI_DIAGNOSTIC_DIR')
        if not directory:
            return None
        return cls(directory, metadata)

    def begin(self, request):
        self.sequence += 1
        if self.pending.full():
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 100 == 0:
                logging.warning('Diagnostic queue full: dropped %d records', self.dropped)
            return None
        return {'sequence': self.sequence,
                'received_at': datetime.now(timezone.utc).isoformat(),
                '_start': time.perf_counter(), 'request': copy.deepcopy(request)}

    def finish(self, record, response):
        if record is None:
            return
        record['elapsed_ms'] = (time.perf_counter() - record.pop('_start')) * 1000
        record['response'] = copy.deepcopy(response)
        try:
            self.pending.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            logging.warning('Diagnostic queue full: dropped %d records', self.dropped)

    def _write_loop(self):
        while not self.stopping.is_set() or not self.pending.empty():
            try:
                record = self.pending.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                summary = summarize(record)
                summary['dropped_records'] = self.dropped
                summary['raw_file'] = None
                if self.written_bytes < self.max_bytes:
                    payload = pack(record)
                    if self.written_bytes + len(payload) <= self.max_bytes:
                        filename = f"{record['sequence']:08d}.msgpack"
                        temporary = self.directory / (filename + '.partial')
                        temporary.write_bytes(payload)
                        temporary.rename(self.directory / filename)
                        self.written_bytes += len(payload)
                        summary['raw_file'] = filename
                    else:
                        self.written_bytes = self.max_bytes
                        logging.warning('Diagnostic raw budget reached; continuing summaries only')
                with (self.directory / 'summary.jsonl').open('a') as stream:
                    stream.write(json.dumps(summary, default=str) + '\n')
            except Exception:
                logging.exception('Diagnostic recording failed; inference continues')
            finally:
                self.pending.task_done()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=2)
