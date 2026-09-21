"""Unchanged TCP/MessagePack contract through the real simulated dual worker."""
from contextlib import chdir
from pathlib import Path
import queue
import socket
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
from client_single_piper_assisted_rtc_interactive import parse_args
from ruri.client.controllers.single_piper_assisted import SinglePiperAssistedController
from ruri.client.controllers.single_piper.normalization import normalize_pose
from ruri.client.schedulers.assisted import run_assisted_rtc
from ruri.client.utils import inference_client
from ruri.common.zmq import send, recv


def udp_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(('127.0.0.1', 0))
        return f'udp://127.0.0.1:{s.getsockname()[1]}'


class DualRemoteRTCTests(unittest.TestCase):
    def test_tcp_policy_intervention_resume_and_no_recording(self):
        ready = queue.Queue()
        stop = threading.Event()
        wire, phases, errors, robots = [], [], [], []

        def server():
            context = zmq.Context()
            reply = context.socket(zmq.REP)
            reply.setsockopt(zmq.LINGER, 0)
            reply.setsockopt(zmq.RCVTIMEO, 100)
            port = reply.bind_to_random_port('tcp://127.0.0.1')
            ready.put(f'tcp://127.0.0.1:{port}')
            try:
                while not stop.is_set():
                    try:
                        request = recv(reply)
                    except zmq.Again:
                        continue
                    if request['type'] == 'metadata':
                        send(reply, {'inputs': {key: {} for key in (
                            'observation.state', 'observation.images.top', 'observation.images.wrist')},
                            'outputs': {'output_chunk_size': 10},
                            'policy': {'checkpoint_path': 'test-rtc'}})
                    else:
                        wire.append((time.monotonic(), request))
                        time.sleep(.025)  # Old actions continue during the remote request.
                        send(reply, {'action_chunk': np.tile(request['observation.state'], (10, 1)),
                                     'rtc.applied': 'context.rtc.prev_chunk_left_over' in request,
                                     'rtc.method': 'inference-time'})
            except Exception as exc:
                errors.append(exc)
            finally:
                reply.close(); context.term()

        server_thread = threading.Thread(target=server, daemon=True)
        server_thread.start()
        endpoint = ready.get(timeout=2.)
        args = parse_args(['--policy-endpoint', endpoint, '--simulate', '--no-startup-home',
                           '--prompt', 'A0', '--seconds', '3.6'], interactive=False)
        args.command_address, args.telemetry_address = udp_address(), udp_address()
        args.camera_width, args.camera_height = 32, 24

        def factory(config):
            robot = SinglePiperAssistedController(config)
            original = robot.get_sample
            def sample():
                result = original()
                phases.append((time.monotonic(), result['telemetry']['control']['authority_phase']))
                return result
            robot.get_sample = sample
            robots.append(robot)
            return robot

        def intervene():
            try:
                deadline = time.monotonic()+5.
                while len(wire) < 3:
                    if time.monotonic() > deadline:
                        raise AssertionError('remote inference did not start')
                    time.sleep(.01)
                robot = robots[0]
                leader = robot._io.snapshot()['leader']
                q = np.asarray(leader['q']); q[0] += .25
                robot.simulate_leader(normalize_pose(q, leader['gripper_width']), duration_s=.8)
            except Exception as exc:
                errors.append(exc)

        operator = threading.Thread(target=intervene, daemon=True)
        try:
            with tempfile.TemporaryDirectory() as directory, chdir(directory):
                with inference_client.connect(args) as policy:
                    operator.start()
                    stats = run_assisted_rtc(controller=factory, policy=policy, args=args)
                self.assertEqual(list(Path(directory).iterdir()), [])
        finally:
            stop.set()
            server_thread.join(2.)
            if operator.ident is not None:
                operator.join(2.)
        if errors:
            raise errors[0]
        self.assertFalse(robots[0].is_connected)
        self.assertGreater(stats['rtc_results'], 0)
        self.assertTrue({'policy', 'blend', 'human', 'recover'} <= {p for _, p in phases})
        self.assertEqual(phases[-1][1], 'policy')
        human_start = next(t for t, p in phases if p == 'human')
        recovery_start = next(t for t, p in phases if p == 'recover')
        resumed = next(t for t, p in phases if t > recovery_start and p == 'policy')
        self.assertTrue(any('context.rtc.prev_chunk_left_over' in x for t, x in wire if t < human_start))
        self.assertTrue(any(human_start < t < recovery_start for t, _ in wire))
        # Exclude boundary scheduling races by a control tick.
        for t, request in wire:
            if human_start+.04 < t < recovery_start:
                self.assertNotIn('context.rtc.prev_chunk_left_over', request)
        self.assertTrue(any('context.rtc.prev_chunk_left_over' in x for t, x in wire if t > resumed))
        allowed = {'type', 'prompt', 'observation.state', 'observation.images.top',
                   'observation.images.wrist', 'context.rtc.prev_chunk_left_over',
                   'context.rtc.consumed_steps', 'context.rtc.estimated_inference_delay_steps'}
        for _, request in wire:
            self.assertLessEqual(set(request), allowed)
            self.assertEqual(request['prompt'], 'A0')
            self.assertEqual(request['observation.state'].shape, (7,))
            for name in ('top', 'wrist'):
                self.assertEqual(request[f'observation.images.{name}'].shape, (24, 32, 3))
                self.assertEqual(request[f'observation.images.{name}'].dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
