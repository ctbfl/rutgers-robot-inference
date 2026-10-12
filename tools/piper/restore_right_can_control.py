#!/usr/bin/env python3
"""Clear a released teach-mode latch on the registered right Piper (V189).

Read-only by default. --execute --arm-supported performs ONE torque-off reset,
restores CAN/MIT with calibrated gravity and measured-position support, then
disconnects leaving the final MIT hold enabled. No home, gripper or left-arm
commands are sent. Physically support the right arm throughout recovery.
"""

import argparse
from contextlib import suppress
import time

import numpy as np

from ruri.client.controllers.single_piper.discovery import list_can_interfaces, read_can_hardware_id
from ruri.client.controllers.single_piper.hardware_registry import find_arm_registration
from ruri.client.controllers.single_piper.mit import leader_follower as mit
from ruri.client.controllers.single_piper.mit.leader_gravity_check import (
    CALIBRATION, hardware_owner, require_stationary,
)


def describe(arm):
    status = arm.get_arm_status()
    if status is None:
        raise mit.TeleopAbort('No right-arm status feedback')
    m = status.msg
    print(f'ctrl={m.ctrl_mode} mode={m.mode_feedback} status={m.arm_status} '
          f'teach={m.teach_status} enabled={arm.get_joints_enable_status_list()}', flush=True)


def fresh_sample(arm):
    sample = mit.read_sample(arm, 'right recovery')
    status = arm.get_arm_status()
    if status is None or status.msg.err_code != 0:
        raise mit.TeleopAbort('Missing or faulted right-arm status')
    ages = time.time() - np.asarray((*sample.stamps, status.timestamp))
    if not np.isfinite(ages).all() or np.any(ages < -.05) or np.any(ages > .2):
        raise mit.TeleopAbort('Stale right-arm feedback')
    if np.max(np.abs(sample.qd)) > mit.DEFAULT_MAX_JOINT_SPEED:
        raise mit.TeleopAbort('Right arm is moving too quickly for recovery')
    # Encoder readings at a mechanical stop can differ slightly from the old
    # measured range. Keep the command within the existing hardware limits.
    hold = np.clip(sample.q, arm._lower, arm._upper)
    # Allow up to half a degree of stop/readback calibration discrepancy.
    # This does not widen command limits: every hold target stays clamped.
    if np.max(np.abs(hold - sample.q)) > np.deg2rad(.5):
        raise mit.TeleopAbort('Move the supported arm away from its joint stops before recovery')
    return sample


def restore(arm, gravity, *, supported=False):
    if not supported:
        raise ValueError('Physically support the right arm before its torque-off reset')
    sample = fresh_sample(arm)
    require_stationary(sample)
    status = arm.get_arm_status().msg
    if 'NORMAL' not in str(status.arm_status):
        raise mit.TeleopAbort(f'Arm is not healthy: {status.arm_status}; no reset sent')
    enabled = arm.get_joints_enable_status_list()
    if 'CAN' in str(status.ctrl_mode) and enabled and all(enabled):
        mit.require_can_mit_enabled(arm, 'right leader')
        print('Already in enabled CAN/MIT; no reset needed.', flush=True)
        return
    if 'TEACH' not in str(status.ctrl_mode) or 'STOP_RECORDING' not in str(status.teach_status):
        raise mit.TeleopAbort('Expected released TEACHING_MODE / STOP_RECORDING; no reset sent')
    # Check the current calibration output before sending any command.
    mit.bounded_torque(gravity.torque(sample.q), 'right recovery')
    attempted = False
    try:
        attempted = True
        if any(enabled):
            arm.disable()
        deadline = time.monotonic() + 1.0
        while True:
            enabled = arm.get_joints_enable_status_list()
            if enabled is not None and len(enabled) == 6 and not any(enabled):
                break
            if time.monotonic() >= deadline:
                raise mit.TeleopAbort('Right-arm disable was not acknowledged')
            time.sleep(.02)
        stable_since = None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            sample = fresh_sample(arm)
            if np.max(np.abs(sample.qd)) <= .03:
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= .2:
                    break
            else:
                stable_since = None
            time.sleep(.02)
        else:
            raise mit.TeleopAbort('Supported arm did not settle for recovery')
        arm.set_follower_mode()
        arm.set_auto_set_motion_mode_enabled(True)
        hold = np.clip(sample.q, arm._lower, arm._upper)
        mit.send_position_impedance(arm, sample, hold, np.zeros(6),
                                    gravity, 3.0, .5, .15, 'right recovery')
        confirmation = mit.MITStartupConfirmation(arm)
        ready_at = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            enabled = arm.get_joints_enable_status_list()
            if enabled is None or not all(enabled):
                if ready_at is not None:
                    raise mit.TeleopAbort('Right arm lost enable during recovery hold')
                arm.enable()
            else:
                sample = fresh_sample(arm)
                if np.max(np.abs(sample.q - hold)) > .1:
                    raise mit.TeleopAbort('Right arm moved more than 0.1 rad during recovery')
                if confirmation.auto_mode_disabled:
                    mit.require_can_mit_enabled(arm, 'right recovery')
                mit.send_position_impedance(arm, sample, hold, np.zeros(6),
                                            gravity, 3.0, .5, .15, 'right recovery')
                if ready_at is None and confirmation.observe():
                    ready_at = time.monotonic()
                if ready_at is not None and time.monotonic() - ready_at >= 1.0:
                    describe(arm)
                    print('CAN/MIT restored. Right arm left holding its measured pose; restart collection.', flush=True)
                    return
            time.sleep(.02)
        raise mit.TeleopAbort('CAN/MIT recovery timed out; no automatic reset retry')
    except BaseException:
        if attempted:
            with suppress(Exception):
                arm.disable()
            print('Recovery incomplete; right-arm disable requested. Keep supporting it.', flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--arm-supported', action='store_true', help='confirm physical support during the torque-off reset')
    args = parser.parse_args()
    if args.execute and not args.arm_supported:
        parser.error('--execute requires --arm-supported; the right arm briefly loses motor support')
    registration = find_arm_registration('right', 'secondary')
    interfaces = [name for name in list_can_interfaces()
                  if read_can_hardware_id(name) == registration.can_hardware_id]
    if len(interfaces) != 1:
        raise RuntimeError(f'Expected one registered right-arm adapter; found {interfaces}')
    with hardware_owner():
        arm = mit.build_arm(interfaces[0])
        try:
            describe(arm)
            if args.execute:
                restore(arm, mit.GravityCompensator(CALIBRATION), supported=args.arm_supported)
            else:
                print('Read-only. Support the arm, then use --execute --arm-supported to restore CAN/MIT.')
        finally:
            arm.disconnect()


if __name__ == '__main__':
    main()
