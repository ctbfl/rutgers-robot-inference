# Right-leader gravity audit for firmware 1.9.1

The validated firmware 1.9.1 setup keeps pyAgxArm `799b8412fbe8b9156bc9892d3dbeb2df7e98be71`
and the packaged `calibration_leader.json` together, while selecting `PiperFW.V189`.
Do not add the latest SDK's `c` conversion to this calibration as an isolated change.
Offline compatibility checks and operator trials are recorded below. The operator
has accepted gravity-only dragging, dual-arm teleoperation, gripper synchronization
and force feedback, and data collection on the new firmware.

## Calibration provenance and numerical checks

- Physical arm: `right_arm`, USB-CAN `usb:1d50:606f:002B00464759530920353131`.
- Source dataset: `/mnt/data/datasets/leader_gravity_id/20260916_211345`.
- Its metadata records firmware `S-V1.8-2`, the same SDK commit `799b841`, and
  effort units `current*k*b`. These are SDK estimates derived from current,
  not independent measured joint torques.
- All **387,242 samples in 350 segments** exactly satisfy the stored-current
  conversion: J1–J3 `4.725 * current_A`; J4–J6 `0.95844 * current_A`.
- The packaged calibration is the later fit described in its
  `gravity_fit_method`, with payload mass 0.474471567 kg and the no-gripper URDF.
  The dataset metadata's calibration hash identifies the collection-time model,
  not this subsequent fit.
- Current calibration SHA-256:
  `e60dfec0b98dd202e9ef25bc059ab2f52bfbd36529048cce4ed37c4157da3fc1`.
- Current URDF SHA-256:
  `318b5dae9a34bcef82e4dcad706e9be4475abae047732afefa8d6538200831b1`.
- Recomputed gravity plus existing 0.2 damping at **4,028 sampled states** (one
  per 100 records within each segment) stayed within the existing ceilings.
  Peak absolute commands J1–J6 were approximately
  `[0.103, 8.839, 5.013, 0.555, 1.185, 0.151]`, versus
  `[10, 12, 10, 3, 3, 3]`. These are the retained legacy SDK torque units.
- The model's analytic torque matched its numerical potential-energy gradient
  over 20 sampled poses to `1.70e-9` before applying the calibration sign/scale.
  This checks model arithmetic, not physical payload accuracy.
- At home, the calibrated command is approximately
  `[0, 2.793, -4.856, 0.0088, -1.110, 0.0173]`.

The numerical audit is saved in
`/tmp/piper-gravity-torque-audit-20260920.json`; its generating script is
`/tmp/audit_piper_gravity_191.py`. No CAN socket or camera was opened by that audit.

## Firmware encoding versus SDK coefficient changes

The official [firmware reference](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/piper/firmware_reference.md)
states that firmware from 1.8.3 handles the `b` conversion in the master, and
1.8.8/1.8.9 uses the 12-bit MIT torque field and mode `0x06`.
With the pinned SDK, the command chain is therefore:

| Pair | SDK wire torque | Master behavior | Ideal current intent |
| --- | --- | --- | --- |
| DEFAULT / firmware 1.8.2 | `t_ff / b`, 8-bit | transparent `b` | `t_ff / (b*k)` |
| V189 / firmware 1.9.1 | `t_ff`, 12-bit | applies `b` | `t_ff / (b*k)` |

Thus no extra factor of four belongs in the application for J1–J3.
Actual encoded values differ because of quantization. The real vendored codecs
are tested offline at the frame boundary for both signs on all six joints.
The table's current equivalence follows the documented master behavior;
it does not independently measure the firmware's current regulation.

As inspected on 2026-09-20, the latest upstream
[coefficient definitions](https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/api/constants.py)
add `c=0.813252` on Piper J4–J6 and change their `k` from `0.95844` to `0.9576`.
The [V188 driver](https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/protocols/can_protocol/drivers/piper/versions/v188/driver.py)
then encodes `t_ff/c`. Consequently, using unchanged application numbers with
that SDK would multiply wrist wire commands by **1.229631**, while reported
effort per amp would become **0.7787701152**, or **0.812539** of the old value.
Those are calculated unit-conversion effects, not newly measured robot forces.
Migrating SDK coefficients requires jointly reviewing gravity calibration,
software damping, torque ceilings, friction parameters, and logged effort units.

## Prepared operator trial

`ruri.client.controllers.single_piper.mit.leader_gravity_check` is offline by
default. It prints the coefficient contract, calibration hash and home torque;
it rejects mismatched `k/b/c` before hardware access.

```sh
python -m ruri.client.controllers.single_piper.mit.leader_gravity_check
```

The following command is for the operator-supported stage only. Stop the
existing position worker explicitly first; the tool refuses competing writers
and resolves the right arm by USB identity. No firmware configuration or CAN
restart is performed.

```sh
python -m ruri.client.controllers.single_piper.mit.leader_gravity_check \
  --execute --seconds 30 --log logs/right-gravity-fw191.jsonl
```

For operator-controlled timing on this workstation, run:

```sh
./tools/piper/leader_gravity_test
```

This wrapper selects the environment and waits for Enter **before hardware
access**. The operator supports the right arm, presses Enter, and begins dragging
when `GRAVITY ACTIVE` appears. The 60-second timer starts at the actual first
gravity-only command. Ctrl+C ends early; the normal final-position hold applies.

The start must already be stationary and inside the arm's measured limits. The trial uses the existing
fresh MIT enable handshake, holds the measured pose for 2 seconds, removes
position stiffness over 2 seconds, then runs the actual `send_leader()` path
for the requested duration: `kp=kd=0`, `t_ff=gravity(q)-0.2*qd`. Existing measured
arm limits, speed ceiling and feed-forward torque ceilings remain in effect.
There are no follower, gripper, camera, dither or friction-compensation commands.

Every tick records position, velocity, requested torque, current and legacy
effort. The current read and the effort snapshot are sequential, not an atomic
sample. Each motor's feedback and arm status must be recent; error codes abort.

Timeout, SIGINT or SIGTERM ends the free mode. When fresh healthy feedback is
available, the process captures the measured pose and sends position hold for
0.5 seconds before disconnecting. It does not home or disable torque. Hardware
remains energized in the final command; this is not a persistent host control
service. If feedback/state prevents a hold, the failure is reported and the
operator must support the arm. No automatic restart/re-enable is attempted.

The earlier right-arm 20-degree position test used the follower-only entry's
default follower calibration. It established MIT position tracking, not the
leader's gravity-only balance. This trial explicitly selects the leader model.

## Offline validation result

On 2026-09-20, all **233 repository tests passed** in 19.796 seconds, including
13 gravity-trial tests and two additional real-codec tests for the leader law
and DEFAULT/V189 current intent. The simulated trial covers the actual startup
handshake, release ramp, completed/aborted position hold, stale motor feedback,
error codes, torque ceilings, manual excursion, wrong-arm rejection and the
offline default. This was the preparation result before the first hardware trial.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:vendor/pyAgxArm \
  python -m unittest discover -s tests -q
```

Full test output: `/tmp/ruri-gravity-full-tests.log`.

## First hardware trial and operator timing change

The first requested 60-second trial entered gravity-only mode, then ended early
when J3 crossed an extra trial-only 20-degree excursion check during the
operator's drag. The final hold target was approximately
`[0, 0.09470, -0.35606, 0, 0.02836, -0.01044]` rad. The run did **not** complete
60 seconds; its log is `/tmp/right-gravity-fw191-60s-20260920.jsonl`.

The operator requested direct local control of timing. The terminal wrapper now
waits for Enter; the extra 20-degree excursion check and home-only admission
have been removed. Starts still require stationary fresh feedback, valid measured
joint limits and bounded gravity torque. Existing motor-speed, torque, firmware,
feedback and writer-exclusion checks remain. These changes were prepared offline;
no second gravity trial was automatically started.

## Operator acceptance on 2026-09-20

- Both arms completed the sequential 20-degree MIT joint-position checks and
  returned home. J3 used the negative direction within its nominal range.
- The operator independently launched the revised terminal gravity trial and
  confirmed the right leader's balance and drag feel. The local log
  `logs/leader-gravity-1789940890189522786.jsonl` records approximately 41 seconds
  of gravity-only control at 100 Hz, an interrupted outcome, and a final position
  hold. This was not a completed 60-second run.
- The operator confirmed the persistent example's dual-arm teleoperation,
  gripper synchronization, and gripper force feedback worked normally.
- After reviewing the collection reset-home path, the operator also confirmed
  data collection worked normally. No new collection log was inspected for this
  acceptance statement.
- The collection reset path was checked offline with nine RURI request/worker
  tests and three workflow recorder-order tests, all passing. Reset homes the
  leader while the follower tracks it, through the shared V189 MIT driver.
- Before the firmware migration checkpoint commit, the final RURI suite passed
  all **237 tests** in 20.610 seconds. Output:
  `/tmp/ruri-fw191-precommit-tests.log`. The terminal launcher also passed
  `sh -n`, and `git diff --check` was clean.

The old identification collector is retained as
`tools/piper/leader_gravity_id_collect_v182_deprecated.py` for historical reference.
It is not adapted to the new firmware; a new collector will be written if needed.
These results do not validate policy execution or assisted policy/human handoffs
on the upgraded hardware.
