Rutgers Robot Inference (RU-RI, ruri) is a lightweight robot inference layer.

Three modules make up the full inference stack:

- `policy_wrapper` runs inference. You write it yourself, generally a new one per model. Think of it as calling the model directly as an API.

- `inference_scheduler` manages timing. It reads sensor data from the robot controller, passes it to the policy wrapper, then composes the returned actions and hands them downstream to the robot controller for execution.

- `robot_setup_controller` owns the hardware and exposes an abstract interface to the layers above it, covering camera sensors, arm control, and so on.

## Choosing a remote policy

List the currently available policies and their complete metadata from the
default internal menu:

```bash
python examples/list_policy_servers.py
```

Then pass the selected endpoint explicitly to a client. Clients intentionally
have no default policy port:

```bash
python examples/client_single_piper_blocking.py \
  --policy-endpoint tcp://arcla.cs.rutgers.edu:5555 \
  --execute-actions-per-chunk 25
```

The server-declared `outputs.output_chunk_size` is authoritative; clients do
not request or override the returned action horizon.

The client connection is a shared utility, not a Scheduler dependency to
inject:

```python
from ruri.client.utils import inference_client

with inference_client.connect(args) as policy:
    scheduler.run(controller=SinglePiperController, policy=policy, args=args)
```

Schedulers own chunk selection, RTC/aggregation and final target clipping.
Controllers validate that final target, convert it to hardware units and send
it; `send_action()` does not silently alter or return the target.

For interactive single-arm RTC, use `examples/client_single_piper_rtc_interactive.py`
with the selected `--policy-endpoint` and the left arm's current `--can-interface`.
Enter the task prompt to start. Press `q` to pause, then enter either a new task
to resume or `home` to return all six joints to zero. Manual home preserves the
gripper target and uses a synchronized trajectory with a maximum joint reference
speed of **1 rad/s**, followed by a measured settling check. It stays paused at
home until another task is entered. During homing, `q` pauses the motion and
`Esc` / Ctrl-C exits through the usual controller recovery. Resuming after home
requests a fresh chunk without an old RTC prefix. Startup and fault recovery
keep their existing homing behavior.

To compare checkpoints without restarting, press `q`, enter `switch 5558`, and
press Enter. This replaces only the current endpoint's port (for example,
`tcp://172.16.68.130:5557` becomes `tcp://172.16.68.130:5558`). The arm holds its
last target while the previous request finishes and the new connection performs
a metadata handshake. The target must declare the same input schema, action
horizon and policy/RTC configuration; the checkpoint path may differ. A successful
switch keeps the task prompt, resets RTC latency estimates, and resumes from a
fresh observation without reusing old actions. A failed switch keeps the old
connection and stays paused; enter a task to resume it or another `switch <port>`
to retry. During the connection attempt, `q` suppresses automatic resumption and
`Esc` / Ctrl-C quits. The scheduler log records port changes and checkpoint metadata.

## Paired Piper interactive inference without recording

```bash
python examples/client_single_piper_assisted_rtc_interactive.py \
  --policy-endpoint tcp://172.16.68.130:5557 \
  --can-interface can0
```

`--can-interface` selects the left/main follower. The right/secondary leader is
discovered by its registered USB hardware ID; optionally specify its current
`--leader-can-interface`. Both IDs are verified before use. Stop the existing
single-arm client before starting this example: the paired worker owns both arms.

Enter the task prompt (for example `A0`). Both arms enable and home using the
persistent teleop MIT sequence, then RTC begins directly. Policy startup and
resume do not wait for measured arm or gripper agreement. Use
`--no-startup-home` to start from nominal-clamped current poses instead. Every
MIT position command passes through the nominal clamp and then that arm's
measured hardware clamp; measured feedback is preserved. The same virtual
reference drives both arms through the validated 100 Hz MIT worker. The right
gripper opens fully during policy tracking. Close it by **1 cm** to request
takeover: the left gripper holds its measured opening, the right actively matches
it, then the right arm enters pure gravity directly. Joint deflection does not
trigger takeover. Once the right arm reaches pure gravity control, the
left blends to its live readings over 0.5 seconds. Quiet right-arm motion for
0.5 seconds starts recovery once the follower reference catches up; impedance
returns over 0.5 seconds. The right gripper reopens fully and re-arms for the next
gesture; its automatic opening does not move the left gripper. Inference continues throughout intervention.
During autonomous policy tracking the original RTC prefix and latency alignment
remain active. Intervention discards old chunks and in-flight results; recovery
uses fresh observations without a stale prefix, then starts a fresh RTC segment.
See [the local control model](docs/piper-shared-reference.md).

`q` pauses both arms; enter a task to resume. At the paused prompt, `switch PORT`
changes checkpoint on the same host and `home` returns both arms to zero at up
to 1 rad/s while preserving the gripper target. Home waits for both arms to
settle and stays paused. A physical correction or `q` pauses homing. Esc / Ctrl-C
uses the assisted worker's supported hold-and-stop path without homing on exit.
Runtime aborts use the persistent teleop recovery: hold both measured poses for
10 seconds, then home with MIT at 0.2 rad/s. The client waits for recovery to
finish. If feedback or CAN/MIT state cannot support recovery, the worker stops.
Policy proposals are clamped and integrated at a bounded reference velocity.
The gravity-handoff path retains feedback, status, speed and feed-forward checks.
Ordinary single-arm startup/recovery behavior is unchanged.

This example composes `SinglePiperAssistedController`, `AssistedScheduler` and
the existing terminal interface. It does not create episodes, datasets, videos,
effort logs or uploads, and does not depend on robot-learning-workflows. Diagnostic
status is printed to the terminal; observations and control traces stay in memory.

## Interactive dual-arm remote RTC

```bash
python examples/client_dual_piper_rtc.py \
  --policy-endpoint tcp://172.16.68.130:5557
```

Use the same endpoint and exact task prompt as the existing single-arm RTC client.
The existing `server_pi05_rtc.py` or training-time RTC server needs no changes.
Enter the task prompt in the terminal to enable/home both arms and start RTC.
Press `q` to pause, then enter another task, `home`, or `switch PORT`.
Esc / Ctrl-C requests HOLD and exits.
It uses the accepted circle controller with the remote policy in place of the
circle policy: right human intervention and left handover stay local. The server
gripper convention stays 0–100, mapped locally to the validated 0–100 mm range
(action 10 commands 10 mm). The right gripper opens to 100 mm, triggers at
90 mm, and aligns to the left at 20 cm/s before gravity takeover. The server
still receives the left arm's 7D state, top/wrist images and the ordinary optional
RTC context, and returns `(H, 7)` actions. There are no intervention/dataset/video
or effort files. `--simulate` uses the same server connection with simulated
arms/images and no hardware. This is a short alias of the interactive entry above.
See [communication and RTC handoff details](docs/piper-assisted-rtc.md).

## Right leader circle demo with gravity handoff

```bash
# Offline FK/IK audit (requires SciPy, provided by the piper-circle extra).
python examples/client_single_piper_shared_circle.py
# Worker + local policy simulation, with automatic artificial intervention.
python examples/client_single_piper_shared_circle.py --simulate --simulate-intervention --seconds 35
# Right arm only: enable, home, circle policy; drag to intervene, stay still to resume.
python examples/client_single_piper_shared_circle.py --execute
```

Default circle: home TCP +5 cm forward / +15 cm up, radius 10 cm, circle angular speed 0.5 rad/s
(12.57 seconds per lap), policy joint velocity limit 0.5 rad/s, and reference
ceiling 4 rad/s, with relaxed pitching orientation. After a filtered joint error
exceeds 5 degrees for 0.1 seconds, leader impedance smoothly fades out over
0.5 seconds. Full human control uses the original gravity-plus-damping teleop
command with motor kp=kd=0. No admittance or position spring resists dragging.
When every joint moves at most 0.1 degrees over 0.5 seconds (`--quiet-angle`),
the reference rebases to the measured pose and impedance returns over 0.5 seconds.
Holding the arm still counts toward recovery. Terminal status prints at 10 Hz.
This entry uses `leader_only=True`; no left CAN or cameras are opened.
Ctrl-C requests HOLD. See [parameters, logging and validation](docs/piper-shared-reference.md).

## Dual-arm circle with right gravity handoff

```bash
python examples/client_dual_piper_shared_circle.py  # offline audit for both arms
python examples/client_dual_piper_shared_circle.py --simulate --simulate-intervention --seconds 9
python examples/client_dual_piper_shared_circle.py --execute  # enable and home both, then run
```

One circle policy feeds both arms the same joint reference during policy tracking
and gripper alignment. All Piper controllers now use a 0–100 mm nominal gripper
range and 0–100 mm per-arm clamps, with normalized gripper values 0–100.
This trial uses a 100 mm right
open target (0.2 N while opening, then a weak spring capped at 0.05 N), and a fixed **90 mm** takeover threshold.
Close the right gripper to 90 mm or below to request takeover, at any speed;
after matching the left gripper, the right arm enters pure gravity directly. Then
the left arm blends from its last sent reference to the **live** right joint readings
over 0.5 seconds, using a quintic curve and its own reference velocity. The left arm
keeps the original single-Piper tracking gains (Kp=10, Kd=0.8) and calibrated
gravity feed-forward, including startup and home. On release, the shared
reference resumes from the last left command once the handover has caught up; right
impedance returns over 0.5 seconds. Rate limiting can extend catch-up time.
This example selects `gravity_handoff=True, human_trigger="gripper"`; the right-only
circle retains joint-deflection entry. Logs: `logs/dual_shared_circle_*.jsonl`. Both circles use each
arm's local base coordinates. The left arm remains a position tracker throughout;
only the right arm enters zero-stiffness gravity dragging.
If a previous failure left a latched electronic stop, `--execute --reset-stop`
explicitly resets stopped arms once, verifies fresh healthy feedback, then starts
the usual enable/home sequence. Ordinary startup does not clear stops.
See [handover equations and timing](docs/piper-shared-reference.md).

## Pi05 sim/real inference server

The sim/real training-time RTC wrapper uses the existing client protocol.
See [checkpoint layout, installation and verification](docs/pi05_sim_real.md).
Select the deployment domain on the server; clients keep their existing
state, camera, prompt and RTC fields.

## Piper installation

The official Agilex SDK is pinned as a Git submodule. It and the Piper-only
Python dependencies are optional:

```bash
git submodule update --init vendor/pyAgxArm
pip install -e '.[piper]'
```

Base RURI imports without the submodule or Piper extra. LeRobot is intentionally
not part of the `piper` extra: install and configure the appropriate LeRobot,
PyTorch/CUDA and RealSense stack yourself.

Piper MIT workers use `PiperFW.V189` and query the actual firmware before enabling
an arm. Missing identification, an unrecognised version, or firmware older than
`S-V1.8-9` stops startup. Both registered arms reported `S-V1.9-1` on 2026-09-20.
The SDK remains pinned to `799b841`; its existing torque units and packaged
gravity calibrations are retained. Updating to the newer SDK
torque coefficients is a separate calibration migration.

The old gravity identification collector is archived as
[`leader_gravity_id_collect_v182_deprecated.py`](tools/piper/leader_gravity_id_collect_v182_deprecated.py).
It retains the v1.8.2 DEFAULT / 8-bit assumptions and is incompatible with the
current V189 controller. A new collector will be written when recalibration is
needed; the existing calibration files remain in use.

The [right-leader gravity audit](docs/notes/2026-09-20-piper-gravity-fw191.md)
checks that pairing against the original current/effort recordings. The separate
`python -m ruri.client.controllers.single_piper.mit.leader_gravity_check` entry
is entirely offline by default. Its operator-supported `--execute --seconds 30`
trial binds the right arm by USB identity, uses `calibration_leader.json`, and
smoothly releases position hold into the existing gravity-plus-damping law.
The trial starts from the current stationary pose within the arm's measured
limits. `./tools/piper/leader_gravity_test` selects the workstation environment
and waits for Enter before starting a 60-second trial; its timer begins only
after gravity-only mode is active. Ctrl+C ends the trial early. It records the
full trial and attempts a fresh position hold on exit;
there is no automatic home, gripper command, camera access, or torque disable.

For the first single-arm MIT check, `policy_controller --hold-only` holds the
measured startup joint positions. It skips homing, opens no policy command
receiver, and sends no gripper commands. It still applies the existing measured
hardware limits, torque ceilings, speed/tracking checks, and feedback watchdog.
Startup requires advancing CAN/MIT/NORMAL status before and after automatic
mode frames are disabled.

Resolve the left/main interface by its registered USB-CAN serial first; the
interface name can change. With the workspace inspected and the arm supported,
the live check can be launched using that interface:

```bash
python -m ruri.client.controllers.single_piper.mit.policy_controller \
  --follower-can <left-arm-interface> --execute --hold-only \
  --diagnostic-log logs/piper-v189-hold.jsonl
```

Omit `--execute` for identification and feedback checks only (a firmware read
request is sent). In hold-only mode, `--seconds N` counts from HOLD ready rather
than waiting for a policy command. The default runs until interrupted. Exiting
does **not** disable joint torque or establish a persistent background hold;
it disconnects after the existing final hold stream. The diagnostic log saves
the last five seconds on exit or failure, including firmware, desired/measured
joints, feed-forward torque, measured effort, and motor current. The default
calibration is for the left/follower arm; a right-arm test must explicitly use
`--follower-calibration` with `calibration_leader.json`.

## Piper leader/follower collection

The standard no-force-feedback MIT path is self-contained in RURI. In the
first terminal, start the controller that owns both CAN buses:

```bash
python examples/single_piper_leader_follower_teleop.py --execute
```

It identifies right/secondary as the leader and left/main as the follower by
the registered USB-CAN hardware IDs. The follower target and follower state are
published in the same MIT telemetry packet.

Install the optional LeRobot adapters into your already-prepared LeRobot 0.6.1
environment:

```bash
pip install -e integrations/lerobot_robot_piper_mit
pip install -e integrations/lerobot_teleoperator_piper_mit
```

Then record in a second terminal:

```bash
lerobot-record \
  --robot.type=piper_mit_observer \
  --robot.id=piper_follower \
  --teleop.type=piper_mit \
  --teleop.id=piper_mit \
  --dataset.repo_id=local/my_dataset \
  --dataset.single_task='describe the task' \
  --dataset.root=/absolute/path/to/my_dataset \
  --dataset.fps=30 \
  --dataset.num_episodes=50 \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.no_stamp=true \
  --dataset.push_to_hub=false
```

The adapters do not open CAN. At each LeRobot frame, RURI latches one current
follower state/target packet and reads the latest head and wrist camera frames.
LeRobot's 30 Hz frame index remains the dataset timebase; camera or telemetry
timestamps are not added to the dataset. Both cameras use RURI's reviewed
`cam_params.json` controls and warm-up procedure.

Legacy adapters for LeRobot 0.5.2 are retained under the corresponding
`*_lerobot0_5_2` integration directories. Install one generation only; both
generations intentionally register the same CLI type names.
