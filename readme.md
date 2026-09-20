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
reference drives both arms directly, through the existing gravity compensation
and 100 Hz MIT worker. Move the right leader to correct the motion; automatic
correction stays in the POLICY session, but the leader uses gravity compensation
and drives the follower. Background inference continues as a preview and cannot
overwrite the manual reference. Release uses 0.4 seconds of quiet leader motion
plus fresh inference; then a new RTC segment starts at the corrected state.
Quiet motion is a release proxy: it cannot distinguish a released hand from a
hand holding the arm still.

`q` pauses both arms; enter a task to resume. At the paused prompt, `switch PORT`
changes checkpoint on the same host and `home` returns both arms to zero at up
to 1 rad/s while preserving the gripper target. Home waits for both arms to
settle and stays paused. A physical correction or `q` pauses homing. Esc / Ctrl-C
uses the assisted worker's supported hold-and-stop path without homing on exit.
Runtime aborts use the persistent teleop recovery: hold both measured poses for
10 seconds, then home with MIT at 0.2 rad/s. The client waits for recovery to
finish. If feedback or CAN/MIT state cannot support recovery, the worker stops.
Policy targets are clamped and sent directly; their distance from measured
positions does not itself trigger an abort.
Ordinary single-arm startup/recovery behavior is unchanged.

This example composes `SinglePiperAssistedController`, `AssistedScheduler` and
the existing terminal interface. It does not create episodes, datasets, videos,
effort logs or uploads, and does not depend on robot-learning-workflows. Diagnostic
status is printed to the terminal; observations and control traces stay in memory.

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
