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
