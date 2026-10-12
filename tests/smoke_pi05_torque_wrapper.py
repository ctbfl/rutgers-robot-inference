"""Real-checkpoint validation of the torque-history wrapper. No robot.

Runs in the OpenPI environment on one free GPU and serves over a real ZeroMQ
socket on localhost for the round-trip checks:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tests/smoke_pi05_torque_wrapper.py \
        --output /tmp/torque_smoke.json

Checks:
  * a client-style 30 Hz effort buffer, sent through the wrapper, gives the model
    exactly the normalized state the training pipeline built for that frame --
    mid-episode and 0.1 s into an episode;
  * a fixed RNG gives identical actions twice; changing only the effort history
    changes them;
  * sampled actions track the recorded ones;
  * training-time RTC pins the committed prefix, in raw action space, and the
    pinned rows do not move when the history changes;
  * metadata and inference survive the real server and inference_client.
"""

import argparse
import json
import logging
import socket
import threading
import time
from pathlib import Path

import numpy as np

from ruri.common import zmq as codec
from ruri.server.serve import handle_request, serve
from ruri.server.wrappers.pi05.pi05_torque import (
    ASSET_ID, EFFORT_HISTORY, EFFORT_TIMES, Pi05TrainRTCTorqueWrapper,
)

DEFAULT_CHECKPOINT = (
    "/common/users/jh2400/training_results/pi05_train_rtc/"
    "260915_whole_v1_torque_rtc_h30_4gpu_bs128/checkpoints/"
    "pi05_whole_v1_torque_rtc_h30/whole_v1_torque_rtc_h30_4gpu_bs128/10000"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    results: dict = {"checkpoint": args.checkpoint, "checks": {}}
    failures: list[str] = []

    def check(name, ok, **detail):
        results["checks"][name] = {"ok": bool(ok), **detail}
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
        if not ok:
            failures.append(name)

    start = time.perf_counter()
    wrapper = Pi05TrainRTCTorqueWrapper(args.checkpoint, num_denoising_steps=10, warmup=True)
    results["startup_s"] = time.perf_counter() - start

    import jax
    from openpi import transforms
    from openpi.training import data_loader
    from pi05_torque import data_patch
    from pi05_torque.config import make_train_config

    # Training-side pipeline, reading norm stats from the same checkpoint.
    config = make_train_config("pi05_whole_v1_torque_rtc_h30", "smoke", Path(args.checkpoint) / "assets")
    dc = config.data.create(config.assets_dirs, config.model)
    data_patch.install(config.model.action_horizon)
    dataset = data_loader.create_torch_dataset(dc, config.model.action_horizon, config.model)
    inner = data_patch._unwrap(dataset)
    train_transform = transforms.compose([
        *dc.repack_transforms.inputs, *dc.data_transforms.inputs,
        transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs,
    ])
    ep_from = np.asarray(inner.episode_data_index["from"])
    ep_to = np.asarray(inner.episode_data_index["to"])

    def rgb(value):
        image = np.asarray(value)
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = (255 * image).astype(np.uint8)  # what LiberoInputs does in training
        return np.ascontiguousarray(image)

    def client_request(index, *, effort_from=None, clock=5000.0):
        """What a 30 Hz client would send at dataset frame `index`, with a cleared-at-start buffer."""
        ep = int(np.searchsorted(ep_to, index, side="right"))
        start_i = int(ep_from[ep])
        rows = np.arange(max(start_i, index - 59), index + 1)
        source = rows if effort_from is None else rows - index + effort_from
        effort = np.stack([np.asarray(inner.hf_dataset[int(i)]["observation.joint_effort"]) for i in source])
        item = dataset[index]
        return {
            "observation.state": np.asarray(item["observation.state"], dtype=np.float32),
            "observation.images.top": rgb(item["observation.images.top"]),
            "observation.images.wrist": rgb(item["observation.images.hand"]),
            EFFORT_HISTORY: effort.astype(np.float32),
            EFFORT_TIMES: clock + (rows - start_i).astype(np.float64) / 30.0,
            "prompt": item["prompt"],
        }, item

    def model_inputs(request):
        mapped = {wrapper.INPUT_MAPPING.get(k, k): v for k, v in request.items()}
        obs = wrapper._to_openpi_observation(mapped, wrapper._resolve_prompt(mapped))
        return wrapper.policy._input_transform(dict(obs))

    # --- train/serve equivalence, mid-episode and at the episode start
    long_eps = [e for e in range(len(ep_from)) if ep_to[e] - ep_from[e] > 200]
    mid = int(ep_from[long_eps[10]]) + 120
    early = int(ep_from[long_eps[20]]) + 3
    for label, index in (("mid-episode", mid), ("episode start + 3 frames", early)):
        request, item = client_request(index)
        served = model_inputs(request)
        trained = train_transform(dict(item))
        state_diff = float(np.max(np.abs(np.asarray(served["state"]) - np.asarray(trained["state"]))))
        prompt_same = bool(np.array_equal(served["tokenized_prompt"], trained["tokenized_prompt"]))
        image_same = all(np.array_equal(np.asarray(served["image"][k]), np.asarray(trained["image"][k]))
                         for k in trained["image"])
        check(f"served == trained model inputs ({label})",
              state_diff == 0.0 and prompt_same and image_same,
              index=index, state_max_abs_diff=state_diff, prompt_equal=prompt_same, images_equal=image_same,
              state_dim=int(np.asarray(served["state"]).shape[0]))

    # --- plain inference: determinism, accuracy, torque sensitivity
    request, item = client_request(mid)
    gt = None
    if mid + 30 <= int(ep_to[long_eps[10]]):
        gt = np.stack([np.asarray(inner.hf_dataset[i]["action"]) for i in range(mid, mid + 30)]).astype(np.float32)

    def run(req, key=0):
        wrapper.policy._rng = jax.random.key(key)
        return wrapper.infer(req)

    first = run(request)
    again = run(request)
    actions = first["action_chunk"]
    check("action chunk shape and finiteness", actions.shape == (30, 7) and np.all(np.isfinite(actions)),
          shape=list(actions.shape), infer_ms=first["timing.infer_ms"])
    check("fixed RNG is deterministic", np.array_equal(actions, again["action_chunk"]))
    check("torque diagnostics for a full buffer",
          first["torque.slots_from_oldest_row"] == 0 and first["torque.max_hold_ms"] < 1.0
          and first["torque.clipped_values"] == 0,
          **{k: v for k, v in first.items() if k.startswith("torque.")})
    if gt is not None:
        err = np.abs(actions - gt)
        check("sampled actions track the recording", float(err[:, :6].mean()) < 5.0,
              joint_mean_abs_err_deg=float(err[:, :6].mean()), gripper_mean_abs_err=float(err[:, 6].mean()))
    other_episode = int(ep_from[long_eps[40]]) + 150
    swapped, _ = client_request(mid, effort_from=other_episode)
    shifted = run(swapped)["action_chunk"]
    shift = float(np.abs(shifted - actions)[:, :6].mean())
    check("changing only the effort history changes the actions", shift > 0.05, joint_mean_abs_shift_deg=shift)

    # --- training-time RTC in raw action space
    consumed, delay = 8, 4
    rtc_request = {
        **request,
        "context.rtc.prev_chunk_left_over": actions[consumed:].copy(),
        "context.rtc.consumed_steps": consumed,
        "context.rtc.estimated_inference_delay_steps": delay,
    }
    conditioned = run(rtc_request, key=1)
    pinned_err = float(np.max(np.abs(conditioned["action_chunk"][:delay] - actions[consumed:consumed + delay])))
    check("RTC applied and pins the committed prefix",
          conditioned["rtc.applied"] and conditioned["rtc.execute_from"] == delay and pinned_err < 1e-3,
          prefix_max_abs_err=pinned_err, infer_ms=conditioned["timing.infer_ms"])
    conditioned_swap = run({**rtc_request, EFFORT_HISTORY: swapped[EFFORT_HISTORY]}, key=1)
    prefix_moved = float(np.max(np.abs(conditioned_swap["action_chunk"][:delay] - conditioned["action_chunk"][:delay])))
    tail_moved = float(np.abs(conditioned_swap["action_chunk"][delay:] - conditioned["action_chunk"][delay:])[:, :6].mean())
    check("pinned rows ignore the history, free rows follow it",
          prefix_moved < 1e-3 and tail_moved > 0.02, prefix_max_move=prefix_moved, tail_mean_move_deg=tail_moved)

    # --- the wire, through the real server and client
    metadata = codec.unpack(codec.pack(handle_request(wrapper, {"type": "metadata"})))
    json.dumps(metadata)
    check("metadata carries the torque contract",
          metadata["policy"]["torque"]["training_q01"] is not None
          and EFFORT_HISTORY in metadata["inputs"] and metadata["outputs"]["output_chunk_size"] == 30,
          asset_id=ASSET_ID)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=serve, kwargs={"policy": wrapper, "bind_address": f"tcp://127.0.0.1:{port}"},
                     daemon=True).start()
    from ruri.client.utils import inference_client
    with inference_client.connect(f"tcp://127.0.0.1:{port}") as client:
        remote_meta = client.metadata
        remote = client.infer(request)
    check("real server round trip",
          remote_meta["policy"]["name"] == "pi05_train_rtc_torque"
          and remote["action_chunk"].shape == (30, 7) and remote["torque.history_rows"] == len(request[EFFORT_TIMES]),
          output_chunk_size=remote_meta["outputs"]["output_chunk_size"])

    results["metadata"] = metadata
    results["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
