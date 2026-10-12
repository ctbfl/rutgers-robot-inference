"""Opt-in paired eager/compiled benchmark on held-out windows. No robot I/O.

Reports whole-wrapper P50/P95, model time, compile warmup cost, peak VRAM,
compiled/eager action differences, and independent changed-input repeatability.
Results are saved after each stage so a compiler failure is not lost.
"""

import argparse
import json
import logging
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np

from ruri.server.wrappers.fastwam import FastWAMWrapper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="New JSON file, refuses to overwrite")
    p.add_argument("--fastwam-root", default="/common/users/jh2400/FastWAM")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, nargs="+", default=[20, 10])
    p.add_argument("--max-normalized-diff", type=float, default=0.01,
                   help="Compiled/eager regression tolerance, not a robot safety limit")
    args = p.parse_args()
    if min(args.iters, args.windows, args.warmup, *args.steps) < 1 or args.iters < args.windows:
        p.error("Counts must be positive and iters >= windows")
    if not np.isfinite(args.max_normalized_diff) or args.max_normalized_diff <= 0:
        p.error("max-normalized-diff must be finite and positive")
    output = Path(args.output).resolve()
    if output.exists():
        p.error(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch

    torch.set_num_threads(4)
    wrapper = FastWAMWrapper(args.checkpoint, fastwam_root=args.fastwam_root,
        default_prompt="insert the row of five objects, from left to right, into the bottom row of holes on the manipulation-net board.",
        warmup=False, compile_action_infer=False)
    runtime = wrapper._runtime
    from hydra.utils import instantiate
    from fastwam.utils import misc

    report = {"checkpoint": str(runtime.checkpoint_path), "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "windows": [], "stages": [], "status": "running"}

    def save():
        output.write_text(json.dumps(report, indent=2))

    with tempfile.TemporaryDirectory(prefix="fastwam-compile-data-") as folder:
        misc.register_work_dir(folder)
        ds = instantiate(runtime.cfg.data.val, pretrained_norm_stats=str(runtime.stats_path),
                         text_embedding_cache_dir=str(runtime.text_cache_dir))
        base = ds.lerobot_dataset
        processor = base.processor
        requests, ground_truth = [], []
        # Interior, well-separated windows; shared across every timed setting.
        indices = np.linspace(64, len(ds) - 65, args.windows, dtype=int)
        for index in indices:
            try:
                base.processor = None
                raw = base[int(index)]
            finally:
                base.processor = processor
            if bool(raw["action_is_pad"].any()):
                raise ValueError(f"Selected padded action window at {index}; choose different windows")
            assert raw["idx"] == int(index), "Dataset retried a different window"
            state = raw["state"]["default"][0].numpy().copy()
            requests.append({"observation.state": state,
                "observation.images.top": raw["images"]["top"][0].permute(1, 2, 0).numpy().copy(),
                "observation.images.wrist": raw["images"]["hand"][0].permute(1, 2, 0).numpy().copy()})
            ground_truth.append(raw["action"]["default"].numpy().copy())
            report["windows"].append(int(index))
        references = {}
        for compiled in (False, True):
            for steps in args.steps:
                runtime.compile_action_infer = compiled
                wrapper.diffusion_steps = steps
                stage = {"compiled": compiled, "steps": steps, "warmup_ms": [], "samples": []}
                report["stages"].append(stage)
                save()
                print(f"BEGIN compiled={compiled} steps={steps}", flush=True)
                try:
                    torch.cuda.reset_peak_memory_stats()
                    for i in range(args.warmup):
                        start = time.perf_counter()
                        wrapper.infer(requests[i % len(requests)])
                        stage["warmup_ms"].append((time.perf_counter() - start) * 1000)
                        print(f"WARMUP {i + 1}: {stage['warmup_ms'][-1]:.1f} ms", flush=True)
                        save()
                    first_actions = {}
                    for i in range(args.iters):
                        j = i % len(requests)
                        response = wrapper.infer(requests[j])
                        action = response["action_chunk"].copy()
                        assert action.shape == (runtime.horizon, 7) and np.isfinite(action).all()
                        row = {"window": j, "wrapper_ms": response["timing.wrapper_ms"],
                               "model_ms": response["timing.infer_ms"],
                               "joint_mae_to_gt": float(np.abs(action[:, :6] - ground_truth[j][:, :6]).mean()),
                               "gripper_mae_to_gt": float(np.abs(action[:, 6] - ground_truth[j][:, 6]).mean())}
                        if j in first_actions:
                            row["repeat_max_abs_diff"] = float(np.abs(action - first_actions[j]).max())
                            assert row["repeat_max_abs_diff"] == 0, "Changed-input graph replay is not repeatable"
                        first_actions[j] = action
                        if compiled:
                            diff = action - references[(steps, j)]
                            normalized_diff = diff * runtime.action_norm.scale.numpy()
                            row.update(joint_max_abs_diff=float(np.abs(diff[:, :6]).max()),
                                       joint_mean_abs_diff=float(np.abs(diff[:, :6]).mean()),
                                       gripper_max_abs_diff=float(np.abs(diff[:, 6]).max()),
                                       normalized_max_abs_diff=float(np.abs(normalized_diff).max()))
                            if row["normalized_max_abs_diff"] > args.max_normalized_diff:
                                stage["samples"].append(row)
                                raise AssertionError("Compiled/eager action difference exceeds regression tolerance")
                        else:
                            references[(steps, j)] = action
                        stage["samples"].append(row)
                    for field in ("wrapper_ms", "model_ms"):
                        values = [row[field] for row in stage["samples"]]
                        stage[field] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
                                        "mean": float(np.mean(values))}
                    stage["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                    stage["status"] = "ok"
                    save()
                    print(json.dumps({k: v for k, v in stage.items() if k != "samples"}), flush=True)
                except Exception:
                    stage["status"] = "failed"
                    stage["error"] = traceback.format_exc()
                    report["status"] = "failed"
                    save()
                    raise
    report["status"] = "complete"
    save()
    print(f"BENCHMARK_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
