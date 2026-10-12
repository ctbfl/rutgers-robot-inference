# FastWAM action-only server

Run in the FastWAM environment with RURI's base dependencies (`numpy`,
`pyzmq`, `msgpack`). No Piper SDK/submodule is needed on the policy server.
No training/DeepSpeed process is started. The server uses one selected GPU.

```bash
GPU=0 bash examples/shell_scripts/serve_fastwam.sh
```

Defaults: final 20k export of `260906_verified_zero2_row1`, port 5560,
20 diffusion steps, seed 42, compiled action inference, three startup warmups.
Checkpoint/prompt defaults live
only in `examples/server_fastwam.py`; all shell arguments pass through:

```bash
GPU=0 bash examples/shell_scripts/serve_fastwam.sh \
  --checkpoint /path/to/run/checkpoints/weights/step_016000.pt \
  --bind tcp://127.0.0.1:5560
```

The default bind exposes the unauthenticated RURI service on all interfaces,
like the other policy examples. Use loopback or a trusted private network.
Loading and warmup finish before the endpoint is registered with RURI's menu.

## Compilation and latency

Compilation is enabled by default (`compile_action_infer=True` in the wrapper;
`--compile-action-infer` in the entry point). It uses FastWAM's native
`torch.compile(mode="reduce-overhead", fullgraph=True)` video-prefill and
action-denoising path, including its CUDA Graph buffer-lifetime handling. It
does not reuse previous observations' video K/V or previous action chunks.
The VAE/input preparation remains on the native eager path.

Keep the default startup warmup: the first pass compiles, subsequent passes
capture/replay CUDA Graphs before the server advertises readiness. The initial
cold-cache compiled request took about **74 seconds** in our experiment;
this is NOT steady-state latency. Start the client only after server readiness.
`--warmup-iters` changes the default three passes; `--no-warmup` defers all this
cost to the first client request and is not recommended for serving.

The shell launcher stores Inductor/Triton caches under
`/common/users/jh2400/cache/{torchinductor_fastwam,triton_fastwam}` (override
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to relocate). Repeated process
startup can reuse compiled kernels, but must still warm up the process's graphs.
No package upgrade or model retraining is required.

To explicitly disable compilation for diagnosis:

```bash
GPU=0 bash examples/shell_scripts/serve_fastwam.sh --no-compile-action-infer
```

Compilation errors fail visibly; the wrapper does not silently fall back to
eager while advertising compiled execution. Metadata reports the selected
compilation flag/mode and warmup count. CUDA is required for this accelerated path.

Paired benchmark on GPU 0 (RTX PRO 6000 Blackwell Max-Q), BF16, 20k checkpoint,
four held-out validation windows, 20 measured requests per setting:

| Sampling steps | Compilation | Whole-wrapper P50 | P95 |
|---|---|---:|---:|
| 20 | off | 558.1 ms | 563.7 ms |
| 20 | on | 128.7 ms | 128.8 ms |
| 10 | off | 299.1 ms | 300.4 ms |
| 10 | on | 78.2 ms | 78.6 ms |

Thus compilation alone speeds up the unchanged 20-step policy by about **4.3x**.
The default remains 20 steps. `--diffusion-steps 10` is available, but reducing
steps changes the sampling algorithm and needs a wider quality evaluation.
Unlike the eager/native smoke test, compiled/eager BF16 actions are not
bit-identical: the maximum difference was 0.003907 in model-normalized action
units, or 0.299 joint / 0.151 gripper units in the calibrated Piper action space
(NOT degrees or millimetres). Fixed-seed repeated requests remained bit-identical
within each mode while alternating observations. Four windows do not establish
closed-loop task success. Full experiment details: [compile benchmark](fastwam_compile_benchmark.md).

## Assets and supported checkpoint schema

- A complete `mot` + `proprio_encoder` weights export, **not** a DeepSpeed state
  directory or legacy video-only checkpoint. Both modules load strictly.
- The run's saved `config.yaml` and `dataset_stats.json`, inferred from the
  standard export layout; override with `--config` and `--stats` if relocated.
- Frozen VAE weights under `FASTWAM_ROOT/checkpoints`, or `--model-base-path`.
  Fine-tuned exports replace both experts, so their original pretrained weights
  are not loaded again. Automatic downloads are disabled.
- Training text-embedding cache, located through the saved config or
  `--text-cache-dir`. No online T5 encoder is loaded.

Only the reviewed single-Piper schema is supported: seven state/action values,
six chunk-relative joint dimensions plus absolute gripper, global normalization,
top/hand cameras concatenated horizontally, first-frame-causal FastWAM. Unknown
schemas fail rather than silently changing action meaning. Horizon comes from
the run config (`num_frames - 1`), not a client override.

## Wire contract

Input keys are `observation.state` (seven finite numeric values),
`observation.images.top` and `observation.images.wrist` (RGB uint8 HWC arrays),
plus optional `prompt`. Wrist maps to training's hand camera. Image resolution
may vary; preprocessing follows the saved validation transforms and final
resize/crop. RGB vs BGR cannot be detected automatically.

The state and returned actions use the **same calibrated Piper coordinates as
the training dataset and current client**, not radians/degrees. Hardware
calibration normalization is distinct from the model's dataset-statistics
normalization. No old/new calibration conversion is applied.

After inverse model normalization, each row's first six dimensions is added
to the *same original input state*. These are not incremental deltas to sum
over time. The gripper remains absolute. The response contains:

- `action_chunk`: CPU float32 NumPy `[horizon, 7]`, currently `[32, 7]`;
- `prompt`: effective requested task string;
- `timing.infer_ms`: synchronized model execution time;
- `timing.wrapper_ms`: whole request time, including preprocessing/postprocessing.

No chunk truncation, smoothing, hardware clipping or video generation occurs in
the wrapper. Schedulers select/compose/clip final targets; controllers validate
and send them. RTC context is rejected: this wrapper does not implement RTC.

## Prompt and client integration

The default task is exactly the row-of-five instruction used in training:

```text
insert the row of five objects, from left to right, into the bottom row of holes on the manipulation-net board.
```

The wrapper applies FastWAM's training prompt prefix once, and also accepts the
already-prefixed form used by offline evaluation. A request with no prompt uses
the server default. Other prompts need an exact matching embedding cache file;
uncached prompts raise a clear error. Padding and attention masks match training.
`--random-seed` requests fresh noise; otherwise the fixed seed makes repeated
observations reproducible. No previous action chunk or image cache is reused.

Existing Piper client examples have a **different default prompt**; explicitly
pass the row-of-five task when connecting them. Start validation with
`DummyController` and `BlockingScheduler`; importing the wrapper does not touch
hardware. Real robot execution is a separate operational step, not part of the
smoke test. Check measured inference latency against the chosen execution prefix
and client timeout before choosing rolling/temporal ensemble scheduling. A
32-step horizon at 30 Hz is only about 1.07 seconds and does not imply that
inference meets that budget. Current connection utility defaults to a 10 s
request timeout and accepts `policy_timeout_s` through its args object.

## Verification

Base-only contract tests, including real loopback ZeroMQ + dummy scheduler:

```bash
PYTHONPATH=src python -m unittest tests.test_fastwam_wrapper -v
```

Opt-in GPU smoke test (no hardware, no persistent service registration):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  /common/users/jh2400/conda_envs/fastwam/bin/python tests/smoke_fastwam_wrapper.py \
  --checkpoint /common/users/jh2400/training_results/fastwam/260906_verified_zero2_row1/checkpoints/weights/step_020000.pt
```

This smoke test defaults to eager mode for exact native comparison. Add
`--compile-action-infer` to also exercise the default compiled constructor and
warmup, comparing against eager native actions with a 0.01 model-normalized
regression tolerance (not a physical safety limit).

It uses a held-out validation window, compares image/state/text preprocessing
exactly to the native dataset, checks action-transform inversion, compares
wrapper actions to native inference at the same seed, and checks repeatability.
This validates integration, not closed-loop task success or robot safety.

Initial eager verification on 2026-09-06: 130 tests passed (including the optional CPU
backend tests). The 20k checkpoint on validation index 0 matched native
preprocessing exactly and native action inference with max absolute difference
0.0; fixed-seed repeat was exact. GPU 0 (RTX PRO 6000 Blackwell Max-Q), 20
diffusion steps: first measured model call 735 ms, repeated warm model call
542 ms, warm whole-wrapper call 544 ms. These are individual smoke-test timings,
not a latency distribution or a closed-loop performance claim. The FastWAM
environment needed `pyzmq` installed; no training packages were upgraded.
