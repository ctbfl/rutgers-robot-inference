# FastWAM compilation benchmark — 2026-09-06

No physical robot connection, no persistent policy server, no training changes.
All 134 unit/regression tests passed after integration. A separate fresh-process
compiled constructor/warmup smoke test passed on validation index 0: 127.61 ms
warm whole-wrapper latency, maximum eager/compiled normalized action difference
0.00195323, and exact fixed-seed repeatability. It reused the persistent kernel
cache and created fresh per-process CUDA Graphs.

## Setup

- RURI `FastWAMWrapper`, checkpoint `260906_verified_zero2_row1/checkpoints/weights/step_020000.pt`.
- GPU 0: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.
- PyTorch `2.7.1+cu128`, BF16, four CPU threads, seed 42, batch size one.
- Top/hand RGB input, concatenated 224×448; 32 actions × 7 dimensions.
- Cached text embeddings; no online T5 encoding, video generation or network timing.
- Validation indices: 64, 4468, 8872, 13276. All have unpadded actions.
- Four settings share weights and observations; three warmups and 20 measured
  requests each. Requests alternate between windows; first compile costs are
  excluded from steady-state measurements. CPU wall timers synchronize CUDA.
- Raw per-request results: `logs/fastwam_compile_benchmark_v1.json` (local,
  ignored runtime artifact). The executable benchmark is versioned under tests.

## Latency

| Steps | Compiled | Wrapper P50 / P95 | Model P50 | Peak allocated GPU memory |
|---|---|---|---:|---:|
| 20 | no | 558.06 / 563.73 ms | 555.49 ms | 12.661 GiB |
| 20 | yes | 128.74 / 128.79 ms | 126.11 ms | 12.758 GiB |
| 10 | no | 299.05 / 300.42 ms | 296.71 ms | 12.661 GiB |
| 10 | yes | 78.17 / 78.63 ms | 75.49 ms | 12.661 GiB |

The first cold compiled 20-step request took 73.88 s; the second 943.6 ms and
third 129.6 ms. Compilation happens before serving by default. Compiler caches
help later startup, but CUDA Graphs still need per-process warmup. Peak memory
is PyTorch's allocated-memory measure, not total process memory from nvidia-smi.

## Numerical checks

Compiled/eager differences at the same seed and sampling-step count:

| Steps | Maximum joint difference | Maximum gripper difference | Maximum model-normalized difference |
|---|---:|---:|---:|
| 20 | 0.29818 | 0.15086 | 0.00390634 |
| 10 | 0.29818 | 0.15086 | 0.00390634 |

Joint and gripper values above use the calibrated dataset action convention,
not degrees or millimetres. Compilation/fusion changes BF16 rounding: outputs
are close but not bit-identical to eager. Within each mode, the fixed-seed
repeat difference was exactly zero despite intervening requests with other
observations. Every output was finite and had shape `[32, 7]`.

Open-loop absolute-target MAE on these four windows only:

| Steps | Compiled | Joint MAE | Gripper MAE |
|---|---|---:|---:|
| 20 | no | 1.08812 | 2.26505 |
| 20 | yes | 1.07785 | 2.27065 |
| 10 | no | 1.05578 | 2.31255 |
| 10 | yes | 1.04735 | 2.32203 |

This small diagnostic sample is not enough to rank policy quality or establish
real-robot safety/success. Keep 20 steps as the serving default; acceleration
alone supplies the 4.3x latency gain without reducing denoising steps.

## Reproduce

Use a new output filename on every run (the benchmark refuses overwrites):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
TORCHINDUCTOR_CACHE_DIR=/common/users/jh2400/cache/torchinductor_fastwam \
TRITON_CACHE_DIR=/common/users/jh2400/cache/triton_fastwam \
/common/users/jh2400/conda_envs/fastwam/bin/python tests/benchmark_fastwam_compile.py \
  --checkpoint /common/users/jh2400/training_results/fastwam/260906_verified_zero2_row1/checkpoints/weights/step_020000.pt \
  --output logs/fastwam_compile_benchmark_new.json
```

The script saves progress/failures after each stage, verifies finite outputs
and deterministic changed-input replay, and checks compiled/eager normalized
differences against a configurable 0.01 regression tolerance. This threshold
is a software-integration check, not a robot safety specification.
