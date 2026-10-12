#!/usr/bin/env bash
# Python owns checkpoint, prompt and port defaults. All CLI arguments pass through.
set -euo pipefail
RURI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-/common/users/jh2400/conda_envs/fastwam/bin/python}"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="$RURI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/common/users/jh2400/cache/torchinductor_fastwam}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/common/users/jh2400/cache/triton_fastwam}"
exec "$FASTWAM_PYTHON" "$RURI_ROOT/examples/server_fastwam.py" "$@"
