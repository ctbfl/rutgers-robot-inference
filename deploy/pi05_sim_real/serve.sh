#!/usr/bin/env bash
# Layout: ~/rutgers-robot-inference and ~/ruri-deps; all paths are overridable.
set -euo pipefail
RURI_SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RURI_DEPLOY_ROOT="${RURI_DEPLOY_ROOT:-$(dirname -- "$RURI_SOURCE_ROOT")/ruri-deps}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$RURI_DEPLOY_ROOT/openpi-cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$RURI_DEPLOY_ROOT/venv/bin/ruri-serve-pi05-sim-real" \
  --checkpoint "${RURI_CHECKPOINT:-$RURI_DEPLOY_ROOT/checkpoints/10000}" \
  --domain "${RURI_DOMAIN:-real}" "$@"
