#!/usr/bin/env bash
# Usage: GPU=0 PORT=5561 bash examples/shell_scripts/serve_pi05_effort.sh plain STEP_DIR
#        GPU=0 PORT=5562 bash examples/shell_scripts/serve_pi05_effort.sh rtc STEP_DIR
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {plain|rtc} CHECKPOINT_STEP_DIR [server flags...]" >&2
  exit 2
fi
variant="$1"
checkpoint="$2"
shift 2
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.2}"
export PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/common/users/jh2400/cache/openpi}"
export HF_HOME="${HF_HOME:-/common/users/jh2400/cache/huggingface}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${VENV:-/common/home/jh2400/projects/openpi/.venv}/bin/python" \
  "$repo_dir/examples/server_pi05_effort.py" \
  --variant "$variant" --checkpoint "$checkpoint" \
  --bind "${BIND:-tcp://*:${PORT:-5561}}" "$@"
