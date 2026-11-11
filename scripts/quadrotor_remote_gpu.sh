#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
helper="$script_dir/../../../scripts/quadrotor_remote_gpu.sh"

if [[ ! -x "$helper" ]]; then
  echo "Remote GPU helper not found at: $helper" >&2
  echo "Run from the main checkout instead:" >&2
  echo "  /Users/lasse/Documents/programming/machine-learning/reinforcement-learning/misc/PufferLib/scripts/quadrotor_remote_gpu.sh" >&2
  exit 1
fi

exec "$helper" "$@"
