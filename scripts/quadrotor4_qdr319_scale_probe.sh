#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt}"
episodes="${QDR319_EPISODES:-2048}"
agents="${QDR319_TOTAL_AGENTS:-2048}"
threads="${QDR319_THREADS:-16}"
buffers="${QDR319_BUFFERS:-2}"
out_dir="${QDR319_OUT_DIR:-logs/qdr319}"

mkdir -p "$out_dir"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=".venv/bin:$CUDA_HOME/bin:$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" bash build.sh quadrotor

common_args=(
  --mode procedural_house_multizone
  --seed 88
  --episodes "$episodes"
  --total-agents "$agents"
  --num-buffers "$buffers"
  --num-threads "$threads"
  --max-eval-steps 300000
  --speed-fraction 0.50
  --spawn-radius-min 0.0
  --explore-goal-fraction 0.99
  --explore-frontier-target-mode onboard
  --explore-reachable-coverage
  --explore-onboard-path-lookahead 2
  --explore-onboard-region-commit
  --explore-onboard-region-commit-stall-ticks 90
  --explore-onboard-region-tracker
  --explore-onboard-passage-quality
  --explore-onboard-passage-weight 0.35
  --deploy-patch-obs
  --deploy-onboard-target-obs
  --safety-barrier-actions
  --safety-slide
  --safety-cone-ray-stop 0.30
  --safety-cone-deg 60
  --safety-slow-ray 0.80
)

run_case() {
  local label="$1"
  local room_size="$2"
  local cell_size="$3"
  local max_steps="$4"
  local log_prefix="$out_dir/$label"

  echo "qdr319_case label=$label room_size=$room_size cell_size=$cell_size max_steps=$max_steps episodes=$episodes"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    --room-size "$room_size" \
    --cell-size "$cell_size" \
    --max-steps "$max_steps" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    --ignore-checkpoint-env-config \
    --room-size "$room_size" \
    --cell-size "$cell_size" \
    --max-steps "$max_steps" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

run_case room10_cell035_steps4096 10.0 0.35 4096
run_case room10_cell035_steps3072 10.0 0.35 3072
run_case room12_cell040_steps4096 12.0 0.40 4096
