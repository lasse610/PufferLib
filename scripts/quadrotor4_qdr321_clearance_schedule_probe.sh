#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt}"
episodes="${QDR321_EPISODES:-2048}"
agents="${QDR321_TOTAL_AGENTS:-2048}"
threads="${QDR321_THREADS:-16}"
buffers="${QDR321_BUFFERS:-2}"
out_dir="${QDR321_OUT_DIR:-logs/qdr321}"

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
  --room-size 12.0
  --cell-size 0.40
  --max-steps 1200
  --speed-fraction 0.70
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
  --safety-cone-deg 60
)

run_profile() {
  local label="$1"
  local cone_stop="$2"
  local slow_ray="$3"
  local log_prefix="$out_dir/$label"

  echo "qdr321_profile label=$label cone_stop=$cone_stop slow_ray=$slow_ray episodes=$episodes"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    --safety-cone-ray-stop "$cone_stop" \
    --safety-slow-ray "$slow_ray" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    --ignore-checkpoint-env-config \
    --safety-cone-ray-stop "$cone_stop" \
    --safety-slow-ray "$slow_ray" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

run_profile slow120_cone030 0.30 1.20
run_profile slow120_cone035 0.35 1.20
