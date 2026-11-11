#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt}"
episodes="${QDR324_EPISODES:-4096}"
agents="${QDR324_TOTAL_AGENTS:-2048}"
threads="${QDR324_THREADS:-16}"
buffers="${QDR324_BUFFERS:-2}"
out_dir="${QDR324_OUT_DIR:-logs/qdr324}"

mkdir -p "$out_dir"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=".venv/bin:$CUDA_HOME/bin:$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" bash build.sh quadrotor

base_args=(
  --seed 88
  --episodes "$episodes"
  --total-agents "$agents"
  --num-buffers "$buffers"
  --num-threads "$threads"
  --max-eval-steps 300000
  --room-size 12.0
  --cell-size 0.40
  --max-steps 1200
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
  local mode="$2"
  local speed_fraction="$3"
  shift 3
  local log_prefix="$out_dir/$label"

  echo "qdr324_min_ray_cap label=$label mode=$mode speed_fraction=$speed_fraction episodes=$episodes extra=$*"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${base_args[@]}" \
    --mode "$mode" \
    --speed-fraction "$speed_fraction" \
    "$@" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${base_args[@]}" \
    --mode "$mode" \
    --ignore-checkpoint-env-config \
    --speed-fraction "$speed_fraction" \
    "$@" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

run_case p3_speed070_cap018045 procedural_house_multizone_p3 0.70 \
  --safety-min-ray-speed-cap \
  --safety-min-ray-speed-stop 0.18 \
  --safety-min-ray-speed-slow 0.45

run_case p3_speed090_cap018045 procedural_house_multizone_p3 0.90 \
  --safety-min-ray-speed-cap \
  --safety-min-ray-speed-stop 0.18 \
  --safety-min-ray-speed-slow 0.45

run_case p3_speed090_cap020060 procedural_house_multizone_p3 0.90 \
  --safety-min-ray-speed-cap \
  --safety-min-ray-speed-stop 0.20 \
  --safety-min-ray-speed-slow 0.60

run_case aggregate_speed070_cap018045 procedural_house_multizone 0.70 \
  --safety-min-ray-speed-cap \
  --safety-min-ray-speed-stop 0.18 \
  --safety-min-ray-speed-slow 0.45
