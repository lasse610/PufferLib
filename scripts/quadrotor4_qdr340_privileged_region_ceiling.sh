#!/usr/bin/env bash
set -euo pipefail

episodes="${QDR340_EPISODES:-2048}"
agents="${QDR340_TOTAL_AGENTS:-2048}"
threads="${QDR340_THREADS:-16}"
buffers="${QDR340_BUFFERS:-2}"
out_dir="${QDR340_OUT_DIR:-logs/qdr340}"
speed="${QDR340_SPEED:-0.72}"
horizons="${QDR340_HORIZONS:-1200,1600}"
frontiers="${QDR340_FRONTIERS:-onboard,region,unexplored}"
build_args="${QDR340_BUILD_ARGS:-}"

mkdir -p "$out_dir"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=".venv/bin:../../.venv/bin:$CUDA_HOME/bin:$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" bash build.sh quadrotor $build_args

common_args=(
  --seed 88
  --episodes "$episodes"
  --total-agents "$agents"
  --num-buffers "$buffers"
  --num-threads "$threads"
  --max-eval-steps 300000
  --mode procedural_house_multizone_p3
  --room-size 12.0
  --cell-size 0.40
  --spawn-radius-min 0.0
  --speed-fraction "$speed"
  --explore-goal-fraction 0.99
  --explore-reachable-coverage
  --safety-barrier-actions
  --safety-slide
  --safety-cone-ray-stop 0.30
  --safety-cone-deg 60
  --safety-slow-ray 0.80
  --safety-min-ray-speed-cap
  --safety-min-ray-speed-stop 0.18
  --safety-min-ray-speed-slow 0.45
)

onboard_args=(
  --explore-onboard-path-lookahead 4
  --explore-onboard-adaptive-lookahead
  --explore-onboard-adaptive-lookahead-min 2
  --explore-onboard-adaptive-lookahead-max 4
  --explore-onboard-adaptive-clearance 0.55
  --explore-onboard-adaptive-turn-dot 0.55
  --explore-onboard-region-commit
  --explore-onboard-region-commit-stall-ticks 90
  --explore-onboard-region-tracker
  --explore-onboard-passage-quality
  --explore-onboard-passage-weight 0.35
  --deploy-patch-obs
  --deploy-onboard-target-obs
)

label_float() {
  printf '%s' "$1" | tr -d '.'
}

run_case() {
  local frontier="$1"
  local horizon="$2"
  shift 2
  local log_prefix
  log_prefix="$out_dir/p3_h${horizon}_speed$(label_float "$speed")_${frontier}_oracle"

  echo "qdr340_privileged_region_ceiling frontier=$frontier horizon=$horizon speed=$speed episodes=$episodes"
  if [[ "$frontier" == "onboard" ]]; then
    scripts/quadrotor4.sh explore-v0-target-oracle-eval \
      "${common_args[@]}" \
      "${onboard_args[@]}" \
      --explore-frontier-target-mode onboard \
      --max-steps "$horizon" \
      "$@" \
      2>&1 | tee "${log_prefix}.log"
  else
    scripts/quadrotor4.sh explore-v0-target-oracle-eval \
      "${common_args[@]}" \
      --explore-frontier-target-mode "$frontier" \
      --max-steps "$horizon" \
      "$@" \
      2>&1 | tee "${log_prefix}.log"
  fi
}

IFS=',' read -r -a frontier_items <<< "$frontiers"
IFS=',' read -r -a horizon_items <<< "$horizons"
for horizon in "${horizon_items[@]}"; do
  [[ -z "$horizon" ]] && continue
  for frontier in "${frontier_items[@]}"; do
    [[ -z "$frontier" ]] && continue
    run_case "$frontier" "$horizon" "$@"
  done
done
