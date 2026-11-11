#!/usr/bin/env bash
set -euo pipefail

checkpoint="experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  checkpoint="$1"
  shift
fi

episodes="${QDR346_EPISODES:-2048}"
agents="${QDR346_TOTAL_AGENTS:-2048}"
threads="${QDR346_THREADS:-16}"
buffers="${QDR346_BUFFERS:-2}"
out_dir="${QDR346_OUT_DIR:-logs/qdr346}"
speed="${QDR346_SPEED:-0.72}"
cases="${QDR346_CASES:-aggregate_h1200,aggregate_h1600}"
build_args="${QDR346_BUILD_ARGS:-}"

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
  --max-eval-steps 500000
  --room-size 12.0
  --cell-size 0.40
  --spawn-radius-min 0.0
  --speed-fraction "$speed"
  --explore-goal-fraction 0.99
  --explore-frontier-target-mode onboard
  --explore-reachable-coverage
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
  --explore-onboard-frontier-graph
  --explore-onboard-graph-gain-weight 0.32
  --explore-onboard-graph-cluster-weight 0.12
  --explore-onboard-graph-depth-weight 0.045
  --explore-onboard-graph-active-bonus 0.0
  --explore-onboard-graph-stale-penalty 0.0
  --explore-onboard-graph-failure-penalty 0.0
  --explore-onboard-graph-attempt-penalty 0.0
  --explore-onboard-graph-gain-cap 192
  --deploy-patch-obs
  --deploy-onboard-target-obs
  --safety-barrier-actions
  --safety-slide
  --safety-cone-ray-stop 0.30
  --safety-cone-deg 60
  --safety-slow-ray 0.80
  --safety-min-ray-speed-cap
  --safety-min-ray-speed-stop 0.18
  --safety-min-ray-speed-slow 0.45
)

label_float() {
  printf '%s' "$1" | tr -d '.'
}

case_mode() {
  case "$1" in
    p3_h1200|p3_h1600) printf 'procedural_house_multizone_p3' ;;
    aggregate_h1200|aggregate_h1600) printf 'procedural_house_multizone' ;;
    *) echo "unknown QDR346 case: $1" >&2; exit 2 ;;
  esac
}

case_horizon() {
  case "$1" in
    p3_h1200|aggregate_h1200) printf '1200' ;;
    p3_h1600|aggregate_h1600) printf '1600' ;;
    *) echo "unknown QDR346 case: $1" >&2; exit 2 ;;
  esac
}

run_case() {
  local case_name="$1"
  shift
  local mode horizon log_prefix
  mode="$(case_mode "$case_name")"
  horizon="$(case_horizon "$case_name")"
  log_prefix="$out_dir/${case_name}_speed$(label_float "$speed")_graph_no_persistence"

  echo "qdr346_graph_baseline case=$case_name mode=$mode horizon=$horizon speed=$speed episodes=$episodes checkpoint=$checkpoint"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    --mode "$mode" \
    --max-steps "$horizon" \
    "$@" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    --mode "$mode" \
    --max-steps "$horizon" \
    --ignore-checkpoint-env-config \
    "$@" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

IFS=',' read -r -a case_items <<< "$cases"
for case_name in "${case_items[@]}"; do
  [[ -z "$case_name" ]] && continue
  run_case "$case_name" "$@"
done
