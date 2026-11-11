#!/usr/bin/env bash
set -euo pipefail

checkpoint="experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  checkpoint="$1"
  shift
fi

episodes="${QDR345_EPISODES:-2048}"
agents="${QDR345_TOTAL_AGENTS:-2048}"
threads="${QDR345_THREADS:-16}"
buffers="${QDR345_BUFFERS:-2}"
out_dir="${QDR345_OUT_DIR:-logs/qdr345}"
speed="${QDR345_SPEED:-0.72}"
horizons="${QDR345_HORIZONS:-1200}"
cases="${QDR345_CASES:-default,no_gain,no_persistence,score_only}"
build_args="${QDR345_BUILD_ARGS:-}"

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

graph_args_for_case() {
  local variant="$1"
  case "$variant" in
    default)
      graph_args=(
        --explore-onboard-graph-gain-weight 0.32
        --explore-onboard-graph-cluster-weight 0.12
        --explore-onboard-graph-depth-weight 0.045
        --explore-onboard-graph-active-bonus 0.75
        --explore-onboard-graph-stale-penalty 0.020
        --explore-onboard-graph-failure-penalty 0.70
        --explore-onboard-graph-attempt-penalty 0.12
        --explore-onboard-graph-gain-cap 192
      )
      ;;
    no_gain)
      graph_args=(
        --explore-onboard-graph-gain-weight 0.0
        --explore-onboard-graph-cluster-weight 0.12
        --explore-onboard-graph-depth-weight 0.045
        --explore-onboard-graph-active-bonus 0.75
        --explore-onboard-graph-stale-penalty 0.020
        --explore-onboard-graph-failure-penalty 0.70
        --explore-onboard-graph-attempt-penalty 0.12
        --explore-onboard-graph-gain-cap 192
      )
      ;;
    no_persistence)
      graph_args=(
        --explore-onboard-graph-gain-weight 0.32
        --explore-onboard-graph-cluster-weight 0.12
        --explore-onboard-graph-depth-weight 0.045
        --explore-onboard-graph-active-bonus 0.0
        --explore-onboard-graph-stale-penalty 0.0
        --explore-onboard-graph-failure-penalty 0.0
        --explore-onboard-graph-attempt-penalty 0.0
        --explore-onboard-graph-gain-cap 192
      )
      ;;
    score_only)
      graph_args=(
        --explore-onboard-graph-gain-weight 0.0
        --explore-onboard-graph-cluster-weight 0.0
        --explore-onboard-graph-depth-weight 0.045
        --explore-onboard-graph-active-bonus 0.0
        --explore-onboard-graph-stale-penalty 0.0
        --explore-onboard-graph-failure-penalty 0.0
        --explore-onboard-graph-attempt-penalty 0.0
        --explore-onboard-graph-gain-cap 192
      )
      ;;
    *)
      echo "unknown QDR345 case: $variant" >&2
      exit 2
      ;;
  esac
}

run_case() {
  local variant="$1"
  local horizon="$2"
  shift 2
  local log_prefix
  local graph_args=()
  graph_args_for_case "$variant"
  log_prefix="$out_dir/p3_h${horizon}_speed$(label_float "$speed")_${variant}"

  echo "qdr345_frontier_graph_ablation case=$variant horizon=$horizon speed=$speed episodes=$episodes checkpoint=$checkpoint"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    "${graph_args[@]}" \
    --max-steps "$horizon" \
    "$@" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    "${graph_args[@]}" \
    --max-steps "$horizon" \
    --ignore-checkpoint-env-config \
    "$@" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

IFS=',' read -r -a case_items <<< "$cases"
IFS=',' read -r -a horizon_items <<< "$horizons"
for horizon in "${horizon_items[@]}"; do
  [[ -z "$horizon" ]] && continue
  for variant in "${case_items[@]}"; do
    [[ -z "$variant" ]] && continue
    run_case "$variant" "$horizon" "$@"
  done
done
