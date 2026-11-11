#!/usr/bin/env bash
set -euo pipefail

checkpoint="experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  checkpoint="$1"
  shift
fi

episodes="${QDR344_EPISODES:-2048}"
agents="${QDR344_TOTAL_AGENTS:-2048}"
threads="${QDR344_THREADS:-16}"
buffers="${QDR344_BUFFERS:-2}"
out_dir="${QDR344_OUT_DIR:-logs/qdr344}"
speed="${QDR344_SPEED:-0.72}"
horizons="${QDR344_HORIZONS:-1200,1600}"
build_args="${QDR344_BUILD_ARGS:-}"
gain_weight="${QDR344_GRAPH_GAIN_WEIGHT:-0.32}"
cluster_weight="${QDR344_GRAPH_CLUSTER_WEIGHT:-0.12}"
depth_weight="${QDR344_GRAPH_DEPTH_WEIGHT:-0.045}"
active_bonus="${QDR344_GRAPH_ACTIVE_BONUS:-0.75}"
stale_penalty="${QDR344_GRAPH_STALE_PENALTY:-0.020}"
failure_penalty="${QDR344_GRAPH_FAILURE_PENALTY:-0.70}"
attempt_penalty="${QDR344_GRAPH_ATTEMPT_PENALTY:-0.12}"
gain_cap="${QDR344_GRAPH_GAIN_CAP:-192}"

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
  --explore-onboard-graph-gain-weight "$gain_weight"
  --explore-onboard-graph-cluster-weight "$cluster_weight"
  --explore-onboard-graph-depth-weight "$depth_weight"
  --explore-onboard-graph-active-bonus "$active_bonus"
  --explore-onboard-graph-stale-penalty "$stale_penalty"
  --explore-onboard-graph-failure-penalty "$failure_penalty"
  --explore-onboard-graph-attempt-penalty "$attempt_penalty"
  --explore-onboard-graph-gain-cap "$gain_cap"
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

run_case() {
  local horizon="$1"
  shift
  local log_prefix
  log_prefix="$out_dir/p3_h${horizon}_speed$(label_float "$speed")_frontier_graph"

  echo "qdr344_frontier_graph horizon=$horizon speed=$speed episodes=$episodes checkpoint=$checkpoint"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    --max-steps "$horizon" \
    "$@" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    --max-steps "$horizon" \
    --ignore-checkpoint-env-config \
    "$@" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

IFS=',' read -r -a horizon_items <<< "$horizons"
for horizon in "${horizon_items[@]}"; do
  [[ -z "$horizon" ]] && continue
  run_case "$horizon" "$@"
done
