#!/usr/bin/env bash
set -euo pipefail

checkpoint="experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  checkpoint="$1"
  shift
fi

episodes="${QDR332_EPISODES:-2048}"
agents="${QDR332_TOTAL_AGENTS:-2048}"
threads="${QDR332_THREADS:-16}"
buffers="${QDR332_BUFFERS:-2}"
out_dir="${QDR332_OUT_DIR:-logs/qdr332}"
speeds="${QDR332_SPEEDS:-0.75,0.80,0.85,0.90}"
modes="${QDR332_MODES:-p3,aggregate}"
build_args="${QDR332_BUILD_ARGS:-}"

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
  --room-size 12.0
  --cell-size 0.40
  --max-steps 1200
  --spawn-radius-min 0.0
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

mode_name() {
  case "$1" in
    p3) printf 'procedural_house_multizone_p3' ;;
    aggregate) printf 'procedural_house_multizone' ;;
    *) echo "unknown mode alias: $1" >&2; exit 1 ;;
  esac
}

speed_label() {
  printf '%s' "$1" | tr -d '.'
}

run_case() {
  local speed="$1"
  local mode_label="$2"
  shift 2
  local mode log_prefix suffix
  mode="$(mode_name "$mode_label")"
  suffix="$(speed_label "$speed")"
  log_prefix="$out_dir/${mode_label}_speed${suffix}_adaptive_l2l4_balanced"

  echo "qdr332_adaptive_speed_headroom label=${mode_label}_speed${speed} mode=$mode speed=$speed episodes=$episodes"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    --mode "$mode" \
    --speed-fraction "$speed" \
    "$@" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    --mode "$mode" \
    --speed-fraction "$speed" \
    --ignore-checkpoint-env-config \
    "$@" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

IFS=',' read -r -a speed_items <<< "$speeds"
IFS=',' read -r -a mode_items <<< "$modes"
for speed in "${speed_items[@]}"; do
  [[ -z "$speed" ]] && continue
  for mode_label in "${mode_items[@]}"; do
    [[ -z "$mode_label" ]] && continue
    run_case "$speed" "$mode_label" "$@"
  done
done
