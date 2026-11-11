#!/usr/bin/env bash
set -euo pipefail

checkpoint="experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  checkpoint="$1"
  shift
fi

episodes="${QDR336_EPISODES:-2048}"
agents="${QDR336_TOTAL_AGENTS:-2048}"
threads="${QDR336_THREADS:-16}"
buffers="${QDR336_BUFFERS:-2}"
out_dir="${QDR336_OUT_DIR:-logs/qdr336}"
profiles="${QDR336_PROFILES:-light,balanced,strong}"
modes="${QDR336_MODES:-p3}"
speed="${QDR336_SPEED:-0.75}"
build_args="${QDR336_BUILD_ARGS:-}"

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
  --safety-corridor-width-actions
)

mode_name() {
  case "$1" in
    p3) printf 'procedural_house_multizone_p3' ;;
    aggregate) printf 'procedural_house_multizone' ;;
    *) echo "unknown mode alias: $1" >&2; exit 1 ;;
  esac
}

profile_args() {
  case "$1" in
    light)
      printf '%s\n' \
        "--safety-corridor-width-near 0.50" \
        "--safety-corridor-width-front 0.75" \
        "--safety-corridor-width-side-stop 0.22" \
        "--safety-corridor-width-side-slow 0.55" \
        "--safety-corridor-width-min-scale 0.45" \
        "--safety-corridor-width-max-turn-deg 40" \
        "--safety-corridor-width-progress-floor 0.18"
      ;;
    balanced)
      printf '%s\n' \
        "--safety-corridor-width-near 0.55" \
        "--safety-corridor-width-front 0.70" \
        "--safety-corridor-width-side-stop 0.24" \
        "--safety-corridor-width-side-slow 0.65" \
        "--safety-corridor-width-min-scale 0.35" \
        "--safety-corridor-width-max-turn-deg 55" \
        "--safety-corridor-width-progress-floor 0.10"
      ;;
    strong)
      printf '%s\n' \
        "--safety-corridor-width-near 0.60" \
        "--safety-corridor-width-front 0.65" \
        "--safety-corridor-width-side-stop 0.28" \
        "--safety-corridor-width-side-slow 0.75" \
        "--safety-corridor-width-min-scale 0.30" \
        "--safety-corridor-width-max-turn-deg 65" \
        "--safety-corridor-width-progress-floor 0.05"
      ;;
    *) echo "unknown profile: $1" >&2; exit 1 ;;
  esac
}

speed_label() {
  printf '%s' "$1" | tr -d '.'
}

run_case() {
  local profile="$1"
  local mode_label="$2"
  shift 2
  local mode suffix log_prefix
  mode="$(mode_name "$mode_label")"
  suffix="$(speed_label "$speed")"
  log_prefix="$out_dir/${mode_label}_speed${suffix}_${profile}_corridor_width"
  read -r -a extra <<< "$(profile_args "$profile" | tr '\n' ' ')"

  echo "qdr336_corridor_width_selector label=${mode_label}_${profile} mode=$mode speed=$speed episodes=$episodes"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    --mode "$mode" \
    "${extra[@]}" \
    "$@" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    --mode "$mode" \
    --ignore-checkpoint-env-config \
    "${extra[@]}" \
    "$@" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

IFS=',' read -r -a profile_items <<< "$profiles"
IFS=',' read -r -a mode_items <<< "$modes"
for profile in "${profile_items[@]}"; do
  [[ -z "$profile" ]] && continue
  for mode_label in "${mode_items[@]}"; do
    [[ -z "$mode_label" ]] && continue
    run_case "$profile" "$mode_label" "$@"
  done
done
