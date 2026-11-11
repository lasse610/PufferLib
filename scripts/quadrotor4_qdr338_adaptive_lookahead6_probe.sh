#!/usr/bin/env bash
set -euo pipefail

checkpoint="experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  checkpoint="$1"
  shift
fi

episodes="${QDR338_EPISODES:-2048}"
agents="${QDR338_TOTAL_AGENTS:-2048}"
threads="${QDR338_THREADS:-16}"
buffers="${QDR338_BUFFERS:-2}"
out_dir="${QDR338_OUT_DIR:-logs/qdr338}"
profiles="${QDR338_PROFILES:-strict,very_strict,ultra}"
modes="${QDR338_MODES:-p3}"
speed="${QDR338_SPEED:-0.72}"
build_args="${QDR338_BUILD_ARGS:-}"

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
  --explore-onboard-path-lookahead 6
  --explore-onboard-adaptive-lookahead
  --explore-onboard-adaptive-lookahead-min 2
  --explore-onboard-adaptive-lookahead-max 6
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

profile_args() {
  case "$1" in
    strict)
      printf '%s\n' \
        "--explore-onboard-adaptive-clearance 0.70" \
        "--explore-onboard-adaptive-turn-dot 0.70"
      ;;
    very_strict)
      printf '%s\n' \
        "--explore-onboard-adaptive-clearance 0.85" \
        "--explore-onboard-adaptive-turn-dot 0.80"
      ;;
    ultra)
      printf '%s\n' \
        "--explore-onboard-adaptive-clearance 1.00" \
        "--explore-onboard-adaptive-turn-dot 0.85"
      ;;
    *) echo "unknown profile: $1" >&2; exit 1 ;;
  esac
}

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
  local profile="$1"
  local mode_label="$2"
  shift 2
  local mode suffix log_prefix
  mode="$(mode_name "$mode_label")"
  suffix="$(speed_label "$speed")"
  log_prefix="$out_dir/${mode_label}_speed${suffix}_${profile}_adaptive_l2l6"
  read -r -a extra <<< "$(profile_args "$profile" | tr '\n' ' ')"

  echo "qdr338_adaptive_lookahead6 label=${mode_label}_${profile} mode=$mode speed=$speed episodes=$episodes"
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
