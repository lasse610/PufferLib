#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt}"
episodes="${QDR331_EPISODES:-2048}"
agents="${QDR331_TOTAL_AGENTS:-2048}"
threads="${QDR331_THREADS:-16}"
buffers="${QDR331_BUFFERS:-2}"
out_dir="${QDR331_OUT_DIR:-logs/qdr331}"
profiles="${QDR331_PROFILES:-balanced,strict}"
build_args="${QDR331_BUILD_ARGS:-}"

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
  --speed-fraction 0.70
  --explore-goal-fraction 0.99
  --explore-frontier-target-mode onboard
  --explore-reachable-coverage
  --explore-onboard-path-lookahead 4
  --explore-onboard-adaptive-lookahead
  --explore-onboard-adaptive-lookahead-min 2
  --explore-onboard-adaptive-lookahead-max 4
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

profile_clearance() {
  case "$1" in
    permissive) printf '0.45' ;;
    balanced) printf '0.55' ;;
    strict) printf '0.65' ;;
    *) echo "unknown profile: $1" >&2; exit 1 ;;
  esac
}

profile_turn_dot() {
  case "$1" in
    permissive) printf '0.35' ;;
    balanced) printf '0.55' ;;
    strict) printf '0.65' ;;
    *) echo "unknown profile: $1" >&2; exit 1 ;;
  esac
}

run_case() {
  local profile="$1"
  local mode_label="$2"
  local mode="$3"
  local clearance turn_dot log_prefix
  clearance="$(profile_clearance "$profile")"
  turn_dot="$(profile_turn_dot "$profile")"
  log_prefix="$out_dir/${mode_label}_${profile}_adaptive_l2l4"

  echo "qdr331_adaptive_lookahead label=${mode_label}_${profile} mode=$mode clearance=$clearance turn_dot=$turn_dot episodes=$episodes"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    --mode "$mode" \
    --explore-onboard-adaptive-clearance "$clearance" \
    --explore-onboard-adaptive-turn-dot "$turn_dot" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    --mode "$mode" \
    --ignore-checkpoint-env-config \
    --explore-onboard-adaptive-clearance "$clearance" \
    --explore-onboard-adaptive-turn-dot "$turn_dot" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

IFS=',' read -r -a profile_items <<< "$profiles"
for profile in "${profile_items[@]}"; do
  [[ -z "$profile" ]] && continue
  run_case "$profile" p3 procedural_house_multizone_p3
  run_case "$profile" aggregate procedural_house_multizone
done
