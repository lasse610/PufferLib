#!/usr/bin/env bash
set -euo pipefail

checkpoint="experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  checkpoint="$1"
  shift
fi

episodes="${QDR339_EPISODES:-2048}"
agents="${QDR339_TOTAL_AGENTS:-2048}"
threads="${QDR339_THREADS:-16}"
buffers="${QDR339_BUFFERS:-2}"
out_dir="${QDR339_OUT_DIR:-logs/qdr339}"
speed="${QDR339_SPEED:-0.72}"
horizon="${QDR339_HORIZON:-1200}"
modes="${QDR339_MODES:-p3}"
profiles="${QDR339_PROFILES:-local,balanced,regional}"
build_args="${QDR339_BUILD_ARGS:-}"

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
  --max-steps "$horizon"
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
  --explore-onboard-budget-frontier
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

set_profile_args() {
  case "$1" in
    local)
      profile_extra=(
        --explore-onboard-budget-gain-weight 1.00
        --explore-onboard-budget-cluster-weight 0.20
        --explore-onboard-budget-cell-weight 0.45
        --explore-onboard-budget-depth-bias 3.0
        --explore-onboard-budget-depth-weight 1.25
      )
      ;;
    balanced)
      profile_extra=(
        --explore-onboard-budget-gain-weight 1.00
        --explore-onboard-budget-cluster-weight 0.35
        --explore-onboard-budget-cell-weight 0.25
        --explore-onboard-budget-depth-bias 6.0
        --explore-onboard-budget-depth-weight 1.00
      )
      ;;
    regional)
      profile_extra=(
        --explore-onboard-budget-gain-weight 1.15
        --explore-onboard-budget-cluster-weight 0.65
        --explore-onboard-budget-cell-weight 0.10
        --explore-onboard-budget-depth-bias 10.0
        --explore-onboard-budget-depth-weight 0.80
      )
      ;;
    *)
      echo "unknown profile: $1" >&2
      exit 1
      ;;
  esac
}

label_float() {
  printf '%s' "$1" | tr -d '.'
}

run_case() {
  local profile="$1"
  local mode_label="$2"
  shift 2
  local mode log_prefix profile_extra
  mode="$(mode_name "$mode_label")"
  set_profile_args "$profile"
  log_prefix="$out_dir/${mode_label}_h${horizon}_speed$(label_float "$speed")_${profile}_budget_frontier"

  echo "qdr339_budget_frontier label=${mode_label}_${profile} mode=$mode speed=$speed horizon=$horizon episodes=$episodes"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    "${profile_extra[@]}" \
    --mode "$mode" \
    "$@" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    "${profile_extra[@]}" \
    --mode "$mode" \
    --ignore-checkpoint-env-config \
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
