#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt}"
episodes="${QDR330_EPISODES:-2048}"
agents="${QDR330_TOTAL_AGENTS:-2048}"
threads="${QDR330_THREADS:-16}"
buffers="${QDR330_BUFFERS:-2}"
out_dir="${QDR330_OUT_DIR:-logs/qdr330}"
cap_pairs="${QDR330_CAP_PAIRS:-020050,020060}"
build_args="${QDR330_BUILD_ARGS:-}"

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
)

cap_stop() {
  case "$1" in
    020050|020060) printf '0.20' ;;
    022060) printf '0.22' ;;
    *) echo "unknown cap pair: $1" >&2; exit 1 ;;
  esac
}

cap_slow() {
  case "$1" in
    020050) printf '0.50' ;;
    020060|022060) printf '0.60' ;;
    *) echo "unknown cap pair: $1" >&2; exit 1 ;;
  esac
}

run_case() {
  local mode_label="$1"
  local mode="$2"
  local cap="$3"
  local stop slow log_prefix
  stop="$(cap_stop "$cap")"
  slow="$(cap_slow "$cap")"
  log_prefix="$out_dir/${mode_label}_l4_cap${cap}"

  echo "qdr330_l4_safety label=${mode_label}_l4_cap${cap} mode=$mode stop=$stop slow=$slow episodes=$episodes"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    --mode "$mode" \
    --safety-min-ray-speed-stop "$stop" \
    --safety-min-ray-speed-slow "$slow" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    --mode "$mode" \
    --ignore-checkpoint-env-config \
    --safety-min-ray-speed-stop "$stop" \
    --safety-min-ray-speed-slow "$slow" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

IFS=',' read -r -a cap_items <<< "$cap_pairs"
for cap in "${cap_items[@]}"; do
  [[ -z "$cap" ]] && continue
  run_case p3 procedural_house_multizone_p3 "$cap"
  run_case aggregate procedural_house_multizone "$cap"
done
