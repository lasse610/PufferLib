#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt}"
episodes="${QDR329_EPISODES:-2048}"
agents="${QDR329_TOTAL_AGENTS:-2048}"
threads="${QDR329_THREADS:-16}"
buffers="${QDR329_BUFFERS:-2}"
out_dir="${QDR329_OUT_DIR:-logs/qdr329}"
lookaheads="${QDR329_LOOKAHEADS:-2,4,6}"
aggregate_lookaheads="${QDR329_AGG_LOOKAHEADS:-4,6}"
build_args="${QDR329_BUILD_ARGS:-}"

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

run_case() {
  local label="$1"
  local mode="$2"
  local lookahead="$3"
  local log_prefix="$out_dir/$label"

  echo "qdr329_path_lookahead label=$label mode=$mode lookahead=$lookahead episodes=$episodes"
  scripts/quadrotor4.sh explore-v0-target-oracle-eval \
    "${common_args[@]}" \
    --mode "$mode" \
    --explore-onboard-path-lookahead "$lookahead" \
    2>&1 | tee "${log_prefix}_oracle.log"

  scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${common_args[@]}" \
    --mode "$mode" \
    --ignore-checkpoint-env-config \
    --explore-onboard-path-lookahead "$lookahead" \
    2>&1 | tee "${log_prefix}_qdr316.log"
}

IFS=',' read -r -a lookahead_items <<< "$lookaheads"
for lookahead in "${lookahead_items[@]}"; do
  [[ -z "$lookahead" ]] && continue
  run_case "p3_l${lookahead}_speed070_cap018045" procedural_house_multizone_p3 "$lookahead"
done

IFS=',' read -r -a aggregate_items <<< "$aggregate_lookaheads"
for lookahead in "${aggregate_items[@]}"; do
  [[ -z "$lookahead" ]] && continue
  run_case "aggregate_l${lookahead}_speed070_cap018045" procedural_house_multizone "$lookahead"
done
