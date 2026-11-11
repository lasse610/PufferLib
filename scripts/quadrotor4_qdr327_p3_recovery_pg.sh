#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt}"
hard_states="${QDR327_HARD_STATES:-64}"
agents="${QDR327_TOTAL_AGENTS:-1024}"
threads="${QDR327_THREADS:-16}"
buffers="${QDR327_BUFFERS:-2}"
updates="${QDR327_PG_UPDATES:-120}"
out_dir="${QDR327_OUT_DIR:-logs/qdr327}"
output="${QDR327_OUTPUT:-experiments/qdr327/qdr327_p3_recovery_pg_s88.pt}"
build_args="${QDR327_BUILD_ARGS:-}"

mkdir -p "$out_dir"
mkdir -p "$(dirname "$output")"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=".venv/bin:../../.venv/bin:$CUDA_HOME/bin:$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" bash build.sh quadrotor $build_args

echo "qdr327_p3_recovery_pg hard_states=$hard_states agents=$agents updates=$updates output=$output"
scripts/quadrotor4.sh explore-v0-recovery-replay "$checkpoint" \
  --output "$output" \
  --mode procedural_house_multizone_p3 \
  --seed 88 \
  --hard-states "$hard_states" \
  --collect-steps 24000 \
  --capture-cooldown 48 \
  --total-agents "$agents" \
  --num-buffers "$buffers" \
  --num-threads "$threads" \
  --room-size 12.0 \
  --cell-size 0.40 \
  --max-steps 1200 \
  --max-eval-steps 300000 \
  --spawn-radius-min 0.0 \
  --speed-fraction 0.70 \
  --explore-goal-fraction 0.99 \
  --explore-frontier-target-mode onboard \
  --explore-reachable-coverage \
  --explore-onboard-path-lookahead 2 \
  --explore-onboard-region-commit \
  --explore-onboard-region-commit-stall-ticks 90 \
  --explore-onboard-region-tracker \
  --explore-onboard-passage-quality \
  --explore-onboard-passage-weight 0.35 \
  --deploy-patch-obs \
  --deploy-onboard-target-obs \
  --safety-barrier-actions \
  --safety-slide \
  --safety-cone-ray-stop 0.30 \
  --safety-cone-deg 60 \
  --safety-slow-ray 0.80 \
  --safety-min-ray-speed-cap \
  --safety-min-ray-speed-stop 0.18 \
  --safety-min-ray-speed-slow 0.45 \
  --ignore-checkpoint-env-config \
  --recovery-stall-steps 90 \
  --recovery-duration-steps 96 \
  --recovery-near-wall 0.40 \
  --recovery-progress-delta 0.002 \
  --controllers base,stateful \
  --replay-agents 64 \
  --replay-horizon 256 \
  --pg-updates "$updates" \
  --pg-batch-size 32 \
  --train-horizon 96 \
  --pg-action-std 0.15 \
  --pg-learning-rate 0.0003 \
  --pg-gamma 0.98 \
  --pg-coverage-weight 300.0 \
  --pg-clearance-weight 0.40 \
  --pg-danger-weight 4.0 \
  --pg-step-penalty 0.003 \
  --pg-success-bonus 2.0 \
  --pg-crash-penalty 4.0 \
  --pg-timeout-penalty 0.5 \
  --bc-warmup-epochs 3 \
  --bc-warmup-target base \
  --print-interval 20 \
  2>&1 | tee "$out_dir/p3_recovery_pg.log"
