#!/usr/bin/env bash
set -euo pipefail

hard_states="${QDR326_HARD_STATES:-16}"
agents="${QDR326_TOTAL_AGENTS:-512}"
threads="${QDR326_THREADS:-16}"
buffers="${QDR326_BUFFERS:-2}"
out_dir="${QDR326_OUT_DIR:-logs/qdr326}"

mkdir -p "$out_dir"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=".venv/bin:$CUDA_HOME/bin:$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" bash build.sh quadrotor

echo "qdr326_p3_mpc_replay hard_states=$hard_states agents=$agents"
scripts/quadrotor4.sh explore-v0-onboard-hardstate-probe \
  --mode procedural_house_multizone_p3 \
  --seed 88 \
  --hard-states "$hard_states" \
  --collect-steps 20000 \
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
  --stall-steps 90 \
  --stall-clearance 0.40 \
  --replay-horizon 192 \
  --mpc-rollout-steps 8 \
  --mpc-commit-steps 1 \
  --mpc-speed-fractions 1.0,0.65,0.35 \
  2>&1 | tee "$out_dir/p3_mpc_replay.log"
