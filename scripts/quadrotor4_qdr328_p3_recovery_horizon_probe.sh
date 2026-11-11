#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt}"
hard_states="${QDR328_HARD_STATES:-64}"
agents="${QDR328_TOTAL_AGENTS:-1024}"
threads="${QDR328_THREADS:-16}"
buffers="${QDR328_BUFFERS:-2}"
out_dir="${QDR328_OUT_DIR:-logs/qdr328}"
horizons="${QDR328_EXTRA_HORIZONS:-512,1024,2048}"
build_args="${QDR328_BUILD_ARGS:-}"

mkdir -p "$out_dir"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=".venv/bin:../../.venv/bin:$CUDA_HOME/bin:$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" bash build.sh quadrotor $build_args

echo "qdr328_p3_recovery_horizon_probe hard_states=$hard_states agents=$agents horizons=$horizons"
scripts/quadrotor4.sh explore-v0-recovery-replay "$checkpoint" \
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
  --extra-replay-horizons "$horizons" \
  --pg-updates 0 \
  2>&1 | tee "$out_dir/p3_recovery_horizon_probe.log"
