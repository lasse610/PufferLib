#!/usr/bin/env bash
set -euo pipefail

checkpoint="experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  checkpoint="$1"
  shift
fi
mode="${QDR331_VISUAL_MODE:-procedural_house_multizone}"
build="${QDR331_VISUAL_BUILD:-0}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=".venv/bin:../../.venv/bin:$CUDA_HOME/bin:$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

if [[ "$build" == "1" ]]; then
  PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" bash build.sh quadrotor --cpu
fi

scripts/quadrotor4.sh explore-v0-target-visual "$checkpoint" \
  --ignore-checkpoint-env-config \
  --mode "$mode" \
  --room-size "${QDR331_VISUAL_ROOM_SIZE:-12.0}" \
  --cell-size "${QDR331_VISUAL_CELL_SIZE:-0.40}" \
  --max-steps "${QDR331_VISUAL_MAX_STEPS:-1200}" \
  --max-eval-steps "${QDR331_VISUAL_MAX_EVAL_STEPS:-100000}" \
  --spawn-radius-min 0.0 \
  --speed-fraction "${QDR331_VISUAL_SPEED:-0.70}" \
  --explore-goal-fraction "${QDR331_VISUAL_GOAL:-0.99}" \
  --explore-frontier-target-mode onboard \
  --explore-reachable-coverage \
  --explore-onboard-path-lookahead 4 \
  --explore-onboard-adaptive-lookahead \
  --explore-onboard-adaptive-lookahead-min 2 \
  --explore-onboard-adaptive-lookahead-max 4 \
  --explore-onboard-adaptive-clearance 0.55 \
  --explore-onboard-adaptive-turn-dot 0.55 \
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
  --restart-on-stall-steps "${QDR331_VISUAL_RESTART_ON_STALL_STEPS:-300}" \
  --restart-on-stall-delta "${QDR331_VISUAL_RESTART_ON_STALL_DELTA:-0.002}" \
  --wall-stall-debug-steps "${QDR331_VISUAL_WALL_STALL_STEPS:-90}" \
  --wall-stall-clearance "${QDR331_VISUAL_WALL_STALL_CLEARANCE:-0.35}" \
  --render-every "${QDR331_VISUAL_RENDER_EVERY:-1}" \
  --episodes "${QDR331_VISUAL_EPISODES:-0}" \
  "$@"
