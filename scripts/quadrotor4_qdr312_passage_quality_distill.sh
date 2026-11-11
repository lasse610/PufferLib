#!/usr/bin/env bash
set -euo pipefail

seed="${QDR312_SEED:-88}"
updates="${QDR312_UPDATES:-800}"
agents="${QDR312_TOTAL_AGENTS:-1024}"
buffers="${QDR312_NUM_BUFFERS:-2}"
threads="${QDR312_NUM_THREADS:-16}"

base_checkpoint="${QDR312_INIT:-experiments/qdr300/checkpoints/qdr300_region_commit_barrier_ft_qdr285_s88_u000100.pt}"
output="${QDR312_OUTPUT:-experiments/qdr312/qdr312_passage_quality_distill_s${seed}.pt}"
checkpoint_dir="${QDR312_CHECKPOINT_DIR:-experiments/qdr312/checkpoints}"
python_bin="${PYTHON:-.venv/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$(dirname "$python_bin"):$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" bash build.sh quadrotor

PYTHON="$python_bin" scripts/quadrotor4.sh explore-v0-target-distill \
  --output "$output" \
  --init-checkpoint "$base_checkpoint" \
  --anchor-checkpoint "$base_checkpoint" \
  --anchor-modes \
    procedural_indoor procedural_indoor_p0 procedural_indoor_p2 \
    maze center_block doorway corridor random_blocks layout_mixed \
  --anchor-loss-weight 0.5 \
  --deploy-patch-obs \
  --deploy-onboard-target-obs \
  --explore-reachable-coverage \
  --explore-frontier-target-mode onboard \
  --explore-onboard-path-lookahead 2 \
  --explore-onboard-region-commit \
  --explore-onboard-region-commit-stall-ticks 90 \
  --explore-onboard-region-tracker \
  --explore-onboard-passage-quality \
  --explore-onboard-passage-weight 0.35 \
  --predict-frontier-sectors \
  --frontier-sector-loss-weight 2.0 \
  --predict-frontier-vector \
  --frontier-vector-loss-weight 1.0 \
  --frontier-vector-dist-loss-weight 2.0 \
  --unclamped-frontier-vector-dist-loss \
  --use-pred-vector-fallback \
  --pred-vector-fallback-primary-threshold 0.6 \
  --pred-vector-fallback-min-coverage 0.25 \
  --pred-vector-fallback-min-dist 0.02 \
  --train-modes \
    procedural_indoor_hard_p1 procedural_indoor_hard_p1 procedural_indoor_hard_p1 \
    procedural_indoor_hard_p2 procedural_indoor_hard_p3 procedural_indoor_hard_p0 \
    procedural_indoor_hard procedural_indoor_p1 procedural_indoor_p3 \
    procedural_indoor maze center_block doorway corridor random_blocks layout_mixed \
  --mode procedural_indoor_hard_p1 \
  --seed "$seed" \
  --seq-updates "$updates" \
  --teacher-rollout-updates 80 \
  --max-student-rollout-prob 0.5 \
  --learning-rate 0.00005 \
  --total-agents "$agents" \
  --num-buffers "$buffers" \
  --num-threads "$threads" \
  --seq-len 64 \
  --max-steps 2048 \
  --max-eval-steps 50000 \
  --room-size 6.0 \
  --cell-size 0.30 \
  --explore-goal-fraction 0.95 \
  --spawn-radius-min 0.0 \
  --speed-fraction 0.50 \
  --safety-barrier-actions \
  --safety-slide \
  --safety-cone-ray-stop 0.30 \
  --safety-cone-deg 60 \
  --safety-slow-ray 0.80 \
  --checkpoint-interval 100 \
  --checkpoint-dir "$checkpoint_dir" \
  --print-interval 50 \
  --eval-after \
  --episodes 4096 \
  --drain-initial-episodes
