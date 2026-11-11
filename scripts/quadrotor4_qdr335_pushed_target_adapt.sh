#!/usr/bin/env bash
set -euo pipefail

seed="${QDR335_SEED:-88}"
updates="${QDR335_UPDATES:-200}"
agents="${QDR335_TOTAL_AGENTS:-2048}"
buffers="${QDR335_NUM_BUFFERS:-2}"
threads="${QDR335_NUM_THREADS:-16}"
train_speed="${QDR335_TRAIN_SPEED:-0.75}"
eval_speeds="${QDR335_EVAL_SPEEDS:-0.72,0.75}"
eval_modes="${QDR335_EVAL_MODES:-p3,aggregate}"
eval_episodes="${QDR335_EVAL_EPISODES:-2048}"
base_checkpoint="${QDR335_INIT:-experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt}"
anchor_checkpoint="${QDR335_ANCHOR:-$base_checkpoint}"
output="${QDR335_OUTPUT:-experiments/qdr335/qdr335_pushed_target_adapt_s${seed}.pt}"
checkpoint_dir="${QDR335_CHECKPOINT_DIR:-experiments/qdr335/checkpoints}"
out_dir="${QDR335_OUT_DIR:-logs/qdr335}"
build_args="${QDR335_BUILD_ARGS:-}"
skip_train="${QDR335_SKIP_TRAIN:-0}"
skip_eval="${QDR335_SKIP_EVAL:-0}"
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
elif [[ -x "../../.venv/bin/python" ]]; then
  python_bin="../../.venv/bin/python"
else
  python_bin="python"
fi

mkdir -p "$out_dir" "$(dirname "$output")" "$checkpoint_dir"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$(dirname "$python_bin"):$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" bash build.sh quadrotor $build_args

common_env=(
  --ignore-checkpoint-env-config
  --deploy-patch-obs
  --deploy-onboard-target-obs
  --explore-reachable-coverage
  --explore-frontier-target-mode onboard
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
  --room-size 12.0
  --cell-size 0.40
  --max-steps 1200
  --spawn-radius-min 0.0
  --explore-goal-fraction 0.99
  --safety-barrier-actions
  --safety-slide
  --safety-cone-ray-stop 0.30
  --safety-cone-deg 60
  --safety-slow-ray 0.80
  --safety-min-ray-speed-cap
  --safety-min-ray-speed-stop 0.18
  --safety-min-ray-speed-slow 0.45
)

if [[ "$skip_train" != "1" ]]; then
  echo "qdr335_pushed_target_adapt train seed=$seed updates=$updates agents=$agents train_speed=$train_speed"
  PYTHON="$python_bin" scripts/quadrotor4.sh explore-v0-target-distill \
    --output "$output" \
    --init-checkpoint "$base_checkpoint" \
    --anchor-checkpoint "$anchor_checkpoint" \
    --anchor-modes \
      procedural_indoor_hard_p1 procedural_indoor_hard_p2 procedural_indoor_hard_p3 \
      procedural_indoor_hard_p0 procedural_indoor procedural_indoor_p0 \
      procedural_indoor_p2 maze center_block doorway corridor random_blocks layout_mixed \
    --anchor-loss-weight 0.45 \
    --train-modes \
      procedural_house_multizone_p3 procedural_house_multizone_p3 procedural_house_multizone_p3 \
      procedural_house_multizone_p3 procedural_house_multizone procedural_house_multizone \
      procedural_house_multizone_p0 procedural_house_multizone_p1 procedural_house_multizone_p2 \
    --hard-modes procedural_house_multizone_p3 procedural_house_multizone \
    --hard-example-weight 2.0 \
    --hard-angle-deg 40 \
    --hard-clearance-gap 0.12 \
    --hard-low-speed 0.10 \
    --hard-min-coverage 0.35 \
    --mode procedural_house_multizone_p3 \
    --seed "$seed" \
    --seq-updates "$updates" \
    --teacher-rollout-updates 40 \
    --max-student-rollout-prob 0.55 \
    --learning-rate 0.00002 \
    --total-agents "$agents" \
    --num-buffers "$buffers" \
    --num-threads "$threads" \
    --seq-len 64 \
    --speed-fraction "$train_speed" \
    --max-eval-steps 50000 \
    --checkpoint-interval 50 \
    --checkpoint-dir "$checkpoint_dir" \
    --print-interval 20 \
    --episodes "$eval_episodes" \
    --drain-initial-episodes \
    "${common_env[@]}" \
    2>&1 | tee "$out_dir/train_s${seed}.log"
fi

if [[ "$skip_eval" == "1" ]]; then
  exit 0
fi

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

run_eval_case() {
  local checkpoint="$1"
  local speed="$2"
  local mode_label="$3"
  local mode suffix log_prefix
  mode="$(mode_name "$mode_label")"
  suffix="$(speed_label "$speed")"
  log_prefix="$out_dir/${mode_label}_speed${suffix}_qdr335"

  echo "qdr335_pushed_target_adapt eval checkpoint=$checkpoint mode=$mode speed=$speed episodes=$eval_episodes"
  PYTHON="$python_bin" scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    --seed "$seed" \
    --episodes "$eval_episodes" \
    --total-agents "$agents" \
    --num-buffers "$buffers" \
    --num-threads "$threads" \
    --max-eval-steps 300000 \
    --mode "$mode" \
    --speed-fraction "$speed" \
    "${common_env[@]}" \
    2>&1 | tee "${log_prefix}.log"
}

IFS=',' read -r -a speed_items <<< "$eval_speeds"
IFS=',' read -r -a mode_items <<< "$eval_modes"
for speed in "${speed_items[@]}"; do
  [[ -z "$speed" ]] && continue
  for mode_label in "${mode_items[@]}"; do
    [[ -z "$mode_label" ]] && continue
    run_eval_case "$output" "$speed" "$mode_label"
  done
done
