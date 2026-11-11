#!/usr/bin/env bash
set -euo pipefail

seed="${QDR342_SEED:-88}"
updates="${QDR342_UPDATES:-250}"
agents="${QDR342_TOTAL_AGENTS:-2048}"
buffers="${QDR342_NUM_BUFFERS:-2}"
threads="${QDR342_NUM_THREADS:-16}"
teacher_updates="${QDR342_TEACHER_UPDATES:-60}"
student_prob="${QDR342_STUDENT_PROB:-0.45}"
seq_len="${QDR342_SEQ_LEN:-96}"
learning_rate="${QDR342_LR:-0.00003}"
train_speed="${QDR342_TRAIN_SPEED:-0.72}"
eval_speed="${QDR342_EVAL_SPEED:-0.72}"
eval_episodes="${QDR342_EVAL_EPISODES:-2048}"
output="${QDR342_OUTPUT:-experiments/qdr342/qdr342_priv_unexplored_distill_s${seed}.pt}"
checkpoint_dir="${QDR342_CHECKPOINT_DIR:-experiments/qdr342/checkpoints}"
out_dir="${QDR342_OUT_DIR:-logs/qdr342}"
build_args="${QDR342_BUILD_ARGS:-}"
checkpoint_interval="${QDR342_CHECKPOINT_INTERVAL:-50}"
print_interval="${QDR342_PRINT_INTERVAL:-20}"
skip_train="${QDR342_SKIP_TRAIN:-0}"
skip_eval="${QDR342_SKIP_EVAL:-0}"
init_checkpoint="${QDR342_INIT:-}"

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
  --explore-reachable-coverage
  --explore-frontier-target-mode unexplored
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

train_extra=(--ignore-checkpoint-env-config)
if [[ -n "$init_checkpoint" ]]; then
  train_extra+=(
    --init-checkpoint "$init_checkpoint"
  )
fi

if [[ "$skip_train" != "1" ]]; then
  echo "qdr342_privileged_unexplored_distill train seed=$seed updates=$updates agents=$agents speed=$train_speed teacher_updates=$teacher_updates student_prob=$student_prob init=${init_checkpoint:-from_zero}"
  PYTHON="$python_bin" scripts/quadrotor4.sh explore-v0-target-distill \
    --output "$output" \
    "${train_extra[@]}" \
    --deploy-patch-obs \
    --deploy-coverage-obs \
    --train-modes \
      procedural_house_multizone_p3 procedural_house_multizone_p3 procedural_house_multizone_p3 \
      procedural_house_multizone_p3 procedural_house_multizone procedural_house_multizone \
      procedural_house_multizone_p0 procedural_house_multizone_p1 procedural_house_multizone_p2 \
    --mode procedural_house_multizone_p3 \
    --seed "$seed" \
    --seq-updates "$updates" \
    --teacher-rollout-updates "$teacher_updates" \
    --max-student-rollout-prob "$student_prob" \
    --learning-rate "$learning_rate" \
    --total-agents "$agents" \
    --num-buffers "$buffers" \
    --num-threads "$threads" \
    --seq-len "$seq_len" \
    --speed-fraction "$train_speed" \
    --max-eval-steps 50000 \
    --checkpoint-interval "$checkpoint_interval" \
    --checkpoint-dir "$checkpoint_dir" \
    --print-interval "$print_interval" \
    --episodes "$eval_episodes" \
    --drain-initial-episodes \
    "${common_env[@]}" \
    2>&1 | tee "$out_dir/train_s${seed}.log"
fi

if [[ "$skip_eval" == "1" ]]; then
  exit 0
fi

run_eval_case() {
  local checkpoint="$1"
  local mode_label="$2"
  local mode log_prefix
  case "$mode_label" in
    p3) mode="procedural_house_multizone_p3" ;;
    aggregate) mode="procedural_house_multizone" ;;
    *) echo "unknown mode alias: $mode_label" >&2; exit 1 ;;
  esac
  log_prefix="$out_dir/${mode_label}_speed$(printf '%s' "$eval_speed" | tr -d '.')_qdr342"

  echo "qdr342_privileged_unexplored_distill eval checkpoint=$checkpoint mode=$mode speed=$eval_speed episodes=$eval_episodes"
  PYTHON="$python_bin" scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    --seed "$seed" \
    --episodes "$eval_episodes" \
    --total-agents "$agents" \
    --num-buffers "$buffers" \
    --num-threads "$threads" \
    --max-eval-steps 300000 \
    --mode "$mode" \
    --speed-fraction "$eval_speed" \
    --ignore-checkpoint-env-config \
    --deploy-patch-obs \
    --deploy-coverage-obs \
    "${common_env[@]}" \
    2>&1 | tee "${log_prefix}.log"
}

run_eval_case "$output" p3
run_eval_case "$output" aggregate
