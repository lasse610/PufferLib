#!/usr/bin/env bash
set -euo pipefail

seed="${QDR348_SEED:-88}"
updates="${QDR348_UPDATES:-300}"
agents="${QDR348_TOTAL_AGENTS:-2048}"
buffers="${QDR348_NUM_BUFFERS:-2}"
threads="${QDR348_NUM_THREADS:-16}"
teacher_updates="${QDR348_TEACHER_UPDATES:-50}"
student_prob="${QDR348_STUDENT_PROB:-0.75}"
seq_len="${QDR348_SEQ_LEN:-96}"
learning_rate="${QDR348_LR:-0.0001}"
train_speed="${QDR348_TRAIN_SPEED:-0.72}"
eval_speed="${QDR348_EVAL_SPEED:-0.72}"
eval_episodes="${QDR348_EVAL_EPISODES:-2048}"
output="${QDR348_OUTPUT:-experiments/qdr348/qdr348_graph_candidate_selector_s${seed}.pt}"
checkpoint_dir="${QDR348_CHECKPOINT_DIR:-experiments/qdr348/checkpoints}"
out_dir="${QDR348_OUT_DIR:-logs/qdr348}"
checkpoint_interval="${QDR348_CHECKPOINT_INTERVAL:-50}"
print_interval="${QDR348_PRINT_INTERVAL:-20}"
skip_train="${QDR348_SKIP_TRAIN:-0}"
skip_eval="${QDR348_SKIP_EVAL:-0}"
build_args="${QDR348_BUILD_ARGS:-}"
init_checkpoint="${QDR348_INIT:-}"

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

deploy_args=(
  --deploy-graph-candidate-obs
  --deploy-coverage-obs
)

common_env=(
  --explore-reachable-coverage
  --explore-frontier-target-mode unexplored
  --explore-onboard-graph-candidate-obs
  --explore-onboard-passage-quality
  --explore-onboard-passage-weight 0.35
  --explore-onboard-frontier-graph
  --explore-onboard-graph-gain-weight 0.32
  --explore-onboard-graph-cluster-weight 0.12
  --explore-onboard-graph-depth-weight 0.045
  --explore-onboard-graph-active-bonus 0.0
  --explore-onboard-graph-stale-penalty 0.0
  --explore-onboard-graph-failure-penalty 0.0
  --explore-onboard-graph-attempt-penalty 0.0
  --explore-onboard-graph-gain-cap 192
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

label_args=(
  --predict-graph-candidate
  --graph-candidate-label-align-weight "${QDR348_ALIGN_WEIGHT:-2.0}"
  --graph-candidate-label-clearance-weight "${QDR348_CLEARANCE_WEIGHT:-0.4}"
  --graph-candidate-label-gain-weight "${QDR348_GAIN_WEIGHT:-0.3}"
  --graph-candidate-label-cluster-weight "${QDR348_CLUSTER_WEIGHT:-0.1}"
  --graph-candidate-label-passage-weight "${QDR348_PASSAGE_WEIGHT:-0.2}"
  --graph-candidate-label-selected-weight "${QDR348_SELECTED_WEIGHT:-0.0}"
  --graph-candidate-label-depth-weight "${QDR348_DEPTH_WEIGHT:-0.1}"
  --graph-candidate-label-min-align "${QDR348_MIN_ALIGN:--1.0}"
)

train_extra=(--ignore-checkpoint-env-config)
if [[ -n "$init_checkpoint" ]]; then
  train_extra+=(--init-checkpoint "$init_checkpoint")
fi

if [[ "$skip_train" != "1" ]]; then
  echo "qdr348_candidate_selector train seed=$seed updates=$updates agents=$agents speed=$train_speed init=${init_checkpoint:-from_zero}"
  PYTHON="$python_bin" scripts/quadrotor4.sh explore-v0-target-distill \
    --output "$output" \
    "${train_extra[@]}" \
    "${deploy_args[@]}" \
    "${label_args[@]}" \
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
  log_prefix="$out_dir/${mode_label}_speed$(printf '%s' "$eval_speed" | tr -d '.')_qdr348"

  echo "qdr348_candidate_selector eval checkpoint=$checkpoint mode=$mode speed=$eval_speed episodes=$eval_episodes"
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
    "${deploy_args[@]}" \
    "${label_args[@]}" \
    "${common_env[@]}" \
    2>&1 | tee "${log_prefix}.log"
}

run_eval_case "$output" p3
run_eval_case "$output" aggregate
