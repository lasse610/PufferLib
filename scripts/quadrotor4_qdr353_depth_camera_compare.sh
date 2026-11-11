#!/usr/bin/env bash
set -euo pipefail

variant_args=("$@")
if [[ "${#variant_args[@]}" -eq 0 ]]; then
  variant_args=(all)
fi

variants=()
for variant in "${variant_args[@]}"; do
  case "$variant" in
    all)
      variants+=(baseline cam16 cam20)
      ;;
    baseline|cam16|cam20)
      variants+=("$variant")
      ;;
    *)
      echo "unknown QDR-353 variant: $variant" >&2
      echo "valid variants: baseline cam16 cam20 all" >&2
      exit 1
      ;;
  esac
done

seed="${QDR353_SEED:-88}"
updates="${QDR353_UPDATES:-160}"
agents="${QDR353_TOTAL_AGENTS:-2048}"
buffers="${QDR353_NUM_BUFFERS:-2}"
threads="${QDR353_NUM_THREADS:-16}"
teacher_updates="${QDR353_TEACHER_UPDATES:-50}"
student_prob="${QDR353_STUDENT_PROB:-0.40}"
seq_len="${QDR353_SEQ_LEN:-96}"
learning_rate="${QDR353_LR:-0.00003}"
train_speed="${QDR353_TRAIN_SPEED:-0.72}"
eval_speed="${QDR353_EVAL_SPEED:-0.72}"
eval_episodes="${QDR353_EVAL_EPISODES:-1024}"
checkpoint_interval="${QDR353_CHECKPOINT_INTERVAL:-40}"
print_interval="${QDR353_PRINT_INTERVAL:-20}"
skip_train="${QDR353_SKIP_TRAIN:-0}"
skip_eval="${QDR353_SKIP_EVAL:-0}"
cpu="${QDR353_CPU:-0}"

if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
elif [[ -x "../../.venv/bin/python" ]]; then
  python_bin="../../.venv/bin/python"
else
  python_bin="python"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$(dirname "$python_bin"):$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

build_cmd=(bash build.sh quadrotor)
distill_device_args=()
if [[ "$cpu" == "1" ]]; then
  build_cmd+=(--cpu)
  distill_device_args+=(--cpu)
fi
PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" "${build_cmd[@]}"

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

variant_camera_args() {
  local variant="$1"
  case "$variant" in
    baseline)
      ;;
    cam16)
      echo "--camera-depth-obs --deploy-camera-depth-obs --camera-depth-width 16 --camera-depth-height 12"
      ;;
    cam20)
      echo "--camera-depth-obs --deploy-camera-depth-obs --camera-depth-width 20 --camera-depth-height 15"
      ;;
  esac
}

run_eval_case() {
  local variant="$1"
  local checkpoint="$2"
  local mode_label="$3"
  local mode="$mode_label"
  if [[ "$mode_label" == "p3" ]]; then
    mode="procedural_house_multizone_p3"
  elif [[ "$mode_label" == "aggregate" ]]; then
    mode="procedural_house_multizone"
  fi

  local out_dir="logs/qdr353/$variant"
  local log_prefix="$out_dir/${mode_label}_speed$(printf '%s' "$eval_speed" | tr -d '.')"
  read -r -a camera_args <<<"$(variant_camera_args "$variant")"

  echo "qdr353_depth_camera_compare eval variant=$variant checkpoint=$checkpoint mode=$mode speed=$eval_speed episodes=$eval_episodes"
  PYTHON="$python_bin" scripts/quadrotor4.sh explore-v0-target-distill-eval "$checkpoint" \
    "${distill_device_args[@]}" \
    --seed "$seed" \
    --episodes "$eval_episodes" \
    --total-agents "$agents" \
    --num-buffers "$buffers" \
    --num-threads "$threads" \
    --max-eval-steps 300000 \
    --mode "$mode" \
    --speed-fraction "$eval_speed" \
    --ignore-checkpoint-env-config \
    --deploy-coverage-obs \
    "${camera_args[@]}" \
    "${common_env[@]}" \
    2>&1 | tee "${log_prefix}.log"
}

for variant in "${variants[@]}"; do
  output="experiments/qdr353/qdr353_depth_${variant}_s${seed}.pt"
  checkpoint_dir="experiments/qdr353/checkpoints/$variant"
  out_dir="logs/qdr353/$variant"
  mkdir -p "$out_dir" "$(dirname "$output")" "$checkpoint_dir"
  read -r -a camera_args <<<"$(variant_camera_args "$variant")"

  if [[ "$skip_train" != "1" ]]; then
    echo "qdr353_depth_camera_compare train variant=$variant seed=$seed updates=$updates agents=$agents speed=$train_speed teacher_updates=$teacher_updates student_prob=$student_prob"
    PYTHON="$python_bin" scripts/quadrotor4.sh explore-v0-target-distill \
      "${distill_device_args[@]}" \
      --output "$output" \
      --ignore-checkpoint-env-config \
      --deploy-coverage-obs \
      "${camera_args[@]}" \
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

  if [[ "$skip_eval" != "1" ]]; then
    run_eval_case "$variant" "$output" p3
    run_eval_case "$variant" "$output" aggregate
  fi
done
