#!/usr/bin/env bash
set -euo pipefail

variant_args=("$@")
if [[ "${#variant_args[@]}" -eq 0 ]]; then
  variant_args=(baseline cam20)
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
      echo "unknown QDR-354 variant: $variant" >&2
      echo "valid variants: baseline cam16 cam20 all" >&2
      exit 1
      ;;
  esac
done

mode="${QDR354_MODE:-maze}"
case "$mode" in
  random_blocks) obstacle_mode=5 ;;
  layout_mixed) obstacle_mode=7 ;;
  maze) obstacle_mode=4 ;;
  procedural_house_multizone) obstacle_mode=28 ;;
  procedural_house_multizone_p3) obstacle_mode=32 ;;
  *)
    echo "unsupported QDR354_MODE=$mode" >&2
    exit 1
    ;;
esac

seed="${QDR354_SEED:-88}"
timesteps="${QDR354_TIMESTEPS:-10000000}"
agents="${QDR354_TOTAL_AGENTS:-4096}"
buffers="${QDR354_NUM_BUFFERS:-2}"
threads="${QDR354_NUM_THREADS:-16}"
horizon="${QDR354_HORIZON:-64}"
minibatch_size="${QDR354_MINIBATCH_SIZE:-32768}"
checkpoint_interval="${QDR354_CHECKPOINT_INTERVAL:-2}"
eval_episodes="${QDR354_EVAL_EPISODES:-2048}"
eval_last="${QDR354_EVAL_LAST:-3}"
eval_every="${QDR354_EVAL_EVERY:-1}"
eval_max_rollouts="${QDR354_EVAL_MAX_ROLLOUTS:-256}"
max_steps="${QDR354_MAX_STEPS:-1024}"
max_vel="${QDR354_MAX_VEL:-0.7}"
max_z_vel="${QDR354_MAX_Z_VEL:-0.25}"
room_size="${QDR354_ROOM_SIZE:-3.0}"
spawn_radius="${QDR354_SPAWN_RADIUS:-2.5}"
spawn_radius_min="${QDR354_SPAWN_RADIUS_MIN:-0.0}"
explore_goal_fraction="${QDR354_GOAL_FRACTION:-0.85}"
explore_memory_obs="${QDR354_MEMORY_OBS:-3}"
cpu="${QDR354_CPU:-0}"
skip_train="${QDR354_SKIP_TRAIN:-0}"
skip_eval="${QDR354_SKIP_EVAL:-0}"

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
train_mode_args=()
if [[ "$cpu" == "1" ]]; then
  build_cmd+=(--cpu)
  train_mode_args+=(--slowly)
fi
PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" "${build_cmd[@]}"

camera_train_args() {
  local variant="$1"
  case "$variant" in
    baseline)
      echo "--env.camera-depth-obs 0"
      ;;
    cam16)
      echo "--env.camera-depth-obs 1 --env.camera-depth-width 16 --env.camera-depth-height 12"
      ;;
    cam20)
      echo "--env.camera-depth-obs 1 --env.camera-depth-width 20 --env.camera-depth-height 15"
      ;;
  esac
}

camera_eval_args() {
  local variant="$1"
  case "$variant" in
    baseline)
      echo "--camera-depth-obs 0"
      ;;
    cam16)
      echo "--camera-depth-obs 1 --camera-depth-width 16 --camera-depth-height 12"
      ;;
    cam20)
      echo "--camera-depth-obs 1 --camera-depth-width 20 --camera-depth-height 15"
      ;;
  esac
}

for variant in "${variants[@]}"; do
  checkpoint_dir="experiments/qdr354/${mode}_${variant}"
  log_dir="logs/qdr354/${mode}_${variant}"
  mkdir -p "$checkpoint_dir" "$log_dir"
  read -r -a train_camera <<<"$(camera_train_args "$variant")"
  read -r -a eval_camera <<<"$(camera_eval_args "$variant")"

  if [[ "$skip_train" != "1" ]]; then
    echo "qdr354_native_ppo_camera train variant=$variant mode=$mode timesteps=$timesteps seed=$seed agents=$agents"
    "$python_bin" -m pufferlib.pufferl train quadrotor \
      "${train_mode_args[@]}" \
      --train.total-timesteps "$timesteps" \
      --train.seed "$seed" \
      --train.horizon "$horizon" \
      --train.minibatch-size "$minibatch_size" \
      --checkpoint-dir "$checkpoint_dir" \
      --log-dir "$log_dir" \
      --checkpoint-interval "$checkpoint_interval" \
      --eval-episodes "$eval_episodes" \
      --vec.total-agents "$agents" \
      --vec.num-buffers "$buffers" \
      --vec.num-threads "$threads" \
      --torch.network MinGRU \
      --policy.use-bias 1 \
      --env.seed "$seed" \
      --env.task-mode 3 \
      --env.action-mode 1 \
      --env.max-steps "$max_steps" \
      --env.max-vel "$max_vel" \
      --env.max-z-vel "$max_z_vel" \
      --env.spawn-radius "$spawn_radius" \
      --env.spawn-radius-min "$spawn_radius_min" \
      --env.spawn-height 1.5 \
      --env.room-size "$room_size" \
      --env.num-rays 24 \
      --env.ray-length 4.0 \
      --env.obstacle-mode "$obstacle_mode" \
      --env.explore-goal-fraction "$explore_goal_fraction" \
      --env.explore-memory-obs "$explore_memory_obs" \
      --env.explore-frontier-progress-scale 0.0 \
      --env.route-wall-obs 1 \
      --env.action-history-obs 1 \
      "${train_camera[@]}"
  fi

  if [[ "$skip_eval" == "1" ]]; then
    continue
  fi

  run_dir="$(find "$checkpoint_dir/quadrotor" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  if [[ -z "$run_dir" ]]; then
    echo "no native checkpoint run found under $checkpoint_dir/quadrotor" >&2
    exit 1
  fi
  echo "qdr354_native_ppo_camera eval variant=$variant run_dir=$run_dir mode=$mode"
  PYTHON="$python_bin" scripts/quadrotor4.sh explore-v0-eval-checkpoints "$run_dir" \
    --episodes "$eval_episodes" \
    --total-agents "$agents" \
    --num-buffers "$buffers" \
    --num-threads "$threads" \
    --horizon "$horizon" \
    --max-rollouts "$eval_max_rollouts" \
    --network MinGRU \
    --use-bias 1 \
    --seed "$seed" \
    --obstacle-mode "$mode" \
    --max-steps "$max_steps" \
    --max-vel "$max_vel" \
    --max-z-vel "$max_z_vel" \
    --explore-goal-fraction "$explore_goal_fraction" \
    --explore-memory-obs "$explore_memory_obs" \
    --every "$eval_every" \
    --last "$eval_last" \
    "${eval_camera[@]}" \
    2>&1 | tee "$log_dir/eval_last${eval_last}_s${seed}.log"
done
