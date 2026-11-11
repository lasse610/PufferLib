#!/usr/bin/env bash
set -euo pipefail

variant_args=("$@")
if [[ "${#variant_args[@]}" -eq 0 ]]; then
  variant_args=(rays)
fi

variants=()
for variant in "${variant_args[@]}"; do
  case "$variant" in
    all)
      variants+=(rays depth20 randomized)
      ;;
    rays|depth20|randomized)
      variants+=("$variant")
      ;;
    *)
      echo "unknown AFL-001 variant: $variant" >&2
      echo "valid variants: rays depth20 randomized all" >&2
      exit 1
      ;;
  esac
done

mode="${AFL001_MODE:-layout_mixed}"
case "$mode" in
  none|open) obstacle_mode=0 ;;
  doorway) obstacle_mode=2 ;;
  corridor) obstacle_mode=3 ;;
  maze) obstacle_mode=4 ;;
  random_blocks) obstacle_mode=5 ;;
  layout_mixed) obstacle_mode=7 ;;
  procedural_indoor) obstacle_mode=8 ;;
  procedural_indoor_hard) obstacle_mode=13 ;;
  *)
    echo "unsupported AFL001_MODE=$mode" >&2
    exit 1
    ;;
esac

seed="${AFL001_SEED:-88}"
experiment="${AFL_EXPERIMENT:-afl001}"
timesteps="${AFL001_TIMESTEPS:-20000000}"
agents="${AFL001_TOTAL_AGENTS:-4096}"
buffers="${AFL001_NUM_BUFFERS:-2}"
threads="${AFL001_NUM_THREADS:-16}"
horizon="${AFL001_HORIZON:-64}"
minibatch_size="${AFL001_MINIBATCH_SIZE:-32768}"
checkpoint_interval="${AFL001_CHECKPOINT_INTERVAL:-2}"
eval_episodes="${AFL001_EVAL_EPISODES:-4096}"
eval_last="${AFL001_EVAL_LAST:-3}"
eval_every="${AFL001_EVAL_EVERY:-1}"
eval_max_rollouts="${AFL001_EVAL_MAX_ROLLOUTS:-256}"
eval_fresh_process="${AFL001_EVAL_FRESH_PROCESS:-1}"
max_steps="${AFL001_MAX_STEPS:-768}"
max_vel="${AFL001_MAX_VEL:-1.4}"
max_z_vel="${AFL001_MAX_Z_VEL:-0.35}"
room_size="${AFL001_ROOM_SIZE:-3.0}"
spawn_radius="${AFL001_SPAWN_RADIUS:-2.5}"
spawn_radius_min="${AFL001_SPAWN_RADIUS_MIN:-0.0}"
route_goal_count="${AFL001_ROUTE_GOAL_COUNT:-3}"
route_goal_count_min="${AFL001_ROUTE_GOAL_COUNT_MIN:-0}"
route_target_mode="${AFL001_ROUTE_TARGET_MODE:-1}"
route_blocked_target_prob="${AFL001_ROUTE_BLOCKED_TARGET_PROB:-1.0}"
route_progress_scale="${AFL001_ROUTE_PROGRESS_SCALE:-25.0}"
route_clearance_scale="${AFL001_ROUTE_CLEARANCE_SCALE:-0.50}"
route_vel_scale="${AFL001_ROUTE_VEL_SCALE:-0.04}"
cpu="${AFL001_CPU:-0}"
skip_train="${AFL001_SKIP_TRAIN:-0}"
skip_eval="${AFL001_SKIP_EVAL:-0}"

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

variant_train_args() {
  local variant="$1"
  case "$variant" in
    rays)
      echo "--env.num-rays 24 --env.camera-depth-obs 0"
      ;;
    depth20)
      echo "--env.num-rays 0 --env.camera-depth-obs 1 --env.camera-depth-width 20 --env.camera-depth-height 15"
      ;;
    randomized)
      echo "--env.num-rays 24 --env.camera-depth-obs 0 --env.init-vel-radius 0.35 --env.init-omega-radius 0.4 --env.route-target-z-min 1.0 --env.route-target-z-max 2.0 --env.mass-scale-min 0.85 --env.mass-scale-max 1.15 --env.thrust-scale-min 0.9 --env.thrust-scale-max 1.1 --env.drag-scale-min 0.75 --env.drag-scale-max 1.5 --env.motor-tau-scale-min 0.75 --env.motor-tau-scale-max 1.5"
      ;;
  esac
}

variant_eval_args() {
  local variant="$1"
  case "$variant" in
    rays)
      echo "--num-rays 24 --camera-depth-obs 0"
      ;;
    depth20)
      echo "--num-rays 0 --camera-depth-obs 1 --camera-depth-width 20 --camera-depth-height 15"
      ;;
    randomized)
      echo "--num-rays 24 --camera-depth-obs 0 --init-vel-radius 0.35 --init-omega-radius 0.4 --route-target-z-min 1.0 --route-target-z-max 2.0 --mass-scale-min 0.85 --mass-scale-max 1.15 --thrust-scale-min 0.9 --thrust-scale-max 1.1 --drag-scale-min 0.75 --drag-scale-max 1.5 --motor-tau-scale-min 0.75 --motor-tau-scale-max 1.5"
      ;;
  esac
}

for variant in "${variants[@]}"; do
  checkpoint_dir="experiments/agile/${experiment}/${mode}_${variant}"
  log_dir="logs/agile/${experiment}/${mode}_${variant}"
  mkdir -p "$checkpoint_dir" "$log_dir"
  read -r -a train_variant <<<"$(variant_train_args "$variant")"
  read -r -a eval_variant <<<"$(variant_eval_args "$variant")"

  if [[ "$skip_train" != "1" ]]; then
    echo "$experiment agile_v0 train variant=$variant mode=$mode timesteps=$timesteps seed=$seed agents=$agents"
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
      --env.task-mode 4 \
      --env.action-mode 1 \
      --env.max-steps "$max_steps" \
      --env.max-vel "$max_vel" \
      --env.max-z-vel "$max_z_vel" \
      --env.vel-ctrl-ki-z 4.0 \
      --env.spawn-radius "$spawn_radius" \
      --env.spawn-radius-min "$spawn_radius_min" \
      --env.spawn-height 1.5 \
      --env.room-size "$room_size" \
      --env.ray-length 4.0 \
      --env.obstacle-mode "$obstacle_mode" \
      --env.route-progress-scale "$route_progress_scale" \
      --env.route-vel-scale "$route_vel_scale" \
      --env.route-goal-radius 1.0 \
      --env.route-goal-count "$route_goal_count" \
      --env.route-goal-count-min "$route_goal_count_min" \
      --env.route-target-mode "$route_target_mode" \
      --env.route-blocked-target-prob "$route_blocked_target_prob" \
      --env.route-target-z-min 1.5 \
      --env.route-target-z-max 1.5 \
      --env.route-reward-mode 0 \
      --env.route-walls 1 \
      --env.route-direct-obs 1 \
      --env.route-extra-obs 0 \
      --env.route-wall-obs 1 \
      --env.route-perception-obs 0 \
      --env.route-target-obs-mode 1 \
      --env.route-path-margin 0.35 \
      --env.route-clearance-scale "$route_clearance_scale" \
      --env.route-arrival-radius 1.5 \
      --env.route-arrival-bonus 0.04 \
      --env.route-arrival-vel-scale 0.01 \
      --env.route-target-wall-margin 1.0 \
      --env.action-history-obs 1 \
      "${train_variant[@]}"
  fi

  if [[ "$skip_eval" == "1" ]]; then
    continue
  fi

  run_dir="$(find "$checkpoint_dir/quadrotor" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  if [[ -z "$run_dir" ]]; then
    echo "no native checkpoint run found under $checkpoint_dir/quadrotor" >&2
    exit 1
  fi
  echo "$experiment agile_v0 eval variant=$variant run_dir=$run_dir mode=$mode"
  eval_mode_args=()
  if [[ "$eval_fresh_process" == "1" ]]; then
    eval_mode_args+=(--fresh-process)
  fi
  PYTHON="$python_bin" scripts/quadrotor4.sh agile-v0-eval-checkpoints "$run_dir" \
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
    --route-goal-count "$route_goal_count" \
    --route-goal-count-min "$route_goal_count_min" \
    --route-target-mode "$route_target_mode" \
    --route-blocked-target-prob "$route_blocked_target_prob" \
    --route-progress-scale "$route_progress_scale" \
    --route-clearance-scale "$route_clearance_scale" \
    --route-vel-scale "$route_vel_scale" \
    --every "$eval_every" \
    --last "$eval_last" \
    "${eval_mode_args[@]}" \
    "${eval_variant[@]}" \
    2>&1 | tee "$log_dir/eval_last${eval_last}_s${seed}.log"
done
