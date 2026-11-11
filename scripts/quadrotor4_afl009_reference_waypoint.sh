#!/usr/bin/env bash
set -euo pipefail

seed="${AFL009_SEED:-88}"
experiment="${AFL_EXPERIMENT:-afl009}"
scenario="${AFL009_SCENARIO:-reference_waypoint_open3}"
timesteps="${AFL009_TIMESTEPS:-40000000}"
agents="${AFL009_TOTAL_AGENTS:-2048}"
buffers="${AFL009_NUM_BUFFERS:-8}"
threads="${AFL009_NUM_THREADS:-1}"
horizon="${AFL009_HORIZON:-32}"
minibatch_size="${AFL009_MINIBATCH_SIZE:-8192}"
checkpoint_interval="${AFL009_CHECKPOINT_INTERVAL:-5}"
eval_episodes="${AFL009_EVAL_EPISODES:-4096}"
eval_last="${AFL009_EVAL_LAST:-5}"
eval_every="${AFL009_EVAL_EVERY:-1}"
eval_max_rollouts="${AFL009_EVAL_MAX_ROLLOUTS:-256}"
eval_fresh_process="${AFL009_EVAL_FRESH_PROCESS:-1}"
eval_agents="${AFL009_EVAL_TOTAL_AGENTS:-$agents}"
eval_buffers="${AFL009_EVAL_NUM_BUFFERS:-$buffers}"
eval_threads="${AFL009_EVAL_NUM_THREADS:-$threads}"
learning_rate="${AFL009_LEARNING_RATE:-0.0015}"
gamma="${AFL009_GAMMA:-0.99}"
gae_lambda="${AFL009_GAE_LAMBDA:-0.90}"
replay_ratio="${AFL009_REPLAY_RATIO:-1.0}"
clip_coef="${AFL009_CLIP_COEF:-0.10}"
vf_coef="${AFL009_VF_COEF:-1.0}"
vf_clip_coef="${AFL009_VF_CLIP_COEF:-0.20}"
max_grad_norm="${AFL009_MAX_GRAD_NORM:-0.50}"
ent_coef="${AFL009_ENT_COEF:-0.001}"
action_std="${AFL009_ACTION_STD:-0.20}"
train_action_std="${AFL009_TRAIN_ACTION_STD:-1}"
reward_clip="${AFL009_REWARD_CLIP:-1.0}"
max_steps="${AFL009_MAX_STEPS:-1024}"
spawn_radius="${AFL009_SPAWN_RADIUS:-5.0}"
spawn_radius_min="${AFL009_SPAWN_RADIUS_MIN:-0.0}"
spawn_height="${AFL009_SPAWN_HEIGHT:-5.0}"
room_size="${AFL009_ROOM_SIZE:-6.0}"
ceiling_height="${AFL009_CEILING_HEIGHT:-10.0}"
route_goal_count="${AFL009_ROUTE_GOAL_COUNT:-3}"
route_goal_count_min="${AFL009_ROUTE_GOAL_COUNT_MIN:-0}"
route_goal_radius="${AFL009_ROUTE_GOAL_RADIUS:-0.75}"
route_reward_mode="${AFL009_ROUTE_REWARD_MODE:-2}"
route_target_obs_mode="${AFL009_ROUTE_TARGET_OBS_MODE:-1}"
route_step_cost="${AFL009_ROUTE_STEP_COST:-0.004}"
route_target_z_min="${AFL009_ROUTE_TARGET_Z_MIN:-2.0}"
route_target_z_max="${AFL009_ROUTE_TARGET_Z_MAX:-8.0}"
route_progress_scale="${AFL009_ROUTE_PROGRESS_SCALE:-25.0}"
route_vel_scale="${AFL009_ROUTE_VEL_SCALE:-0.02}"
route_arrival_radius="${AFL009_ROUTE_ARRIVAL_RADIUS:-1.25}"
route_arrival_bonus="${AFL009_ROUTE_ARRIVAL_BONUS:-0.02}"
route_arrival_vel_scale="${AFL009_ROUTE_ARRIVAL_VEL_SCALE:-0.005}"
route_arrival_toward_scale="${AFL009_ROUTE_ARRIVAL_TOWARD_SCALE:-0.0}"
route_goal_speed="${AFL009_ROUTE_GOAL_SPEED:-2.0}"
route_target_wall_margin="${AFL009_ROUTE_TARGET_WALL_MARGIN:-0.75}"
route_clearance_radius="${AFL009_ROUTE_CLEARANCE_RADIUS:-0.45}"
route_clearance_scale="${AFL009_ROUTE_CLEARANCE_SCALE:-0.0}"
route_clearance_speed_scale="${AFL009_ROUTE_CLEARANCE_SPEED_SCALE:-0.0}"
route_clearance_speed_min="${AFL009_ROUTE_CLEARANCE_SPEED_MIN:-0.0}"
route_clearance_speed_full="${AFL009_ROUTE_CLEARANCE_SPEED_FULL:-0.0}"
route_walls="${AFL009_ROUTE_WALLS:-0}"
obstacle_mode="${AFL009_OBSTACLE_MODE:-0}"
mass_scale_min="${AFL009_MASS_SCALE_MIN:-0.95}"
mass_scale_max="${AFL009_MASS_SCALE_MAX:-1.05}"
thrust_scale_min="${AFL009_THRUST_SCALE_MIN:-0.95}"
thrust_scale_max="${AFL009_THRUST_SCALE_MAX:-1.05}"
drag_scale_min="${AFL009_DRAG_SCALE_MIN:-0.95}"
drag_scale_max="${AFL009_DRAG_SCALE_MAX:-1.05}"
motor_tau_scale_min="${AFL009_MOTOR_TAU_SCALE_MIN:-0.95}"
motor_tau_scale_max="${AFL009_MOTOR_TAU_SCALE_MAX:-1.05}"
cpu="${AFL009_CPU:-0}"
float_build="${AFL009_FLOAT_BUILD:-1}"
skip_train="${AFL009_SKIP_TRAIN:-0}"
skip_eval="${AFL009_SKIP_EVAL:-0}"
load_model_path="${AFL009_LOAD_MODEL_PATH:-}"

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

checkpoint_dir="experiments/agile/${experiment}/${scenario}"
log_dir="logs/agile/${experiment}/${scenario}"
mkdir -p "$checkpoint_dir" "$log_dir"

build_cmd=(bash build.sh quadrotor)
train_mode_args=()
if [[ "$cpu" == "1" ]]; then
  build_cmd+=(--cpu)
  train_mode_args+=(--slowly)
elif [[ "$float_build" == "1" ]]; then
  build_cmd+=(--float)
fi
train_extra_args=()
if [[ -n "$load_model_path" && "$load_model_path" != "none" ]]; then
  train_extra_args+=(--load-model-path "$load_model_path")
fi
PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" "${build_cmd[@]}"

if [[ "$skip_train" != "1" ]]; then
  echo "$experiment $scenario train timesteps=$timesteps seed=$seed agents=$agents"
  "$python_bin" -m pufferlib.pufferl train quadrotor \
    "${train_mode_args[@]}" \
    --train.total-timesteps "$timesteps" \
    --train.seed "$seed" \
    --train.learning-rate "$learning_rate" \
    --train.gamma "$gamma" \
    --train.gae-lambda "$gae_lambda" \
    --train.replay-ratio "$replay_ratio" \
    --train.clip-coef "$clip_coef" \
    --train.vf-coef "$vf_coef" \
    --train.vf-clip-coef "$vf_clip_coef" \
    --train.max-grad-norm "$max_grad_norm" \
    --train.ent-coef "$ent_coef" \
    --train.action-std "$action_std" \
    --train.train-action-std "$train_action_std" \
    --train.reward-clip "$reward_clip" \
    --train.beta1 0.914207 \
    --train.beta2 0.99988 \
    --train.eps 1e-14 \
    --train.minibatch-size "$minibatch_size" \
    --train.horizon "$horizon" \
    --train.vtrace-rho-clip 3.32822 \
    --train.vtrace-c-clip 4.37219 \
    --train.prio-alpha 0.427293 \
    --train.prio-beta0 1 \
    --checkpoint-dir "$checkpoint_dir" \
    --log-dir "$log_dir" \
    --checkpoint-interval "$checkpoint_interval" \
    --eval-episodes "$eval_episodes" \
    --vec.total-agents "$agents" \
    --vec.num-buffers "$buffers" \
    --vec.num-threads "$threads" \
    --torch.network MLP \
    --policy.hidden-size 128 \
    --policy.num-layers 4 \
    --policy.expansion-factor 1 \
    --policy.use-bias 1 \
    --env.seed "$seed" \
    --env.task-mode 4 \
    --env.action-mode 4 \
    --env.max-steps "$max_steps" \
    --env.action-repeat 1 \
    --env.dynamics-profile 1 \
    --env.physics-dt 0.002 \
    --env.physics-substeps 5 \
    --env.spawn-radius "$spawn_radius" \
    --env.spawn-radius-min "$spawn_radius_min" \
    --env.spawn-height "$spawn_height" \
    --env.init-vel-radius 0.0 \
    --env.init-omega-radius 0.0 \
    --env.room-size "$room_size" \
    --env.ceiling-height "$ceiling_height" \
    --env.num-rays 24 \
    --env.ray-length 4.0 \
    --env.obstacle-mode "$obstacle_mode" \
    --env.route-goal-radius "$route_goal_radius" \
    --env.route-goal-count "$route_goal_count" \
    --env.route-goal-count-min "$route_goal_count_min" \
    --env.route-target-mode 0 \
    --env.route-blocked-target-prob 0.0 \
    --env.route-target-z-min "$route_target_z_min" \
    --env.route-target-z-max "$route_target_z_max" \
    --env.route-reward-mode "$route_reward_mode" \
    --env.route-step-cost "$route_step_cost" \
    --env.route-walls "$route_walls" \
    --env.route-direct-obs 1 \
    --env.route-extra-obs 0 \
    --env.route-wall-obs 1 \
    --env.route-perception-obs 0 \
    --env.route-target-obs-mode "$route_target_obs_mode" \
    --env.route-path-margin 0.35 \
    --env.route-clearance-radius "$route_clearance_radius" \
    --env.route-clearance-scale "$route_clearance_scale" \
    --env.route-clearance-speed-scale "$route_clearance_speed_scale" \
    --env.route-clearance-speed-min "$route_clearance_speed_min" \
    --env.route-clearance-speed-full "$route_clearance_speed_full" \
    --env.route-progress-scale "$route_progress_scale" \
    --env.route-vel-scale "$route_vel_scale" \
    --env.route-arrival-radius "$route_arrival_radius" \
    --env.route-arrival-bonus "$route_arrival_bonus" \
    --env.route-arrival-vel-scale "$route_arrival_vel_scale" \
    --env.route-arrival-toward-scale "$route_arrival_toward_scale" \
    --env.route-goal-speed "$route_goal_speed" \
    --env.route-target-wall-margin "$route_target_wall_margin" \
    --env.action-history-obs 1 \
    --env.mass-scale-min "$mass_scale_min" \
    --env.mass-scale-max "$mass_scale_max" \
    --env.thrust-scale-min "$thrust_scale_min" \
    --env.thrust-scale-max "$thrust_scale_max" \
    --env.drag-scale-min "$drag_scale_min" \
    --env.drag-scale-max "$drag_scale_max" \
    --env.motor-tau-scale-min "$motor_tau_scale_min" \
    --env.motor-tau-scale-max "$motor_tau_scale_max" \
    "${train_extra_args[@]}"
fi

if [[ "$skip_eval" == "1" ]]; then
  exit 0
fi

run_dir="$(find "$checkpoint_dir/quadrotor" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
if [[ -z "$run_dir" ]]; then
  echo "no native checkpoint run found under $checkpoint_dir/quadrotor" >&2
  exit 1
fi

echo "$experiment $scenario eval run_dir=$run_dir"
eval_mode_args=()
if [[ "$eval_fresh_process" == "1" ]]; then
  eval_mode_args+=(--fresh-process)
fi
PYTHON="$python_bin" scripts/quadrotor4.sh reference-waypoint-eval-checkpoints "$run_dir" \
  --episodes "$eval_episodes" \
  --total-agents "$eval_agents" \
  --num-buffers "$eval_buffers" \
  --num-threads "$eval_threads" \
  --horizon "$horizon" \
  --max-rollouts "$eval_max_rollouts" \
  --network MLP \
  --hidden-size 128 \
  --num-layers 4 \
  --use-bias 1 \
  --seed "$seed" \
  --max-steps "$max_steps" \
  --action-repeat 1 \
  --action-mode 4 \
  --dynamics-profile 1 \
  --physics-dt 0.002 \
  --physics-substeps 5 \
  --num-rays 24 \
  --ray-length 4.0 \
  --obstacle-mode "$obstacle_mode" \
  --route-goal-count "$route_goal_count" \
  --route-goal-count-min "$route_goal_count_min" \
  --route-goal-radius "$route_goal_radius" \
  --route-reward-mode "$route_reward_mode" \
  --route-target-obs-mode "$route_target_obs_mode" \
  --route-step-cost "$route_step_cost" \
  --route-target-mode 0 \
  --route-blocked-target-prob 0.0 \
  --route-target-z-min "$route_target_z_min" \
  --route-target-z-max "$route_target_z_max" \
  --route-progress-scale "$route_progress_scale" \
  --route-clearance-radius "$route_clearance_radius" \
  --route-clearance-scale "$route_clearance_scale" \
  --route-clearance-speed-scale "$route_clearance_speed_scale" \
  --route-clearance-speed-min "$route_clearance_speed_min" \
  --route-clearance-speed-full "$route_clearance_speed_full" \
  --route-vel-scale "$route_vel_scale" \
  --route-arrival-radius "$route_arrival_radius" \
  --route-arrival-bonus "$route_arrival_bonus" \
  --route-arrival-vel-scale "$route_arrival_vel_scale" \
  --route-arrival-toward-scale "$route_arrival_toward_scale" \
  --route-goal-speed "$route_goal_speed" \
  --route-walls "$route_walls" \
  --mass-scale-min "$mass_scale_min" \
  --mass-scale-max "$mass_scale_max" \
  --thrust-scale-min "$thrust_scale_min" \
  --thrust-scale-max "$thrust_scale_max" \
  --drag-scale-min "$drag_scale_min" \
  --drag-scale-max "$drag_scale_max" \
  --motor-tau-scale-min "$motor_tau_scale_min" \
  --motor-tau-scale-max "$motor_tau_scale_max" \
  --every "$eval_every" \
  --last "$eval_last" \
  "${eval_mode_args[@]}" \
  2>&1 | tee "$log_dir/eval_last${eval_last}_s${seed}.log"
