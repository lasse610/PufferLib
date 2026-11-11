#!/usr/bin/env bash
set -euo pipefail

seed="${AFL007_SEED:-88}"
experiment="${AFL_EXPERIMENT:-afl007}"
timesteps="${AFL007_TIMESTEPS:-40000000}"
agents="${AFL007_TOTAL_AGENTS:-2048}"
buffers="${AFL007_NUM_BUFFERS:-8}"
threads="${AFL007_NUM_THREADS:-1}"
horizon="${AFL007_HORIZON:-32}"
minibatch_size="${AFL007_MINIBATCH_SIZE:-8192}"
checkpoint_interval="${AFL007_CHECKPOINT_INTERVAL:-5}"
eval_episodes="${AFL007_EVAL_EPISODES:-4096}"
eval_last="${AFL007_EVAL_LAST:-5}"
eval_every="${AFL007_EVAL_EVERY:-1}"
eval_max_rollouts="${AFL007_EVAL_MAX_ROLLOUTS:-256}"
eval_fresh_process="${AFL007_EVAL_FRESH_PROCESS:-1}"
max_steps="${AFL007_MAX_STEPS:-1024}"
spawn_radius="${AFL007_SPAWN_RADIUS:-5.0}"
spawn_radius_min="${AFL007_SPAWN_RADIUS_MIN:-0.0}"
spawn_height="${AFL007_SPAWN_HEIGHT:-5.0}"
ceiling_height="${AFL007_CEILING_HEIGHT:-10.0}"
hover_target_dist="${AFL007_HOVER_TARGET_DIST:-5.0}"
action_mode="${AFL007_ACTION_MODE:-3}"
dynamics_profile="${AFL007_DYNAMICS_PROFILE:-0}"
physics_dt="${AFL007_PHYSICS_DT:-0.05}"
physics_substeps="${AFL007_PHYSICS_SUBSTEPS:-1}"
cpu="${AFL007_CPU:-0}"
skip_train="${AFL007_SKIP_TRAIN:-0}"
skip_eval="${AFL007_SKIP_EVAL:-0}"

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

checkpoint_dir="experiments/agile/${experiment}/drone_hover"
log_dir="logs/agile/${experiment}/drone_hover"
mkdir -p "$checkpoint_dir" "$log_dir"

build_cmd=(bash build.sh quadrotor)
train_mode_args=()
if [[ "$cpu" == "1" ]]; then
  build_cmd+=(--cpu)
  train_mode_args+=(--slowly)
fi
PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" "${build_cmd[@]}"

if [[ "$skip_train" != "1" ]]; then
  echo "$experiment drone_hover train timesteps=$timesteps seed=$seed agents=$agents action_mode=$action_mode dynamics_profile=$dynamics_profile physics_dt=$physics_dt physics_substeps=$physics_substeps"
  "$python_bin" -m pufferlib.pufferl train quadrotor \
    "${train_mode_args[@]}" \
    --train.total-timesteps "$timesteps" \
    --train.seed "$seed" \
    --train.learning-rate 0.00975033 \
    --train.gamma 0.981094 \
    --train.gae-lambda 0.883828 \
    --train.replay-ratio 2.25498 \
    --train.clip-coef 0.067834 \
    --train.vf-coef 4.42222 \
    --train.vf-clip-coef 3.42278 \
    --train.max-grad-norm 0.692201 \
    --train.ent-coef 9.62597e-05 \
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
    --env.task-mode 5 \
    --env.action-mode "$action_mode" \
    --env.max-steps "$max_steps" \
    --env.action-repeat 1 \
    --env.dynamics-profile "$dynamics_profile" \
    --env.physics-dt "$physics_dt" \
    --env.physics-substeps "$physics_substeps" \
    --env.spawn-radius "$spawn_radius" \
    --env.spawn-radius-min "$spawn_radius_min" \
    --env.spawn-height "$spawn_height" \
    --env.init-vel-radius 0.0 \
    --env.init-omega-radius 0.0 \
    --env.room-size 0.0 \
    --env.ceiling-height "$ceiling_height" \
    --env.num-rays 24 \
    --env.ray-length 4.0 \
    --env.obstacle-mode 0 \
    --env.route-direct-obs 0 \
    --env.route-wall-obs 1 \
    --env.route-perception-obs 0 \
    --env.route-target-obs-mode 1 \
    --env.action-history-obs 1 \
    --env.mass-scale-min 0.95 \
    --env.mass-scale-max 1.05 \
    --env.thrust-scale-min 0.95 \
    --env.thrust-scale-max 1.05 \
    --env.drag-scale-min 0.95 \
    --env.drag-scale-max 1.05 \
    --env.motor-tau-scale-min 0.95 \
    --env.motor-tau-scale-max 1.05 \
    --env.drone-alpha-dist 0.782192 \
    --env.drone-alpha-hover 0.071445 \
    --env.drone-alpha-omega 0.00135588 \
    --env.drone-alpha-shaping 3.9754 \
    --env.drone-hover-target-dist "$hover_target_dist" \
    --env.drone-hover-dist 0.1 \
    --env.drone-hover-omega 0.1 \
    --env.drone-hover-vel 0.1
fi

if [[ "$skip_eval" == "1" ]]; then
  exit 0
fi

run_dir="$(find "$checkpoint_dir/quadrotor" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
if [[ -z "$run_dir" ]]; then
  echo "no native checkpoint run found under $checkpoint_dir/quadrotor" >&2
  exit 1
fi

echo "$experiment drone_hover eval run_dir=$run_dir"
eval_mode_args=()
if [[ "$eval_fresh_process" == "1" ]]; then
  eval_mode_args+=(--fresh-process)
fi
eval_cmd="drone-hover-eval-checkpoints"
if [[ "$action_mode" == "4" && "$dynamics_profile" == "1" ]]; then
  eval_cmd="drone-reference-eval-checkpoints"
fi
PYTHON="$python_bin" scripts/quadrotor4.sh "$eval_cmd" "$run_dir" \
  --episodes "$eval_episodes" \
  --total-agents "$agents" \
  --num-buffers "$buffers" \
  --num-threads "$threads" \
  --horizon "$horizon" \
  --max-rollouts "$eval_max_rollouts" \
  --network MLP \
  --num-layers 4 \
  --use-bias 1 \
  --seed "$seed" \
  --action-mode "$action_mode" \
  --dynamics-profile "$dynamics_profile" \
  --physics-dt "$physics_dt" \
  --physics-substeps "$physics_substeps" \
  --every "$eval_every" \
  --last "$eval_last" \
  "${eval_mode_args[@]}" \
  2>&1 | tee "$log_dir/eval_last${eval_last}_s${seed}.log"
