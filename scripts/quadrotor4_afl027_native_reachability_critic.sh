#!/usr/bin/env bash
set -euo pipefail

seed="${AFL027_SEED:-88}"
scenario="${AFL027_SCENARIO:-native_p1_reachability_critic_sanity}"
timesteps="${AFL027_TIMESTEPS:-20000000}"
agents="${AFL027_TOTAL_AGENTS:-2048}"
buffers="${AFL027_NUM_BUFFERS:-8}"
threads="${AFL027_NUM_THREADS:-1}"
horizon="${AFL027_HORIZON:-32}"
minibatch_size="${AFL027_MINIBATCH_SIZE:-8192}"
checkpoint_interval="${AFL027_CHECKPOINT_INTERVAL:-25}"
eval_episodes="${AFL027_EVAL_EPISODES:-2048}"
max_steps="${AFL027_MAX_STEPS:-1024}"
obstacle_mode="${AFL027_OBSTACLE_MODE:-35}"
asymmetric_critic="${AFL027_ASYMMETRIC_CRITIC:-1}"
cpu="${AFL027_CPU:-0}"

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

build_cmd=(bash build.sh quadrotor --float)
if [[ "$cpu" == "1" ]]; then
  build_cmd=(bash build.sh quadrotor --cpu)
fi
PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" "${build_cmd[@]}"

checkpoint_dir="experiments/agile/afl027/${scenario}"
log_dir="logs/agile/afl027/${scenario}"
mkdir -p "$checkpoint_dir" "$log_dir"

echo "afl027 ${scenario} train timesteps=${timesteps} seed=${seed} obstacle_mode=${obstacle_mode}"
"$python_bin" -m pufferlib.pufferl train quadrotor \
  --slowly \
  --train.asymmetric-critic "$asymmetric_critic" \
  --train.asymmetric-critic-route-direct-obs 0 \
  --train.asymmetric-critic-route-privileged-obs 1 \
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
  --train.action-std 1.0 \
  --train.reward-clip 1.0 \
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
  --env.spawn-radius 5.0 \
  --env.spawn-radius-min 0.0 \
  --env.spawn-height 5.0 \
  --env.init-vel-radius 0.0 \
  --env.init-omega-radius 0.0 \
  --env.room-size 6.0 \
  --env.ceiling-height 10.0 \
  --env.num-rays 24 \
  --env.ray-length 4.0 \
  --env.obstacle-mode "$obstacle_mode" \
  --env.route-walls 0 \
  --env.route-direct-obs 1 \
  --env.route-privileged-obs 0 \
  --env.route-extra-obs 0 \
  --env.route-wall-obs 1 \
  --env.route-perception-obs 0 \
  --env.route-target-obs-mode 3 \
  --env.route-goal-radius 0.75 \
  --env.route-goal-count 3 \
  --env.route-goal-count-min 0 \
  --env.route-target-mode 0 \
  --env.route-blocked-target-prob 0.0 \
  --env.route-target-z-min 2.0 \
  --env.route-target-z-max 8.0 \
  --env.route-reward-mode 4 \
  --env.route-step-cost 0.0 \
  --env.route-path-margin 0.35 \
  --env.route-clearance-radius 0.80 \
  --env.route-clearance-scale 0.60 \
  --env.route-progress-scale 25.0 \
  --env.route-vel-scale 0.02 \
  --env.route-arrival-radius 1.5 \
  --env.route-arrival-bonus 0.02 \
  --env.route-arrival-vel-scale 0.08 \
  --env.route-arrival-toward-scale 0.0 \
  --env.route-goal-speed 2.0 \
  --env.action-history-obs 1 \
  --env.mass-scale-min 0.95 \
  --env.mass-scale-max 1.05 \
  --env.thrust-scale-min 0.95 \
  --env.thrust-scale-max 1.05 \
  --env.drag-scale-min 0.95 \
  --env.drag-scale-max 1.05 \
  --env.motor-tau-scale-min 0.95 \
  --env.motor-tau-scale-max 1.05 \
  2>&1 | tee "$log_dir/train_s${seed}.log"
