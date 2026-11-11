#!/usr/bin/env bash
set -euo pipefail

seed="${AFL026_SEED:-88}"
scenario="${AFL026_SCENARIO:-reference_waypoint_p0_reachability_critic}"
run_id="${AFL026_RUN_ID:-$(date +%s)}"
updates="${AFL026_UPDATES:-610}"
agents="${AFL026_TOTAL_AGENTS:-2048}"
buffers="${AFL026_NUM_BUFFERS:-4}"
threads="${AFL026_NUM_THREADS:-1}"
horizon="${AFL026_HORIZON:-32}"
ppo_epochs="${AFL026_PPO_EPOCHS:-1}"
save_interval="${AFL026_SAVE_INTERVAL:-50}"
print_interval="${AFL026_PRINT_INTERVAL:-5}"
max_steps="${AFL026_MAX_STEPS:-1024}"
obstacle_mode="${AFL026_OBSTACLE_MODE:-34}"
route_walls="${AFL026_ROUTE_WALLS:-0}"
eval_episodes="${AFL026_EVAL_EPISODES:-4096}"
max_eval_steps="${AFL026_MAX_EVAL_STEPS:-30000}"
cpu="${AFL026_CPU:-0}"

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
  train_mode_args+=(--cpu)
fi
PUFFERLIB_CPU_ADVANTAGE="${PUFFERLIB_CPU_ADVANTAGE:-1}" "${build_cmd[@]}"

output_dir="experiments/agile/afl026/${scenario}/quadrotor/${run_id}"
log_dir="logs/agile/afl026/${scenario}"
mkdir -p "$output_dir" "$log_dir"

echo "afl026 ${scenario} train seed=${seed} updates=${updates} agents=${agents} horizon=${horizon}"
"$python_bin" scripts/quadrotor4_asym_ppo.py \
  "${train_mode_args[@]}" \
  --task route \
  --seed "$seed" \
  --output "$output_dir" \
  --checkpoint-prefix afl026_asym_ppo \
  --updates "$updates" \
  --horizon "$horizon" \
  --ppo-epochs "$ppo_epochs" \
  --total-agents "$agents" \
  --num-buffers "$buffers" \
  --num-threads "$threads" \
  --max-steps "$max_steps" \
  --save-interval "$save_interval" \
  --print-interval "$print_interval" \
  --eval-after \
  --episodes "$eval_episodes" \
  --max-eval-steps "$max_eval_steps" \
  --action-mode 4 \
  --dynamics-profile 1 \
  --physics-dt 0.002 \
  --physics-substeps 5 \
  --spawn-radius 5.0 \
  --spawn-radius-min 0.0 \
  --spawn-height 5.0 \
  --room-size 6.0 \
  --ceiling-height 10.0 \
  --num-rays 24 \
  --ray-length 4.0 \
  --obstacle-mode "$obstacle_mode" \
  --route-walls "$route_walls" \
  --route-goal-count 3 \
  --route-goal-count-min 0 \
  --route-goal-radius 0.75 \
  --route-path-margin 0.35 \
  --route-target-mode 0 \
  --route-blocked-target-prob 0.0 \
  --target-z-min 2.0 \
  --target-z-max 8.0 \
  --route-target-obs-mode 3 \
  --route-reward-mode 4 \
  --route-step-cost 0.0 \
  --route-progress-scale 25.0 \
  --route-clearance-radius 0.80 \
  --route-clearance-scale 0.60 \
  --route-vel-scale 0.02 \
  --route-arrival-radius 1.5 \
  --route-arrival-bonus 0.02 \
  --route-arrival-vel-scale 0.08 \
  --route-arrival-toward-scale 0.0 \
  --route-goal-speed 2.0 \
  --mass-scale-min 0.95 \
  --mass-scale-max 1.05 \
  --thrust-scale-min 0.95 \
  --thrust-scale-max 1.05 \
  --drag-scale-min 0.95 \
  --drag-scale-max 1.05 \
  --motor-tau-scale-min 0.95 \
  --motor-tau-scale-max 1.05 \
  --network MLP \
  --critic-network MLP \
  --hidden-size 128 \
  --num-layers 4 \
  --critic-num-layers 4 \
  --action-std 1.0 \
  --actor-lr "${AFL026_ACTOR_LR:-0.0001}" \
  --critic-lr "${AFL026_CRITIC_LR:-0.001}" \
  --gamma 0.981094 \
  --gae-lambda 0.883828 \
  --clip-coef 0.067834 \
  --ent-coef 9.62597e-05 \
  --kl-coef 0.0 \
  --max-grad-norm 0.692201 \
  2>&1 | tee "$log_dir/train_s${seed}_${run_id}.log"

echo "afl026 output_dir=${output_dir}"
