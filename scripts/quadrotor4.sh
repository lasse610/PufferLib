#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/quadrotor4.sh smoke
  scripts/quadrotor4.sh build-cpu
  scripts/quadrotor4.sh train-torch [puffer args...]
  scripts/quadrotor4.sh build-gpu
  scripts/quadrotor4.sh train-gpu [puffer args...]
  scripts/quadrotor4.sh agile-v0-train [rays|depth20|randomized|all]
  scripts/quadrotor4.sh agile-v0-eval [checkpoint|latest] [eval args...]
  scripts/quadrotor4.sh agile-v0-eval-checkpoints [run|latest] [eval args...]
  scripts/quadrotor4.sh drone-hover-train
  scripts/quadrotor4.sh drone-hover-eval [checkpoint|latest] [eval args...]
  scripts/quadrotor4.sh drone-hover-eval-checkpoints [run|latest] [eval args...]
  scripts/quadrotor4.sh drone-reference-train
  scripts/quadrotor4.sh drone-reference-eval [checkpoint|latest] [eval args...]
  scripts/quadrotor4.sh drone-reference-eval-checkpoints [run|latest] [eval args...]
  scripts/quadrotor4.sh reference-waypoint-train
  scripts/quadrotor4.sh reference-waypoint-p0-reachability-train
  scripts/quadrotor4.sh reference-waypoint-native-reachability-train
  scripts/quadrotor4.sh reference-waypoint-eval [checkpoint|latest] [eval args...]
  scripts/quadrotor4.sh reference-waypoint-eval-checkpoints [run|latest] [eval args...]
  scripts/quadrotor4.sh reference-waypoint-visual [checkpoint] [visual args...]
  scripts/quadrotor4.sh reference-waypoint-rollout-probe [checkpoint] [probe args...]
  scripts/quadrotor4.sh reference-waypoint-bottleneck-probe [checkpoint|afl038] [probe args...]
  scripts/quadrotor4.sh reference-waypoint-bottleneck-trace [checkpoint|afl038] [trace args...]
  scripts/quadrotor4.sh explore-v0-train [seed] [timesteps]
  scripts/quadrotor4.sh explore-v0-obstacle-train [qdr-name] [obstacle-mode] [seed] [timesteps]
  scripts/quadrotor4.sh explore-v0-memory-train [seed] [timesteps]
  scripts/quadrotor4.sh explore-v0-frontier-progress-train [seed] [timesteps]
  scripts/quadrotor4.sh explore-v0-frontier-target-train [seed] [timesteps]
  scripts/quadrotor4.sh explore-v0-frontier-path-train [seed] [timesteps]
  scripts/quadrotor4.sh explore-v0-frontier-path-progress-train [seed] [timesteps]
  scripts/quadrotor4.sh explore-v0-layout-train [seed] [timesteps]
  scripts/quadrotor4.sh explore-v0-eval [checkpoint|latest] [eval args...]
  scripts/quadrotor4.sh explore-v0-eval-checkpoints [run|latest] [eval args...]
  scripts/quadrotor4.sh explore-v0-gauntlet [checkpoint|latest] [gauntlet args...]
  scripts/quadrotor4.sh explore-v0-maze-probe [probe args...]
  scripts/quadrotor4.sh explore-v0-frontier-teacher [probe args...]
  scripts/quadrotor4.sh explore-v0-frontier-obs-probe [probe args...]
  scripts/quadrotor4.sh explore-v0-depth-camera-probe [probe args...]
  scripts/quadrotor4.sh explore-v0-frontier-distill [distill args...]
  scripts/quadrotor4.sh explore-v0-target-distill [distill args...]
  scripts/quadrotor4.sh explore-v0-target-distill-eval [checkpoint] [eval args...]
  scripts/quadrotor4.sh explore-v0-target-balanced-eval [checkpoint] [eval args...]
  scripts/quadrotor4.sh explore-v0-target-oracle-eval [eval args...]
  scripts/quadrotor4.sh explore-v0-onboard-frontier-verify [verify args...]
  scripts/quadrotor4.sh explore-v0-planner-probe [probe args...]
  scripts/quadrotor4.sh explore-v0-planner-state-verify [verify args...]
  scripts/quadrotor4.sh explore-v0-viewpoint-candidate-probe [probe args...]
  scripts/quadrotor4.sh explore-v0-onboard-hardstate-probe [probe args...]
  scripts/quadrotor4.sh explore-v0-onboard-visual [visual args...]
  scripts/quadrotor4.sh explore-v0-target-diagnose [checkpoint] [diagnostic args...]
  scripts/quadrotor4.sh explore-v0-target-hardstate-probe [checkpoint] [probe args...]
  scripts/quadrotor4.sh explore-v0-target-hardstate-train [train args...]
  scripts/quadrotor4.sh explore-v0-target-horizon-probe [checkpoint] [probe args...]
  scripts/quadrotor4.sh explore-v0-target-visual [checkpoint] [visual args...]
  scripts/quadrotor4.sh explore-v0-recovery-policy-train [checkpoint] [train args...]
  scripts/quadrotor4.sh explore-v0-recovery-replay [checkpoint] [replay args...]
  scripts/quadrotor4.sh eval-headless [checkpoint|latest] [eval args...]
  scripts/quadrotor4.sh eval-checkpoints [run|latest] [eval args...]
  scripts/quadrotor4.sh visual-eval [checkpoint] [visual args...]
  scripts/quadrotor4.sh distill-train [distill args...]
  scripts/quadrotor4.sh distill-train-seq [distill args...]
  scripts/quadrotor4.sh distill-eval [checkpoint|teacher] [distill eval args...]
  scripts/quadrotor4.sh distill-convert [input.pt] [output.bin] [convert args...]
  scripts/quadrotor4.sh asym-ppo [asym args...]
  scripts/quadrotor4.sh failure-recovery [recovery args...]
  scripts/quadrotor4.sh replay-probe [checkpoint] [probe args...]
  scripts/quadrotor4.sh hardstate-teacher-probe [checkpoint] [probe args...]

Environment:
  PYTHON             Python executable. Defaults to .venv, parent .venv, then python.
  QUAD_TIMESTEPS     Default train timesteps for train-gpu/train-torch.
EOF
}

if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
elif [[ -x "../../.venv/bin/python" ]]; then
  python_bin="../../.venv/bin/python"
else
  python_bin="python"
fi

python_dir="$(dirname "$(command -v "$python_bin" 2>/dev/null || echo "$python_bin")")"
export PATH="$python_dir:$PATH"
export PYTHONPATH="${PYTHONPATH:-.}"

cmd="${1:-}"
if [[ -z "$cmd" || "$cmd" == "-h" || "$cmd" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

run_explore_v0_train() {
  local seed="$1"
  local timesteps="$2"
  local checkpoint_dir="$3"
  local log_dir="$4"
  local obstacle_mode="$5"
  local explore_memory_obs="${6:-0}"
  local frontier_progress_scale="${7:-0.0}"

  bash build.sh quadrotor
  "$python_bin" -m pufferlib.pufferl train quadrotor \
    --train.total-timesteps "$timesteps" \
    --checkpoint-dir "$checkpoint_dir" \
    --log-dir "$log_dir" \
    --checkpoint-interval 5 \
    --eval-episodes 4096 \
    --env.seed "$seed" \
    --train.seed "$seed" \
    --torch.network MinGRU \
    --policy.use-bias 1 \
    --env.task-mode 3 \
    --env.action-mode 1 \
    --env.max-steps 1024 \
    --env.max-vel 0.7 \
    --env.max-z-vel 0.25 \
    --env.spawn-radius 2.5 \
    --env.spawn-radius-min 0.0 \
    --env.spawn-height 1.5 \
    --env.room-size 3.0 \
    --env.num-rays 24 \
    --env.ray-length 4.0 \
    --env.obstacle-mode "$obstacle_mode" \
    --env.explore-goal-fraction 0.85 \
    --env.explore-memory-obs "$explore_memory_obs" \
    --env.explore-frontier-progress-scale "$frontier_progress_scale" \
    --env.route-wall-obs 1 \
    --env.action-history-obs 1
}

case "$cmd" in
  build-cpu)
    bash build.sh quadrotor --cpu
    ;;
  smoke)
    bash build.sh quadrotor --cpu
    "$python_bin" -c "import numpy as np; import pufferlib.models as m; import pufferlib._C as C; from pufferlib.pufferl import load_config; args=load_config('quadrotor'); args['vec']['total_agents']=8; args['vec']['num_buffers']=1; v=C.create_vec(args, 0); print(C.env_name, hasattr(m, 'MinGRU'), v.obs_size, v.num_atns, v.act_sizes, v.obs_dtype); v.reset(); a=np.zeros((8, 4), dtype='float32'); v.cpu_step(a.__array_interface__['data'][0]); v.close()"
    "$python_bin" -m pufferlib.pufferl train quadrotor --slowly \
      --vec.total-agents 64 \
      --vec.num-threads 1 \
      --train.horizon 16 \
      --train.minibatch-size 1024 \
      --train.total-timesteps 2048 \
      --checkpoint-interval 999 \
      --eval-episodes 16
    ;;
  build-gpu)
    bash build.sh quadrotor
    ;;
  train-gpu)
    timesteps="${QUAD_TIMESTEPS:-50000000}"
    bash build.sh quadrotor
    "$python_bin" -m pufferlib.pufferl train quadrotor \
      --train.total-timesteps "$timesteps" "$@"
    ;;
  agile-v0-train)
    PYTHON="$python_bin" scripts/quadrotor4_afl001_agile_v0.sh "$@"
    ;;
  agile-v0-eval)
    checkpoint="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_headless.py "$checkpoint" \
      --agile-v0 "$@"
    ;;
  agile-v0-eval-checkpoints)
    run="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_checkpoints.py "$run" \
      --agile-v0 "$@"
    ;;
  drone-hover-train)
    PYTHON="$python_bin" scripts/quadrotor4_afl007_drone_hover.sh "$@"
    ;;
  drone-hover-eval)
    checkpoint="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_headless.py "$checkpoint" \
      --drone-hover "$@"
    ;;
  drone-hover-eval-checkpoints)
    run="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_checkpoints.py "$run" \
      --drone-hover "$@"
    ;;
  drone-reference-train)
    PYTHON="$python_bin" scripts/quadrotor4_afl008_drone_reference.sh "$@"
    ;;
  drone-reference-eval)
    checkpoint="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_headless.py "$checkpoint" \
      --drone-reference "$@"
    ;;
  drone-reference-eval-checkpoints)
    run="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_checkpoints.py "$run" \
      --drone-reference "$@"
    ;;
  reference-waypoint-train)
    PYTHON="$python_bin" scripts/quadrotor4_afl009_reference_waypoint.sh "$@"
    ;;
  reference-waypoint-p0-reachability-train)
    PYTHON="$python_bin" scripts/quadrotor4_afl026_p0_reachability_critic.sh "$@"
    ;;
  reference-waypoint-native-reachability-train)
    PYTHON="$python_bin" scripts/quadrotor4_afl027_native_reachability_critic.sh "$@"
    ;;
  reference-waypoint-eval)
    checkpoint="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_headless.py "$checkpoint" \
      --reference-waypoint \
      --network MLP \
      --hidden-size 128 \
      --num-layers 4 \
      --use-bias 1 \
      "$@"
    ;;
  reference-waypoint-eval-checkpoints)
    run="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_checkpoints.py "$run" \
      --reference-waypoint \
      --network MLP \
      --hidden-size 128 \
      --num-layers 4 \
      --use-bias 1 \
      "$@"
    ;;
  reference-waypoint-visual)
    checkpoint="${1:?missing checkpoint path}"
    shift
    "$python_bin" scripts/quadrotor4_reference_waypoint_visual.py "$checkpoint" "$@"
    ;;
  reference-waypoint-rollout-probe)
    checkpoint="${1:?missing checkpoint path}"
    shift
    "$python_bin" scripts/quadrotor4_rollout_probe.py "$checkpoint" "$@"
    ;;
  reference-waypoint-bottleneck-probe)
    checkpoint="${1:-afl038}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_bottleneck_probe.py "$checkpoint" "$@"
    ;;
  reference-waypoint-bottleneck-trace)
    checkpoint="${1:-afl038}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_bottleneck_trace.py "$checkpoint" "$@"
    ;;
  explore-v0-train)
    seed="${1:-88}"
    timesteps="${2:-${QUAD_TIMESTEPS:-25000000}}"
    run_explore_v0_train "$seed" "$timesteps" experiments/qdr156 logs/qdr156 5
    ;;
  explore-v0-obstacle-train)
    qdr_name="${1:?missing qdr-name, e.g. qdr158}"
    obstacle_mode="${2:?missing obstacle-mode, e.g. 4 or layout_mixed id}"
    seed="${3:-88}"
    timesteps="${4:-${QUAD_TIMESTEPS:-25000000}}"
    run_explore_v0_train "$seed" "$timesteps" "experiments/$qdr_name" "logs/$qdr_name" "$obstacle_mode"
    ;;
  explore-v0-layout-train)
    seed="${1:-88}"
    timesteps="${2:-${QUAD_TIMESTEPS:-25000000}}"
    run_explore_v0_train "$seed" "$timesteps" experiments/qdr157 logs/qdr157 7
    ;;
  explore-v0-memory-train)
    seed="${1:-88}"
    timesteps="${2:-${QUAD_TIMESTEPS:-25000000}}"
    run_explore_v0_train "$seed" "$timesteps" experiments/qdr161 logs/qdr161 4 1
    ;;
  explore-v0-frontier-progress-train)
    seed="${1:-88}"
    timesteps="${2:-${QUAD_TIMESTEPS:-25000000}}"
    run_explore_v0_train "$seed" "$timesteps" experiments/qdr162 logs/qdr162 4 1 1.0
    ;;
  explore-v0-frontier-target-train)
    seed="${1:-88}"
    timesteps="${2:-${QUAD_TIMESTEPS:-25000000}}"
    run_explore_v0_train "$seed" "$timesteps" experiments/qdr163 logs/qdr163 4 2 1.0
    ;;
  explore-v0-frontier-path-train)
    seed="${1:-88}"
    timesteps="${2:-${QUAD_TIMESTEPS:-25000000}}"
    run_explore_v0_train "$seed" "$timesteps" experiments/qdr165 logs/qdr165 4 3 1.0
    ;;
  explore-v0-frontier-path-progress-train)
    seed="${1:-88}"
    timesteps="${2:-${QUAD_TIMESTEPS:-25000000}}"
    run_explore_v0_train "$seed" "$timesteps" experiments/qdr166 logs/qdr166 4 3 1.0
    ;;
  explore-v0-eval)
    checkpoint="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_headless.py "$checkpoint" \
      --explore-v0 "$@"
    ;;
  explore-v0-eval-checkpoints)
    run="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_checkpoints.py "$run" \
      --explore-v0 "$@"
    ;;
  explore-v0-gauntlet)
    checkpoint="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_explore_gauntlet.py "$checkpoint" "$@"
    ;;
  explore-v0-maze-probe)
    "$python_bin" scripts/quadrotor4_maze_frontier_probe.py "$@"
    ;;
  explore-v0-frontier-teacher)
    "$python_bin" scripts/quadrotor4_frontier_teacher_probe.py "$@"
    ;;
  explore-v0-frontier-obs-probe)
    "$python_bin" scripts/quadrotor4_frontier_obs_probe.py "$@"
    ;;
  explore-v0-depth-camera-probe)
    "$python_bin" scripts/quadrotor4_depth_camera_probe.py "$@"
    ;;
  explore-v0-frontier-distill)
    "$python_bin" scripts/quadrotor4_frontier_distill.py "$@"
    ;;
  explore-v0-target-distill)
    "$python_bin" scripts/quadrotor4_frontier_target_distill.py train "$@"
    ;;
  explore-v0-target-distill-eval)
    checkpoint="${1:?missing checkpoint}"
    shift
    "$python_bin" scripts/quadrotor4_frontier_target_distill.py eval "$checkpoint" "$@"
    ;;
  explore-v0-target-balanced-eval)
    checkpoint="${1:?missing checkpoint, or pass --oracle as the first argument}"
    shift
    "$python_bin" scripts/quadrotor4_target_balanced_eval.py "$checkpoint" "$@"
    ;;
  explore-v0-target-oracle-eval)
    "$python_bin" scripts/quadrotor4_frontier_target_distill.py oracle-eval "$@"
    ;;
  explore-v0-onboard-frontier-verify)
    "$python_bin" scripts/quadrotor4_onboard_frontier_verify.py "$@"
    ;;
  explore-v0-planner-probe)
    "$python_bin" scripts/quadrotor4_planner_probe.py "$@"
    ;;
  explore-v0-planner-state-verify)
    "$python_bin" scripts/quadrotor4_planner_state_verify.py "$@"
    ;;
  explore-v0-viewpoint-candidate-probe)
    "$python_bin" scripts/quadrotor4_viewpoint_candidate_probe.py "$@"
    ;;
  explore-v0-onboard-hardstate-probe)
    "$python_bin" scripts/quadrotor4_onboard_hardstate_probe.py "$@"
    ;;
  explore-v0-onboard-visual)
    "$python_bin" scripts/quadrotor4_target_visual_eval.py \
      --controller oracle \
      --explore-frontier-target-mode onboard \
      --explore-reachable-coverage \
      --mode procedural_indoor_hard_p1 \
      --room-size 6.0 \
      --explore-goal-fraction 0.98 \
      --max-steps 4096 \
      --max-eval-steps 4096 \
      --restart-on-stall-steps 3000 \
      "$@"
    ;;
  explore-v0-target-diagnose)
    checkpoint="${1:?missing checkpoint}"
    shift
    "$python_bin" scripts/quadrotor4_frontier_target_distill.py diagnose "$checkpoint" "$@"
    ;;
  explore-v0-target-hardstate-probe)
    checkpoint="${1:?missing checkpoint}"
    shift
    "$python_bin" scripts/quadrotor4_target_hardstate_probe.py "$checkpoint" "$@"
    ;;
  explore-v0-target-hardstate-train)
    "$python_bin" scripts/quadrotor4_target_hardstate_train.py "$@"
    ;;
  explore-v0-target-horizon-probe)
    "$python_bin" scripts/quadrotor4_target_horizon_probe.py "$@"
    ;;
  explore-v0-target-visual)
    checkpoint=""
    if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
      checkpoint="$1"
      shift
    fi
    if [[ -n "$checkpoint" ]]; then
      "$python_bin" scripts/quadrotor4_target_visual_eval.py "$checkpoint" "$@"
    else
      "$python_bin" scripts/quadrotor4_target_visual_eval.py --controller oracle "$@"
    fi
    ;;
  explore-v0-recovery-policy-train)
    checkpoint="${1:?missing checkpoint}"
    shift
    "$python_bin" scripts/quadrotor4_recovery_policy_train.py "$checkpoint" "$@"
    ;;
  explore-v0-recovery-replay)
    checkpoint="${1:?missing checkpoint}"
    shift
    "$python_bin" scripts/quadrotor4_recovery_replay.py "$checkpoint" "$@"
    ;;
  train-torch)
    timesteps="${QUAD_TIMESTEPS:-50000000}"
    bash build.sh quadrotor --float
    "$python_bin" -m pufferlib.pufferl train quadrotor --slowly \
      --train.total-timesteps "$timesteps" "$@"
    ;;
  eval-headless)
    checkpoint="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_headless.py "$checkpoint" "$@"
    ;;
  eval-checkpoints)
    run="${1:-latest}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    "$python_bin" scripts/quadrotor4_eval_checkpoints.py "$run" "$@"
    ;;
  visual-eval)
    checkpoint=""
    if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
      checkpoint="$1"
      shift
    fi
    if [[ -n "$checkpoint" ]]; then
      "$python_bin" scripts/quadrotor4_visual_eval.py "$checkpoint" "$@"
    else
      "$python_bin" scripts/quadrotor4_visual_eval.py "$@"
    fi
    ;;
  distill-train)
    "$python_bin" scripts/quadrotor4_privileged_distill.py train "$@"
    ;;
  distill-train-seq)
    "$python_bin" scripts/quadrotor4_privileged_distill.py train-seq "$@"
    ;;
  distill-eval)
    checkpoint="${1:-}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    if [[ "$checkpoint" == "teacher" ]]; then
      "$python_bin" scripts/quadrotor4_privileged_distill.py eval --eval-policy teacher "$@"
    elif [[ -n "$checkpoint" ]]; then
      "$python_bin" scripts/quadrotor4_privileged_distill.py eval "$checkpoint" "$@"
    else
      usage
      exit 1
    fi
    ;;
  distill-convert)
    "$python_bin" scripts/quadrotor4_convert_distill_checkpoint.py "$@"
    ;;
  native-action-std-patch)
    "$python_bin" scripts/quadrotor4_patch_native_action_std.py "$@"
    ;;
  asym-ppo)
    "$python_bin" scripts/quadrotor4_asym_ppo.py "$@"
    ;;
  failure-recovery)
    "$python_bin" scripts/quadrotor4_failure_recovery.py "$@"
    ;;
  replay-probe)
    checkpoint=""
    if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
      checkpoint="$1"
      shift
    fi
    if [[ -n "$checkpoint" ]]; then
      "$python_bin" scripts/quadrotor4_replay_probe.py "$checkpoint" "$@"
    else
      "$python_bin" scripts/quadrotor4_replay_probe.py "$@"
    fi
    ;;
  hardstate-teacher-probe)
    checkpoint=""
    if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
      checkpoint="$1"
      shift
    fi
    if [[ -n "$checkpoint" ]]; then
      "$python_bin" scripts/quadrotor4_hardstate_teacher_probe.py "$checkpoint" "$@"
    else
      "$python_bin" scripts/quadrotor4_hardstate_teacher_probe.py "$@"
    fi
    ;;
  *)
    usage
    exit 1
    ;;
esac
