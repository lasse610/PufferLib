#!/usr/bin/env python3
import argparse
import sys

from pufferlib.pufferl import _load_weights_if_requested, _resolve_backend, unroll_nested_dict

from quadrotor4_eval_headless import (
    REFERENCE_WAYPOINT_PROFILE,
    load_quadrotor_config,
    parse_obstacle_mode,
    set_if_not_none,
)


def configure_args(parsed):
    args = load_quadrotor_config()
    args["vec"]["total_agents"] = parsed.total_agents
    args["vec"]["num_buffers"] = parsed.num_buffers
    args["vec"]["num_threads"] = parsed.num_threads
    args["train"]["total_timesteps"] = parsed.total_timesteps
    args["train"]["horizon"] = parsed.horizon
    args["train"]["learning_rate"] = parsed.learning_rate
    args["train"]["gamma"] = parsed.gamma
    args["train"]["gae_lambda"] = parsed.gae_lambda
    args["train"]["replay_ratio"] = parsed.replay_ratio
    args["train"]["clip_coef"] = parsed.clip_coef
    args["train"]["vf_coef"] = parsed.vf_coef
    args["train"]["vf_clip_coef"] = parsed.vf_clip_coef
    args["train"]["max_grad_norm"] = parsed.max_grad_norm
    args["train"]["ent_coef"] = parsed.ent_coef
    args["train"]["action_std"] = parsed.action_std
    args["train"]["train_action_std"] = parsed.train_action_std
    args["train"]["reward_clip"] = parsed.reward_clip
    args["train"]["minibatch_size"] = parsed.minibatch_size
    args["train"]["vtrace_rho_clip"] = parsed.vtrace_rho_clip
    args["train"]["vtrace_c_clip"] = parsed.vtrace_c_clip
    args["train"]["prio_alpha"] = parsed.prio_alpha
    args["train"]["prio_beta0"] = parsed.prio_beta0
    args["train"]["beta1"] = parsed.beta1
    args["train"]["beta2"] = parsed.beta2
    args["train"]["eps"] = parsed.eps
    args["env"]["seed"] = parsed.seed
    args["train"]["seed"] = parsed.seed
    args["torch"]["network"] = "MLP"
    args["policy"]["hidden_size"] = 128
    args["policy"]["num_layers"] = 4
    args["policy"]["expansion_factor"] = 1
    args["policy"]["use_bias"] = 1
    args["env"].update(REFERENCE_WAYPOINT_PROFILE)

    env = args["env"]
    env["max_steps"] = parsed.max_steps
    env["spawn_radius"] = parsed.spawn_radius
    env["spawn_radius_min"] = parsed.spawn_radius_min
    env["spawn_height"] = parsed.spawn_height
    env["room_size"] = parsed.room_size
    env["ceiling_height"] = parsed.ceiling_height
    env["obstacle_mode"] = parsed.obstacle_mode
    env["route_goal_count"] = parsed.route_goal_count
    env["route_goal_count_min"] = parsed.route_goal_count_min
    env["route_goal_radius"] = parsed.route_goal_radius
    env["route_reward_mode"] = parsed.route_reward_mode
    env["route_target_obs_mode"] = parsed.route_target_obs_mode
    env["route_step_cost"] = parsed.route_step_cost
    env["route_target_z_min"] = parsed.route_target_z_min
    env["route_target_z_max"] = parsed.route_target_z_max
    env["route_progress_scale"] = parsed.route_progress_scale
    env["route_vel_scale"] = parsed.route_vel_scale
    env["route_arrival_radius"] = parsed.route_arrival_radius
    env["route_arrival_bonus"] = parsed.route_arrival_bonus
    env["route_arrival_vel_scale"] = parsed.route_arrival_vel_scale
    env["route_arrival_toward_scale"] = parsed.route_arrival_toward_scale
    env["route_goal_speed"] = parsed.route_goal_speed
    env["route_target_wall_margin"] = parsed.route_target_wall_margin
    env["route_clearance_radius"] = parsed.route_clearance_radius
    env["route_clearance_scale"] = parsed.route_clearance_scale
    env["route_clearance_speed_scale"] = parsed.route_clearance_speed_scale
    env["route_clearance_speed_min"] = parsed.route_clearance_speed_min
    env["route_clearance_speed_full"] = parsed.route_clearance_speed_full
    env["route_walls"] = parsed.route_walls
    env["mass_scale_min"] = parsed.mass_scale_min
    env["mass_scale_max"] = parsed.mass_scale_max
    env["thrust_scale_min"] = parsed.thrust_scale_min
    env["thrust_scale_max"] = parsed.thrust_scale_max
    env["drag_scale_min"] = parsed.drag_scale_min
    env["drag_scale_max"] = parsed.drag_scale_max
    env["motor_tau_scale_min"] = parsed.motor_tau_scale_min
    env["motor_tau_scale_max"] = parsed.motor_tau_scale_max
    set_if_not_none(args, "reset_state", parsed.reset_state)
    return args


def print_logs(label, logs):
    flat = dict(unroll_nested_dict(logs))
    keys = (
        "agent_steps",
        "epoch",
        "env/n",
        "env/success",
        "env/score",
        "env/crashed",
        "env/obstacle_hit",
        "env/ground_hit",
        "env/timeout",
        "env/avg_vel",
        "env/route_goals",
        "env/min_clearance",
        "env/action_saturation",
        "env/action_delta",
        "loss/entropy",
        "loss/policy_loss",
        "loss/value_loss",
        "loss/approx_kl",
        "loss/clipfrac",
    )
    print(label)
    for key in keys:
        if key in flat:
            print(f"{key} {flat[key]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--rollouts-before", type=int, default=1)
    parser.add_argument("--updates", type=int, default=0)
    parser.add_argument("--rollouts-after", type=int, default=1)
    parser.add_argument("--total-agents", type=int, default=2048)
    parser.add_argument("--num-buffers", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--clip-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--vf-clip-coef", type=float, default=0.20)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--action-std", type=float, default=0.05)
    parser.add_argument("--train-action-std", type=int, default=0)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--beta1", type=float, default=0.914207)
    parser.add_argument("--beta2", type=float, default=0.99988)
    parser.add_argument("--eps", type=float, default=1e-14)
    parser.add_argument("--vtrace-rho-clip", type=float, default=3.32822)
    parser.add_argument("--vtrace-c-clip", type=float, default=4.37219)
    parser.add_argument("--prio-alpha", type=float, default=0.427293)
    parser.add_argument("--prio-beta0", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--spawn-radius", type=float, default=5.0)
    parser.add_argument("--spawn-radius-min", type=float, default=0.0)
    parser.add_argument("--spawn-height", type=float, default=5.0)
    parser.add_argument("--room-size", type=float, default=6.0)
    parser.add_argument("--ceiling-height", type=float, default=10.0)
    parser.add_argument("--obstacle-mode", type=parse_obstacle_mode, default=57)
    parser.add_argument("--route-goal-count", type=int, default=3)
    parser.add_argument("--route-goal-count-min", type=int, default=0)
    parser.add_argument("--route-goal-radius", type=float, default=0.75)
    parser.add_argument("--route-reward-mode", type=int, default=4)
    parser.add_argument("--route-target-obs-mode", type=int, default=3)
    parser.add_argument("--route-step-cost", type=float, default=0.0)
    parser.add_argument("--route-target-z-min", type=float, default=2.0)
    parser.add_argument("--route-target-z-max", type=float, default=8.0)
    parser.add_argument("--route-progress-scale", type=float, default=25.0)
    parser.add_argument("--route-vel-scale", type=float, default=0.02)
    parser.add_argument("--route-arrival-radius", type=float, default=1.5)
    parser.add_argument("--route-arrival-bonus", type=float, default=0.02)
    parser.add_argument("--route-arrival-vel-scale", type=float, default=0.08)
    parser.add_argument("--route-arrival-toward-scale", type=float, default=0.0)
    parser.add_argument("--route-goal-speed", type=float, default=3.0)
    parser.add_argument("--route-target-wall-margin", type=float, default=0.75)
    parser.add_argument("--route-clearance-radius", type=float, default=0.80)
    parser.add_argument("--route-clearance-scale", type=float, default=0.60)
    parser.add_argument("--route-clearance-speed-scale", type=float, default=0.02)
    parser.add_argument("--route-clearance-speed-min", type=float, default=0.90)
    parser.add_argument("--route-clearance-speed-full", type=float, default=1.60)
    parser.add_argument("--route-walls", type=int, default=0)
    parser.add_argument("--mass-scale-min", type=float, default=0.0)
    parser.add_argument("--mass-scale-max", type=float, default=0.0)
    parser.add_argument("--thrust-scale-min", type=float, default=0.0)
    parser.add_argument("--thrust-scale-max", type=float, default=0.0)
    parser.add_argument("--drag-scale-min", type=float, default=0.0)
    parser.add_argument("--drag-scale-max", type=float, default=0.0)
    parser.add_argument("--motor-tau-scale-min", type=float, default=0.0)
    parser.add_argument("--motor-tau-scale-max", type=float, default=0.0)
    parser.add_argument("--reset-state", type=int, default=None)
    parser.add_argument("--slowly", action="store_true")
    parsed = parser.parse_args()

    args = configure_args(parsed)
    if parsed.slowly:
        args["slowly"] = True

    backend = _resolve_backend(args)
    pufferl = backend.create_pufferl(args)
    try:
        _load_weights_if_requested(backend, pufferl, args, parsed.checkpoint)
        print(f"backend {backend.__name__ if hasattr(backend, '__name__') else backend}")
        print(f"reset_state {args['reset_state']}")
        print(f"agents {args['vec']['total_agents']} buffers {args['vec']['num_buffers']} horizon {args['train']['horizon']}")
        for idx in range(parsed.rollouts_before):
            backend.rollouts(pufferl)
            print_logs(f"before_rollout {idx + 1}", backend.log(pufferl))
        for idx in range(parsed.updates):
            backend.train(pufferl)
            print_logs(f"after_update {idx + 1}", backend.log(pufferl))
            if idx + 1 < parsed.updates:
                backend.rollouts(pufferl)
        for idx in range(parsed.rollouts_after):
            backend.rollouts(pufferl)
            print_logs(f"after_rollout {idx + 1}", backend.log(pufferl))
    finally:
        backend.close(pufferl)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
