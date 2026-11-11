#!/usr/bin/env python3
import argparse
import sys
import torch


OBSTACLE_MODES = {
    "gate_slalom": 33,
    "gate_slalom_p0": 34,
    "gate_slalom_p1": 35,
    "gate_slalom_p2": 36,
    "gate_slalom_p3": 37,
    "gate_slalom_easy_mix": 38,
    "gate_slalom_p1_p0_mix": 39,
    "p1_p0_mix": 39,
    "agile_geometry": 40,
    "agile_course": 40,
    "creative_course": 40,
    "agile_geometry_p0": 41,
    "agile_geometry_p1": 42,
    "agile_geometry_p2": 43,
    "agile_geometry_p3": 44,
    "agile_geometry_intro": 45,
    "agile_course_intro": 45,
    "agile_pillar_weave": 46,
    "agile_pillars": 46,
    "agile_pillar_l0": 47,
    "agile_pillar_l1": 48,
    "agile_pillar_l2": 49,
    "agile_pillar_l3": 50,
    "agile_pillar_easy_mix": 51,
    "agile_pillar_curriculum": 52,
    "agile_pillar_l2_open": 53,
    "agile_pillar_l2_marker_tiny": 54,
    "agile_pillar_l2_marker_small": 55,
    "agile_pillar_l2_marker_large": 56,
    "agile_pillar_bottleneck_curriculum": 57,
}


def parse_obstacle_mode(value):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return OBSTACLE_MODES[value]
    except KeyError as e:
        valid = ", ".join(sorted(OBSTACLE_MODES))
        raise argparse.ArgumentTypeError(
            f"unknown obstacle mode {value!r}; valid names: {valid}"
        ) from e


def configure_args(parsed):
    from pufferlib.pufferl import load_config

    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        args = load_config("quadrotor")
    finally:
        sys.argv = old_argv
    args["slowly"] = True
    args["load_model_path"] = None
    args["reset_state"] = False
    args["render_mode"] = "human" if parsed.render else "None"
    args["train"]["horizon"] = 1
    args["vec"]["total_agents"] = 1
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 1
    args["torch"]["network"] = "MLP"
    args["policy"]["hidden_size"] = 128
    args["policy"]["num_layers"] = 4
    args["policy"]["expansion_factor"] = 1
    args["policy"]["use_bias"] = 1

    args["env"].update({
        "seed": parsed.seed,
        "task_mode": 4,
        "action_mode": 4,
        "max_steps": parsed.max_steps,
        "action_repeat": 1,
        "dynamics_profile": 1,
        "physics_dt": 0.002,
        "physics_substeps": 5,
        "spawn_radius": 5.0,
        "spawn_radius_min": 0.0,
        "spawn_height": 5.0,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 6.0,
        "ceiling_height": 10.0,
        "num_rays": 24,
        "ray_length": 4.0,
        "obstacle_mode": parsed.obstacle_mode,
        "route_goal_radius": 0.75,
        "route_goal_count": 3,
        "route_goal_count_min": 0,
        "route_target_mode": 0,
        "route_blocked_target_prob": 0.0,
        "route_target_z_min": 2.0,
        "route_target_z_max": 8.0,
        "route_reward_mode": 4,
        "route_step_cost": 0.0,
        "route_walls": 0,
        "route_direct_obs": 1,
        "route_extra_obs": 0,
        "route_wall_obs": 1,
        "route_perception_obs": 0,
        "route_target_obs_mode": 3,
        "route_path_margin": 0.35,
        "route_clearance_radius": 0.80,
        "route_clearance_scale": 0.60,
        "route_clearance_speed_scale": parsed.route_clearance_speed_scale,
        "route_clearance_speed_min": parsed.route_clearance_speed_min,
        "route_clearance_speed_full": parsed.route_clearance_speed_full,
        "route_progress_scale": 25.0,
        "route_vel_scale": 0.02,
        "route_arrival_radius": 1.5,
        "route_arrival_bonus": 0.02,
        "route_arrival_vel_scale": 0.08,
        "route_arrival_toward_scale": 0.0,
        "route_goal_speed": parsed.route_goal_speed,
        "route_target_wall_margin": 0.75,
        "action_history_obs": 1,
        "mass_scale_min": parsed.mass_scale_min,
        "mass_scale_max": parsed.mass_scale_max,
        "thrust_scale_min": parsed.thrust_scale_min,
        "thrust_scale_max": parsed.thrust_scale_max,
        "drag_scale_min": parsed.drag_scale_min,
        "drag_scale_max": parsed.drag_scale_max,
        "motor_tau_scale_min": parsed.motor_tau_scale_min,
        "motor_tau_scale_max": parsed.motor_tau_scale_max,
    })
    return args


def reset_done_state(state, done):
    if not isinstance(state, tuple) or not torch.is_tensor(done) or not done.any():
        return state
    out = []
    for value in state:
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == done.shape[0]:
            value = value.clone()
            value[:, done] = 0
        out.append(value)
    return tuple(out)


def main():
    parser = argparse.ArgumentParser(description="Visual eval for AFL reference-waypoint checkpoints.")
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--max-steps", type=int, default=1536)
    parser.add_argument("--route-clearance-speed-scale", type=float, default=0.0)
    parser.add_argument("--route-clearance-speed-min", type=float, default=0.0)
    parser.add_argument("--route-clearance-speed-full", type=float, default=0.0)
    parser.add_argument("--route-goal-speed", type=float, default=2.0)
    parser.add_argument("--obstacle-mode", type=parse_obstacle_mode, default=35)
    parser.add_argument("--mass-scale-min", type=float, default=0.95)
    parser.add_argument("--mass-scale-max", type=float, default=1.05)
    parser.add_argument("--thrust-scale-min", type=float, default=0.95)
    parser.add_argument("--thrust-scale-max", type=float, default=1.05)
    parser.add_argument("--drag-scale-min", type=float, default=0.95)
    parser.add_argument("--drag-scale-max", type=float, default=1.05)
    parser.add_argument("--motor-tau-scale-min", type=float, default=0.95)
    parser.add_argument("--motor-tau-scale-max", type=float, default=1.05)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parsed = parser.parse_args()

    from pufferlib.pufferl import _resolve_backend
    from pufferlib.torch_pufferl import _actions_for_vec_step

    args = configure_args(parsed)
    backend = _resolve_backend(args)
    pufferl = backend.create_pufferl(args)
    try:
        backend.load_weights(pufferl, parsed.checkpoint)
        policy = pufferl.policy
        policy.eval()
        state = policy.initial_state(pufferl.total_agents, pufferl.device)
        completed = 0

        while completed < parsed.episodes:
            if parsed.render:
                backend.render(pufferl, 0)

            with torch.no_grad():
                logits, _, state = policy.forward_eval(pufferl.vec_obs, state)
                if isinstance(logits, torch.distributions.Normal):
                    actions = logits.sample() if parsed.stochastic else logits.mean
                else:
                    actions = logits

            actions = _actions_for_vec_step(actions)
            pufferl._vec.cpu_step(actions.data_ptr())
            done = pufferl.vec_terminals > 0.5
            state = reset_done_state(state, done)

            logs = pufferl._vec.log()
            n = int(logs.get("n", 0))
            for _ in range(n):
                completed += 1
                print(
                    f"episode={completed} "
                    f"success={logs.get('success', 0.0):.4f} "
                    f"crashed={logs.get('crashed', 0.0):.4f} "
                    f"obstacle={logs.get('obstacle_hit', 0.0):.4f} "
                    f"ground={logs.get('ground_hit', 0.0):.4f} "
                    f"timeout={logs.get('timeout', 0.0):.4f} "
                    f"goals={logs.get('route_goals', 0.0):.4f}",
                    flush=True,
                )
    finally:
        backend.close(pufferl)


if __name__ == "__main__":
    main()
