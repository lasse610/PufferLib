#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import torch


OBSTACLE_MODES = {
    "none": 0,
    "center_block": 1,
    "doorway": 2,
    "corridor": 3,
    "maze": 4,
    "random_blocks": 5,
    "mixed_open_blocks": 6,
    "layout_mixed": 7,
    "procedural_indoor": 8,
    "procedural_indoor_p0": 9,
    "procedural_indoor_p1": 10,
    "procedural_indoor_p2": 11,
    "procedural_indoor_p3": 12,
    "procedural_indoor_hard": 13,
    "procedural_indoor_hard_p0": 14,
    "procedural_indoor_hard_p1": 15,
    "procedural_indoor_hard_p2": 16,
    "procedural_indoor_hard_p3": 17,
    "procedural_indoor_tight": 18,
    "procedural_indoor_tight_p0": 19,
    "procedural_indoor_tight_p1": 20,
    "procedural_indoor_tight_p2": 21,
    "procedural_indoor_tight_p3": 22,
    "procedural_house": 23,
    "procedural_house_p0": 24,
    "procedural_house_p1": 25,
    "procedural_house_p2": 26,
    "procedural_house_p3": 27,
    "procedural_house_multizone": 28,
    "procedural_house_multizone_p0": 29,
    "procedural_house_multizone_p1": 30,
    "procedural_house_multizone_p2": 31,
    "procedural_house_multizone_p3": 32,
    "gate_slalom": 33,
    "gates": 33,
    "slalom": 33,
    "creative_obstacles": 33,
    "gate_slalom_p0": 34,
    "gate_slalom_p1": 35,
    "gate_slalom_p2": 36,
    "gate_slalom_p3": 37,
    "gate_slalom_easy_mix": 38,
    "gate_slalom_curriculum": 38,
    "easy_gates": 38,
    "gate_slalom_p1_p0_mix": 39,
    "p1_p0_mix": 39,
    "p1_gate_intro": 39,
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

STRESS_PROFILES = {
    "direct_one_goal": {},
    "direct_goals3": {"route_goal_count": 3, "max_steps": 1024},
    "direct_goals5": {"route_goal_count": 5, "max_steps": 1536},
    "direct_z_band": {
        "route_goal_count": 1,
        "route_target_z_min": 1.0,
        "route_target_z_max": 2.0,
    },
    "direct_dynamics_light": {
        "mass_scale_min": 0.85,
        "mass_scale_max": 1.15,
        "thrust_scale_min": 0.9,
        "thrust_scale_max": 1.1,
        "drag_scale_min": 0.75,
        "drag_scale_max": 1.5,
        "motor_tau_scale_min": 0.75,
        "motor_tau_scale_max": 1.5,
    },
    "direct_combo": {
        "route_goal_count": 3,
        "max_steps": 1024,
        "route_target_z_min": 1.0,
        "route_target_z_max": 2.0,
        "mass_scale_min": 0.85,
        "mass_scale_max": 1.15,
        "thrust_scale_min": 0.9,
        "thrust_scale_max": 1.1,
        "drag_scale_min": 0.75,
        "drag_scale_max": 1.5,
        "motor_tau_scale_min": 0.75,
        "motor_tau_scale_max": 1.5,
    },
}

EXPLORE_V0_PROFILE = {
    "task_mode": 3,
    "action_mode": 1,
    "max_steps": 1024,
    "max_vel": 0.7,
    "max_z_vel": 0.25,
    "spawn_radius": 2.5,
    "spawn_radius_min": 0.0,
    "spawn_height": 1.5,
    "room_size": 3.0,
    "num_rays": 24,
    "ray_length": 4.0,
    "obstacle_mode": 5,
    "explore_goal_fraction": 0.85,
    "route_wall_obs": 1,
    "action_history_obs": 1,
}

AGILE_V0_PROFILE = {
    "task_mode": 4,
    "action_mode": 1,
    "max_steps": 768,
    "max_vel": 1.4,
    "max_z_vel": 0.35,
    "vel_ctrl_ki_z": 4.0,
    "spawn_radius": 2.5,
    "spawn_radius_min": 0.0,
    "spawn_height": 1.5,
    "init_vel_radius": 0.0,
    "init_omega_radius": 0.0,
    "room_size": 3.0,
    "num_rays": 24,
    "ray_length": 4.0,
    "obstacle_mode": 7,
    "route_goal_radius": 1.0,
    "route_goal_count": 3,
    "route_goal_count_min": 0,
    "route_target_mode": 1,
    "route_blocked_target_prob": 1.0,
    "route_target_z_min": 1.5,
    "route_target_z_max": 1.5,
    "route_reward_mode": 0,
    "route_walls": 1,
    "route_direct_obs": 1,
    "route_extra_obs": 0,
    "route_wall_obs": 1,
    "route_perception_obs": 0,
    "route_target_obs_mode": 1,
    "route_path_margin": 0.35,
    "route_clearance_radius": 0.45,
    "route_clearance_scale": 0.50,
    "route_arrival_radius": 1.5,
    "route_arrival_bonus": 0.04,
    "route_arrival_vel_scale": 0.01,
    "route_target_wall_margin": 1.0,
    "action_history_obs": 1,
}

DRONE_HOVER_PROFILE = {
    "task_mode": 5,
    "action_mode": 3,
    "max_steps": 1024,
    "action_repeat": 1,
    "spawn_radius": 5.0,
    "spawn_radius_min": 0.0,
    "spawn_height": 5.0,
    "init_vel_radius": 0.0,
    "init_omega_radius": 0.0,
    "room_size": 0.0,
    "ceiling_height": 10.0,
    "num_rays": 24,
    "ray_length": 4.0,
    "obstacle_mode": 0,
    "route_direct_obs": 0,
    "route_wall_obs": 1,
    "route_perception_obs": 0,
    "route_target_obs_mode": 1,
    "action_history_obs": 1,
    "mass_scale_min": 0.95,
    "mass_scale_max": 1.05,
    "thrust_scale_min": 0.95,
    "thrust_scale_max": 1.05,
    "drag_scale_min": 0.95,
    "drag_scale_max": 1.05,
    "motor_tau_scale_min": 0.95,
    "motor_tau_scale_max": 1.05,
    "drone_alpha_dist": 0.782192,
    "drone_alpha_hover": 0.071445,
    "drone_alpha_omega": 0.00135588,
    "drone_alpha_shaping": 3.9754,
    "drone_hover_target_dist": 5.0,
    "drone_hover_dist": 0.1,
    "drone_hover_omega": 0.1,
    "drone_hover_vel": 0.1,
}

DRONE_REFERENCE_PROFILE = {
    **DRONE_HOVER_PROFILE,
    "action_mode": 4,
    "dynamics_profile": 1,
    "physics_dt": 0.002,
    "physics_substeps": 5,
}

REFERENCE_WAYPOINT_PROFILE = {
    "task_mode": 4,
    "action_mode": 4,
    "max_steps": 1024,
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
    "obstacle_mode": 0,
    "route_goal_radius": 0.75,
    "route_goal_count": 3,
    "route_goal_count_min": 0,
    "route_target_mode": 0,
    "route_blocked_target_prob": 0.0,
    "route_target_z_min": 2.0,
    "route_target_z_max": 8.0,
    "route_reward_mode": 2,
    "route_walls": 0,
    "route_direct_obs": 1,
    "route_extra_obs": 0,
    "route_wall_obs": 1,
    "route_perception_obs": 0,
    "route_target_obs_mode": 1,
    "route_path_margin": 0.35,
    "route_clearance_radius": 0.45,
    "route_clearance_scale": 0.0,
    "route_vel_scale": 0.02,
    "route_arrival_radius": 1.25,
    "route_arrival_bonus": 0.02,
    "route_arrival_vel_scale": 0.005,
    "route_target_wall_margin": 0.75,
    "action_history_obs": 1,
    "mass_scale_min": 0.95,
    "mass_scale_max": 1.05,
    "thrust_scale_min": 0.95,
    "thrust_scale_max": 1.05,
    "drag_scale_min": 0.95,
    "drag_scale_max": 1.05,
    "motor_tau_scale_min": 0.95,
    "motor_tau_scale_max": 1.05,
}


def load_quadrotor_config():
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        from pufferlib.pufferl import load_config
        return load_config("quadrotor")
    finally:
        sys.argv = old_argv


def resolve_checkpoint(args, checkpoint):
    if checkpoint != "latest":
        return checkpoint

    pattern = os.path.join(args["checkpoint_dir"], args["env_name"], "**", "*.bin")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No .bin checkpoints found under {args['checkpoint_dir']}/{args['env_name']}")
    return max(candidates, key=os.path.getctime)


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


def detach_state(state):
    if not isinstance(state, tuple):
        return state
    out = []
    for value in state:
        out.append(value.detach() if torch.is_tensor(value) else value)
    return tuple(out)


def accumulate_logs(accum, logs):
    n = float(logs.get("n", 0.0))
    if n <= 0:
        return accum
    if accum is None:
        accum = {"n": 0.0, "sums": {}}
    for key, value in logs.items():
        if key == "n":
            continue
        accum["sums"][key] = accum["sums"].get(key, 0.0) + float(value) * n
    accum["n"] += n
    return accum


def finalize_logs(accum):
    if accum is None or accum["n"] <= 0:
        return {}
    return {"n": accum["n"], **{
        key: value / accum["n"] for key, value in accum["sums"].items()
    }}


def parse_obstacle_mode(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    if value not in OBSTACLE_MODES:
        valid = ", ".join(sorted(OBSTACLE_MODES))
        raise argparse.ArgumentTypeError(f"unknown obstacle mode {value!r}; valid: {valid}")
    return OBSTACLE_MODES[value]


def set_if_not_none(env, key, value):
    if value is not None:
        env[key] = value


def deterministic_torch_eval(pufferl, episodes, max_steps, log_interval):
    from pufferlib.torch_pufferl import _actions_for_vec_step

    policy = pufferl.policy
    policy.eval()
    state = policy.initial_state(pufferl.total_agents, pufferl.device)
    accum = None
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            logits, _, state = policy.forward_eval(pufferl.vec_obs, state)
            actions = logits.mean if isinstance(logits, torch.distributions.Normal) else logits

        actions_flat = _actions_for_vec_step(actions)
        if pufferl.gpu:
            actions_flat = actions_flat.cuda()
            pufferl._vec.gpu_step(actions_flat.data_ptr())
            torch.cuda.synchronize()
        else:
            pufferl._vec.cpu_step(actions_flat.data_ptr())

        done = pufferl.vec_terminals > 0.5
        state = reset_done_state(state, done)
        state = detach_state(state)
        if step % log_interval == 0:
            accum = accumulate_logs(accum, pufferl._vec.log())
            if finalize_logs(accum).get("n", 0.0) >= episodes:
                break

    accum = accumulate_logs(accum, pufferl._vec.log())
    return finalize_logs(accum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="latest")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--total-agents", type=int, default=4096)
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-rollouts", type=int, default=256)
    parser.add_argument("--network", choices=("MinGRU", "MLP", "GELU"))
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--use-bias", type=int, choices=(0, 1))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--route-direct-obs", type=int, choices=(0, 1))
    parser.add_argument("--route-wall-obs", type=int, choices=(0, 1))
    parser.add_argument("--route-target-obs-mode", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--route-perception-obs", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--action-history-obs", type=int, choices=(0, 1))
    parser.add_argument("--num-rays", type=int)
    parser.add_argument("--ray-length", type=float)
    parser.add_argument("--camera-depth-obs", type=int, choices=(0, 1))
    parser.add_argument("--camera-depth-width", type=int)
    parser.add_argument("--camera-depth-height", type=int)
    parser.add_argument("--camera-depth-fov-x-deg", type=float)
    parser.add_argument("--camera-depth-fov-y-deg", type=float)
    parser.add_argument("--stress-profile", choices=sorted(STRESS_PROFILES), default=None)
    parser.add_argument("--explore-v0", action="store_true")
    parser.add_argument("--agile-v0", action="store_true")
    parser.add_argument("--drone-hover", action="store_true")
    parser.add_argument("--drone-reference", action="store_true")
    parser.add_argument("--reference-waypoint", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--action-repeat", type=int)
    parser.add_argument("--action-mode", type=int)
    parser.add_argument("--dynamics-profile", type=int)
    parser.add_argument("--physics-dt", type=float)
    parser.add_argument("--physics-substeps", type=int)
    parser.add_argument("--init-vel-radius", type=float)
    parser.add_argument("--init-omega-radius", type=float)
    parser.add_argument("--max-vel", type=float)
    parser.add_argument("--max-z-vel", type=float)
    parser.add_argument("--obstacle-mode", type=parse_obstacle_mode)
    parser.add_argument("--explore-goal-fraction", type=float)
    parser.add_argument("--explore-memory-obs", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--explore-frontier-progress-scale", type=float)
    parser.add_argument("--route-goal-count", type=int)
    parser.add_argument("--route-goal-count-min", type=int)
    parser.add_argument("--route-goal-radius", type=float)
    parser.add_argument("--route-target-mode", type=int, choices=(0, 1))
    parser.add_argument("--route-blocked-target-prob", type=float)
    parser.add_argument("--route-reward-mode", type=int)
    parser.add_argument("--route-step-cost", type=float)
    parser.add_argument("--route-walls", type=int)
    parser.add_argument("--route-progress-scale", type=float)
    parser.add_argument("--route-clearance-radius", type=float)
    parser.add_argument("--route-clearance-scale", type=float)
    parser.add_argument("--route-clearance-speed-scale", type=float)
    parser.add_argument("--route-clearance-speed-min", type=float)
    parser.add_argument("--route-clearance-speed-full", type=float)
    parser.add_argument("--route-vel-scale", type=float)
    parser.add_argument("--route-arrival-radius", type=float)
    parser.add_argument("--route-arrival-bonus", type=float)
    parser.add_argument("--route-arrival-vel-scale", type=float)
    parser.add_argument("--route-arrival-toward-scale", type=float)
    parser.add_argument("--route-goal-speed", type=float)
    parser.add_argument("--route-target-z-min", type=float)
    parser.add_argument("--route-target-z-max", type=float)
    parser.add_argument("--mass-scale-min", type=float)
    parser.add_argument("--mass-scale-max", type=float)
    parser.add_argument("--thrust-scale-min", type=float)
    parser.add_argument("--thrust-scale-max", type=float)
    parser.add_argument("--drag-scale-min", type=float)
    parser.add_argument("--drag-scale-max", type=float)
    parser.add_argument("--motor-tau-scale-min", type=float)
    parser.add_argument("--motor-tau-scale-max", type=float)
    parser.add_argument("--drone-alpha-dist", type=float)
    parser.add_argument("--drone-alpha-hover", type=float)
    parser.add_argument("--drone-alpha-omega", type=float)
    parser.add_argument("--drone-alpha-shaping", type=float)
    parser.add_argument("--drone-hover-target-dist", type=float)
    parser.add_argument("--drone-hover-dist", type=float)
    parser.add_argument("--drone-hover-omega", type=float)
    parser.add_argument("--drone-hover-vel", type=float)
    parser.add_argument("--slowly", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parsed = parser.parse_args()

    from pufferlib.pufferl import _resolve_backend, unroll_nested_dict

    args = load_quadrotor_config()
    args["reset_state"] = False
    if parsed.slowly or parsed.deterministic:
        args["slowly"] = True
    args["vec"]["total_agents"] = parsed.total_agents
    args["vec"]["num_buffers"] = parsed.num_buffers
    args["vec"]["num_threads"] = parsed.num_threads
    args["train"]["horizon"] = parsed.horizon
    if parsed.network is not None:
        args["torch"]["network"] = parsed.network
    if parsed.hidden_size is not None:
        args["policy"]["hidden_size"] = parsed.hidden_size
    if parsed.num_layers is not None:
        args["policy"]["num_layers"] = parsed.num_layers
    if parsed.use_bias is not None:
        args["policy"]["use_bias"] = parsed.use_bias
    if parsed.seed is not None:
        args["env"]["seed"] = parsed.seed
    if parsed.stress_profile is not None:
        args["env"].update(STRESS_PROFILES[parsed.stress_profile])
    if parsed.explore_v0:
        args["env"].update(EXPLORE_V0_PROFILE)
    if parsed.agile_v0:
        args["env"].update(AGILE_V0_PROFILE)
    if parsed.drone_hover:
        args["env"].update(DRONE_HOVER_PROFILE)
    if parsed.drone_reference:
        args["env"].update(DRONE_REFERENCE_PROFILE)
    if parsed.reference_waypoint:
        args["env"].update(REFERENCE_WAYPOINT_PROFILE)
    if parsed.route_direct_obs is not None:
        args["env"]["route_direct_obs"] = parsed.route_direct_obs
    if parsed.route_wall_obs is not None:
        args["env"]["route_wall_obs"] = parsed.route_wall_obs
    if parsed.route_target_obs_mode is not None:
        args["env"]["route_target_obs_mode"] = parsed.route_target_obs_mode
    if parsed.route_perception_obs is not None:
        args["env"]["route_perception_obs"] = parsed.route_perception_obs
    if parsed.action_history_obs is not None:
        args["env"]["action_history_obs"] = parsed.action_history_obs
    set_if_not_none(args["env"], "num_rays", parsed.num_rays)
    set_if_not_none(args["env"], "ray_length", parsed.ray_length)
    env = args["env"]
    set_if_not_none(env, "camera_depth_obs", parsed.camera_depth_obs)
    set_if_not_none(env, "camera_depth_width", parsed.camera_depth_width)
    set_if_not_none(env, "camera_depth_height", parsed.camera_depth_height)
    set_if_not_none(env, "camera_depth_fov_x_deg", parsed.camera_depth_fov_x_deg)
    set_if_not_none(env, "camera_depth_fov_y_deg", parsed.camera_depth_fov_y_deg)
    set_if_not_none(env, "max_steps", parsed.max_steps)
    set_if_not_none(env, "action_repeat", parsed.action_repeat)
    set_if_not_none(env, "action_mode", parsed.action_mode)
    set_if_not_none(env, "dynamics_profile", parsed.dynamics_profile)
    set_if_not_none(env, "physics_dt", parsed.physics_dt)
    set_if_not_none(env, "physics_substeps", parsed.physics_substeps)
    set_if_not_none(env, "init_vel_radius", parsed.init_vel_radius)
    set_if_not_none(env, "init_omega_radius", parsed.init_omega_radius)
    set_if_not_none(env, "max_vel", parsed.max_vel)
    set_if_not_none(env, "max_z_vel", parsed.max_z_vel)
    set_if_not_none(env, "obstacle_mode", parsed.obstacle_mode)
    set_if_not_none(env, "explore_goal_fraction", parsed.explore_goal_fraction)
    set_if_not_none(env, "explore_memory_obs", parsed.explore_memory_obs)
    set_if_not_none(env, "explore_frontier_progress_scale", parsed.explore_frontier_progress_scale)
    set_if_not_none(env, "route_goal_count", parsed.route_goal_count)
    set_if_not_none(env, "route_goal_count_min", parsed.route_goal_count_min)
    set_if_not_none(env, "route_goal_radius", parsed.route_goal_radius)
    set_if_not_none(env, "route_target_mode", parsed.route_target_mode)
    set_if_not_none(env, "route_blocked_target_prob", parsed.route_blocked_target_prob)
    set_if_not_none(env, "route_reward_mode", parsed.route_reward_mode)
    set_if_not_none(env, "route_step_cost", parsed.route_step_cost)
    set_if_not_none(env, "route_walls", parsed.route_walls)
    set_if_not_none(env, "route_progress_scale", parsed.route_progress_scale)
    set_if_not_none(env, "route_clearance_radius", parsed.route_clearance_radius)
    set_if_not_none(env, "route_clearance_scale", parsed.route_clearance_scale)
    set_if_not_none(env, "route_clearance_speed_scale", parsed.route_clearance_speed_scale)
    set_if_not_none(env, "route_clearance_speed_min", parsed.route_clearance_speed_min)
    set_if_not_none(env, "route_clearance_speed_full", parsed.route_clearance_speed_full)
    set_if_not_none(env, "route_vel_scale", parsed.route_vel_scale)
    set_if_not_none(env, "route_arrival_radius", parsed.route_arrival_radius)
    set_if_not_none(env, "route_arrival_bonus", parsed.route_arrival_bonus)
    set_if_not_none(env, "route_arrival_vel_scale", parsed.route_arrival_vel_scale)
    set_if_not_none(env, "route_arrival_toward_scale", parsed.route_arrival_toward_scale)
    set_if_not_none(env, "route_goal_speed", parsed.route_goal_speed)
    set_if_not_none(env, "route_target_z_min", parsed.route_target_z_min)
    set_if_not_none(env, "route_target_z_max", parsed.route_target_z_max)
    set_if_not_none(env, "mass_scale_min", parsed.mass_scale_min)
    set_if_not_none(env, "mass_scale_max", parsed.mass_scale_max)
    set_if_not_none(env, "thrust_scale_min", parsed.thrust_scale_min)
    set_if_not_none(env, "thrust_scale_max", parsed.thrust_scale_max)
    set_if_not_none(env, "drag_scale_min", parsed.drag_scale_min)
    set_if_not_none(env, "drag_scale_max", parsed.drag_scale_max)
    set_if_not_none(env, "motor_tau_scale_min", parsed.motor_tau_scale_min)
    set_if_not_none(env, "motor_tau_scale_max", parsed.motor_tau_scale_max)
    set_if_not_none(env, "drone_alpha_dist", parsed.drone_alpha_dist)
    set_if_not_none(env, "drone_alpha_hover", parsed.drone_alpha_hover)
    set_if_not_none(env, "drone_alpha_omega", parsed.drone_alpha_omega)
    set_if_not_none(env, "drone_alpha_shaping", parsed.drone_alpha_shaping)
    set_if_not_none(env, "drone_hover_target_dist", parsed.drone_hover_target_dist)
    set_if_not_none(env, "drone_hover_dist", parsed.drone_hover_dist)
    set_if_not_none(env, "drone_hover_omega", parsed.drone_hover_omega)
    set_if_not_none(env, "drone_hover_vel", parsed.drone_hover_vel)

    checkpoint = resolve_checkpoint(args, parsed.checkpoint)
    backend = _resolve_backend(args)
    pufferl = backend.create_pufferl(args)
    try:
        backend.load_weights(pufferl, checkpoint)
        if parsed.deterministic:
            env_logs = deterministic_torch_eval(
                pufferl,
                parsed.episodes,
                parsed.max_rollouts * parsed.horizon,
                parsed.horizon,
            )
            logs = {f"env/{key}": value for key, value in env_logs.items()}
        else:
            logs = {}
            for _ in range(parsed.max_rollouts):
                backend.rollouts(pufferl)
                logs = dict(unroll_nested_dict(backend.eval_log(pufferl)))
                if int(logs.get("env/n", 0)) >= parsed.episodes:
                    break

        keys = (
            "env/n",
            "env/perf",
            "env/success",
            "env/score",
            "env/episode_length",
            "env/crashed",
            "env/obstacle_hit",
            "env/wall_hit",
            "env/ground_hit",
            "env/timeout",
            "env/drone_oob",
            "env/final_dist",
            "env/final_z_error",
            "env/final_vel",
            "env/settled",
            "env/hover_rate",
            "env/avg_vel",
            "env/coverage",
            "env/explored_cells",
            "env/new_cell_rate",
            "env/route_goals",
            "env/min_clearance",
            "env/path_length",
            "env/path_efficiency",
            "env/action_saturation",
            "env/action_delta",
        )
        print(f"checkpoint {checkpoint}")
        for key in keys:
            if key in logs:
                print(f"{key[4:] if key.startswith('env/') else key} {logs[key]:.6g}")
    finally:
        backend.close(pufferl)


if __name__ == "__main__":
    main()
