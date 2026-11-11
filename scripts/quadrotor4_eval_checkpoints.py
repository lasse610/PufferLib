#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
import sys


METRIC_KEYS = (
    "env/n",
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


def checkpoint_step(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    try:
        return int(stem)
    except ValueError:
        return -1


def resolve_run_dir(args, run):
    root = os.path.join(args["checkpoint_dir"], args["env_name"])
    if run == "latest":
        candidates = [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint runs found under {root}")
        return max(candidates, key=os.path.getctime)

    if os.path.isdir(run):
        return run

    candidate = os.path.join(root, run)
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(f"Could not resolve run '{run}' under {root}")


def select_checkpoints(run_dir, every, limit, last):
    checkpoints = sorted(glob.glob(os.path.join(run_dir, "*.bin")), key=checkpoint_step)
    if every > 1:
        checkpoints = checkpoints[::every]
    if last is not None:
        checkpoints = checkpoints[-last:]
    if limit is not None:
        checkpoints = checkpoints[:limit]
    if not checkpoints:
        raise FileNotFoundError(f"No .bin checkpoints found under {run_dir}")
    return checkpoints


def evaluate_checkpoint(backend, args, checkpoint, episodes, max_rollouts):
    pufferl = backend.create_pufferl(args)
    try:
        backend.load_weights(pufferl, checkpoint)
        logs = {}
        from pufferlib.pufferl import unroll_nested_dict
        for _ in range(max_rollouts):
            backend.rollouts(pufferl)
            logs = dict(unroll_nested_dict(backend.eval_log(pufferl)))
            if int(logs.get("env/n", 0)) >= episodes:
                break

        return {key: float(logs[key]) for key in METRIC_KEYS if key in logs}
    finally:
        backend.close(pufferl)


def add_forwarded_arg(cmd, flag, value):
    if value is not None:
        cmd.extend((flag, str(value)))


def evaluate_checkpoint_fresh_process(parsed, checkpoint):
    script = os.path.join(os.path.dirname(__file__), "quadrotor4_eval_headless.py")
    cmd = [
        sys.executable,
        script,
        checkpoint,
        "--episodes", str(parsed.episodes),
        "--total-agents", str(parsed.total_agents),
        "--num-buffers", str(parsed.num_buffers),
        "--num-threads", str(parsed.num_threads),
        "--horizon", str(parsed.horizon),
        "--max-rollouts", str(parsed.max_rollouts),
    ]

    if parsed.explore_v0:
        cmd.append("--explore-v0")
    if parsed.agile_v0:
        cmd.append("--agile-v0")
    if parsed.drone_hover:
        cmd.append("--drone-hover")
    if parsed.drone_reference:
        cmd.append("--drone-reference")
    if parsed.reference_waypoint:
        cmd.append("--reference-waypoint")
    if parsed.slowly:
        cmd.append("--slowly")

    for flag, value in (
        ("--network", parsed.network),
        ("--hidden-size", parsed.hidden_size),
        ("--num-layers", parsed.num_layers),
        ("--use-bias", parsed.use_bias),
        ("--seed", parsed.seed),
        ("--route-direct-obs", parsed.route_direct_obs),
        ("--route-wall-obs", parsed.route_wall_obs),
        ("--route-target-obs-mode", parsed.route_target_obs_mode),
        ("--route-perception-obs", parsed.route_perception_obs),
        ("--action-history-obs", parsed.action_history_obs),
        ("--num-rays", parsed.num_rays),
        ("--ray-length", parsed.ray_length),
        ("--camera-depth-obs", parsed.camera_depth_obs),
        ("--camera-depth-width", parsed.camera_depth_width),
        ("--camera-depth-height", parsed.camera_depth_height),
        ("--camera-depth-fov-x-deg", parsed.camera_depth_fov_x_deg),
        ("--camera-depth-fov-y-deg", parsed.camera_depth_fov_y_deg),
        ("--max-steps", parsed.max_steps),
        ("--action-repeat", parsed.action_repeat),
        ("--action-mode", parsed.action_mode),
        ("--dynamics-profile", parsed.dynamics_profile),
        ("--physics-dt", parsed.physics_dt),
        ("--physics-substeps", parsed.physics_substeps),
        ("--init-vel-radius", parsed.init_vel_radius),
        ("--init-omega-radius", parsed.init_omega_radius),
        ("--max-vel", parsed.max_vel),
        ("--max-z-vel", parsed.max_z_vel),
        ("--obstacle-mode", parsed.obstacle_mode),
        ("--explore-goal-fraction", parsed.explore_goal_fraction),
        ("--explore-memory-obs", parsed.explore_memory_obs),
        ("--explore-frontier-progress-scale", parsed.explore_frontier_progress_scale),
        ("--route-goal-count", parsed.route_goal_count),
        ("--route-goal-count-min", parsed.route_goal_count_min),
        ("--route-goal-radius", parsed.route_goal_radius),
        ("--route-target-mode", parsed.route_target_mode),
        ("--route-blocked-target-prob", parsed.route_blocked_target_prob),
        ("--route-reward-mode", parsed.route_reward_mode),
        ("--route-step-cost", parsed.route_step_cost),
        ("--route-walls", parsed.route_walls),
        ("--route-progress-scale", parsed.route_progress_scale),
        ("--route-clearance-radius", parsed.route_clearance_radius),
        ("--route-clearance-scale", parsed.route_clearance_scale),
        ("--route-clearance-speed-scale", parsed.route_clearance_speed_scale),
        ("--route-clearance-speed-min", parsed.route_clearance_speed_min),
        ("--route-clearance-speed-full", parsed.route_clearance_speed_full),
        ("--route-vel-scale", parsed.route_vel_scale),
        ("--route-arrival-radius", parsed.route_arrival_radius),
        ("--route-arrival-bonus", parsed.route_arrival_bonus),
        ("--route-arrival-vel-scale", parsed.route_arrival_vel_scale),
        ("--route-arrival-toward-scale", parsed.route_arrival_toward_scale),
        ("--route-goal-speed", parsed.route_goal_speed),
        ("--route-target-z-min", parsed.route_target_z_min),
        ("--route-target-z-max", parsed.route_target_z_max),
        ("--mass-scale-min", parsed.mass_scale_min),
        ("--mass-scale-max", parsed.mass_scale_max),
        ("--thrust-scale-min", parsed.thrust_scale_min),
        ("--thrust-scale-max", parsed.thrust_scale_max),
        ("--drag-scale-min", parsed.drag_scale_min),
        ("--drag-scale-max", parsed.drag_scale_max),
        ("--motor-tau-scale-min", parsed.motor_tau_scale_min),
        ("--motor-tau-scale-max", parsed.motor_tau_scale_max),
        ("--drone-alpha-dist", parsed.drone_alpha_dist),
        ("--drone-alpha-hover", parsed.drone_alpha_hover),
        ("--drone-alpha-omega", parsed.drone_alpha_omega),
        ("--drone-alpha-shaping", parsed.drone_alpha_shaping),
        ("--drone-hover-target-dist", parsed.drone_hover_target_dist),
        ("--drone-hover-dist", parsed.drone_hover_dist),
        ("--drone-hover-omega", parsed.drone_hover_omega),
        ("--drone-hover-vel", parsed.drone_hover_vel),
    ):
        add_forwarded_arg(cmd, flag, value)

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    allowed = {key[4:] for key in METRIC_KEYS if key.startswith("env/")}
    metrics = {}
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in allowed:
            continue
        try:
            metrics[f"env/{parts[0]}"] = float(parts[1])
        except ValueError:
            continue
    return metrics


def fmt(value):
    if value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", default="latest")
    parser.add_argument("--episodes", type=int, default=1024)
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
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--last", type=int)
    parser.add_argument("--slowly", action="store_true")
    parser.add_argument("--fresh-process", action="store_true",
                        help="Evaluate each checkpoint in a separate process. Slower, but avoids cross-checkpoint eval state.")
    parsed = parser.parse_args()

    if parsed.every < 1:
        raise ValueError("--every must be >= 1")
    if parsed.limit is not None and parsed.limit < 1:
        raise ValueError("--limit must be >= 1")
    if parsed.last is not None and parsed.last < 1:
        raise ValueError("--last must be >= 1")

    from pufferlib.pufferl import _resolve_backend

    args = load_quadrotor_config()
    args["reset_state"] = False
    if parsed.slowly:
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
    env = args["env"]
    if parsed.explore_v0:
        env.update(EXPLORE_V0_PROFILE)
    if parsed.agile_v0:
        env.update(AGILE_V0_PROFILE)
    if parsed.drone_hover:
        env.update(DRONE_HOVER_PROFILE)
    if parsed.drone_reference:
        env.update(DRONE_REFERENCE_PROFILE)
    if parsed.reference_waypoint:
        env.update(REFERENCE_WAYPOINT_PROFILE)
    for key, value in (
        ("seed", parsed.seed),
        ("route_direct_obs", parsed.route_direct_obs),
        ("route_wall_obs", parsed.route_wall_obs),
        ("route_target_obs_mode", parsed.route_target_obs_mode),
        ("route_perception_obs", parsed.route_perception_obs),
        ("action_history_obs", parsed.action_history_obs),
        ("num_rays", parsed.num_rays),
        ("ray_length", parsed.ray_length),
        ("camera_depth_obs", parsed.camera_depth_obs),
        ("camera_depth_width", parsed.camera_depth_width),
        ("camera_depth_height", parsed.camera_depth_height),
        ("camera_depth_fov_x_deg", parsed.camera_depth_fov_x_deg),
        ("camera_depth_fov_y_deg", parsed.camera_depth_fov_y_deg),
        ("max_steps", parsed.max_steps),
        ("action_repeat", parsed.action_repeat),
        ("action_mode", parsed.action_mode),
        ("dynamics_profile", parsed.dynamics_profile),
        ("physics_dt", parsed.physics_dt),
        ("physics_substeps", parsed.physics_substeps),
        ("init_vel_radius", parsed.init_vel_radius),
        ("init_omega_radius", parsed.init_omega_radius),
        ("max_vel", parsed.max_vel),
        ("max_z_vel", parsed.max_z_vel),
        ("obstacle_mode", parsed.obstacle_mode),
        ("explore_goal_fraction", parsed.explore_goal_fraction),
        ("explore_memory_obs", parsed.explore_memory_obs),
        ("explore_frontier_progress_scale", parsed.explore_frontier_progress_scale),
        ("route_goal_count", parsed.route_goal_count),
        ("route_goal_count_min", parsed.route_goal_count_min),
        ("route_goal_radius", parsed.route_goal_radius),
        ("route_target_mode", parsed.route_target_mode),
        ("route_blocked_target_prob", parsed.route_blocked_target_prob),
        ("route_reward_mode", parsed.route_reward_mode),
        ("route_step_cost", parsed.route_step_cost),
        ("route_walls", parsed.route_walls),
        ("route_progress_scale", parsed.route_progress_scale),
        ("route_clearance_radius", parsed.route_clearance_radius),
        ("route_clearance_scale", parsed.route_clearance_scale),
        ("route_clearance_speed_scale", parsed.route_clearance_speed_scale),
        ("route_clearance_speed_min", parsed.route_clearance_speed_min),
        ("route_clearance_speed_full", parsed.route_clearance_speed_full),
        ("route_vel_scale", parsed.route_vel_scale),
        ("route_arrival_radius", parsed.route_arrival_radius),
        ("route_arrival_bonus", parsed.route_arrival_bonus),
        ("route_arrival_vel_scale", parsed.route_arrival_vel_scale),
        ("route_arrival_toward_scale", parsed.route_arrival_toward_scale),
        ("route_goal_speed", parsed.route_goal_speed),
        ("route_target_z_min", parsed.route_target_z_min),
        ("route_target_z_max", parsed.route_target_z_max),
        ("mass_scale_min", parsed.mass_scale_min),
        ("mass_scale_max", parsed.mass_scale_max),
        ("thrust_scale_min", parsed.thrust_scale_min),
        ("thrust_scale_max", parsed.thrust_scale_max),
        ("drag_scale_min", parsed.drag_scale_min),
        ("drag_scale_max", parsed.drag_scale_max),
        ("motor_tau_scale_min", parsed.motor_tau_scale_min),
        ("motor_tau_scale_max", parsed.motor_tau_scale_max),
        ("drone_alpha_dist", parsed.drone_alpha_dist),
        ("drone_alpha_hover", parsed.drone_alpha_hover),
        ("drone_alpha_omega", parsed.drone_alpha_omega),
        ("drone_alpha_shaping", parsed.drone_alpha_shaping),
        ("drone_hover_target_dist", parsed.drone_hover_target_dist),
        ("drone_hover_dist", parsed.drone_hover_dist),
        ("drone_hover_omega", parsed.drone_hover_omega),
        ("drone_hover_vel", parsed.drone_hover_vel),
    ):
        if value is not None:
            env[key] = value

    run_dir = resolve_run_dir(args, parsed.run)
    checkpoints = select_checkpoints(run_dir, parsed.every, parsed.limit, parsed.last)
    backend = None if parsed.fresh_process else _resolve_backend(args)

    rows = []
    try:
        for checkpoint in checkpoints:
            if parsed.fresh_process:
                metrics = evaluate_checkpoint_fresh_process(parsed, checkpoint)
            else:
                metrics = evaluate_checkpoint(
                    backend, args, checkpoint, parsed.episodes, parsed.max_rollouts)
            row = {
                "checkpoint": os.path.basename(checkpoint),
                "step": checkpoint_step(checkpoint),
            }
            row.update({key[4:] if key.startswith("env/") else key: value
                        for key, value in metrics.items()})
            rows.append(row)
    finally:
        pass

    columns = (
        "checkpoint",
        "step",
        "n",
        "success",
        "score",
        "episode_length",
        "crashed",
        "obstacle_hit",
        "wall_hit",
        "ground_hit",
        "timeout",
        "drone_oob",
        "final_dist",
        "final_z_error",
        "final_vel",
        "settled",
        "hover_rate",
        "avg_vel",
        "coverage",
        "explored_cells",
        "new_cell_rate",
        "route_goals",
        "min_clearance",
        "path_length",
        "path_efficiency",
        "action_saturation",
        "action_delta",
    )
    print(f"run {run_dir}")
    print(",".join(columns))
    for row in rows:
        print(",".join(fmt(row.get(column, "")) for column in columns))

    best = max(rows, key=lambda row: row.get("success", float("-inf")))
    print(
        "best "
        f"{best['checkpoint']} "
        f"success={best.get('success', 0.0):.6g} "
        f"coverage={best.get('coverage', 0.0):.6g} "
        f"obstacle_hit={best.get('obstacle_hit', 0.0):.6g} "
        f"drone_oob={best.get('drone_oob', 0.0):.6g} "
        f"score={best.get('score', 0.0):.6g}"
    )


if __name__ == "__main__":
    main()
