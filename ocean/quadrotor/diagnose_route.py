import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pufferlib.ocean.quadrotor.eval_hover import (
    RAY_OBS_START,
    deterministic_action,
    initial_policy_state,
    load_policy,
    make_env,
    pd_hover_action,
    ray_pd_action,
    ray_reactive_action,
    reset_done_policy_state,
    zero_action,
)


DEFAULT_CHECKPOINT = (
    "experiments/quadrotor_precision_earlystop_20260516_1710/"
    "plain_075/model_puffer_quadrotor_000003.pt"
)

STATUS_NAMES = {
    0: "none",
    1: "ground",
    2: "wall",
    3: "obstacle",
    4: "diverged",
    10: "success",
    11: "timeout",
}

BASE_ARGS = {
    "render": False,
    "device": "cpu",
    "seed": 1,
    "num_envs": 64,
    "max_steps": 1024,
    "action_repeat": 1,
    "max_rate": 1.0,
    "max_vel": 1.0,
    "max_z_vel": 0.75,
    "spawn_radius": 2.5,
    "spawn_height": 1.5,
    "init_vel_radius": 0.0,
    "init_omega_radius": 0.0,
    "room_size": 3.0,
    "ceiling_height": 3.0,
    "num_rays": 24,
    "ray_length": 4.0,
    "mask_rays": False,
    "hover_radius": 0.2,
    "obstacle_mode": "random_blocks",
    "obstacle_block_prob": 1.0,
    "obs_noise": 0.0,
    "action_noise": 0.0,
    "action_delay": 0,
    "action_history_obs": 0,
    "mass_scale": 1.0,
    "inertia_scale": 0.0,
    "thrust_scale": 1.0,
    "torque_scale": 0.0,
    "drag_scale": 1.0,
    "motor_tau_scale": 1.0,
    "explore_cell_size": 0.25,
    "explore_goal_fraction": 0.85,
    "route_goal_radius": 1.0,
    "route_goal_count": 3,
    "route_step_cost": 0.004,
    "route_proximity_reward": 0.02,
    "route_dist_scale": 0.01,
    "route_progress_scale": 25.0,
    "route_crash_reward": -24.0,
    "route_crash_goal_scale": 4.0,
    "route_crash_dist_scale": 1.5,
    "route_clearance_radius": 0.45,
    "route_clearance_scale": 0.35,
    "route_reference_clearance_radius": 0.45,
    "route_reference_clearance_scale": 0.0,
    "route_path_margin": 0.0,
    "route_vel_scale": 0.04,
    "route_arrival_radius": 0.0,
    "route_arrival_bonus": 0.0,
    "route_arrival_vel_scale": 0.0,
    "route_arrival_toward_scale": 0.0,
    "route_target_wall_margin": 0.0,
    "route_goal_speed": 2.0,
    "route_goal_bonus": 1.0,
    "route_reach_bonus": 5.0,
    "route_reward_mode": "dense",
    "route_target_z_min": 0.0,
    "route_target_z_max": 0.0,
    "route_target_mode": "random",
    "route_blocked_target_prob": 1.0,
    "route_walls": 1,
    "route_direct_obs": 0,
    "route_extra_obs": 0,
    "route_wall_obs": 0,
    "route_perception_obs": 0,
    "route_target_obs_mode": "legacy",
    "task_mode": "indoor_route",
    "action_mode": "motor",
    "hidden_size": 128,
    "action_std": 0.1,
    "rnn_name": "Recurrent",
    "rnn_input_size": 128,
    "rnn_hidden_size": 128,
    "pd_kp_pos": 2.0,
    "pd_max_vel": 3.0,
    "pd_kp_vel": 6.0,
    "pd_max_tilt_acc": 9.81 * 0.364,
    "pd_max_z_acc": 9.81 * 0.5,
    "pd_kp_att": 20.0,
    "pd_kd_att": 5.0,
    "pd_avoid_radius": 1.0,
    "pd_avoid_gain": 1.0,
}


WAYPOINT_FIELDS = [
    "episode",
    "env",
    "waypoint",
    "outcome",
    "status",
    "reached",
    "steps",
    "start_dist",
    "min_dist",
    "min_dist_step",
    "speed_at_min",
    "toward_speed_at_min",
    "vz_at_min",
    "min_x",
    "min_y",
    "min_z",
    "event_dist",
    "event_speed",
    "crossed_goal_plane",
    "cross_dist",
    "cross_speed",
    "max_overshoot",
    "visible_frac",
    "obs_target_final_frac",
    "visible_at_min",
    "min_ray_at_min",
    "target_ray_at_min",
    "front_ray_at_min",
    "clearance_at_min",
    "wall_clearance_at_min",
    "wall_side_at_min",
    "room_clearance_at_min",
    "room_boundary_at_min",
    "start_x",
    "start_y",
    "start_z",
    "start_wall_clearance",
    "start_wall_side",
    "final_x",
    "final_y",
    "final_z",
    "final_dist",
    "final_speed",
    "final_vz",
    "final_toward_speed",
    "final_clearance",
    "final_wall_clearance",
    "final_wall_side",
    "final_room_clearance",
    "final_room_boundary",
    "final_state_source",
    "target_x",
    "target_y",
    "target_z",
    "target_wall_clearance",
    "obs_target_start_x",
    "obs_target_start_y",
    "obs_target_start_z",
    "obs_target_start_wall_clearance",
    "obs_target_start_target_dist",
    "obs_target_start_is_final",
    "obs_target_min_x",
    "obs_target_min_y",
    "obs_target_min_z",
    "obs_target_min_wall_clearance",
    "obs_target_min_target_dist",
    "obs_target_min_is_final",
]

EPISODE_FIELDS = [
    "episode",
    "env",
    "status",
    "success",
    "goals",
    "final_dist",
    "return",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose route waypoint arrival, overshoot, visibility, and crash modes."
    )
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--policy",
        choices=("auto", "checkpoint", "pd", "ray", "raypd", "zero"),
        default="auto",
    )
    parser.add_argument("--episodes", type=int, default=256, help="Total completed episodes to collect.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--route-goal-radius", type=float, default=1.0)
    parser.add_argument("--route-goal-count", type=int, default=3)
    parser.add_argument("--route-goal-speed", type=float, default=2.0)
    parser.add_argument("--route-goal-bonus", type=float, default=1.0)
    parser.add_argument("--route-reach-bonus", type=float, default=5.0)
    parser.add_argument("--route-arrival-radius", type=float, default=0.0)
    parser.add_argument("--route-arrival-bonus", type=float, default=0.0)
    parser.add_argument("--route-arrival-vel-scale", type=float, default=0.0)
    parser.add_argument("--route-arrival-toward-scale", type=float, default=0.0)
    parser.add_argument("--route-target-wall-margin", type=float, default=0.0)
    parser.add_argument("--route-reward-mode", default="dense")
    parser.add_argument("--route-target-z-min", type=float, default=0.0)
    parser.add_argument("--route-target-z-max", type=float, default=0.0)
    parser.add_argument("--route-target-mode", default="random")
    parser.add_argument("--route-blocked-target-prob", type=float, default=1.0)
    parser.add_argument("--route-reference-clearance-radius", type=float, default=0.45)
    parser.add_argument("--route-reference-clearance-scale", type=float, default=0.0)
    parser.add_argument("--route-path-margin", type=float, default=0.0)
    parser.add_argument("--route-walls", type=int, default=1)
    parser.add_argument("--route-direct-obs", type=int, default=0)
    parser.add_argument("--route-extra-obs", type=int, default=0)
    parser.add_argument("--route-wall-obs", type=int, default=0)
    parser.add_argument("--route-perception-obs", type=int, default=0)
    parser.add_argument("--route-target-obs-mode", default="legacy")
    parser.add_argument("--obstacle-mode", default="random_blocks")
    parser.add_argument("--obstacle-block-prob", type=float, default=1.0)
    parser.add_argument("--num-rays", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=4.0)
    parser.add_argument("--room-size", type=float, default=3.0)
    parser.add_argument("--ceiling-height", type=float, default=3.0)
    parser.add_argument("--spawn-radius", type=float, default=2.5)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--max-z-vel", type=float, default=0.75)
    parser.add_argument("--vel-ctrl-kp-vel", type=float, default=0.0)
    parser.add_argument("--vel-ctrl-ki-z", type=float, default=0.0)
    parser.add_argument("--vel-ctrl-int-limit-z", type=float, default=0.0)
    parser.add_argument("--vel-ctrl-max-z-acc", type=float, default=0.0)
    parser.add_argument("--action-mode", default="motor")
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--action-std", type=float, default=0.1)
    parser.add_argument("--rnn-name", default="Recurrent")
    parser.add_argument("--action-history-obs", type=int, default=0)
    parser.add_argument("--mask-rays", action="store_true")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def make_runtime_args(cli):
    values = dict(BASE_ARGS)
    for key in (
        "num_envs",
        "seed",
        "device",
        "max_steps",
        "route_goal_radius",
        "route_goal_count",
        "route_goal_speed",
        "route_goal_bonus",
        "route_reach_bonus",
        "route_arrival_radius",
        "route_arrival_bonus",
        "route_arrival_vel_scale",
        "route_arrival_toward_scale",
        "route_target_wall_margin",
        "route_reward_mode",
        "route_target_z_min",
        "route_target_z_max",
        "route_target_mode",
        "route_blocked_target_prob",
        "route_reference_clearance_radius",
        "route_reference_clearance_scale",
        "route_path_margin",
        "route_walls",
        "route_direct_obs",
        "route_extra_obs",
        "route_wall_obs",
        "route_perception_obs",
        "route_target_obs_mode",
        "obstacle_mode",
        "obstacle_block_prob",
        "num_rays",
        "ray_length",
        "room_size",
        "ceiling_height",
        "spawn_radius",
        "max_vel",
        "max_z_vel",
        "vel_ctrl_kp_vel",
        "vel_ctrl_ki_z",
        "vel_ctrl_int_limit_z",
        "vel_ctrl_max_z_acc",
        "action_mode",
        "action_repeat",
        "action_std",
        "rnn_name",
        "action_history_obs",
        "mask_rays",
    ):
        values[key] = getattr(cli, key)
    return SimpleNamespace(**values)


def status_name(value):
    return STATUS_NAMES.get(int(value), f"status_{int(value)}")


def as_vec3(value):
    return np.asarray(value, dtype=np.float32)


def wall_clearance_xy(pos, room_size):
    if room_size <= 0.0:
        return math.nan
    pos = as_vec3(pos)
    return float(room_size - max(abs(float(pos[0])), abs(float(pos[1]))))


def wall_side_xy(pos, room_size):
    if room_size <= 0.0:
        return "none"
    pos = as_vec3(pos)
    distances = {
        "+x": room_size - float(pos[0]),
        "-x": room_size + float(pos[0]),
        "+y": room_size - float(pos[1]),
        "-y": room_size + float(pos[1]),
    }
    return min(distances, key=distances.get)


def room_clearance(pos, args):
    pos = as_vec3(pos)
    distances = []
    if args.room_size > 0.0:
        distances.extend([
            args.room_size - float(pos[0]),
            args.room_size + float(pos[0]),
            args.room_size - float(pos[1]),
            args.room_size + float(pos[1]),
        ])
    if args.ceiling_height > 0.0:
        distances.append(args.ceiling_height - float(pos[2]))
    distances.append(float(pos[2]))
    return float(min(distances)) if distances else math.nan


def room_boundary(pos, args):
    pos = as_vec3(pos)
    distances = {}
    if args.room_size > 0.0:
        distances.update({
            "+x": args.room_size - float(pos[0]),
            "-x": args.room_size + float(pos[0]),
            "+y": args.room_size - float(pos[1]),
            "-y": args.room_size + float(pos[1]),
        })
    if args.ceiling_height > 0.0:
        distances["ceiling"] = args.ceiling_height - float(pos[2])
    distances["ground"] = float(pos[2])
    return min(distances, key=distances.get) if distances else "none"


def ray_metrics(obs_row, state, args):
    if args.num_rays <= 0:
        return math.nan, math.nan, math.nan

    rays = np.asarray(
        obs_row[RAY_OBS_START:RAY_OBS_START + args.num_rays],
        dtype=np.float32,
    ) * args.ray_length
    if rays.size == 0:
        return math.nan, math.nan, math.nan

    pos = as_vec3(state["pos"])
    target = as_vec3(state["target"])
    delta = target[:2] - pos[:2]
    angle = math.atan2(float(delta[1]), float(delta[0]))
    if angle < 0.0:
        angle += 2.0 * math.pi
    target_idx = int(round(angle / (2.0 * math.pi) * args.num_rays)) % args.num_rays
    return float(np.min(rays)), float(rays[target_idx]), float(rays[0])


def new_track(env_id, episode_id, state, args):
    pos = as_vec3(state["pos"])
    target = as_vec3(state["target"])
    obs_target = as_vec3(state["obs_target"])
    axis = target - pos
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        axis_norm = 1.0
    axis_unit = axis / axis_norm
    signed = float(np.dot(pos - target, axis_unit))
    waypoint = int(state["route_goals_reached"])
    return {
        "episode": episode_id,
        "env": env_id,
        "waypoint": waypoint,
        "axis_unit": axis_unit,
        "start_pos": pos,
        "target": target,
        "obs_target_start": obs_target,
        "start_dist": float(state["direct_dist"]),
        "start_wall_clearance": wall_clearance_xy(pos, args.room_size),
        "start_wall_side": wall_side_xy(pos, args.room_size),
        "target_wall_clearance": wall_clearance_xy(target, args.room_size),
        "obs_target_start_wall_clearance": wall_clearance_xy(obs_target, args.room_size),
        "obs_target_start_target_dist": float(np.linalg.norm(obs_target - target)),
        "steps": 0,
        "min_dist": float("inf"),
        "min_dist_step": -1,
        "speed_at_min": 0.0,
        "toward_speed_at_min": 0.0,
        "vz_at_min": 0.0,
        "min_pos": pos,
        "visible_at_min": 0,
        "min_ray_at_min": math.nan,
        "target_ray_at_min": math.nan,
        "front_ray_at_min": math.nan,
        "clearance_at_min": math.nan,
        "wall_clearance_at_min": math.nan,
        "wall_side_at_min": "none",
        "room_clearance_at_min": math.nan,
        "room_boundary_at_min": "none",
        "obs_target_min": obs_target,
        "obs_target_min_wall_clearance": wall_clearance_xy(obs_target, args.room_size),
        "obs_target_min_target_dist": float(np.linalg.norm(obs_target - target)),
        "visible_steps": 0,
        "obs_target_final_steps": 0,
        "last_pos": pos,
        "last_dist": float(state["direct_dist"]),
        "last_speed": float(state["speed"]),
        "last_vz": float(as_vec3(state["vel"])[2]),
        "last_toward_speed": 0.0,
        "last_clearance": float(state["clearance"]),
        "last_wall_clearance": wall_clearance_xy(pos, args.room_size),
        "last_wall_side": wall_side_xy(pos, args.room_size),
        "last_room_clearance": room_clearance(pos, args),
        "last_room_boundary": room_boundary(pos, args),
        "crossed_goal_plane": 0,
        "cross_dist": math.nan,
        "cross_speed": math.nan,
        "max_overshoot": max(0.0, signed),
        "prev_signed": signed,
    }


def update_track(track, state, obs_row, args):
    track["steps"] += 1
    pos = as_vec3(state["pos"])
    vel = as_vec3(state["vel"])
    target = as_vec3(state["target"])
    target_delta = target - pos
    dist = float(state["direct_dist"])
    speed = float(state["speed"])
    visible = int(state["target_visible"])
    min_ray, target_ray, front_ray = ray_metrics(obs_row, state, args)
    toward_speed = 0.0
    if dist > 1e-6:
        toward_speed = float(np.dot(vel, target_delta) / dist)

    obs_target = as_vec3(state["obs_target"])
    if float(np.linalg.norm(obs_target - target)) < 0.05:
        track["obs_target_final_steps"] += 1
    if visible:
        track["visible_steps"] += 1

    signed = float(np.dot(pos - target, track["axis_unit"]))
    track["max_overshoot"] = max(track["max_overshoot"], signed)
    if not track["crossed_goal_plane"] and track["prev_signed"] < 0.0 <= signed:
        track["crossed_goal_plane"] = 1
        track["cross_dist"] = dist
        track["cross_speed"] = speed
    track["prev_signed"] = signed

    if dist < track["min_dist"]:
        track["min_dist"] = dist
        track["min_dist_step"] = track["steps"]
        track["speed_at_min"] = speed
        track["toward_speed_at_min"] = toward_speed
        track["vz_at_min"] = float(vel[2])
        track["min_pos"] = pos
        track["visible_at_min"] = visible
        track["min_ray_at_min"] = min_ray
        track["target_ray_at_min"] = target_ray
        track["front_ray_at_min"] = front_ray
        track["clearance_at_min"] = float(state["clearance"])
        track["wall_clearance_at_min"] = wall_clearance_xy(pos, args.room_size)
        track["wall_side_at_min"] = wall_side_xy(pos, args.room_size)
        track["room_clearance_at_min"] = room_clearance(pos, args)
        track["room_boundary_at_min"] = room_boundary(pos, args)
        track["obs_target_min"] = obs_target
        track["obs_target_min_wall_clearance"] = wall_clearance_xy(obs_target, args.room_size)
        track["obs_target_min_target_dist"] = float(np.linalg.norm(obs_target - target))

    track["last_pos"] = pos
    track["last_dist"] = dist
    track["last_speed"] = speed
    track["last_vz"] = float(vel[2])
    track["last_toward_speed"] = toward_speed
    track["last_clearance"] = float(state["clearance"])
    track["last_wall_clearance"] = wall_clearance_xy(pos, args.room_size)
    track["last_wall_side"] = wall_side_xy(pos, args.room_size)
    track["last_room_clearance"] = room_clearance(pos, args)
    track["last_room_boundary"] = room_boundary(pos, args)


def finish_track(track, status, reached, args, event_state=None):
    steps = max(int(track["steps"]), 1)
    target = track["target"]
    start_pos = track["start_pos"]
    obs_target_start = track["obs_target_start"]
    obs_target_min = track["obs_target_min"]
    final_pos = track["last_pos"]
    final_dist = track["last_dist"]
    final_speed = track["last_speed"]
    final_vz = track["last_vz"]
    final_toward_speed = track["last_toward_speed"]
    final_clearance = track["last_clearance"]
    final_wall_clearance = track["last_wall_clearance"]
    final_wall_side = track["last_wall_side"]
    final_room_clearance = track["last_room_clearance"]
    final_room_boundary = track["last_room_boundary"]
    final_state_source = "last_live"
    event_dist = math.nan
    event_speed = math.nan
    if event_state is not None and (reached or "last_episode_pos" not in event_state):
        final_pos = as_vec3(event_state["pos"])
        final_dist = float(event_state["direct_dist"])
        final_speed = float(event_state["speed"])
        final_vz = float(as_vec3(event_state["vel"])[2])
        final_clearance = float(event_state["clearance"])
        final_wall_clearance = wall_clearance_xy(final_pos, args.room_size)
        final_wall_side = wall_side_xy(final_pos, args.room_size)
        final_room_clearance = room_clearance(final_pos, args)
        final_room_boundary = room_boundary(final_pos, args)
        final_state_source = "event"
        final_target_delta = target - final_pos
        if final_dist > 1e-6:
            final_toward_speed = float(np.dot(as_vec3(event_state["vel"]), final_target_delta) / final_dist)
    elif event_state is not None and "last_episode_pos" in event_state:
        final_pos = as_vec3(event_state["last_episode_pos"])
        final_dist = float(event_state["last_episode_dist"])
        final_vel = as_vec3(event_state["last_episode_vel"])
        final_speed = float(np.linalg.norm(final_vel))
        final_vz = float(final_vel[2])
        final_clearance = float(event_state["last_episode_clearance"])
        final_wall_clearance = wall_clearance_xy(final_pos, args.room_size)
        final_wall_side = wall_side_xy(final_pos, args.room_size)
        final_room_clearance = room_clearance(final_pos, args)
        final_room_boundary = room_boundary(final_pos, args)
        final_state_source = "last_episode"
        final_target_delta = target - final_pos
        if final_dist > 1e-6:
            final_toward_speed = float(np.dot(final_vel, final_target_delta) / final_dist)
    if reached and event_state is not None:
        event_dist = float(event_state["route_goal_event_dist"])
        event_speed = float(event_state["route_goal_event_speed"])
    outcome = "reached" if reached else f"failed_{status}"
    return {
        "episode": track["episode"],
        "env": track["env"],
        "waypoint": track["waypoint"],
        "outcome": outcome,
        "status": status,
        "reached": int(reached),
        "steps": track["steps"],
        "start_dist": track["start_dist"],
        "min_dist": track["min_dist"],
        "min_dist_step": track["min_dist_step"],
        "speed_at_min": track["speed_at_min"],
        "toward_speed_at_min": track["toward_speed_at_min"],
        "vz_at_min": track["vz_at_min"],
        "min_x": float(track["min_pos"][0]),
        "min_y": float(track["min_pos"][1]),
        "min_z": float(track["min_pos"][2]),
        "event_dist": event_dist,
        "event_speed": event_speed,
        "crossed_goal_plane": track["crossed_goal_plane"],
        "cross_dist": track["cross_dist"],
        "cross_speed": track["cross_speed"],
        "max_overshoot": max(0.0, float(track["max_overshoot"])),
        "visible_frac": track["visible_steps"] / steps,
        "obs_target_final_frac": track["obs_target_final_steps"] / steps,
        "visible_at_min": track["visible_at_min"],
        "min_ray_at_min": track["min_ray_at_min"],
        "target_ray_at_min": track["target_ray_at_min"],
        "front_ray_at_min": track["front_ray_at_min"],
        "clearance_at_min": track["clearance_at_min"],
        "wall_clearance_at_min": track["wall_clearance_at_min"],
        "wall_side_at_min": track["wall_side_at_min"],
        "room_clearance_at_min": track["room_clearance_at_min"],
        "room_boundary_at_min": track["room_boundary_at_min"],
        "start_x": float(start_pos[0]),
        "start_y": float(start_pos[1]),
        "start_z": float(start_pos[2]),
        "start_wall_clearance": track["start_wall_clearance"],
        "start_wall_side": track["start_wall_side"],
        "final_x": float(final_pos[0]),
        "final_y": float(final_pos[1]),
        "final_z": float(final_pos[2]),
        "final_dist": final_dist,
        "final_speed": final_speed,
        "final_vz": final_vz,
        "final_toward_speed": final_toward_speed,
        "final_clearance": final_clearance,
        "final_wall_clearance": final_wall_clearance,
        "final_wall_side": final_wall_side,
        "final_room_clearance": final_room_clearance,
        "final_room_boundary": final_room_boundary,
        "final_state_source": final_state_source,
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "target_z": float(target[2]),
        "target_wall_clearance": track["target_wall_clearance"],
        "obs_target_start_x": float(obs_target_start[0]),
        "obs_target_start_y": float(obs_target_start[1]),
        "obs_target_start_z": float(obs_target_start[2]),
        "obs_target_start_wall_clearance": track["obs_target_start_wall_clearance"],
        "obs_target_start_target_dist": track["obs_target_start_target_dist"],
        "obs_target_start_is_final": int(track["obs_target_start_target_dist"] < 0.05),
        "obs_target_min_x": float(obs_target_min[0]),
        "obs_target_min_y": float(obs_target_min[1]),
        "obs_target_min_z": float(obs_target_min[2]),
        "obs_target_min_wall_clearance": track["obs_target_min_wall_clearance"],
        "obs_target_min_target_dist": track["obs_target_min_target_dist"],
        "obs_target_min_is_final": int(track["obs_target_min_target_dist"] < 0.05),
    }


def action_for_policy(policy_name, policy, obs, env, args, state, step):
    action_obs = obs
    if args.mask_rays and args.num_rays > 0:
        action_obs = obs.copy()
        action_obs[:, RAY_OBS_START:RAY_OBS_START + args.num_rays] = 1.0

    if policy_name == "zero":
        return zero_action(env)
    if policy_name == "ray":
        return ray_reactive_action(action_obs, env, args, step)
    if policy_name == "pd":
        return pd_hover_action(action_obs, env, args)
    if policy_name == "raypd":
        return ray_pd_action(action_obs, env, args)
    return deterministic_action(policy, action_obs, env, args, state)


def numeric(value):
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.6f}"
    return str(value)


def write_tsv(path, rows, fields):
    with path.open("w") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(numeric(row.get(field, "")) for field in fields) + "\n")


def percent(value):
    return f"{100.0 * value:.1f}%"


def summarize(episodes, waypoints, args):
    total_episodes = len(episodes)
    total_waypoints = len(waypoints)
    reached = [row for row in waypoints if row["reached"]]
    failed = [row for row in waypoints if not row["reached"]]
    successes = [row for row in episodes if row["success"]]

    def rate(rows, pred):
        if not rows:
            return 0.0
        return sum(1 for row in rows if pred(row)) / len(rows)

    def mean(rows, key):
        vals = [float(row[key]) for row in rows if row.get(key, "") != "" and not math.isnan(float(row[key]))]
        return float(np.mean(vals)) if vals else 0.0

    def quantiles(rows, key):
        vals = [float(row[key]) for row in rows if row.get(key, "") != "" and not math.isnan(float(row[key]))]
        if not vals:
            return {"p50": 0.0, "p90": 0.0}
        arr = np.asarray(vals, dtype=np.float32)
        return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}

    by_status = {}
    for row in episodes:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1

    by_goals = {}
    for row in episodes:
        goals = int(row["goals"])
        by_goals[goals] = by_goals.get(goals, 0) + 1

    failed_final_wall_sides = {}
    for row in failed:
        side = row["final_wall_side"]
        failed_final_wall_sides[side] = failed_final_wall_sides.get(side, 0) + 1

    failed_final_boundaries = {}
    for row in failed:
        side = row["final_room_boundary"]
        failed_final_boundaries[side] = failed_final_boundaries.get(side, 0) + 1

    wall_failed_final_wall_sides = {}
    for row in failed:
        if row["status"] != "wall":
            continue
        side = row["final_wall_side"]
        wall_failed_final_wall_sides[side] = wall_failed_final_wall_sides.get(side, 0) + 1

    wall_failed_final_boundaries = {}
    for row in failed:
        if row["status"] != "wall":
            continue
        side = row["final_room_boundary"]
        wall_failed_final_boundaries[side] = wall_failed_final_boundaries.get(side, 0) + 1

    failed_by_waypoint = {}
    obstacle_failed_by_waypoint = {}
    for row in failed:
        waypoint = int(row["waypoint"])
        failed_by_waypoint[waypoint] = failed_by_waypoint.get(waypoint, 0) + 1
        if row["status"] == "obstacle":
            obstacle_failed_by_waypoint[waypoint] = (
                obstacle_failed_by_waypoint.get(waypoint, 0) + 1
            )

    reached_by_waypoint = {}
    for row in reached:
        waypoint = int(row["waypoint"])
        reached_by_waypoint.setdefault(waypoint, []).append(row)

    near_threshold = args.route_goal_radius + 0.25
    summary = {
        "episodes": total_episodes,
        "episode_success_rate": len(successes) / total_episodes if total_episodes else 0.0,
        "episode_status_counts": by_status,
        "episode_goal_counts": by_goals,
        "waypoints": total_waypoints,
        "waypoint_reach_rate": len(reached) / total_waypoints if total_waypoints else 0.0,
        "failed_waypoints": len(failed),
        "failed_by_waypoint": failed_by_waypoint,
        "obstacle_failed_by_waypoint": obstacle_failed_by_waypoint,
        "failed_after_first_goal_rate": rate(failed, lambda row: int(row["waypoint"]) >= 1),
        "failed_near_miss_rate": rate(failed, lambda row: row["min_dist"] < near_threshold),
        "failed_crossed_goal_plane_rate": rate(failed, lambda row: bool(row["crossed_goal_plane"])),
        "failed_hidden_majority_rate": rate(failed, lambda row: row["visible_frac"] < 0.5),
        "failed_obstacle_or_wall_episode_rate": rate(
            failed,
            lambda row: row["status"] in ("obstacle", "wall"),
        ),
        "mean_failed_min_dist": mean(failed, "min_dist"),
        "mean_failed_speed_at_min": mean(failed, "speed_at_min"),
        "mean_failed_vz_at_min": mean(failed, "vz_at_min"),
        "mean_failed_clearance_at_min": mean(failed, "clearance_at_min"),
        "mean_failed_wall_clearance_at_min": mean(failed, "wall_clearance_at_min"),
        "mean_failed_room_clearance_at_min": mean(failed, "room_clearance_at_min"),
        "mean_failed_final_wall_clearance": mean(failed, "final_wall_clearance"),
        "mean_failed_final_room_clearance": mean(failed, "final_room_clearance"),
        "mean_wall_failed_final_wall_clearance": mean(
            [row for row in failed if row["status"] == "wall"],
            "final_wall_clearance",
        ),
        "mean_wall_failed_final_room_clearance": mean(
            [row for row in failed if row["status"] == "wall"],
            "final_room_clearance",
        ),
        "mean_wall_failed_final_z": mean(
            [row for row in failed if row["status"] == "wall"],
            "final_z",
        ),
        "mean_wall_failed_final_vz": mean(
            [row for row in failed if row["status"] == "wall"],
            "final_vz",
        ),
        "mean_wall_failed_min_z": mean(
            [row for row in failed if row["status"] == "wall"],
            "min_z",
        ),
        "mean_wall_failed_vz_at_min": mean(
            [row for row in failed if row["status"] == "wall"],
            "vz_at_min",
        ),
        "mean_failed_target_wall_clearance": mean(failed, "target_wall_clearance"),
        "mean_reached_target_wall_clearance": mean(reached, "target_wall_clearance"),
        "mean_failed_obs_target_start_wall_clearance": mean(
            failed,
            "obs_target_start_wall_clearance",
        ),
        "mean_reached_obs_target_start_wall_clearance": mean(
            reached,
            "obs_target_start_wall_clearance",
        ),
        "mean_failed_obs_target_start_target_dist": mean(
            failed,
            "obs_target_start_target_dist",
        ),
        "mean_reached_obs_target_start_target_dist": mean(
            reached,
            "obs_target_start_target_dist",
        ),
        "failed_final_wall_side_counts": failed_final_wall_sides,
        "failed_final_boundary_counts": failed_final_boundaries,
        "wall_failed_final_wall_side_counts": wall_failed_final_wall_sides,
        "wall_failed_final_boundary_counts": wall_failed_final_boundaries,
        "mean_obstacle_failed_clearance_at_min": mean(
            [row for row in failed if row["status"] == "obstacle"],
            "clearance_at_min",
        ),
        "mean_reached_event_speed": mean(reached, "event_speed"),
        "mean_first_goal_event_speed": mean(reached_by_waypoint.get(0, []), "event_speed"),
        "mean_later_goal_event_speed": mean(
            [row for waypoint, rows in reached_by_waypoint.items() if waypoint >= 1 for row in rows],
            "event_speed",
        ),
        "failed_min_dist_quantiles": quantiles(failed, "min_dist"),
        "failed_clearance_quantiles": quantiles(failed, "clearance_at_min"),
        "reached_event_speed_quantiles": quantiles(reached, "event_speed"),
        "mean_failed_toward_speed_at_min": mean(failed, "toward_speed_at_min"),
        "mean_failed_visible_frac": mean(failed, "visible_frac"),
        "mean_reached_visible_frac": mean(reached, "visible_frac"),
        "mean_failed_obs_target_final_frac": mean(failed, "obs_target_final_frac"),
        "mean_reached_obs_target_final_frac": mean(reached, "obs_target_final_frac"),
    }
    return summary


def print_summary(summary, examples):
    print()
    print("Route arrival diagnostics")
    print("-------------------------")
    print(f"episodes: {summary['episodes']}")
    print(f"episode success: {percent(summary['episode_success_rate'])}")
    print(f"episode statuses: {summary['episode_status_counts']}")
    print(f"episode goals: {summary['episode_goal_counts']}")
    print(f"waypoints: {summary['waypoints']}")
    print(f"waypoint reach: {percent(summary['waypoint_reach_rate'])}")
    print(f"failed by waypoint: {summary['failed_by_waypoint']}")
    print(f"obstacle failures by waypoint: {summary['obstacle_failed_by_waypoint']}")
    print(f"failed after first goal: {percent(summary['failed_after_first_goal_rate'])}")
    print(f"failed near miss: {percent(summary['failed_near_miss_rate'])}")
    print(f"failed crossed goal plane: {percent(summary['failed_crossed_goal_plane_rate'])}")
    print(f"failed hidden majority: {percent(summary['failed_hidden_majority_rate'])}")
    print(f"failed obstacle/wall episodes: {percent(summary['failed_obstacle_or_wall_episode_rate'])}")
    print(f"failed min_dist mean: {summary['mean_failed_min_dist']:.3f}")
    print(f"failed speed_at_min mean: {summary['mean_failed_speed_at_min']:.3f}")
    print(f"failed vz_at_min mean: {summary['mean_failed_vz_at_min']:.3f}")
    print(f"failed toward_speed_at_min mean: {summary['mean_failed_toward_speed_at_min']:.3f}")
    print(f"failed clearance_at_min mean: {summary['mean_failed_clearance_at_min']:.3f}")
    print(f"failed wall_clearance_at_min mean: {summary['mean_failed_wall_clearance_at_min']:.3f}")
    print(f"failed room_clearance_at_min mean: {summary['mean_failed_room_clearance_at_min']:.3f}")
    print(f"failed final_wall_clearance mean: {summary['mean_failed_final_wall_clearance']:.3f}")
    print(f"failed final_room_clearance mean: {summary['mean_failed_final_room_clearance']:.3f}")
    print(f"wall failed final_wall_clearance mean: {summary['mean_wall_failed_final_wall_clearance']:.3f}")
    print(f"wall failed final_room_clearance mean: {summary['mean_wall_failed_final_room_clearance']:.3f}")
    print(f"wall failed final_z mean: {summary['mean_wall_failed_final_z']:.3f}")
    print(f"wall failed final_vz mean: {summary['mean_wall_failed_final_vz']:.3f}")
    print(f"wall failed min_z mean: {summary['mean_wall_failed_min_z']:.3f}")
    print(f"wall failed vz_at_min mean: {summary['mean_wall_failed_vz_at_min']:.3f}")
    print(f"failed target_wall_clearance mean: {summary['mean_failed_target_wall_clearance']:.3f}")
    print(f"reached target_wall_clearance mean: {summary['mean_reached_target_wall_clearance']:.3f}")
    print(
        "failed obs_target_start_wall_clearance mean: "
        f"{summary['mean_failed_obs_target_start_wall_clearance']:.3f}"
    )
    print(
        "reached obs_target_start_wall_clearance mean: "
        f"{summary['mean_reached_obs_target_start_wall_clearance']:.3f}"
    )
    print(
        "failed obs_target_start_target_dist mean: "
        f"{summary['mean_failed_obs_target_start_target_dist']:.3f}"
    )
    print(
        "reached obs_target_start_target_dist mean: "
        f"{summary['mean_reached_obs_target_start_target_dist']:.3f}"
    )
    print(f"failed final wall sides: {summary['failed_final_wall_side_counts']}")
    print(f"failed final room boundaries: {summary['failed_final_boundary_counts']}")
    print(f"wall failed final wall sides: {summary['wall_failed_final_wall_side_counts']}")
    print(f"wall failed final room boundaries: {summary['wall_failed_final_boundary_counts']}")
    print(f"obstacle failed clearance_at_min mean: {summary['mean_obstacle_failed_clearance_at_min']:.3f}")
    print(f"failed visible_frac mean: {summary['mean_failed_visible_frac']:.3f}")
    print(f"reached visible_frac mean: {summary['mean_reached_visible_frac']:.3f}")
    print(f"failed obs_target_final_frac mean: {summary['mean_failed_obs_target_final_frac']:.3f}")
    print(f"reached obs_target_final_frac mean: {summary['mean_reached_obs_target_final_frac']:.3f}")
    print(f"reached event_speed mean: {summary['mean_reached_event_speed']:.3f}")
    print(f"first goal event_speed mean: {summary['mean_first_goal_event_speed']:.3f}")
    print(f"later goal event_speed mean: {summary['mean_later_goal_event_speed']:.3f}")

    if not examples:
        return

    fields = [
        "episode",
        "env",
        "waypoint",
        "status",
        "min_dist",
        "speed_at_min",
        "toward_speed_at_min",
        "vz_at_min",
        "min_z",
        "crossed_goal_plane",
        "cross_dist",
        "visible_frac",
        "target_ray_at_min",
        "clearance_at_min",
        "wall_clearance_at_min",
        "room_clearance_at_min",
        "final_wall_clearance",
        "final_wall_side",
        "final_room_clearance",
        "final_room_boundary",
        "final_z",
        "final_vz",
        "target_wall_clearance",
        "obs_target_start_wall_clearance",
        "obs_target_start_target_dist",
    ]
    widths = {field: len(field) for field in fields}
    rendered = []
    for row in examples:
        out = {}
        for field in fields:
            value = numeric(row.get(field, ""))
            out[field] = value
            widths[field] = max(widths[field], len(value))
        rendered.append(out)

    print()
    print("Closest failed waypoints")
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in rendered:
        print("  ".join(row[field].ljust(widths[field]) for field in fields))


def main():
    cli = parse_args()
    args = make_runtime_args(cli)
    policy_name = cli.policy
    if policy_name == "auto":
        policy_name = cli.checkpoint if cli.checkpoint in ("zero", "pd", "ray", "raypd") else "checkpoint"

    if cli.out_dir is None:
        out_dir = Path(f"experiments/quadrotor_route_diagnostics_{time.strftime('%Y%m%d_%H%M%S')}")
    else:
        out_dir = cli.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    policy = None
    if policy_name == "checkpoint":
        checkpoint = Path(cli.checkpoint)
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        policy = load_policy(env, str(checkpoint), args)

    state = initial_policy_state(args)
    debug = env.debug()
    episode_ids = list(range(args.num_envs))
    next_episode_id = args.num_envs
    tracks = [new_track(i, episode_ids[i], debug[i], args) for i in range(args.num_envs)]
    episodes = []
    waypoints = []

    max_policy_steps = args.max_steps * max(2, math.ceil(cli.episodes / args.num_envs) + 4)
    for step in range(max_policy_steps):
        debug = env.debug()
        for i, row in enumerate(debug):
            update_track(tracks[i], row, obs[i], args)

        actions = action_for_policy(policy_name, policy, obs, env, args, state, step)
        obs, _, terminals, truncations, _ = env.step(actions)
        done = np.logical_or(terminals, truncations).astype(bool)
        post = env.debug()

        for i, row in enumerate(post):
            event = bool(row["route_goal_event"])
            if event:
                status = "reached"
                if done[i]:
                    status = status_name(row["last_episode_status"])
                waypoints.append(finish_track(tracks[i], status, True, args, row))

            if done[i]:
                status = status_name(row["last_episode_status"])
                if not event:
                    waypoints.append(finish_track(tracks[i], status, False, args, row))
                episodes.append(
                    {
                        "episode": episode_ids[i],
                        "env": i,
                        "status": status,
                        "success": int(status == "success"),
                        "goals": int(row["last_episode_goals"]),
                        "final_dist": float(row["last_episode_dist"]),
                        "return": float(row["last_episode_return"]),
                    }
                )
                episode_ids[i] = next_episode_id
                next_episode_id += 1
                tracks[i] = new_track(i, episode_ids[i], row, args)
            elif event:
                tracks[i] = new_track(i, episode_ids[i], row, args)

        reset_done_policy_state(state, done)
        if len(episodes) >= cli.episodes:
            break

    env.close()

    summary = summarize(episodes, waypoints, args)
    failed = sorted(
        [row for row in waypoints if not row["reached"]],
        key=lambda row: (row["min_dist"], -row["speed_at_min"]),
    )
    examples = failed[: cli.examples]

    write_tsv(out_dir / "waypoints.tsv", waypoints, WAYPOINT_FIELDS)
    write_tsv(out_dir / "episodes.tsv", episodes, EPISODE_FIELDS)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "examples.json").write_text(json.dumps(examples, indent=2, sort_keys=True))

    if cli.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary, examples)
        print()
        print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
