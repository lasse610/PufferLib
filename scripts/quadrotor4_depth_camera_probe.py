#!/usr/bin/env python3
import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


OBSTACLE_MODES = {
    "center_block": 1,
    "doorway": 2,
    "corridor": 3,
    "maze": 4,
    "random_blocks": 5,
    "layout_mixed": 7,
    "procedural_indoor": 8,
    "procedural_indoor_p0": 9,
    "procedural_indoor_p1": 10,
    "procedural_indoor_p2": 11,
    "procedural_indoor_p3": 12,
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
}

PATCH_START = 60
DEPTH_DEFAULT_WIDTH = 16
DEPTH_DEFAULT_HEIGHT = 12
DEPTH_MAX_WIDTH = 20
DEPTH_MAX_HEIGHT = 15
DEPTH_STRIDE = DEPTH_MAX_WIDTH
DEPTH_SIZE = DEPTH_STRIDE * DEPTH_MAX_HEIGHT
DEPTH_START = PATCH_START
DEPTH_END = DEPTH_START + DEPTH_SIZE


def load_quadrotor_config():
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        from pufferlib.pufferl import load_config

        return load_config("quadrotor")
    finally:
        sys.argv = old_argv


def ptr_array(ptr, ctype, shape):
    size = int(np.prod(shape))
    raw = ctypes.cast(int(ptr), ctypes.POINTER(ctype * size)).contents
    return np.ctypeslib.as_array(raw).reshape(shape)


def make_args(cli, mode, camera_depth_obs=True):
    args = load_quadrotor_config()
    args["reset_state"] = False
    args["vec"]["total_agents"] = cli.total_agents
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = cli.num_threads
    args["train"]["horizon"] = cli.horizon
    args["env"].update({
        "seed": cli.seed,
        "task_mode": 3,
        "action_mode": 1,
        "max_steps": cli.max_steps,
        "action_repeat": cli.action_repeat,
        "max_vel": cli.max_vel,
        "max_z_vel": cli.max_z_vel,
        "spawn_radius": cli.spawn_radius,
        "spawn_radius_min": 0.0,
        "spawn_height": cli.spawn_height,
        "room_size": cli.room_size,
        "ceiling_height": cli.ceiling_height,
        "num_rays": cli.num_rays,
        "ray_length": cli.ray_length,
        "camera_depth_obs": int(camera_depth_obs),
        "camera_depth_width": cli.camera_depth_width,
        "camera_depth_height": cli.camera_depth_height,
        "camera_depth_fov_x_deg": cli.fov_x_deg,
        "camera_depth_fov_y_deg": cli.fov_y_deg,
        "obstacle_mode": OBSTACLE_MODES[mode],
        "explore_cell_size": cli.cell_size,
        "explore_goal_fraction": cli.explore_goal_fraction,
        "explore_memory_obs": 3,
        "explore_frontier_target_mode": 3,
        "route_wall_obs": 1,
        "action_history_obs": 1,
    })
    return args


def camera_shape(cli):
    width = max(1, min(int(cli.camera_depth_width), DEPTH_MAX_WIDTH))
    height = max(1, min(int(cli.camera_depth_height), DEPTH_MAX_HEIGHT))
    return width, height


def extract_depth_frames(obs, cli):
    width, height = camera_shape(cli)
    reserved = obs[:, DEPTH_START:DEPTH_END].copy().reshape(
        obs.shape[0], DEPTH_MAX_HEIGHT, DEPTH_STRIDE
    )
    return reserved[:, :height, :width]


def ascii_depth(frame):
    chars = "@%#*+=-:. "
    rows = []
    for row in frame:
        line = []
        for value in row:
            idx = int(np.clip((1.0 - float(value)) * (len(chars) - 1), 0, len(chars) - 1))
            line.append(chars[idx])
        rows.append("".join(line))
    return "\n".join(rows)


def validate_frames(frames, cli, mode):
    if not np.isfinite(frames).all():
        raise RuntimeError(f"{mode}: depth frame contains NaN/Inf")
    min_value = float(frames.min())
    max_value = float(frames.max())
    if min_value < -1e-5 or max_value > 1.00001:
        raise RuntimeError(
            f"{mode}: depth frame outside [0, 1], min={min_value:.6f} max={max_value:.6f}"
        )

    dynamic_range = float(np.max(frames, axis=(1, 2)).mean() - np.min(frames, axis=(1, 2)).mean())
    near_fraction = float((frames < cli.near_hit_threshold).mean())
    if dynamic_range < cli.min_dynamic_range:
        raise RuntimeError(
            f"{mode}: depth frame is nearly constant, dynamic_range={dynamic_range:.6f}"
        )
    if near_fraction <= 0.0:
        raise RuntimeError(
            f"{mode}: no depth samples hit before {cli.near_hit_threshold:.3f} normalized range"
        )


def run_mode(cli, mode):
    import pufferlib._C as C

    args = make_args(cli, mode, camera_depth_obs=True)
    vec = C.create_vec(args, 0)
    obs_size = vec.obs_size
    obs = ptr_array(vec.obs_ptr, ctypes.c_float, (cli.total_agents, vec.obs_size))
    actions = np.zeros((cli.total_agents, 4), dtype=np.float32)
    try:
        vec.reset()
        for _ in range(cli.warmup_steps):
            vec.cpu_step(actions.ctypes.data)
        frames = extract_depth_frames(obs, cli)
        front_lidar = obs[:, 26].copy()
        validate_frames(frames, cli, mode)
        logs = vec.log()
    finally:
        vec.close()

    disabled_diff = None
    if cli.compare_disabled:
        args = make_args(cli, mode, camera_depth_obs=False)
        vec = C.create_vec(args, 0)
        disabled_obs = ptr_array(vec.obs_ptr, ctypes.c_float, (cli.total_agents, vec.obs_size))
        try:
            vec.reset()
            disabled = extract_depth_frames(disabled_obs, cli)
            disabled_diff = float(np.mean(np.abs(frames - disabled)))
        finally:
            vec.close()

    width, height = camera_shape(cli)
    print(
        f"mode={mode} obs_size={obs_size} camera={width}x{height} "
        f"min={frames.min():.4f} max={frames.max():.4f} mean={frames.mean():.4f} "
        f"center={frames[:, height // 2, width // 2].mean():.4f} "
        f"front_lidar={front_lidar.mean():.4f} "
        f"near_frac={(frames < cli.near_hit_threshold).mean():.4f} "
        f"coverage={float(logs.get('coverage', 0.0)):.4f}"
    )
    if disabled_diff is not None:
        print(f"disabled_patch_mean_abs_diff={disabled_diff:.4f}")

    for agent in range(min(cli.print_agents, cli.total_agents)):
        print(f"agent={agent}")
        print(ascii_depth(frames[agent]))


def main():
    parser = argparse.ArgumentParser(
        description="Validate the opt-in egocentric depth-camera observation."
    )
    parser.add_argument("--modes", nargs="+", choices=sorted(OBSTACLE_MODES),
        default=("doorway", "maze", "procedural_house_multizone_p3"))
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--total-agents", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--room-size", type=float, default=3.0)
    parser.add_argument("--ceiling-height", type=float, default=3.0)
    parser.add_argument("--spawn-radius", type=float, default=2.5)
    parser.add_argument("--spawn-height", type=float, default=1.5)
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--explore-goal-fraction", type=float, default=0.85)
    parser.add_argument("--num-rays", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=4.0)
    parser.add_argument("--max-vel", type=float, default=0.7)
    parser.add_argument("--max-z-vel", type=float, default=0.25)
    parser.add_argument("--camera-depth-width", type=int, default=DEPTH_DEFAULT_WIDTH)
    parser.add_argument("--camera-depth-height", type=int, default=DEPTH_DEFAULT_HEIGHT)
    parser.add_argument("--fov-x-deg", type=float, default=90.0)
    parser.add_argument("--fov-y-deg", type=float, default=60.0)
    parser.add_argument("--near-hit-threshold", type=float, default=0.995)
    parser.add_argument("--min-dynamic-range", type=float, default=0.02)
    parser.add_argument("--print-agents", type=int, default=1)
    parser.add_argument("--compare-disabled", action="store_true")
    cli = parser.parse_args()

    for mode in cli.modes:
        run_mode(cli, mode)


if __name__ == "__main__":
    main()
