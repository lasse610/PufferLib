#!/usr/bin/env python3
import argparse
import ctypes
import sys

import numpy as np


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
}


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


def make_args(cli):
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
        "ceiling_height": 3.0,
        "num_rays": cli.num_rays,
        "ray_length": cli.ray_length,
        "obstacle_mode": OBSTACLE_MODES[cli.mode],
        "explore_cell_size": cli.cell_size,
        "explore_goal_fraction": cli.explore_goal_fraction,
        "explore_memory_obs": cli.explore_memory_obs,
        "explore_frontier_progress_scale": cli.frontier_progress_scale,
        "route_wall_obs": 1,
        "action_history_obs": 1,
    })
    return args


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
    return {
        "n": accum["n"],
        **{key: value / accum["n"] for key, value in accum["sums"].items()},
    }


def fill_actions(obs, actions, cli):
    direction = obs[:, 14:16]
    exists = obs[:, 18:19] > 0.5
    norm = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
    unit = direction / norm

    xy = unit * cli.speed_fraction
    if cli.controller == "avoid":
        front = obs[:, 19:20]
        left = obs[:, 20:21]
        right = obs[:, 21:22]
        blocked = front < cli.clearance_stop
        caution = front < cli.clearance_slow
        side = np.where(left >= right, 1.0, -1.0)
        xy[:, 0:1] *= np.where(caution, cli.slow_fraction, 1.0)
        xy[:, 0:1] = np.where(blocked, np.minimum(xy[:, 0:1], 0.0), xy[:, 0:1])
        xy[:, 1:2] += np.where(caution, side * cli.avoid_side_fraction, 0.0)

    actions[:, :] = 0.0
    actions[:, 0:2] = np.clip(np.where(exists, xy, 0.0), -1.0, 1.0)
    actions[:, 3] = np.clip(unit[:, 1] * cli.yaw_fraction, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Drive exploration from the deployable frontier-target observation"
    )
    parser.add_argument("--controller", choices=("direct", "avoid"), default="direct")
    parser.add_argument("--mode", choices=sorted(OBSTACLE_MODES), default="maze")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--total-agents", type=int, default=1024)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-rollouts", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--room-size", type=float, default=3.0)
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--spawn-radius", type=float, default=2.5)
    parser.add_argument("--spawn-height", type=float, default=1.5)
    parser.add_argument("--num-rays", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=4.0)
    parser.add_argument("--max-vel", type=float, default=0.7)
    parser.add_argument("--max-z-vel", type=float, default=0.25)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--explore-goal-fraction", type=float, default=0.85)
    parser.add_argument("--frontier-progress-scale", type=float, default=1.0)
    parser.add_argument("--explore-memory-obs", type=int, choices=(2, 3), default=2)
    parser.add_argument("--speed-fraction", type=float, default=0.55)
    parser.add_argument("--slow-fraction", type=float, default=0.25)
    parser.add_argument("--avoid-side-fraction", type=float, default=0.45)
    parser.add_argument("--yaw-fraction", type=float, default=0.35)
    parser.add_argument("--clearance-slow", type=float, default=0.22)
    parser.add_argument("--clearance-stop", type=float, default=0.12)
    cli = parser.parse_args()

    import pufferlib._C as C

    args = make_args(cli)
    vec = C.create_vec(args, 0)
    obs = ptr_array(vec.obs_ptr, ctypes.c_float, (cli.total_agents, vec.obs_size))
    actions = np.zeros((cli.total_agents, 4), dtype=np.float32)
    accum = None
    try:
        vec.reset()
        for step in range(1, cli.max_rollouts * cli.horizon + 1):
            fill_actions(obs, actions, cli)
            vec.cpu_step(actions.ctypes.data)
            if step % cli.horizon == 0:
                accum = accumulate_logs(accum, vec.log())
                if finalize_logs(accum).get("n", 0.0) >= cli.episodes:
                    break
        accum = accumulate_logs(accum, vec.log())
    finally:
        vec.close()

    logs = finalize_logs(accum)
    keys = (
        "n",
        "success",
        "score",
        "crashed",
        "obstacle_hit",
        "wall_hit",
        "ground_hit",
        "timeout",
        "coverage",
        "explored_cells",
        "new_cell_rate",
        "min_clearance",
        "action_saturation",
        "action_delta",
    )
    print(f"controller {cli.controller}")
    print(f"mode {cli.mode}")
    print(f"explore_memory_obs {cli.explore_memory_obs}")
    for key in keys:
        if key in logs:
            print(f"{key} {logs[key]:.6g}")


if __name__ == "__main__":
    main()
