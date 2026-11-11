#!/usr/bin/env python3
import argparse
import ctypes
import sys
from collections import deque

import numpy as np

from quadrotor4_maze_frontier_probe import (
    build_world,
    precompute_visibility,
)


POS_OBS_SCALE = 5.0


def body_to_world(quat, vec):
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    vx = vec[:, 0]
    vy = vec[:, 1]
    vz = vec[:, 2]

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    out = np.empty_like(vec)
    out[:, 0] = vx + w * tx + (y * tz - z * ty)
    out[:, 1] = vy + w * ty + (z * tx - x * tz)
    out[:, 2] = vz + w * tz + (x * ty - y * tx)
    return out


def world_to_body(quat, vec):
    conj = quat.copy()
    conj[:, 1:4] *= -1.0
    return body_to_world(conj, vec)


def nearest_cell(x, y, xx, yy, traversable, room_size, cell_size):
    iy = int(np.clip(np.floor((y + room_size) / cell_size), 0, traversable.shape[0] - 1))
    ix = int(np.clip(np.floor((x + room_size) / cell_size), 0, traversable.shape[1] - 1))
    if traversable[iy, ix]:
        return iy, ix
    candidates = np.argwhere(traversable)
    d2 = (xx[candidates[:, 0], candidates[:, 1]] - x) ** 2 + (
        yy[candidates[:, 0], candidates[:, 1]] - y
    ) ** 2
    best = int(np.argmin(d2))
    return int(candidates[best, 0]), int(candidates[best, 1])


def bfs_path(start, goal, traversable):
    parent = {}
    seen = {start}
    queue = deque([start])
    while queue:
        cy, cx = queue.popleft()
        if (cy, cx) == goal:
            break
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if ny < 0 or nx < 0 or ny >= traversable.shape[0] or nx >= traversable.shape[1]:
                continue
            nxt = (ny, nx)
            if traversable[ny, nx] and nxt not in seen:
                seen.add(nxt)
                parent[nxt] = (cy, cx)
                queue.append(nxt)
    if goal not in seen:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def distance_grid(start, traversable):
    dist = -np.ones(traversable.shape, dtype=np.int32)
    dist[start] = 0
    queue = deque([start])
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if ny < 0 or nx < 0 or ny >= traversable.shape[0] or nx >= traversable.shape[1]:
                continue
            if traversable[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[cy, cx] + 1
                queue.append((ny, nx))
    return dist


def select_path(current, seen, poses, masks, pose_to_index, traversable):
    dist = distance_grid(current, traversable)
    best = None
    for pose, mask in zip(poses, masks):
        d = int(dist[pose])
        if d < 0:
            continue
        new_cells = int((mask & ~seen).sum())
        if new_cells <= 0:
            continue
        score = new_cells / max(d, 1)
        if best is None or score > best[0]:
            best = (score, pose, new_cells)
    if best is None:
        return []
    return bfs_path(current, best[1], traversable)


def make_native_args(cli):
    from pufferlib.pufferl import load_config

    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        args = load_config("quadrotor")
    finally:
        sys.argv = old_argv
    args["reset_state"] = False
    args["vec"]["total_agents"] = 1
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 1
    args["train"]["horizon"] = 64
    args["env"].update({
        "seed": cli.seed,
        "task_mode": 3,
        "action_mode": 1,
        "max_steps": cli.max_steps,
        "action_repeat": 10,
        "max_vel": cli.max_vel,
        "max_z_vel": 0.25,
        "spawn_radius": cli.spawn_radius,
        "spawn_radius_min": 0.0,
        "spawn_height": cli.spawn_height,
        "room_size": cli.room_size,
        "ceiling_height": 3.0,
        "num_rays": cli.num_rays,
        "ray_length": cli.ray_length,
        "obstacle_mode": {"center_block": 1, "doorway": 2, "corridor": 3, "maze": 4}[cli.mode],
        "explore_cell_size": cli.cell_size,
        "explore_goal_fraction": cli.explore_goal_fraction,
        "route_wall_obs": 1,
        "action_history_obs": 1,
    })
    return args


def ptr_array(ptr, ctype, shape):
    size = int(np.prod(shape))
    raw = ctypes.cast(int(ptr), ctypes.POINTER(ctype * size)).contents
    return np.ctypeslib.as_array(raw).reshape(shape)


def pose_from_obs(obs, cli):
    wall_start = 26 + cli.num_rays
    x = (obs[:, wall_start + 1] - obs[:, wall_start]) * cli.room_size
    y = (obs[:, wall_start + 3] - obs[:, wall_start + 2]) * cli.room_size
    z = obs[:, wall_start + 5] * 3.0
    return np.column_stack([x, y]).astype(np.float32), z.reshape(-1, 1).astype(np.float32)


def run_episode(vec, obs, rewards, terminals, actions, cli, world):
    boxes, coords, xx, yy, valid_cells, drone_cells, poses, masks, pose_to_index = world
    vec.reset()
    seen = np.zeros_like(valid_cells, dtype=bool)
    path = []
    reached = 0
    stale = 0
    max_coverage = 0.0
    status = "timeout"

    for step in range(cli.max_steps):
        current = obs.copy()
        xy, z = pose_from_obs(current, cli)
        quat = current[:, 6:10]
        cell = nearest_cell(
            float(xy[0, 0]), float(xy[0, 1]), xx, yy, drone_cells,
            cli.room_size, cli.cell_size)
        seen |= masks[pose_to_index[cell]]
        coverage = float(seen.sum() / max(int(valid_cells.sum()), 1))
        max_coverage = max(max_coverage, coverage)
        if coverage >= cli.explore_goal_fraction and cli.stop_on_mirror_success:
            status = "mirror_success"
            break

        if not path or np.linalg.norm(xy[0] - np.array([xx[path[0]], yy[path[0]]], dtype=np.float32)) < cli.waypoint_radius:
            if path:
                path.pop(0)
                reached += 1
            if not path:
                path = select_path(cell, seen, poses, masks, pose_to_index, drone_cells)
                if path and path[0] == cell:
                    path = path[1:]
        if not path:
            action = np.zeros((1, 4), dtype=np.float32)
            stale += 1
        else:
            target = np.array([xx[path[0]], yy[path[0]]], dtype=np.float32)
            delta = target - xy[0]
            dist = float(np.linalg.norm(delta))
            speed = min(cli.max_vel, cli.kp_xy * dist)
            desired_world = np.zeros((1, 3), dtype=np.float32)
            if dist > 1e-6:
                desired_world[0, 0:2] = delta / dist * speed
            desired_world[0, 2] = np.clip(
                cli.kp_z * (cli.spawn_height - float(z[0, 0])),
                -0.25,
                0.25,
            )
            desired_body = world_to_body(quat, desired_world)
            actions[:] = 0.0
            actions[0, 0] = np.clip(desired_body[0, 0] / cli.max_vel, -1.0, 1.0)
            actions[0, 1] = np.clip(desired_body[0, 1] / cli.max_vel, -1.0, 1.0)
            actions[0, 2] = np.clip(desired_world[0, 2] / 0.25, -1.0, 1.0)

        if not path:
            actions[:] = 0.0
        vec.cpu_step(actions.ctypes.data)
        if float(terminals[0]) > 0.5:
            status = "success" if float(rewards[0]) > 0.0 else ("timeout" if step + 1 >= cli.max_steps else "crash")
            break
    return {
        "status": status,
        "steps": step + 1,
        "coverage": max_coverage,
        "waypoints": reached,
        "stale_steps": stale,
    }


def print_table(rows, columns):
    print(",".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        print(",".join(values))


def main():
    parser = argparse.ArgumentParser(description="Privileged frontier-teacher probe in the actual quadrotor env")
    parser.add_argument("--mode", default="maze", choices=("maze", "center_block", "doorway", "corridor"))
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--room-size", type=float, default=3.0)
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--coverage-margin", type=float, default=0.30)
    parser.add_argument("--drone-radius", type=float, default=0.12)
    parser.add_argument("--num-rays", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=4.0)
    parser.add_argument("--spawn-radius", type=float, default=2.5)
    parser.add_argument("--spawn-height", type=float, default=1.5)
    parser.add_argument("--explore-goal-fraction", type=float, default=0.85)
    parser.add_argument("--max-vel", type=float, default=0.7)
    parser.add_argument("--kp-xy", type=float, default=1.6)
    parser.add_argument("--kp-z", type=float, default=1.5)
    parser.add_argument("--waypoint-radius", type=float, default=0.18)
    parser.add_argument("--stop-on-mirror-success", action="store_true")
    cli = parser.parse_args()

    boxes, coords, xx, yy, valid_cells, drone_cells = build_world(cli)
    poses, masks = precompute_visibility(cli, boxes, xx, yy, valid_cells, drone_cells)
    pose_to_index = {pose: idx for idx, pose in enumerate(poses)}
    world = (boxes, coords, xx, yy, valid_cells, drone_cells, poses, masks, pose_to_index)

    import pufferlib._C as C

    native_args = make_native_args(cli)
    vec = C.create_vec(native_args, 0)
    obs = ptr_array(vec.obs_ptr, ctypes.c_float, (1, vec.obs_size))
    rewards = ptr_array(vec.rewards_ptr, ctypes.c_float, (1,))
    terminals = ptr_array(vec.terminals_ptr, ctypes.c_float, (1,))
    actions = np.zeros((1, 4), dtype=np.float32)
    rows = []
    try:
        for episode in range(cli.episodes):
            row = run_episode(vec, obs, rewards, terminals, actions, cli, world)
            row["episode"] = episode
            rows.append(row)
    finally:
        vec.close()

    n = max(len(rows), 1)
    summary = {
        "episodes": len(rows),
        "success": sum(row["status"] == "success" for row in rows) / n,
        "mirror_success": sum(row["status"] == "mirror_success" for row in rows) / n,
        "crash": sum(row["status"] == "crash" for row in rows) / n,
        "timeout": sum(row["status"] == "timeout" for row in rows) / n,
        "coverage": float(np.mean([row["coverage"] for row in rows])),
        "p10_coverage": float(np.quantile([row["coverage"] for row in rows], 0.10)),
        "steps": float(np.mean([row["steps"] for row in rows])),
        "waypoints": float(np.mean([row["waypoints"] for row in rows])),
        "stale_steps": float(np.mean([row["stale_steps"] for row in rows])),
    }
    print_table([summary], (
        "episodes",
        "success",
        "mirror_success",
        "crash",
        "timeout",
        "coverage",
        "p10_coverage",
        "steps",
        "waypoints",
        "stale_steps",
    ))
    print()
    print_table(rows[: min(10, len(rows))], (
        "episode",
        "status",
        "steps",
        "coverage",
        "waypoints",
        "stale_steps",
    ))


if __name__ == "__main__":
    main()
