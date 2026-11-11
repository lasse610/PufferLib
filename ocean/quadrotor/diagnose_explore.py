import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pufferlib.ocean.quadrotor.eval_hover import (
    body_to_world,
    deterministic_action,
    initial_policy_state,
    load_policy,
    make_env,
    pd_hover_action,
    ray_reactive_action,
    reset_done_policy_state,
    zero_action,
)
from pufferlib.ocean.quadrotor.validate_explore import BASE_ARGS, DEFAULT_CHECKPOINT


POS_OBS_SCALE = 5.0

CASES = {
    "open_full": {
        "obstacle_mode": "none",
        "spawn_radius": 2.5,
        "seed": 4,
    },
    "blocks_center": {
        "obstacle_mode": "random_blocks",
        "spawn_radius": 1.0,
        "seed": 4,
    },
    "blocks_full": {
        "obstacle_mode": "random_blocks",
        "spawn_radius": 2.5,
        "seed": 4,
    },
    "blocks_edge": {
        "obstacle_mode": "random_blocks",
        "spawn_radius": 2.75,
        "seed": 4,
    },
}


def make_args(cli, case):
    values = dict(BASE_ARGS)
    values.update(CASES[case])
    values.update(
        {
            "device": cli.device,
            "num_envs": cli.num_envs,
            "episodes": cli.episodes,
            "max_steps": cli.max_steps,
            "max_vel": cli.max_vel,
            "explore_goal_fraction": cli.explore_goal_fraction,
            "blocked_clearance": cli.blocked_clearance,
            "render": False,
        }
    )
    return SimpleNamespace(**values)


def ray_masks(num_rays):
    angles = np.arange(num_rays, dtype=np.float32) * (2.0 * np.pi / num_rays)
    angles = (angles + np.pi) % (2.0 * np.pi) - np.pi
    return angles, {
        "front": np.cos(angles) > 0.707,
        "left": np.sin(angles) > 0.5,
        "right": np.sin(angles) < -0.5,
    }


def episode_cells(xy, room_size, cell_size, grid_dim):
    idx = np.floor((xy + room_size) / cell_size).astype(np.int32)
    idx = np.clip(idx, 0, grid_dim - 1)
    return idx[:, 0], idx[:, 1]


def policy_action(kind, policy, obs, env, args, state, step):
    if kind == "zero":
        return zero_action(env)
    if kind == "ray":
        return ray_reactive_action(obs, env, args, step)
    if kind == "pd":
        return pd_hover_action(obs, env, args)
    return deterministic_action(policy, obs, env, args, state)


def status_from_terminal(reward, episode_len, max_steps):
    if reward > 0.0:
        return "success"
    if episode_len >= max_steps:
        return "timeout"
    return "crash"


def safe_mean(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def run_one(label, checkpoint, kind, args):
    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    policy = None if kind in ("zero", "ray", "pd") else load_policy(env, checkpoint, args)
    state = initial_policy_state(args)

    ray_angles, masks = ray_masks(args.num_rays)
    grid_dim = min(int(np.ceil(2.0 * args.room_size / args.explore_cell_size)), 64)
    grid_total = float(grid_dim * grid_dim)
    target_episodes = args.episodes * args.num_envs

    visited = np.zeros((args.num_envs, grid_dim, grid_dim), dtype=bool)
    ep_len = np.zeros(args.num_envs, dtype=np.int32)
    unique_cells = np.zeros(args.num_envs, dtype=np.int32)
    last_new_step = np.zeros(args.num_envs, dtype=np.int32)
    path = np.zeros(args.num_envs, dtype=np.float32)
    prev_xy = np.zeros((args.num_envs, 2), dtype=np.float32)
    start_xy = np.zeros((args.num_envs, 2), dtype=np.float32)

    sums = defaultdict(lambda: np.zeros(args.num_envs, dtype=np.float32))
    counts = defaultdict(lambda: np.zeros(args.num_envs, dtype=np.float32))
    records = []

    max_loops = args.max_steps * args.episodes * 4
    for step in range(max_loops):
        current = obs.copy()
        xy = current[:, 0:2] * args.room_size
        z = args.spawn_height + current[:, 2] * POS_OBS_SCALE
        vel = current[:, 3:6]
        quat = current[:, 6:10]
        rays = current[:, 16 : 16 + args.num_rays] * args.ray_length

        new_episode = ep_len == 0
        if np.any(new_episode):
            visited[new_episode] = False
            unique_cells[new_episode] = 0
            last_new_step[new_episode] = 0
            path[new_episode] = 0.0
            prev_xy[new_episode] = xy[new_episode]
            start_xy[new_episode] = xy[new_episode]
            for value in sums.values():
                value[new_episode] = 0.0
            for value in counts.values():
                value[new_episode] = 0.0

        moved = np.linalg.norm(xy - prev_xy, axis=1)
        path += np.where(new_episode, 0.0, moved)
        prev_xy[:] = xy

        cx, cy = episode_cells(xy, args.room_size, args.explore_cell_size, grid_dim)
        was_new = ~visited[np.arange(args.num_envs), cx, cy]
        visited[np.arange(args.num_envs), cx, cy] = True
        unique_cells += was_new.astype(np.int32)
        last_new_step = np.where(was_new, ep_len, last_new_step)

        actions = policy_action(kind, policy, current, env, args, state, step)
        action_xy_norm = np.linalg.norm(actions[:, 0:2], axis=1)
        action_world = body_to_world(
            quat,
            np.column_stack(
                [
                    actions[:, 0] * args.max_vel,
                    actions[:, 1] * args.max_vel,
                    np.zeros(args.num_envs, dtype=np.float32),
                ]
            ).astype(np.float32),
        )[:, 0:2]

        min_ray = rays.min(axis=1)
        min_ray_idx = rays.argmin(axis=1)
        min_ray_angle = np.rad2deg(ray_angles[min_ray_idx])
        front_clear = rays[:, masks["front"]].min(axis=1)
        left_clear = rays[:, masks["left"]].mean(axis=1)
        right_clear = rays[:, masks["right"]].mean(axis=1)
        side_gap = left_clear - right_clear
        wall_clear = args.room_size - np.maximum(np.abs(xy[:, 0]), np.abs(xy[:, 1]))
        speed = np.linalg.norm(vel, axis=1)

        blocked = front_clear < args.blocked_clearance
        side_known = np.abs(side_gap) > 0.15
        side_choice = np.sign(actions[:, 1])
        side_best = np.sign(side_gap)
        side_aligned = blocked & side_known & (side_choice == side_best)
        side_wrong = blocked & side_known & (side_choice == -side_best)
        hesitant = blocked & (action_xy_norm < 0.15) & (np.abs(actions[:, 3]) < 0.10)
        drive_forward = blocked & (actions[:, 0] > 0.25) & (np.abs(actions[:, 1]) < 0.20)

        pos_norm = np.linalg.norm(xy, axis=1)
        action_norm = np.linalg.norm(action_world, axis=1)
        outward = np.divide(
            np.sum(action_world * xy, axis=1),
            np.maximum(pos_norm * action_norm, 1e-6),
        )
        action_angle = np.rad2deg(np.arctan2(actions[:, 1], actions[:, 0]))
        action_toward_nearest = np.cos(np.deg2rad(action_angle - min_ray_angle))
        near_wall = wall_clear < 0.75

        sums["speed"] += speed
        sums["abs_forward"] += np.abs(actions[:, 0])
        sums["abs_lateral"] += np.abs(actions[:, 1])
        sums["abs_yaw"] += np.abs(actions[:, 3])
        sums["action_xy_norm"] += action_xy_norm
        sums["min_ray"] += min_ray
        sums["front_clear"] += front_clear
        sums["wall_clear"] += wall_clear
        sums["height_error"] += np.abs(z - args.spawn_height)
        counts["steps"] += 1.0
        sums["blocked_steps"] += blocked.astype(np.float32)
        sums["hesitant_blocked"] += hesitant.astype(np.float32)
        sums["drive_forward_blocked"] += drive_forward.astype(np.float32)
        sums["side_aligned"] += side_aligned.astype(np.float32)
        sums["side_wrong"] += side_wrong.astype(np.float32)
        counts["side_known_blocked"] += (blocked & side_known).astype(np.float32)
        sums["near_wall_outward"] += np.where(near_wall, outward, 0.0)
        counts["near_wall"] += near_wall.astype(np.float32)

        obs, rewards, terminals, truncations, infos = env.step(actions)
        done = np.logical_or(terminals, truncations)
        reset_done_policy_state(state, done)

        done_ids = np.flatnonzero(done)
        for env_id in done_ids:
            terminal_len = int(ep_len[env_id] + 1)
            status = status_from_terminal(float(rewards[env_id]), terminal_len, args.max_steps)
            steps = max(float(counts["steps"][env_id]), 1.0)
            blocked_steps = max(float(sums["blocked_steps"][env_id]), 1.0)
            side_cases = max(float(counts["side_known_blocked"][env_id]), 1.0)
            near_wall_steps = max(float(counts["near_wall"][env_id]), 1.0)
            displacement = float(np.linalg.norm(xy[env_id] - start_xy[env_id]))
            records.append(
                {
                    "label": label,
                    "status": status,
                    "episode_length": terminal_len,
                    "reward": float(rewards[env_id]),
                    "unique_cells": int(unique_cells[env_id]),
                    "coverage_proxy": float(unique_cells[env_id] / grid_total),
                    "path_length": float(path[env_id]),
                    "directness": displacement / max(float(path[env_id]), 1e-6),
                    "tail_stale_fraction": float(
                        (terminal_len - int(last_new_step[env_id])) / max(terminal_len, 1)
                    ),
                    "mean_speed": float(sums["speed"][env_id] / steps),
                    "mean_abs_forward": float(sums["abs_forward"][env_id] / steps),
                    "mean_abs_lateral": float(sums["abs_lateral"][env_id] / steps),
                    "mean_abs_yaw": float(sums["abs_yaw"][env_id] / steps),
                    "mean_action_xy_norm": float(sums["action_xy_norm"][env_id] / steps),
                    "mean_min_ray": float(sums["min_ray"][env_id] / steps),
                    "mean_front_clear": float(sums["front_clear"][env_id] / steps),
                    "mean_wall_clear": float(sums["wall_clear"][env_id] / steps),
                    "mean_height_error": float(sums["height_error"][env_id] / steps),
                    "blocked_fraction": float(sums["blocked_steps"][env_id] / steps),
                    "hesitant_when_blocked": float(sums["hesitant_blocked"][env_id] / blocked_steps),
                    "drive_forward_when_blocked": float(
                        sums["drive_forward_blocked"][env_id] / blocked_steps
                    ),
                    "side_alignment": float(sums["side_aligned"][env_id] / side_cases),
                    "side_wrong": float(sums["side_wrong"][env_id] / side_cases),
                    "near_wall_outward": float(sums["near_wall_outward"][env_id] / near_wall_steps),
                    "terminal_min_ray": float(min_ray[env_id]),
                    "terminal_min_ray_angle": float(min_ray_angle[env_id]),
                    "terminal_front_clear": float(front_clear[env_id]),
                    "terminal_wall_clear": float(wall_clear[env_id]),
                    "terminal_action_forward": float(actions[env_id, 0]),
                    "terminal_action_lateral": float(actions[env_id, 1]),
                    "terminal_action_yaw": float(actions[env_id, 3]),
                    "terminal_action_angle": float(action_angle[env_id]),
                    "terminal_action_toward_nearest": float(action_toward_nearest[env_id]),
                    "terminal_xy": [float(xy[env_id, 0]), float(xy[env_id, 1])],
                }
            )

        if len(records) >= target_episodes:
            break

        ep_len += 1
        if done_ids.size:
            ep_len[done_ids] = 0

    env.close()
    return records[:target_episodes]


def by_status(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["status"]].append(record)
    return grouped


def summarize(records):
    n = len(records)
    grouped = by_status(records)
    row = {
        "n": n,
        "success": len(grouped["success"]) / max(n, 1),
        "crash": len(grouped["crash"]) / max(n, 1),
        "timeout": len(grouped["timeout"]) / max(n, 1),
    }
    for key in (
        "episode_length",
        "unique_cells",
        "coverage_proxy",
        "path_length",
        "tail_stale_fraction",
        "mean_speed",
        "mean_abs_forward",
        "mean_abs_lateral",
        "mean_abs_yaw",
        "blocked_fraction",
        "hesitant_when_blocked",
        "drive_forward_when_blocked",
        "side_alignment",
        "side_wrong",
        "near_wall_outward",
    ):
        row[key] = safe_mean(record[key] for record in records)
    return row


def status_summary(records, status):
    group = by_status(records)[status]
    if not group:
        return None
    fields = (
        "episode_length",
        "unique_cells",
        "coverage_proxy",
        "path_length",
        "tail_stale_fraction",
        "mean_speed",
        "mean_abs_lateral",
        "mean_abs_yaw",
        "blocked_fraction",
        "hesitant_when_blocked",
        "drive_forward_when_blocked",
        "terminal_min_ray",
        "terminal_min_ray_angle",
        "terminal_front_clear",
        "terminal_wall_clear",
        "terminal_action_forward",
        "terminal_action_lateral",
        "terminal_action_yaw",
        "terminal_action_angle",
        "terminal_action_toward_nearest",
    )
    row = {"status": status, "n": len(group)}
    row.update({field: safe_mean(record[field] for record in group) for field in fields})
    return row


def print_table(rows, fields):
    widths = {field: len(field) for field in fields}
    formatted = []
    for row in rows:
        out = {}
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            out[field] = str(value)
            widths[field] = max(widths[field], len(out[field]))
        formatted.append(out)

    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in formatted:
        print("  ".join(row[field].ljust(widths[field]) for field in fields))


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose indoor exploration policy failures")
    parser.add_argument("checkpoints", nargs="*", default=[str(DEFAULT_CHECKPOINT)])
    parser.add_argument("--case", choices=sorted(CASES), default="blocks_full")
    parser.add_argument("--include-ray", action="store_true")
    parser.add_argument("--include-zero", action="store_true")
    parser.add_argument("--include-pd", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=768)
    parser.add_argument("--max-vel", type=float, default=0.7)
    parser.add_argument("--explore-goal-fraction", type=float, default=0.85)
    parser.add_argument("--blocked-clearance", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--examples", type=int, default=6)
    return parser.parse_args()


def main():
    cli = parse_args()
    args = make_args(cli, cli.case)

    runs = []
    for checkpoint in cli.checkpoints:
        path = Path(checkpoint)
        runs.append((f"{path.parent.name}/{path.name}", str(path), "checkpoint"))
    if cli.include_ray:
        runs.append(("ray", "ray", "ray"))
    if cli.include_pd:
        runs.append(("pd", "pd", "pd"))
    if cli.include_zero:
        runs.append(("zero", "zero", "zero"))

    all_records = []
    summary_rows = []
    status_rows = []
    for label, checkpoint, kind in runs:
        records = run_one(label, checkpoint, kind, args)
        all_records.extend(records)
        row = {"label": label, "case": cli.case}
        row.update(summarize(records))
        summary_rows.append(row)
        for status in ("success", "crash", "timeout"):
            status_row = status_summary(records, status)
            if status_row:
                status_row["label"] = label
                status_rows.append(status_row)

    print(f"Diagnostic case: {cli.case}")
    print_table(
        summary_rows,
        [
            "label",
            "case",
            "n",
            "success",
            "crash",
            "timeout",
            "episode_length",
            "unique_cells",
            "path_length",
            "tail_stale_fraction",
            "mean_speed",
            "mean_abs_lateral",
            "mean_abs_yaw",
            "blocked_fraction",
            "hesitant_when_blocked",
            "drive_forward_when_blocked",
            "side_alignment",
            "side_wrong",
            "near_wall_outward",
        ],
    )
    print()
    print_table(
        status_rows,
        [
            "label",
            "status",
            "n",
            "episode_length",
            "unique_cells",
            "path_length",
            "tail_stale_fraction",
            "mean_speed",
            "mean_abs_lateral",
            "mean_abs_yaw",
            "blocked_fraction",
            "terminal_min_ray",
            "terminal_min_ray_angle",
            "terminal_front_clear",
            "terminal_wall_clear",
            "terminal_action_forward",
            "terminal_action_lateral",
            "terminal_action_yaw",
            "terminal_action_angle",
            "terminal_action_toward_nearest",
        ],
    )

    failures = [record for record in all_records if record["status"] != "success"]
    failures.sort(key=lambda record: (record["status"] != "crash", record["terminal_min_ray"]))
    if cli.examples > 0 and failures:
        print()
        print_table(
            failures[: cli.examples],
            [
                "label",
                "status",
                "episode_length",
                "unique_cells",
                "path_length",
                "terminal_min_ray",
                "terminal_min_ray_angle",
                "terminal_front_clear",
                "terminal_wall_clear",
                "terminal_action_forward",
                "terminal_action_lateral",
                "terminal_action_yaw",
                "terminal_action_angle",
                "terminal_action_toward_nearest",
            ],
        )

    if cli.json_path:
        cli.json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "case": cli.case,
            "summary": summary_rows,
            "status_summary": status_rows,
            "episodes": all_records,
        }
        cli.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nWrote {cli.json_path}")


if __name__ == "__main__":
    main()
