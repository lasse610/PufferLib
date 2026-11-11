#!/usr/bin/env python3
import argparse
import csv
import math
import sys

import torch

from quadrotor4_frontier_target_distill import (
    FRONTIER_TARGET_MODES,
    OBSTACLE_MODES,
    add_common,
    make_vec_config,
    target_labels,
    target_to_actions,
)
from quadrotor4_privileged_distill import step_vec, tensor_views


def direction_ray(obs_row, direction, num_rays, ray_length):
    if num_rays <= 0:
        return 0.0
    dx, dy = float(direction[0]), float(direction[1])
    angle = math.atan2(dy, dx) % (2.0 * math.pi)
    idx = int(round(angle * (num_rays / (2.0 * math.pi)))) % num_rays
    return float(obs_row[26 + idx]) * ray_length


def make_run_args(parsed):
    parsed.obstacle_mode = OBSTACLE_MODES[parsed.mode]
    parsed.explore_frontier_target_mode = parsed.frontier_mode
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Trace deployable exploration planner state from native observations."
    )
    parser.add_argument("--frontier-mode", choices=tuple(FRONTIER_TARGET_MODES), default="onboard")
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--print-steps", type=int, default=64)
    parser.add_argument("--csv", default="")
    add_common(parser)
    parser.set_defaults(
        cpu=True,
        total_agents=1,
        num_buffers=1,
        num_threads=1,
        max_steps=4096,
        max_eval_steps=4096,
        speed_fraction=0.35,
        spawn_radius_min=0.0,
        explore_goal_fraction=0.98,
        room_size=6.0,
        ray_length=4.0,
        safety_slide=True,
        safety_cone_ray_stop=0.25,
        safety_cone_deg=60.0,
        safety_slow_ray=0.70,
        explore_reachable_coverage=True,
        mode="procedural_indoor_hard_p1",
    )
    parsed = make_run_args(parser.parse_args())

    from pufferlib import _C

    device = torch.device("cpu")
    args = make_vec_config(parsed)
    vec = _C.create_vec(args, 0)
    obs, _, terminals = tensor_views(vec, device)
    rows = []

    previous_coverage = 0.0
    last_progress_step = 0
    try:
        vec.reset()
        for step in range(parsed.steps + 1):
            row_obs = obs[0].detach().cpu()
            direction = row_obs[14:16]
            direction_norm = float(torch.linalg.vector_norm(direction).clamp_min(1e-6))
            unit = direction / direction_norm
            coverage = float(row_obs[17])
            exists = float(row_obs[18])
            target_dist = float(row_obs[16]) * max(2.0 * parsed.room_size, 1e-6)
            rays = row_obs[26 : 26 + parsed.num_rays]
            min_ray = float(rays.min()) * parsed.ray_length if parsed.num_rays > 0 else 0.0
            target_ray = direction_ray(row_obs, unit, parsed.num_rays, parsed.ray_length)
            labels = target_labels(obs, parsed)
            actions = target_to_actions(labels, parsed, obs)
            speed = float(torch.linalg.vector_norm(actions[0, 0:2]))
            yaw = float(actions[0, 3])

            if coverage > previous_coverage + 1.0e-4:
                previous_coverage = coverage
                last_progress_step = step
            stale_age = step - last_progress_step
            record = {
                "step": step,
                "coverage": coverage,
                "exists": exists,
                "target_dist_m": target_dist,
                "target_x": float(unit[0]),
                "target_y": float(unit[1]),
                "target_ray_m": target_ray,
                "min_ray_m": min_ray,
                "action_speed": speed,
                "action_yaw": yaw,
                "stale_age": stale_age,
                "terminal": float(terminals[0]),
            }
            rows.append(record)

            if step % parsed.print_steps == 0 or terminals[0] > 0.5:
                print(
                    "probe "
                    f"step={step} cov={coverage:.4f} exists={exists:.1f} "
                    f"tdist={target_dist:.2f} tray={target_ray:.2f} "
                    f"minray={min_ray:.2f} speed={speed:.3f} "
                    f"stale_age={stale_age} done={float(terminals[0]):.0f}",
                    flush=True,
                )

            if terminals[0] > 0.5 or step >= parsed.steps:
                break
            step_vec(vec, actions, device)
    finally:
        logs = vec.log()
        vec.close()

    keys = (
        "n",
        "success",
        "coverage",
        "timeout",
        "obstacle_hit",
        "wall_hit",
        "new_cell_rate",
        "min_clearance",
        "frontier_count",
        "stale_frontier_count",
        "stale_marks",
        "frontier_cluster_count",
        "onboard_target_exists",
        "onboard_target_age",
        "onboard_target_cluster_cells",
        "blocked_cells",
        "viewpoint_candidate_count",
        "viewpoint_viable_count",
        "viewpoint_used_rate",
        "viewpoint_miss_rate",
        "viewpoint_best_score",
        "viewpoint_best_clearance",
        "viewpoint_best_depth",
        "waypoint_depth",
        "onboard_target_switch_rate",
        "onboard_objective_switch_rate",
        "unproductive_objective_switch_rate",
        "committed_waypoint_used_rate",
        "committed_waypoint_drop_rate",
        "committed_waypoint_age",
    )
    print("summary")
    for key in keys:
        if key in logs:
            print(f"{key} {float(logs[key]):.6g}")

    if parsed.csv:
        with open(parsed.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {parsed.csv}")


if __name__ == "__main__":
    raise SystemExit(main())
