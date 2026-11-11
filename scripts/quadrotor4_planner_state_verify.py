#!/usr/bin/env python3
import argparse
import copy
import csv
import math
import sys

import torch

from quadrotor4_frontier_target_distill import (
    FRONTIER_TARGET_MODES,
    OBSTACLE_MODES,
    add_common,
    accumulate_logs,
    finalize_logs,
    make_vec_config,
    StatefulRecoveryController,
    target_labels,
    target_to_actions,
)
from quadrotor4_privileged_distill import step_vec, tensor_views


METRIC_COLUMNS = (
    "kind",
    "case",
    "seed",
    "n",
    "success",
    "coverage",
    "timeout",
    "obstacle_hit",
    "new_cell_rate",
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
    "absent_under_goal_rate",
    "blocked_target_rate",
    "max_progress_stall",
)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.6g}"


def clone_args(parsed, case, seed):
    run = copy.copy(parsed)
    run.case = case
    run.mode = case
    run.seed = seed
    run.obstacle_mode = OBSTACLE_MODES[case]
    run.explore_frontier_target_mode = parsed.frontier_mode
    return run


def direction_rays(obs, directions, parsed):
    if parsed.num_rays <= 0:
        return torch.zeros((obs.shape[0],), dtype=torch.float32, device=obs.device)

    angles = torch.atan2(directions[:, 1], directions[:, 0])
    angles = torch.remainder(angles, 2.0 * math.pi)
    idx = torch.round(angles * (parsed.num_rays / (2.0 * math.pi))).long()
    idx = torch.remainder(idx, parsed.num_rays)
    rays = obs[:, 26 : 26 + parsed.num_rays].float()
    return rays.gather(1, idx[:, None]).squeeze(1) * parsed.ray_length


def update_sanity(stats, obs, parsed, progress):
    coverage = obs[:, 17].float()
    exists = obs[:, 18].float() > parsed.exists_threshold
    under_goal = coverage < parsed.explore_goal_fraction - parsed.goal_margin
    stats["under_goal"] += int(under_goal.sum().detach().cpu())
    stats["absent_under_goal"] += int((under_goal & ~exists).sum().detach().cpu())

    direction = obs[:, 14:16].float()
    norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
    unit = direction / norm
    target_dist = obs[:, 16].float() * max(2.0 * parsed.room_size, 1e-6)
    target_ray = direction_rays(obs, unit, parsed)
    blocked = exists & (target_dist > parsed.min_blocked_target_dist) & (
        target_ray < parsed.min_target_ray
    )
    stats["target_exists"] += int(exists.sum().detach().cpu())
    stats["blocked_target"] += int(blocked.sum().detach().cpu())

    first_coverage = float(coverage[0].detach().cpu())
    if first_coverage > progress["best"] + parsed.progress_delta:
        progress["best"] = first_coverage
        progress["last_step"] = progress["step"]
    stats["max_progress_stall"] = max(
        stats["max_progress_stall"], progress["step"] - progress["last_step"]
    )


def finish_sanity(row, stats):
    under_goal = max(stats["under_goal"], 1)
    target_exists = max(stats["target_exists"], 1)
    row["absent_under_goal_rate"] = stats["absent_under_goal"] / under_goal
    row["blocked_target_rate"] = stats["blocked_target"] / target_exists
    row["max_progress_stall"] = stats["max_progress_stall"]


def run_oracle_case(parsed):
    from pufferlib import _C

    device = torch.device("cpu")
    args = make_vec_config(parsed)
    vec = _C.create_vec(args, 0)
    obs, _, terminals = tensor_views(vec, device)
    vec.reset()

    accum = None
    stats = {
        "under_goal": 0,
        "absent_under_goal": 0,
        "target_exists": 0,
        "blocked_target": 0,
        "max_progress_stall": 0,
    }
    progress = {"best": 0.0, "last_step": 0, "step": 0}
    action_controller = (
        StatefulRecoveryController(parsed)
        if getattr(parsed, "safety_stateful_recovery", False)
        else None
    )

    try:
        for step in range(1, parsed.max_eval_steps + 1):
            progress["step"] = step
            update_sanity(stats, obs, parsed, progress)
            labels = target_labels(obs, parsed)
            actions = (
                action_controller.actions(labels, obs)
                if action_controller is not None
                else target_to_actions(labels, parsed, obs)
            )
            step_vec(vec, actions, device)
            if action_controller is not None:
                action_controller.reset_done(terminals > 0.5)

            if step % parsed.log_interval == 0 or bool((terminals > 0.5).any()):
                accum = accumulate_logs(accum, vec.log())
                if finalize_logs(accum).get("n", 0.0) >= parsed.episodes:
                    break
        accum = accumulate_logs(accum, vec.log())
    finally:
        vec.close()

    row = {"kind": "oracle", "case": parsed.case, "seed": parsed.seed}
    row.update(finalize_logs(accum))
    finish_sanity(row, stats)
    return row


def run_stall_case(parsed):
    from pufferlib import _C

    run = copy.copy(parsed)
    run.total_agents = 1
    run.num_buffers = 1
    run.num_threads = 1
    run.max_steps = parsed.stall_steps
    run.max_eval_steps = parsed.stall_steps + 8

    device = torch.device("cpu")
    args = make_vec_config(run)
    vec = _C.create_vec(args, 0)
    obs, _, terminals = tensor_views(vec, device)
    vec.reset()
    zeros = torch.zeros((1, 4), dtype=torch.float32, device=device)

    stats = {
        "under_goal": 0,
        "absent_under_goal": 0,
        "target_exists": 0,
        "blocked_target": 0,
        "max_progress_stall": 0,
    }
    progress = {"best": 0.0, "last_step": 0, "step": 0}
    try:
        for step in range(1, run.max_eval_steps + 1):
            progress["step"] = step
            update_sanity(stats, obs, run, progress)
            step_vec(vec, zeros, device)
            if bool((terminals > 0.5).any()):
                break
        logs = vec.log()
    finally:
        vec.close()

    row = {"kind": "stall", "case": parsed.case, "seed": parsed.seed}
    row.update(logs)
    finish_sanity(row, stats)
    return row


def row_failed(row, parsed):
    if row["kind"] == "stall":
        stale_marks = float(row.get("stale_marks", 0.0))
        coverage = float(row.get("coverage", 0.0))
        return stale_marks < parsed.min_stall_stale_marks and coverage < parsed.explore_goal_fraction

    return (
        float(row.get("success", 0.0)) < parsed.min_success
        or float(row.get("coverage", 0.0)) < parsed.min_coverage
        or float(row.get("obstacle_hit", 0.0)) > parsed.max_obstacle_hit
        or float(row.get("absent_under_goal_rate", 0.0)) > parsed.max_absent_under_goal_rate
        or float(row.get("blocked_target_rate", 0.0)) > parsed.max_blocked_target_rate
    )


def print_rows(rows):
    print(",".join(METRIC_COLUMNS))
    for row in rows:
        print(",".join(fmt(row.get(column, "")) for column in METRIC_COLUMNS))


def main():
    parser = argparse.ArgumentParser(
        description="Validate onboard frontier planner state before training against it."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(OBSTACLE_MODES),
        default=["procedural_indoor_hard_p1", "procedural_indoor_hard_p2"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[88])
    parser.add_argument(
        "--frontier-mode",
        choices=tuple(FRONTIER_TARGET_MODES),
        default="onboard",
    )
    parser.add_argument("--include-stall", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stall-steps", type=int, default=180)
    parser.add_argument("--min-stall-stale-marks", type=float, default=1.0)
    parser.add_argument("--min-success", type=float, default=0.20)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    parser.add_argument("--max-obstacle-hit", type=float, default=0.08)
    parser.add_argument("--goal-margin", type=float, default=0.02)
    parser.add_argument("--progress-delta", type=float, default=0.002)
    parser.add_argument("--min-target-ray", type=float, default=0.12)
    parser.add_argument("--min-blocked-target-dist", type=float, default=0.25)
    parser.add_argument("--max-absent-under-goal-rate", type=float, default=0.20)
    parser.add_argument("--max-blocked-target-rate", type=float, default=0.40)
    parser.add_argument("--csv", default="")
    add_common(parser)
    parser.set_defaults(
        cpu=True,
        episodes=64,
        total_agents=64,
        num_buffers=1,
        num_threads=4,
        max_steps=2048,
        max_eval_steps=20000,
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
    )
    parsed = parser.parse_args()

    rows = []
    for case in parsed.cases:
        for seed in parsed.seeds:
            run = clone_args(parsed, case, seed)
            rows.append(run_oracle_case(run))
            if parsed.include_stall:
                rows.append(run_stall_case(run))

    print_rows(rows)

    if parsed.csv:
        with open(parsed.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in METRIC_COLUMNS} for row in rows)
        print(f"wrote {parsed.csv}")

    failed = [row for row in rows if row_failed(row, parsed)]
    if failed:
        print("planner_state_verify_failed", file=sys.stderr)
        return 1

    print("planner_state_verify_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
