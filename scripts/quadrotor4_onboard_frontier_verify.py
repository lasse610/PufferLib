#!/usr/bin/env python3
import argparse
import copy
import sys

from quadrotor4_frontier_target_distill import (
    FRONTIER_TARGET_MODES,
    OBSTACLE_MODES,
    add_common,
    run_oracle_eval,
)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return f"{float(value):.6g}"


def make_run_args(parsed, case, seed, frontier_mode):
    run = copy.copy(parsed)
    run.case = case
    run.mode = case
    run.seed = seed
    run.obstacle_mode = OBSTACLE_MODES[case]
    run.explore_frontier_target_mode = frontier_mode
    return run


def main():
    parser = argparse.ArgumentParser(
        description="Verify the deployable onboard frontier target in the real quadrotor env."
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
    parser.add_argument("--min-success", type=float, default=0.35)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--max-obstacle-hit", type=float, default=0.08)
    parser.add_argument("--verbose-runs", action="store_true")
    add_common(parser)
    parser.set_defaults(
        episodes=512,
        total_agents=512,
        num_buffers=1,
        num_threads=8,
        max_steps=4096,
        max_eval_steps=100000,
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
    failed = False
    for case in parsed.cases:
        for seed in parsed.seeds:
            metrics = run_oracle_eval(
                make_run_args(parsed, case, seed, parsed.frontier_mode)
            )
            row = {"case": case, "seed": seed}
            row.update(metrics)
            rows.append(row)
            success = float(metrics.get("success", 0.0))
            coverage = float(metrics.get("coverage", 0.0))
            obstacle = float(metrics.get("obstacle_hit", 0.0))
            failed = failed or success < parsed.min_success
            failed = failed or coverage < parsed.min_coverage
            failed = failed or obstacle > parsed.max_obstacle_hit

    columns = (
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
        "onboard_target_age",
        "onboard_target_cluster_cells",
        "blocked_cells",
        "viewpoint_candidate_count",
        "viewpoint_viable_count",
        "viewpoint_used_rate",
        "viewpoint_miss_rate",
        "viewpoint_best_clearance",
        "waypoint_depth",
        "onboard_target_switch_rate",
        "onboard_objective_switch_rate",
        "unproductive_objective_switch_rate",
        "committed_waypoint_used_rate",
        "committed_waypoint_drop_rate",
        "committed_waypoint_age",
        "min_clearance",
    )
    print(",".join(columns))
    for row in rows:
        print(",".join(fmt(row.get(column, "")) for column in columns))

    if failed:
        print(
            "verify_failed "
            f"min_success={parsed.min_success:.6g} "
            f"min_coverage={parsed.min_coverage:.6g} "
            f"max_obstacle_hit={parsed.max_obstacle_hit:.6g}",
            file=sys.stderr,
        )
        return 1

    print("verify_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
