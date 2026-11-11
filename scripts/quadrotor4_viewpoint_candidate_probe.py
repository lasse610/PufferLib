#!/usr/bin/env python3
import argparse
import csv
import sys

from quadrotor4_frontier_target_distill import (
    FRONTIER_TARGET_MODES,
    OBSTACLE_MODES,
    add_common,
)
from quadrotor4_planner_state_verify import (
    METRIC_COLUMNS,
    clone_args,
    fmt,
    print_rows,
    row_failed,
    run_oracle_case,
)


def candidate_warnings(row, parsed):
    warnings = []
    if parsed.frontier_mode == "region":
        return warnings
    candidates = float(row.get("viewpoint_candidate_count", 0.0) or 0.0)
    viable = float(row.get("viewpoint_viable_count", 0.0) or 0.0)
    miss_rate = float(row.get("viewpoint_miss_rate", 0.0) or 0.0)
    clearance = float(row.get("viewpoint_best_clearance", 0.0) or 0.0)

    if candidates >= parsed.min_candidate_events:
        viable_fraction = viable / max(candidates, 1.0)
        if viable_fraction < parsed.min_viable_fraction:
            warnings.append(f"low_viable_fraction={viable_fraction:.3f}")
        if miss_rate > parsed.max_viewpoint_miss_rate:
            warnings.append(f"high_miss_rate={miss_rate:.3f}")
        if viable > 0.0 and clearance < parsed.min_viewpoint_clearance:
            warnings.append(f"low_best_clearance={clearance:.3f}")

    return warnings


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Probe QDR-278 onboard frontier/viewpoint candidate quality. "
            "This is a diagnostic gate before training against planner labels."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(OBSTACLE_MODES),
        default=["procedural_indoor_hard_p1", "procedural_indoor_hard_p2"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[88, 90, 91])
    parser.add_argument(
        "--frontier-mode",
        choices=tuple(FRONTIER_TARGET_MODES),
        default="onboard",
    )
    parser.add_argument("--min-candidate-events", type=float, default=1.0)
    parser.add_argument("--min-viable-fraction", type=float, default=0.20)
    parser.add_argument("--max-viewpoint-miss-rate", type=float, default=0.80)
    parser.add_argument("--min-viewpoint-clearance", type=float, default=0.25)
    parser.add_argument("--fail-on-candidate-warning", action="store_true")
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
        min_success=0.20,
        min_coverage=0.60,
        max_obstacle_hit=0.08,
        goal_margin=0.02,
        progress_delta=0.002,
        min_target_ray=0.12,
        min_blocked_target_dist=0.25,
        max_absent_under_goal_rate=0.20,
        max_blocked_target_rate=0.40,
        min_stall_stale_marks=1.0,
    )
    parsed = parser.parse_args()

    rows = []
    warnings = []
    for case in parsed.cases:
        for seed in parsed.seeds:
            run = clone_args(parsed, case, seed)
            row = run_oracle_case(run)
            rows.append(row)
            row_warnings = candidate_warnings(row, parsed)
            if row_warnings:
                warnings.append((row, row_warnings))

    print_rows(rows)

    if parsed.csv:
        with open(parsed.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            writer.writerows(
                {key: row.get(key, "") for key in METRIC_COLUMNS} for row in rows
            )
        print(f"wrote {parsed.csv}")

    for row, row_warnings in warnings:
        print(
            "candidate_warning "
            f"case={row.get('case')} seed={row.get('seed')} "
            + " ".join(row_warnings),
            file=sys.stderr,
        )

    failed = [row for row in rows if row_failed(row, parsed)]
    if failed or (warnings and parsed.fail_on_candidate_warning):
        print("viewpoint_candidate_probe_failed", file=sys.stderr)
        return 1

    print("viewpoint_candidate_probe_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
