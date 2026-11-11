#!/usr/bin/env python3
import argparse
import contextlib
import copy
import io
import sys

from quadrotor4_frontier_target_distill import (
    OBSTACLE_MODES,
    add_common,
    run_eval,
    run_oracle_eval,
)


DEFAULT_CASES = (
    "center_block",
    "doorway",
    "corridor",
    "maze",
    "random_blocks",
    "layout_mixed",
)

SUMMARY_GROUPS = {
    "fixed4": ("center_block", "doorway", "corridor", "maze"),
    "fixed5": ("center_block", "doorway", "corridor", "maze", "random_blocks"),
}

METRICS = (
    "success",
    "coverage",
    "crashed",
    "obstacle_hit",
    "wall_hit",
    "ground_hit",
    "timeout",
    "new_cell_rate",
    "min_clearance",
)


def fmt(value):
    if value is None:
        return ""
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return f"{float(value):.6g}"


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def make_run_args(parsed, case, seed):
    run = copy.copy(parsed)
    run.case = case
    run.mode = case
    run.seed = seed
    run.obstacle_mode = OBSTACLE_MODES[case]
    return run


def evaluate_one(parsed, case, seed):
    run = make_run_args(parsed, case, seed)
    if parsed.verbose_runs:
        return run_oracle_eval(run) if parsed.oracle else run_eval(run)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_oracle_eval(run) if parsed.oracle else run_eval(run)


def case_means(rows):
    out = []
    for case in dict.fromkeys(row["case"] for row in rows):
        selected = [row for row in rows if row["case"] == case]
        merged = {"case": case, "seed": "mean", "n": mean(row.get("n") for row in selected)}
        for metric in METRICS:
            merged[metric] = mean(row.get(metric) for row in selected)
        out.append(merged)
    return out


def print_rows(rows):
    columns = ("case", "seed", "n") + METRICS
    print(",".join(columns))
    for row in rows:
        print(",".join(fmt(row.get(column, "")) for column in columns))


def print_group_summaries(rows):
    means = {row["case"]: row for row in case_means(rows)}
    for name, cases in SUMMARY_GROUPS.items():
        present = [means[case] for case in cases if case in means]
        if len(present) != len(cases):
            continue
        parts = [f"group={name}", f"cases={len(present)}"]
        for metric in ("success", "coverage", "crashed", "obstacle_hit", "timeout", "new_cell_rate"):
            parts.append(f"{metric}={fmt(mean(row.get(metric) for row in present))}")
        print("summary " + " ".join(parts))

    eval_cases = [means[case] for case in means if case != "layout_mixed"]
    if eval_cases:
        parts = ["group=all_nonaggregate", f"cases={len(eval_cases)}"]
        for metric in ("success", "coverage", "crashed", "obstacle_hit", "timeout", "new_cell_rate"):
            parts.append(f"{metric}={fmt(mean(row.get(metric) for row in eval_cases))}")
        print("summary " + " ".join(parts))

    if "layout_mixed" in means:
        row = means["layout_mixed"]
        print(
            "aggregate_layout_mixed "
            f"success={fmt(row.get('success'))} "
            f"coverage={fmt(row.get('coverage'))} "
            f"timeout={fmt(row.get('timeout'))} "
            "note=episode_weighted_not_balanced"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Equal-weight per-layout eval for frontier target-distill checkpoints."
    )
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=tuple(OBSTACLE_MODES), default=list(DEFAULT_CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[88, 90, 91])
    parser.add_argument("--verbose-runs", action="store_true")
    add_common(parser)
    parser.set_defaults(
        episodes=4096,
        total_agents=2048,
        num_buffers=2,
        num_threads=16,
        max_steps=2048,
        max_eval_steps=100000,
        speed_fraction=0.35,
        spawn_radius_min=0.0,
        explore_goal_fraction=0.95,
        safety_cone_ray_stop=0.20,
        safety_cone_deg=60.0,
        safety_slow_ray=0.60,
    )
    parsed = parser.parse_args()
    if not parsed.oracle and not parsed.checkpoint:
        parser.error("checkpoint is required unless --oracle is set")
    if parsed.oracle:
        parsed.checkpoint = "oracle"
    return parsed


def main():
    parsed = parse_args()
    rows = []
    print(f"checkpoint {parsed.checkpoint}")
    for case in parsed.cases:
        for seed in parsed.seeds:
            metrics = evaluate_one(parsed, case, seed)
            row = {"case": case, "seed": seed}
            row.update(metrics)
            rows.append(row)
            print(
                "run "
                f"case={case} seed={seed} "
                f"success={fmt(metrics.get('success'))} "
                f"coverage={fmt(metrics.get('coverage'))} "
                f"timeout={fmt(metrics.get('timeout'))} "
                f"obstacle_hit={fmt(metrics.get('obstacle_hit'))}",
                flush=True,
            )

    print_rows(rows)
    print_group_summaries(rows)


if __name__ == "__main__":
    sys.exit(main())
