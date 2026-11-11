#!/usr/bin/env python3
import argparse
import copy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from quadrotor4_frontier_target_distill import OBSTACLE_MODES, add_common, run_eval  # noqa: E402


def parse_int_list(value):
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one target-distill checkpoint across exploration horizons."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--horizons", type=parse_int_list, default=parse_int_list("1024,2048,4096"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("88,90,91"))
    parser.add_argument("--max-eval-steps-multiplier", type=int, default=4)
    add_common(parser)
    parser.set_defaults(
        cpu=True,
        total_agents=256,
        num_buffers=1,
        num_threads=8,
        mode="procedural_indoor_hard_p1",
        max_steps=2048,
        max_eval_steps=16000,
        room_size=6.0,
        spawn_radius_min=0.0,
        explore_goal_fraction=0.95,
        speed_fraction=0.35,
        safety_cone_ray_stop=0.25,
        safety_cone_deg=60.0,
        safety_slow_ray=0.70,
        drain_initial_episodes=True,
        episodes=256,
    )
    parsed = parser.parse_args()
    parsed.obstacle_mode = OBSTACLE_MODES[parsed.mode]

    rows = []
    for seed in parsed.seeds:
        for horizon in parsed.horizons:
            run = copy.copy(parsed)
            run.seed = seed
            run.max_steps = horizon
            run.max_eval_steps = max(
                parsed.max_eval_steps,
                horizon * parsed.max_eval_steps_multiplier,
            )
            metrics = run_eval(run)
            rows.append((seed, horizon, metrics))

    keys = (
        "n",
        "episode_length",
        "success",
        "timeout",
        "obstacle_hit",
        "coverage",
        "new_cell_rate",
        "min_clearance",
    )
    print("horizon_probe_summary")
    print("seed horizon " + " ".join(keys))
    for seed, horizon, metrics in rows:
        values = " ".join(f"{float(metrics.get(key, 0.0)):.6g}" for key in keys)
        print(f"{seed} {horizon} {values}")


if __name__ == "__main__":
    main()
