#!/usr/bin/env python3
import argparse
import copy
import glob
import os
import sys

from quadrotor4_eval_checkpoints import (
    EXPLORE_V0_PROFILE,
    METRIC_KEYS,
    OBSTACLE_MODES,
    evaluate_checkpoint,
    load_quadrotor_config,
)


CASES = {
    "open": {"obstacle_mode": OBSTACLE_MODES["none"]},
    "center_block": {"obstacle_mode": OBSTACLE_MODES["center_block"]},
    "doorway": {"obstacle_mode": OBSTACLE_MODES["doorway"]},
    "corridor": {"obstacle_mode": OBSTACLE_MODES["corridor"]},
    "maze": {"obstacle_mode": OBSTACLE_MODES["maze"]},
    "random_blocks": {"obstacle_mode": OBSTACLE_MODES["random_blocks"]},
    "layout_mixed": {"obstacle_mode": OBSTACLE_MODES["layout_mixed"]},
    "procedural_indoor": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor"]},
    "procedural_indoor_p0": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_p0"]},
    "procedural_indoor_p1": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_p1"]},
    "procedural_indoor_p2": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_p2"]},
    "procedural_indoor_p3": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_p3"]},
    "procedural_indoor_hard": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_hard"]},
    "procedural_indoor_hard_p0": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_hard_p0"]},
    "procedural_indoor_hard_p1": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_hard_p1"]},
    "procedural_indoor_hard_p2": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_hard_p2"]},
    "procedural_indoor_hard_p3": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_hard_p3"]},
    "procedural_indoor_tight": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_tight"]},
    "procedural_indoor_tight_p0": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_tight_p0"]},
    "procedural_indoor_tight_p1": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_tight_p1"]},
    "procedural_indoor_tight_p2": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_tight_p2"]},
    "procedural_indoor_tight_p3": {"obstacle_mode": OBSTACLE_MODES["procedural_indoor_tight_p3"]},
    "procedural_house": {"obstacle_mode": OBSTACLE_MODES["procedural_house"]},
    "procedural_house_p0": {"obstacle_mode": OBSTACLE_MODES["procedural_house_p0"]},
    "procedural_house_p1": {"obstacle_mode": OBSTACLE_MODES["procedural_house_p1"]},
    "procedural_house_p2": {"obstacle_mode": OBSTACLE_MODES["procedural_house_p2"]},
    "procedural_house_p3": {"obstacle_mode": OBSTACLE_MODES["procedural_house_p3"]},
    "procedural_house_multizone": {"obstacle_mode": OBSTACLE_MODES["procedural_house_multizone"]},
    "procedural_house_multizone_p0": {"obstacle_mode": OBSTACLE_MODES["procedural_house_multizone_p0"]},
    "procedural_house_multizone_p1": {"obstacle_mode": OBSTACLE_MODES["procedural_house_multizone_p1"]},
    "procedural_house_multizone_p2": {"obstacle_mode": OBSTACLE_MODES["procedural_house_multizone_p2"]},
    "procedural_house_multizone_p3": {"obstacle_mode": OBSTACLE_MODES["procedural_house_multizone_p3"]},
}

GAUNTLETS = {
    "core": ("open", "random_blocks", "layout_mixed"),
    "occluded": ("center_block", "doorway", "corridor", "maze", "random_blocks", "layout_mixed"),
    "procedural": (
        "procedural_indoor",
        "procedural_indoor_p0",
        "procedural_indoor_p1",
        "procedural_indoor_p2",
        "procedural_indoor_p3",
    ),
    "procedural_hard": (
        "procedural_indoor_hard",
        "procedural_indoor_hard_p0",
        "procedural_indoor_hard_p1",
        "procedural_indoor_hard_p2",
        "procedural_indoor_hard_p3",
    ),
    "procedural_tight": (
        "procedural_indoor_tight",
        "procedural_indoor_tight_p0",
        "procedural_indoor_tight_p1",
        "procedural_indoor_tight_p2",
        "procedural_indoor_tight_p3",
    ),
    "procedural_house": (
        "procedural_house",
        "procedural_house_p0",
        "procedural_house_p1",
        "procedural_house_p2",
        "procedural_house_p3",
    ),
    "procedural_house_multizone": (
        "procedural_house_multizone",
        "procedural_house_multizone_p0",
        "procedural_house_multizone_p1",
        "procedural_house_multizone_p2",
        "procedural_house_multizone_p3",
    ),
    "full": tuple(CASES),
}


def resolve_checkpoint(args, checkpoint):
    if checkpoint != "latest":
        return checkpoint

    pattern = os.path.join(args["checkpoint_dir"], args["env_name"], "**", "*.bin")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No .bin checkpoints found under {args['checkpoint_dir']}/{args['env_name']}")
    return max(candidates, key=os.path.getctime)


def fmt(value):
    if value == "":
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.6g}"


def set_if_not_none(env, key, value):
    if value is not None:
        env[key] = value


def case_names(parsed):
    if parsed.cases:
        return parsed.cases
    return GAUNTLETS[parsed.gauntlet]


def build_case_args(base_args, parsed, case):
    args = copy.deepcopy(base_args)
    args["reset_state"] = False
    args["vec"]["total_agents"] = parsed.total_agents
    args["vec"]["num_buffers"] = parsed.num_buffers
    args["vec"]["num_threads"] = parsed.num_threads
    args["train"]["horizon"] = parsed.horizon
    if parsed.network is not None:
        args["torch"]["network"] = parsed.network
    if parsed.num_layers is not None:
        args["policy"]["num_layers"] = parsed.num_layers
    if parsed.use_bias is not None:
        args["policy"]["use_bias"] = parsed.use_bias
    if parsed.seed is not None:
        args["env"]["seed"] = parsed.seed

    env = args["env"]
    env.update(EXPLORE_V0_PROFILE)
    env.update(CASES[case])
    set_if_not_none(env, "max_steps", parsed.max_steps)
    set_if_not_none(env, "max_vel", parsed.max_vel)
    set_if_not_none(env, "max_z_vel", parsed.max_z_vel)
    set_if_not_none(env, "num_rays", parsed.num_rays)
    set_if_not_none(env, "ray_length", parsed.ray_length)
    set_if_not_none(env, "explore_goal_fraction", parsed.explore_goal_fraction)
    set_if_not_none(env, "spawn_radius", parsed.spawn_radius)
    set_if_not_none(env, "spawn_radius_min", parsed.spawn_radius_min)
    set_if_not_none(env, "room_size", parsed.room_size)
    return args


def print_table(rows):
    columns = (
        "case",
        "n",
        "success",
        "coverage",
        "new_cell_rate",
        "crashed",
        "obstacle_hit",
        "wall_hit",
        "ground_hit",
        "timeout",
        "min_clearance",
        "score",
    )
    print(",".join(columns))
    for row in rows:
        print(",".join(fmt(row.get(column, "")) for column in columns))


def print_summary(rows):
    metric_names = ("success", "coverage", "new_cell_rate", "crashed", "obstacle_hit", "wall_hit", "timeout")
    summary = {"case": "mean"}
    for metric in metric_names:
        values = [row[metric] for row in rows if metric in row]
        if values:
            summary[metric] = sum(values) / len(values)
    print(
        "summary "
        f"cases={len(rows)} "
        f"success={summary.get('success', 0.0):.6g} "
        f"coverage={summary.get('coverage', 0.0):.6g} "
        f"new_cell_rate={summary.get('new_cell_rate', 0.0):.6g} "
        f"crashed={summary.get('crashed', 0.0):.6g} "
        f"obstacle_hit={summary.get('obstacle_hit', 0.0):.6g} "
        f"timeout={summary.get('timeout', 0.0):.6g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="latest")
    parser.add_argument("--gauntlet", choices=sorted(GAUNTLETS), default="full")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES))
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--total-agents", type=int, default=4096)
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-rollouts", type=int, default=256)
    parser.add_argument("--network", choices=("MinGRU", "MLP", "GELU"))
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--use-bias", type=int, choices=(0, 1))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-vel", type=float)
    parser.add_argument("--max-z-vel", type=float)
    parser.add_argument("--num-rays", type=int)
    parser.add_argument("--ray-length", type=float)
    parser.add_argument("--explore-goal-fraction", type=float)
    parser.add_argument("--spawn-radius", type=float)
    parser.add_argument("--spawn-radius-min", type=float)
    parser.add_argument("--room-size", type=float)
    parsed = parser.parse_args()

    from pufferlib.pufferl import _resolve_backend

    base_args = load_quadrotor_config()
    checkpoint = resolve_checkpoint(base_args, parsed.checkpoint)
    backend = _resolve_backend(base_args)

    rows = []
    for case in case_names(parsed):
        args = build_case_args(base_args, parsed, case)
        metrics = evaluate_checkpoint(
            backend, args, checkpoint, parsed.episodes, parsed.max_rollouts)
        row = {"case": case}
        row.update({key[4:] if key.startswith("env/") else key: value
                    for key, value in metrics.items() if key in METRIC_KEYS})
        rows.append(row)

    print(f"checkpoint {checkpoint}")
    print_table(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
