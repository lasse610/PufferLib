import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from pufferlib.ocean.quadrotor.eval_hover import evaluate, GRAVITY


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "puffer_quadrotor_177887664537"
    / "model_puffer_quadrotor_000010.pt"
)

BASE_ARGS = {
    "render": False,
    "device": "cpu",
    "episodes": 1,
    "num_envs": 128,
    "max_steps": 768,
    "action_repeat": 10,
    "max_rate": 1.0,
    "max_vel": 0.7,
    "max_z_vel": 0.75,
    "spawn_radius": 2.5,
    "spawn_height": 1.5,
    "init_vel_radius": 0.0,
    "init_omega_radius": 0.0,
    "room_size": 3.0,
    "ceiling_height": 3.0,
    "num_rays": 24,
    "ray_length": 4.0,
    "hover_radius": 0.2,
    "obstacle_mode": "none",
    "obstacle_block_prob": 0.5,
    "obs_noise": 0.0,
    "action_noise": 0.0,
    "action_delay": 0,
    "mass_scale": 1.0,
    "inertia_scale": 0.0,
    "thrust_scale": 1.0,
    "torque_scale": 0.0,
    "drag_scale": 1.0,
    "motor_tau_scale": 1.0,
    "explore_cell_size": 0.25,
    "explore_goal_fraction": 0.85,
    "route_goal_radius": 1.0,
    "route_goal_count": 3,
    "route_step_cost": 0.004,
    "route_proximity_reward": 0.02,
    "route_dist_scale": 0.01,
    "route_progress_scale": 25.0,
    "route_crash_reward": -24.0,
    "route_crash_goal_scale": 4.0,
    "route_crash_dist_scale": 1.5,
    "route_clearance_radius": 0.45,
    "route_clearance_scale": 0.35,
    "route_vel_scale": 0.04,
    "route_arrival_radius": 0.0,
    "route_arrival_bonus": 0.0,
    "route_arrival_vel_scale": 0.0,
    "route_arrival_toward_scale": 0.0,
    "route_target_wall_margin": 0.0,
    "route_goal_speed": 2.0,
    "route_goal_bonus": 1.0,
    "route_reach_bonus": 5.0,
    "route_reward_mode": "dense",
    "route_target_mode": "random",
    "route_blocked_target_prob": 1.0,
    "route_walls": 1,
    "route_direct_obs": 0,
    "route_extra_obs": 0,
    "route_wall_obs": 0,
    "route_perception_obs": 0,
    "task_mode": "indoor_explore",
    "action_mode": "velocity",
    "hidden_size": 128,
    "action_std": 0.1,
    "rnn_name": "Recurrent",
    "rnn_input_size": 128,
    "rnn_hidden_size": 128,
    "pd_kp_pos": 2.0,
    "pd_max_vel": 1.0,
    "pd_kp_vel": 6.0,
    "pd_max_tilt_acc": GRAVITY * 0.364,
    "pd_max_z_acc": GRAVITY * 0.5,
    "pd_kp_att": 20.0,
    "pd_kd_att": 5.0,
}

CASE_SETS = {
    "quick": [
        {
            "name": "center_dist",
            "weight": 0.5,
            "spawn_radius": 1.0,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 128,
        },
        {
            "name": "full_dist",
            "weight": 1.0,
            "spawn_radius": 2.5,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 128,
        },
        {
            "name": "hard_prefix",
            "weight": 1.0,
            "spawn_radius": 2.5,
            "seeds": [1, 16],
            "episodes": 3,
            "num_envs": 1,
        },
    ],
    "blocks": [
        {
            "name": "blocks_center",
            "weight": 0.5,
            "obstacle_mode": "random_blocks",
            "spawn_radius": 1.0,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 128,
        },
        {
            "name": "blocks_full",
            "weight": 1.0,
            "obstacle_mode": "random_blocks",
            "spawn_radius": 2.5,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 128,
        },
        {
            "name": "blocks_prefix",
            "weight": 0.25,
            "obstacle_mode": "random_blocks",
            "spawn_radius": 2.5,
            "seeds": [1, 16],
            "episodes": 3,
            "num_envs": 1,
        },
    ],
    "blocks_hard": [
        {
            "name": "blocks_center",
            "weight": 0.75,
            "obstacle_mode": "random_blocks",
            "spawn_radius": 1.0,
            "seeds": [4, 7, 11],
            "episodes": 1,
            "num_envs": 256,
        },
        {
            "name": "blocks_full",
            "weight": 1.0,
            "obstacle_mode": "random_blocks",
            "spawn_radius": 2.5,
            "seeds": [4, 7, 11],
            "episodes": 1,
            "num_envs": 256,
        },
        {
            "name": "blocks_edge",
            "weight": 1.0,
            "obstacle_mode": "random_blocks",
            "spawn_radius": 2.75,
            "seeds": [4, 7, 11],
            "episodes": 1,
            "num_envs": 256,
        },
        {
            "name": "blocks_prefix",
            "weight": 0.25,
            "obstacle_mode": "random_blocks",
            "spawn_radius": 2.5,
            "seeds": [1, 8, 16],
            "episodes": 3,
            "num_envs": 1,
        },
    ],
    "mixed": [
        {
            "name": "open_center",
            "weight": 0.5,
            "obstacle_mode": "none",
            "spawn_radius": 1.0,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 128,
        },
        {
            "name": "open_full",
            "weight": 1.0,
            "obstacle_mode": "none",
            "spawn_radius": 2.5,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 128,
        },
        {
            "name": "blocks_center",
            "weight": 0.5,
            "obstacle_mode": "random_blocks",
            "spawn_radius": 1.0,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 128,
        },
        {
            "name": "blocks_full",
            "weight": 1.0,
            "obstacle_mode": "random_blocks",
            "spawn_radius": 2.5,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 128,
        },
        {
            "name": "mixed_prefix",
            "weight": 1.0,
            "obstacle_mode": "random_blocks",
            "spawn_radius": 2.5,
            "seeds": [1, 8, 16],
            "episodes": 3,
            "num_envs": 1,
        },
    ],
    "full": [
        {
            "name": "center_dist",
            "weight": 0.5,
            "spawn_radius": 1.0,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 256,
        },
        {
            "name": "full_dist",
            "weight": 1.0,
            "spawn_radius": 2.5,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 256,
        },
        {
            "name": "edge_dist",
            "weight": 1.0,
            "spawn_radius": 2.75,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 256,
        },
        {
            "name": "hard_prefix",
            "weight": 1.25,
            "spawn_radius": 2.5,
            "seeds": [1, 8, 16],
            "episodes": 3,
            "num_envs": 1,
        },
        {
            "name": "mixed_prefix",
            "weight": 0.75,
            "spawn_radius": 2.5,
            "seeds": [2, 3, 13, 14, 15],
            "episodes": 3,
            "num_envs": 1,
        },
    ],
    "hard": [
        {
            "name": "edge_dist",
            "weight": 1.0,
            "spawn_radius": 2.75,
            "seeds": [4],
            "episodes": 1,
            "num_envs": 256,
        },
        {
            "name": "hard_prefix",
            "weight": 1.25,
            "spawn_radius": 2.5,
            "seeds": [1, 8, 16],
            "episodes": 3,
            "num_envs": 1,
        },
        {
            "name": "mixed_prefix",
            "weight": 0.75,
            "spawn_radius": 2.5,
            "seeds": [2, 3, 13, 14, 15],
            "episodes": 3,
            "num_envs": 1,
        },
    ],
}


def latest_checkpoint():
    candidates = list((REPO_ROOT / "experiments").glob("puffer_quadrotor_*/model_puffer_quadrotor_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_experiment_dir():
    candidates = list((REPO_ROOT / "experiments").glob("puffer_quadrotor_*"))
    candidates = [path for path in candidates if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_checkpoint_arg(value):
    if value == "latest":
        path = latest_checkpoint()
        if path is None:
            raise SystemExit("No checkpoint found under experiments/")
        return [path]

    if value in ("latest-dir", "latest-experiment", "latest_experiment"):
        path = latest_experiment_dir()
        if path is None:
            raise SystemExit("No experiment directory found under experiments/")
        return sorted(path.glob("model_puffer_quadrotor_*.pt"))

    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if path.is_dir():
        return sorted(path.glob("model_puffer_quadrotor_*.pt"))

    return [path]


def resolve_checkpoints(values, last_n, stride):
    if not values:
        if DEFAULT_CHECKPOINT.exists():
            values = [str(DEFAULT_CHECKPOINT)]
        else:
            values = ["latest"]

    checkpoints = []
    for value in values:
        checkpoints.extend(resolve_checkpoint_arg(value))

    deduped = []
    seen = set()
    for path in checkpoints:
        path = path.resolve()
        if path in seen:
            continue
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        seen.add(path)
        deduped.append(path)

    if stride > 1:
        deduped = deduped[::stride]
    if last_n is not None:
        deduped = deduped[-last_n:]
    return deduped


def short_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def case_args(base, case, seed):
    data = dict(base)
    data.update(
        seed=seed,
        episodes=case["episodes"],
        num_envs=case["num_envs"],
        spawn_radius=case["spawn_radius"],
    )
    for key in (
        "obstacle_mode",
        "room_size",
        "ray_length",
        "num_rays",
        "max_steps",
        "explore_goal_fraction",
    ):
        if key in case:
            data[key] = case[key]
    return SimpleNamespace(**data)


def weighted_mean(rows, key):
    total_n = sum(float(row.get("n", 0.0)) for row in rows)
    if total_n <= 0.0:
        return 0.0
    return sum(float(row.get(key, 0.0)) * float(row.get("n", 0.0)) for row in rows) / total_n


def validation_score(row, goal_fraction):
    success = float(row.get("success", 0.0))
    crashed = float(row.get("crashed", 0.0))
    timeout = float(row.get("timeout", 0.0))
    coverage = float(row.get("coverage", 0.0))
    coverage_term = min(coverage / max(goal_fraction, 1e-6), 1.0)
    return success - 1.5 * crashed - 0.5 * timeout + 0.2 * coverage_term


def aggregate_case(checkpoint_label, case, seed_rows, goal_fraction):
    row = {
        "checkpoint": checkpoint_label,
        "case": case["name"],
        "weight": case["weight"],
        "spawn_radius": case["spawn_radius"],
        "seeds": ",".join(str(seed) for seed in case["seeds"]),
        "n": sum(float(seed_row.get("n", 0.0)) for seed_row in seed_rows),
    }
    for key in (
        "success",
        "crashed",
        "timeout",
        "obstacle_hit",
        "wall_hit",
        "ground_hit",
        "coverage",
        "new_cell_rate",
        "score",
        "avg_vel",
        "action_delta",
    ):
        row[key] = weighted_mean(seed_rows, key)
    row["validation_score"] = validation_score(row, goal_fraction)
    return row


def aggregate_checkpoint(checkpoint_label, case_rows):
    total_weight = sum(float(row["weight"]) for row in case_rows)
    if total_weight <= 0.0:
        total_weight = 1.0

    summary = {
        "checkpoint": checkpoint_label,
        "cases": len(case_rows),
        "n": sum(float(row.get("n", 0.0)) for row in case_rows),
        "validation_score": sum(
            float(row["validation_score"]) * float(row["weight"]) for row in case_rows
        ) / total_weight,
        "cases_failed": sum(1 for row in case_rows if float(row.get("success", 0.0)) < 0.5),
    }
    for key in ("success", "wall_hit", "timeout", "coverage", "new_cell_rate"):
        summary[key] = sum(float(row[key]) * float(row["weight"]) for row in case_rows) / total_weight
    for key in ("crashed", "obstacle_hit", "ground_hit"):
        summary[key] = sum(float(row[key]) * float(row["weight"]) for row in case_rows) / total_weight
    return summary


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(rows, fields):
    widths = {field: len(field) for field in fields}
    rendered = []
    for row in rows:
        out = {}
        for field in fields:
            out[field] = format_value(row.get(field, ""))
            widths[field] = max(widths[field], len(out[field]))
        rendered.append(out)

    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in rendered:
        print("  ".join(row[field].ljust(widths[field]) for field in fields))


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed validation suite for indoor exploration checkpoints")
    parser.add_argument("checkpoints", nargs="*", help="Checkpoint files, experiment directories, or latest")
    parser.add_argument("--suite", choices=sorted(CASE_SETS), default=os.getenv("QUAD_VALIDATE_SUITE", "quick"))
    parser.add_argument("--device", default=os.getenv("QUAD_VALIDATE_DEVICE", "cpu"))
    parser.add_argument("--rnn-name", default=os.getenv("QUAD_VALIDATE_RNN_NAME", "Recurrent"))
    parser.add_argument("--max-vel", type=float, default=float(os.getenv("QUAD_VALIDATE_MAX_VEL", "0.7")))
    parser.add_argument("--max-z-vel", type=float, default=float(os.getenv("QUAD_VALIDATE_MAX_Z_VEL", "0.75")))
    parser.add_argument(
        "--explore-goal-fraction",
        type=float,
        default=float(os.getenv("QUAD_VALIDATE_GOAL_FRACTION", "0.85")),
    )
    parser.add_argument("--checkpoint-stride", type=int, default=int(os.getenv("QUAD_VALIDATE_STRIDE", "1")))
    parser.add_argument("--last-n", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of tables")
    return parser.parse_args()


def main():
    cli = parse_args()
    checkpoints = resolve_checkpoints(cli.checkpoints, cli.last_n, max(cli.checkpoint_stride, 1))
    cases = CASE_SETS[cli.suite]
    base = dict(BASE_ARGS)
    base.update(
        device=cli.device,
        max_vel=cli.max_vel,
        max_z_vel=cli.max_z_vel,
        explore_goal_fraction=cli.explore_goal_fraction,
        rnn_name=cli.rnn_name,
    )

    summaries = []
    all_cases = []
    for checkpoint in checkpoints:
        label = short_path(checkpoint)
        case_rows = []
        for case in cases:
            seed_rows = []
            for seed in case["seeds"]:
                args = case_args(base, case, seed)
                seed_label = f"{label}:{case['name']}:seed{seed}"
                seed_rows.append(evaluate(seed_label, str(checkpoint), args))
            case_rows.append(aggregate_case(label, case, seed_rows, cli.explore_goal_fraction))
        all_cases.extend(case_rows)
        summaries.append(aggregate_checkpoint(label, case_rows))

    summaries.sort(
        key=lambda row: (
            row["validation_score"],
            row["success"],
            -row["crashed"],
            row["coverage"],
        ),
        reverse=True,
    )

    if cli.json:
        print(json.dumps({"summaries": summaries, "cases": all_cases}, indent=2, sort_keys=True))
        return

    print(f"Validation suite: {cli.suite}")
    print()
    if summaries:
        best = summaries[0]
        print(
            "Best checkpoint by validation_score: "
            f"{best['checkpoint']} "
            f"(validation_score={best['validation_score']:.4f}, "
            f"success={best['success']:.4f}, "
            f"crashed={best['crashed']:.4f})"
        )
        print()
    print_table(
        summaries,
        [
            "checkpoint",
            "validation_score",
            "success",
            "crashed",
            "wall_hit",
            "obstacle_hit",
            "timeout",
            "coverage",
            "new_cell_rate",
            "cases_failed",
            "n",
        ],
    )
    print()
    print_table(
        sorted(all_cases, key=lambda row: (row["checkpoint"], row["case"])),
        [
            "checkpoint",
            "case",
            "validation_score",
            "success",
            "crashed",
            "wall_hit",
            "obstacle_hit",
            "timeout",
            "coverage",
            "new_cell_rate",
            "n",
            "seeds",
        ],
    )


if __name__ == "__main__":
    main()
