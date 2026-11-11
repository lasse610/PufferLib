import argparse
import copy
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pufferlib.ocean.quadrotor.diagnose_route import (
    BASE_ARGS,
    EPISODE_FIELDS,
    WAYPOINT_FIELDS,
    action_for_policy,
    finish_track,
    new_track,
    status_name,
    summarize,
    update_track,
    write_tsv,
)
from pufferlib.ocean.quadrotor.eval_hover import (
    initial_policy_state,
    load_policy,
    make_env,
    reset_done_policy_state,
)


OBSTACLE_NAMES = {
    0: "none",
    1: "center_block",
    2: "doorway",
    3: "corridor",
    4: "maze",
    5: "random_blocks",
    6: "mixed_open_blocks",
    7: "layout_mixed",
}


PROFILES = {
    "layout_mixed_goals5": {
        "max_steps": 1536,
        "route_goal_count": 5,
    },
    "layout_mixed_goals10": {
        "max_steps": 2816,
        "route_goal_count": 10,
    },
    "layout_mixed_combo_long": {
        "max_steps": 1536,
        "route_goal_count": 5,
        "route_target_z_min": 1.0,
        "route_target_z_max": 2.0,
        "route_target_wall_margin": 0.50,
        "obs_noise": 0.01,
        "action_noise": 0.02,
        "action_delay": 1,
        "mass_scale": 1.05,
        "inertia_scale": 1.05,
        "thrust_scale": 0.98,
        "torque_scale": 0.98,
        "drag_scale": 1.05,
        "motor_tau_scale": 1.10,
    },
    "layout_mixed_goals3": {
        "max_steps": 1024,
        "route_goal_count": 3,
    },
    "layout_mixed_goals1": {
        "max_steps": 768,
        "route_goal_count": 1,
    },
}


LAYOUT_MIXED_BASE = {
    "num_envs": 1,
    "max_steps": 1536,
    "action_repeat": 1,
    "max_rate": 1.0,
    "max_vel": 1.0,
    "max_z_vel": 0.25,
    "vel_ctrl_kp_vel": 0.0,
    "vel_ctrl_ki_z": 4.0,
    "vel_ctrl_int_limit_z": 0.0,
    "vel_ctrl_max_z_acc": 0.0,
    "spawn_radius": 2.5,
    "spawn_height": 1.5,
    "init_vel_radius": 0.0,
    "init_omega_radius": 0.0,
    "room_size": 3.0,
    "ceiling_height": 3.0,
    "num_rays": 24,
    "ray_length": 4.0,
    "hover_radius": 0.2,
    "obstacle_mode": "layout_mixed",
    "obstacle_block_prob": 0.5,
    "route_goal_radius": 1.0,
    "route_goal_count": 5,
    "route_goal_count_min": 0,
    "route_path_margin": 0.35,
    "route_arrival_radius": 1.5,
    "route_arrival_bonus": 0.04,
    "route_arrival_vel_scale": 0.02,
    "route_target_wall_margin": 1.0,
    "route_target_z_min": 1.5,
    "route_target_z_max": 1.5,
    "route_reward_mode": "dense",
    "route_target_mode": "blocked",
    "route_blocked_target_prob": 1.0,
    "route_walls": 1,
    "route_direct_obs": 0,
    "route_extra_obs": 0,
    "route_wall_obs": 1,
    "route_perception_obs": 0,
    "route_target_obs_mode": "body",
    "task_mode": "indoor_route",
    "action_mode": "velocity",
    "hidden_size": 128,
    "action_std": 0.20,
    "rnn_name": "None",
    "rnn_input_size": 128,
    "rnn_hidden_size": 128,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic paired route diagnostics. Each checkpoint sees "
            "the same one-episode seeds so failures can be compared directly."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint spec as label=path, or just path. Repeat for each policy.",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="layout_mixed_goals5")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-vel", type=float, default=None)
    parser.add_argument("--max-z-vel", type=float, default=None)
    parser.add_argument("--route-goal-count", type=int, default=None)
    parser.add_argument("--route-goal-radius", type=float, default=None)
    parser.add_argument("--route-path-margin", type=float, default=None)
    parser.add_argument("--action-history-obs", type=int, default=None)
    parser.add_argument("--action-scale-xy", type=float, default=1.0)
    parser.add_argument("--action-scale-z", type=float, default=1.0)
    parser.add_argument("--action-scale-yaw", type=float, default=1.0)
    parser.add_argument("--mask-rays", action="store_true")
    return parser.parse_args()


def parse_checkpoint(spec):
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        path = spec
        label = str(Path(path).parent.name + "/" + Path(path).name)
    return label, path


def make_args(cli):
    values = dict(BASE_ARGS)
    values.update(LAYOUT_MIXED_BASE)
    values.update(PROFILES[cli.profile])
    values["seed"] = cli.seed
    values["device"] = cli.device
    values["mask_rays"] = bool(cli.mask_rays)
    for name in (
        "max_steps",
        "max_vel",
        "max_z_vel",
        "route_goal_count",
        "route_goal_radius",
        "route_path_margin",
        "action_history_obs",
    ):
        value = getattr(cli, name)
        if value is not None:
            values[name] = value
    values["action_scale_xy"] = float(cli.action_scale_xy)
    values["action_scale_z"] = float(cli.action_scale_z)
    values["action_scale_yaw"] = float(cli.action_scale_yaw)
    return SimpleNamespace(**values)


def obstacle_name(row):
    mode = int(row.get("active_obstacle_mode", row.get("obstacle_mode", -1)))
    return OBSTACLE_NAMES.get(mode, f"obstacle_{mode}")


def load_policies(checkpoints, args):
    env = make_env(args)
    env.reset(seed=args.seed)
    policies = []
    for label, path in checkpoints:
        if path in ("zero", "pd", "ray", "raypd"):
            policies.append((label, path, None))
            continue
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        policies.append((label, "checkpoint", load_policy(env, str(checkpoint), args)))
    env.close()
    return policies


def add_context(row, label, episode_seed, layout):
    row["checkpoint"] = label
    row["episode_seed"] = episode_seed
    row["layout"] = layout
    return row


def run_episode(policy_name, policy, base_args, label, episode_id, episode_seed):
    args = copy.copy(base_args)
    args.seed = int(episode_seed)
    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    state = initial_policy_state(args)
    debug = env.debug()
    layout = obstacle_name(debug[0])
    track = new_track(0, episode_id, debug[0], args)
    episode_rows = []
    waypoint_rows = []

    for step in range(args.max_steps + 8):
        debug = env.debug()
        update_track(track, debug[0], obs[0], args)
        actions = action_for_policy(policy_name, policy, obs, env, args, state, step)
        if args.action_scale_xy != 1.0:
            actions[:, :2] *= args.action_scale_xy
        if args.action_scale_z != 1.0:
            actions[:, 2] *= args.action_scale_z
        if args.action_scale_yaw != 1.0:
            actions[:, 3] *= args.action_scale_yaw
        actions = np.clip(
            actions,
            env.single_action_space.low,
            env.single_action_space.high,
        ).astype(np.float32)
        obs, _, terminals, truncations, _ = env.step(actions)
        done = bool(terminals[0] or truncations[0])
        post = env.debug()[0]
        event = bool(post["route_goal_event"])

        if event:
            status = "reached"
            if done:
                status = status_name(post["last_episode_status"])
            row = finish_track(track, status, True, args, post)
            waypoint_rows.append(add_context(row, label, episode_seed, layout))

        if done:
            status = status_name(post["last_episode_status"])
            if not event:
                row = finish_track(track, status, False, args, post)
                waypoint_rows.append(add_context(row, label, episode_seed, layout))
            episode = {
                "episode": episode_id,
                "env": 0,
                "status": status,
                "success": int(status == "success"),
                "goals": int(post["last_episode_goals"]),
                "final_dist": float(post["last_episode_dist"]),
                "return": float(post["last_episode_return"]),
            }
            episode_rows.append(add_context(episode, label, episode_seed, layout))
            break

        if event:
            track = new_track(0, episode_id, post, args)
        reset_done_policy_state(state, np.asarray([done], dtype=bool))

    if not episode_rows:
        final_debug = env.debug()[0]
        row = finish_track(track, "timeout", False, args, final_debug)
        waypoint_rows.append(add_context(row, label, episode_seed, layout))
        episode_rows.append(
            add_context(
                {
                    "episode": episode_id,
                    "env": 0,
                    "status": "timeout",
                    "success": 0,
                    "goals": int(track["waypoint"]),
                    "final_dist": float(track["last_dist"]),
                    "return": math.nan,
                },
                label,
                episode_seed,
                layout,
            )
        )
    env.close()
    return episode_rows, waypoint_rows


def count_by(rows, key, pred=lambda row: True):
    counts = {}
    for row in rows:
        if not pred(row):
            continue
        value = row[key]
        counts[value] = counts.get(value, 0) + 1
    return counts


def mean(rows, key):
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if not math.isnan(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else 0.0


def augment_summary(summary, episodes, waypoints):
    failed_waypoints = [row for row in waypoints if not int(row["reached"])]
    obstacle_failed = [row for row in failed_waypoints if row["status"] == "obstacle"]
    summary = dict(summary)
    summary["layout_counts"] = count_by(episodes, "layout")
    summary["failure_layout_counts"] = count_by(failed_waypoints, "layout")
    summary["obstacle_failure_layout_counts"] = count_by(obstacle_failed, "layout")
    summary["mean_goals"] = mean(episodes, "goals")
    summary["mean_return"] = mean(episodes, "return")
    return summary


def paired_summary(labels, episodes_by_label):
    baseline = labels[0]
    base_by_seed = {
        int(row["episode_seed"]): row
        for row in episodes_by_label[baseline]
    }
    out = {}
    for label in labels[1:]:
        counts = {
            "both_success": 0,
            "baseline_only_success": 0,
            "candidate_only_success": 0,
            "both_fail": 0,
            "same_status": 0,
            "candidate_worse_goals": 0,
            "candidate_better_goals": 0,
        }
        examples = []
        for row in episodes_by_label[label]:
            seed = int(row["episode_seed"])
            base = base_by_seed.get(seed)
            if base is None:
                continue
            base_success = bool(base["success"])
            cand_success = bool(row["success"])
            if base_success and cand_success:
                counts["both_success"] += 1
            elif base_success and not cand_success:
                counts["baseline_only_success"] += 1
                examples.append((seed, base, row))
            elif cand_success and not base_success:
                counts["candidate_only_success"] += 1
            else:
                counts["both_fail"] += 1
            if base["status"] == row["status"]:
                counts["same_status"] += 1
            if int(row["goals"]) < int(base["goals"]):
                counts["candidate_worse_goals"] += 1
            elif int(row["goals"]) > int(base["goals"]):
                counts["candidate_better_goals"] += 1
        out[label] = {
            "baseline": baseline,
            "counts": counts,
            "baseline_only_failure_examples": [
                {
                    "episode_seed": seed,
                    "layout": cand["layout"],
                    "baseline_status": base["status"],
                    "candidate_status": cand["status"],
                    "baseline_goals": int(base["goals"]),
                    "candidate_goals": int(cand["goals"]),
                    "candidate_final_dist": float(cand["final_dist"]),
                }
                for seed, base, cand in examples[:12]
            ],
        }
    return out


def print_report(result):
    print()
    print("Paired route checkpoint diagnostics")
    print("-----------------------------------")
    print(f"profile: {result['profile']}")
    print(f"episodes: {result['episodes']}")
    print(f"seed range: {result['seed']}..{result['seed'] + result['episodes'] - 1}")
    print()
    for label, summary in result["summaries"].items():
        print(label)
        print(f"  episode success: {100.0 * summary['episode_success_rate']:.2f}%")
        print(f"  statuses: {summary['episode_status_counts']}")
        print(f"  mean goals: {summary['mean_goals']:.3f}")
        print(f"  failed by waypoint: {summary['failed_by_waypoint']}")
        print(f"  failed layouts: {summary['failure_layout_counts']}")
        print(f"  obstacle failed layouts: {summary['obstacle_failure_layout_counts']}")
        print(f"  failed speed_at_min: {summary['mean_failed_speed_at_min']:.3f}")
        print(f"  failed clearance_at_min: {summary['mean_failed_clearance_at_min']:.3f}")
        print(f"  failed visible_frac: {summary['mean_failed_visible_frac']:.3f}")
        print()
    for label, pair in result["paired"].items():
        print(f"{pair['baseline']} vs {label}")
        print(f"  {pair['counts']}")
        for example in pair["baseline_only_failure_examples"][:5]:
            print(f"  baseline-only example: {example}")
        print()


def main():
    cli = parse_args()
    args = make_args(cli)
    checkpoints = [parse_checkpoint(spec) for spec in cli.checkpoint]
    out_dir = cli.out_dir
    if out_dir is None:
        out_dir = Path(
            f"experiments/quadrotor_route_compare_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = load_policies(checkpoints, args)
    all_episodes = []
    all_waypoints = []
    episodes_by_label = {}
    waypoints_by_label = {}

    for label, policy_name, policy in policies:
        label_episodes = []
        label_waypoints = []
        for episode_id in range(cli.episodes):
            episode_seed = cli.seed + episode_id
            episodes, waypoints = run_episode(
                policy_name, policy, args, label, episode_id, episode_seed
            )
            label_episodes.extend(episodes)
            label_waypoints.extend(waypoints)
        episodes_by_label[label] = label_episodes
        waypoints_by_label[label] = label_waypoints
        all_episodes.extend(label_episodes)
        all_waypoints.extend(label_waypoints)

    summaries = {
        label: augment_summary(summarize(episodes_by_label[label], waypoints_by_label[label], args),
                               episodes_by_label[label], waypoints_by_label[label])
        for label, _, _ in policies
    }
    labels = [label for label, _, _ in policies]
    result = {
        "profile": cli.profile,
        "episodes": cli.episodes,
        "seed": cli.seed,
        "args": vars(args),
        "summaries": summaries,
        "paired": paired_summary(labels, episodes_by_label),
    }

    episode_fields = ["checkpoint", "episode_seed", "layout"] + EPISODE_FIELDS
    waypoint_fields = ["checkpoint", "episode_seed", "layout"] + WAYPOINT_FIELDS
    write_tsv(out_dir / "episodes.tsv", all_episodes, episode_fields)
    write_tsv(out_dir / "waypoints.tsv", all_waypoints, waypoint_fields)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    if cli.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_report(result)
        print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
