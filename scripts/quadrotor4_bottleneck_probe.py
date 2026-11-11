#!/usr/bin/env python3
import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib.pufferl import _resolve_backend, unroll_nested_dict

from quadrotor4_eval_headless import (
    REFERENCE_WAYPOINT_PROFILE,
    deterministic_torch_eval,
    load_quadrotor_config,
    parse_obstacle_mode,
    resolve_checkpoint,
    set_if_not_none,
)


AFL038_CHECKPOINT = (
    "experiments/agile/afl038/"
    "reference_waypoint_p1_speed3_clearance_speed_light_dynamics_ft_conservative/"
    "quadrotor/1782216829045/0000000006029312.bin"
)

AFL049_CHECKPOINT = (
    "experiments/agile/afl049/"
    "reference_waypoint_agile_pillar_bottleneck_fixed_std_float_ft/"
    "quadrotor/1782231138355/0000000022020096.bin"
)

MODE_NAMES = {
    53: "open",
    54: "tiny",
    55: "small",
    56: "large",
    57: "mix",
}


def resolve_named_checkpoint(checkpoint):
    if checkpoint == "afl038":
        return AFL038_CHECKPOINT
    if checkpoint == "afl049":
        return AFL049_CHECKPOINT
    return checkpoint


def parse_float_list(value):
    try:
        return [float(item) for item in value.split(",") if item.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated floats, got {value!r}"
        ) from e


def parse_mode_list(value):
    modes = []
    for item in value.split(","):
        item = item.strip()
        if item:
            modes.append(parse_obstacle_mode(item))
    if not modes:
        raise argparse.ArgumentTypeError("at least one obstacle mode is required")
    return modes


def format_float(value, digits=4):
    if value is None:
        return "nan"
    return f"{float(value):.{digits}f}"


def configure_case_args(parsed, case):
    args = load_quadrotor_config()
    args["reset_state"] = False
    if parsed.deterministic:
        args["slowly"] = True
    args["vec"]["total_agents"] = parsed.total_agents
    args["vec"]["num_buffers"] = parsed.num_buffers
    args["vec"]["num_threads"] = parsed.num_threads
    args["train"]["horizon"] = parsed.horizon
    args["torch"]["network"] = "MLP"
    args["policy"]["hidden_size"] = 128
    args["policy"]["num_layers"] = 4
    args["policy"]["expansion_factor"] = 1
    args["policy"]["use_bias"] = 1
    args["env"].update(REFERENCE_WAYPOINT_PROFILE)

    env = args["env"]
    env.update({
        "seed": parsed.seed + case["index"] * parsed.seed_stride,
        "max_steps": parsed.max_steps,
        "obstacle_mode": case["mode"],
        "route_goal_count": parsed.route_goal_count,
        "route_goal_count_min": 0,
        "route_goal_radius": parsed.route_goal_radius,
        "route_reward_mode": parsed.route_reward_mode,
        "route_target_obs_mode": parsed.route_target_obs_mode,
        "route_walls": parsed.route_walls,
        "route_hardstate_probe": 1,
        "route_hardstate_start_x": case["start_x"],
        "route_hardstate_start_y": case["start_y"],
        "route_hardstate_start_z": parsed.z,
        "route_hardstate_target_x": case["target_x"],
        "route_hardstate_target_y": case["target_y"],
        "route_hardstate_target_z": parsed.z,
        "route_hardstate_yaw_deg": case["yaw_deg"],
        "route_hardstate_jitter_xy": parsed.jitter_xy,
        "route_hardstate_jitter_z": parsed.jitter_z,
        "mass_scale_min": parsed.mass_scale_min,
        "mass_scale_max": parsed.mass_scale_max,
        "thrust_scale_min": parsed.thrust_scale_min,
        "thrust_scale_max": parsed.thrust_scale_max,
        "drag_scale_min": parsed.drag_scale_min,
        "drag_scale_max": parsed.drag_scale_max,
        "motor_tau_scale_min": parsed.motor_tau_scale_min,
        "motor_tau_scale_max": parsed.motor_tau_scale_max,
    })
    set_if_not_none(env, "num_rays", parsed.num_rays)
    set_if_not_none(env, "ray_length", parsed.ray_length)
    set_if_not_none(env, "route_clearance_radius", parsed.route_clearance_radius)
    set_if_not_none(env, "route_clearance_scale", parsed.route_clearance_scale)
    set_if_not_none(env, "route_clearance_speed_scale", parsed.route_clearance_speed_scale)
    set_if_not_none(env, "route_clearance_speed_min", parsed.route_clearance_speed_min)
    set_if_not_none(env, "route_clearance_speed_full", parsed.route_clearance_speed_full)
    set_if_not_none(env, "route_goal_speed", parsed.route_goal_speed)
    return args


def evaluate_case(checkpoint, parsed, case):
    args = configure_case_args(parsed, case)
    checkpoint = resolve_checkpoint(args, checkpoint)
    if parsed.quiet_dashboard:
        import pufferlib.pufferl as pufferl_mod
        pufferl_mod.print_dashboard = lambda *args, **kwargs: None
    backend = _resolve_backend(args)
    pufferl = backend.create_pufferl(args)
    try:
        backend.load_weights(pufferl, checkpoint)
        if parsed.deterministic:
            env_logs = deterministic_torch_eval(
                pufferl,
                parsed.episodes,
                parsed.max_rollouts * parsed.horizon,
                parsed.horizon,
            )
            logs = {f"env/{key}": value for key, value in env_logs.items()}
        else:
            logs = {}
            for _ in range(parsed.max_rollouts):
                backend.rollouts(pufferl)
                logs = dict(unroll_nested_dict(backend.eval_log(pufferl)))
                if int(logs.get("env/n", 0)) >= parsed.episodes:
                    break
    finally:
        backend.close(pufferl)
    return logs


def build_cases(parsed):
    cases = []
    idx = 0
    for mode in parsed.obstacle_modes:
        mode_name = MODE_NAMES.get(mode, str(mode))
        for direction in (-1.0, 1.0):
            direction_name = "lr" if direction > 0.0 else "rl"
            for lateral in parsed.laterals:
                lateral_targets = [-lateral] if parsed.cross else [lateral]
                if parsed.cross and parsed.include_straight:
                    lateral_targets.insert(0, lateral)
                for target_lateral in lateral_targets:
                    start_x = -direction * parsed.start_dist
                    target_x = direction * parsed.target_dist
                    start_y = lateral
                    target_y = target_lateral
                    yaw_deg = math.degrees(math.atan2(
                        target_y - start_y, target_x - start_x))
                    path_name = "cross" if target_lateral != lateral else "straight"
                    cases.append({
                        "index": idx,
                        "label": (
                            f"{mode_name}:{direction_name}:{path_name}:"
                            f"y{lateral:+.2f}->{target_lateral:+.2f}"
                        ),
                        "mode": mode,
                        "start_x": start_x,
                        "start_y": start_y,
                        "target_x": target_x,
                        "target_y": target_y,
                        "yaw_deg": yaw_deg,
                    })
                    idx += 1
    return cases


def print_table(rows):
    headers = (
        "case",
        "n",
        "success",
        "obstacle",
        "timeout",
        "ground",
        "goals",
        "min_clear",
        "avg_vel",
        "ep_len",
        "action_delta",
    )
    widths = {header: len(header) for header in headers}
    formatted = []
    for row in rows:
        values = {
            "case": row["case"],
            "n": format_float(row.get("env/n", 0.0), 0),
            "success": format_float(row.get("env/success")),
            "obstacle": format_float(row.get("env/obstacle_hit")),
            "timeout": format_float(row.get("env/timeout")),
            "ground": format_float(row.get("env/ground_hit")),
            "goals": format_float(row.get("env/route_goals")),
            "min_clear": format_float(row.get("env/min_clearance")),
            "avg_vel": format_float(row.get("env/avg_vel")),
            "ep_len": format_float(row.get("env/episode_length"), 1),
            "action_delta": format_float(row.get("env/action_delta")),
        }
        for key, value in values.items():
            widths[key] = max(widths[key], len(value))
        formatted.append(values)

    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for values in formatted:
        print("  ".join(values[header].ljust(widths[header]) for header in headers))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed bottleneck approach states for AFL direct-motor policies."
    )
    parser.add_argument("checkpoint", nargs="?", default="afl038")
    parser.add_argument(
        "--obstacle-modes",
        type=parse_mode_list,
        default=parse_mode_list("agile_pillar_l2_marker_tiny,agile_pillar_l2_marker_small,agile_pillar_l2_marker_large"),
        help="Comma-separated modes/names. Default: tiny,small,large center markers.",
    )
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--total-agents", type=int, default=512)
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-rollouts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--seed-stride", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=768)
    parser.add_argument("--start-dist", type=float, default=0.8)
    parser.add_argument("--target-dist", type=float, default=0.8)
    parser.add_argument("--laterals", type=parse_float_list, default=parse_float_list("-0.65,-0.45,0.45,0.65"))
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--include-straight", action="store_true")
    parser.add_argument("--z", type=float, default=5.0)
    parser.add_argument("--jitter-xy", type=float, default=0.02)
    parser.add_argument("--jitter-z", type=float, default=0.0)
    parser.add_argument("--route-goal-count", type=int, default=1)
    parser.add_argument("--route-goal-radius", type=float, default=0.75)
    parser.add_argument("--route-reward-mode", type=int, default=4)
    parser.add_argument("--route-target-obs-mode", type=int, default=3)
    parser.add_argument("--route-walls", type=int, default=0)
    parser.add_argument("--route-clearance-radius", type=float, default=0.80)
    parser.add_argument("--route-clearance-scale", type=float, default=0.60)
    parser.add_argument("--route-clearance-speed-scale", type=float, default=0.02)
    parser.add_argument("--route-clearance-speed-min", type=float, default=0.90)
    parser.add_argument("--route-clearance-speed-full", type=float, default=1.60)
    parser.add_argument("--route-goal-speed", type=float, default=3.0)
    parser.add_argument("--num-rays", type=int)
    parser.add_argument("--ray-length", type=float)
    parser.add_argument("--mass-scale-min", type=float, default=0.0)
    parser.add_argument("--mass-scale-max", type=float, default=0.0)
    parser.add_argument("--thrust-scale-min", type=float, default=0.0)
    parser.add_argument("--thrust-scale-max", type=float, default=0.0)
    parser.add_argument("--drag-scale-min", type=float, default=0.0)
    parser.add_argument("--drag-scale-max", type=float, default=0.0)
    parser.add_argument("--motor-tau-scale-min", type=float, default=0.0)
    parser.add_argument("--motor-tau-scale-max", type=float, default=0.0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sort-failures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet-dashboard", action=argparse.BooleanOptionalAction, default=True)
    parsed = parser.parse_args()

    checkpoint = resolve_named_checkpoint(parsed.checkpoint)
    if checkpoint != "latest" and not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)

    cases = build_cases(parsed)
    rows = []
    print(f"checkpoint {checkpoint}")
    print(f"cases {len(cases)} deterministic {int(parsed.deterministic)}")
    for case in cases:
        logs = evaluate_case(checkpoint, parsed, case)
        row = {"case": case["label"], **logs}
        rows.append(row)
        print(
            f"case_done {case['label']} "
            f"success={logs.get('env/success', 0.0):.4f} "
            f"obstacle={logs.get('env/obstacle_hit', 0.0):.4f} "
            f"timeout={logs.get('env/timeout', 0.0):.4f}",
            flush=True,
        )

    if parsed.sort_failures:
        rows.sort(key=lambda row: (
            -float(row.get("env/obstacle_hit", 0.0) + row.get("env/timeout", 0.0)),
            float(row.get("env/success", 0.0)),
            row["case"],
        ))
    print_table(rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
