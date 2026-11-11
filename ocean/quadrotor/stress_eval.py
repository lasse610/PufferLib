import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from pufferlib.ocean.quadrotor.eval_hover import evaluate, print_table, GRAVITY


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINTS = [
    REPO_ROOT / "experiments" / "quadrotor_nav_clone" / "pd_velocity_clone.pt",
    REPO_ROOT
    / "experiments"
    / "puffer_quadrotor_177809531177"
    / "model_puffer_quadrotor_000060.pt",
]

BASE = {
    "render": False,
    "device": "cpu",
    "seed": 1,
    "episodes": 4,
    "num_envs": 128,
    "max_steps": 512,
    "action_repeat": 10,
    "max_rate": 1.0,
    "max_vel": 1.0,
    "max_z_vel": 0.75,
    "spawn_radius": 1.5,
    "spawn_height": 1.5,
    "init_vel_radius": 0.0,
    "init_omega_radius": 0.0,
    "room_size": 2.0,
    "ceiling_height": 3.0,
    "num_rays": 16,
    "ray_length": 3.0,
    "hover_radius": 0.25,
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
    "task_mode": "indoor_nav",
    "action_mode": "velocity",
    "hidden_size": 128,
    "action_std": 0.1,
    "rnn_name": None,
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

SCENARIOS = {
    "open": {},
    "wide_room": {"room_size": 3.0, "spawn_radius": 2.4, "ray_length": 4.0},
    "low_ceiling": {"ceiling_height": 2.0, "spawn_height": 1.1},
    "init_velocity": {"init_vel_radius": 0.5},
    "init_rates": {"init_omega_radius": 1.0},
    "obs_noise": {"obs_noise": 0.03},
    "action_noise": {"action_noise": 0.05},
    "delay_1": {"action_delay": 1},
    "delay_3": {"action_delay": 3},
    "heavy": {"mass_scale": 1.25, "inertia_scale": 1.25},
    "underpowered": {"thrust_scale": 0.85, "torque_scale": 0.85},
    "sluggish_motors": {"motor_tau_scale": 2.0},
    "high_drag": {"drag_scale": 2.0},
    "center_block": {"obstacle_mode": "center_block"},
    "doorway": {"obstacle_mode": "doorway"},
    "corridor": {"obstacle_mode": "corridor"},
    "maze": {"obstacle_mode": "maze", "max_steps": 768},
}


def policy_label(checkpoint):
    if checkpoint in ("pd", "zero"):
        return checkpoint
    path = Path(checkpoint)
    return f"{path.parent.name}/{path.name}"


def make_args(cli, scenario):
    data = dict(BASE)
    data.update(SCENARIOS[scenario])
    data["seed"] = cli.seed
    data["episodes"] = cli.episodes
    data["num_envs"] = cli.num_envs
    data["device"] = cli.device
    data["action_std"] = cli.action_std
    data["rnn_name"] = cli.rnn_name
    data["rnn_input_size"] = cli.rnn_input_size
    data["rnn_hidden_size"] = cli.rnn_hidden_size
    return SimpleNamespace(**data)


def default_checkpoints():
    return [str(path) for path in DEFAULT_CHECKPOINTS if path.exists()]


def parse_args():
    parser = argparse.ArgumentParser(description="Quadrotor stress evaluation gauntlet")
    parser.add_argument("checkpoints", nargs="*", help="model_*.pt checkpoints")
    parser.add_argument("--include-pd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-zero", action="store_true")
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-std", type=float, default=0.1)
    parser.add_argument("--rnn-name", default=None)
    parser.add_argument("--rnn-input-size", type=int, default=128)
    parser.add_argument("--rnn-hidden-size", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    scenarios = cli.scenario or list(SCENARIOS)
    if cli.quick:
        scenarios = ["open", "init_velocity", "delay_1", "underpowered", "doorway"]

    checkpoints = cli.checkpoints or default_checkpoints()
    policies = []
    if cli.include_pd:
        policies.append("pd")
    if cli.include_zero:
        policies.append("zero")
    policies.extend(checkpoints)

    results = []
    for scenario in scenarios:
        args = make_args(cli, scenario)
        for policy in policies:
            label = f"{scenario}/{policy_label(policy)}"
            result = evaluate(label, policy, args)
            result["scenario"] = scenario
            result["policy"] = policy_label(policy)
            results.append(result)

    if cli.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
