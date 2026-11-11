import argparse
from pathlib import Path
from types import SimpleNamespace

from pufferlib.ocean.quadrotor.eval_hover import evaluate, print_table, GRAVITY


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "puffer_quadrotor_177887664537"
    / "model_puffer_quadrotor_000010.pt"
)

SCENARIOS = {
    "hover": {
        "policy": "checkpoint",
        "task_mode": "hold_hover",
        "action_mode": "rate",
        "max_steps": 128,
        "spawn_radius": 0.0,
        "init_vel_radius": 0.05,
        "init_omega_radius": 0.0,
        "room_size": 0.0,
        "num_rays": 0,
    },
    "recovery": {
        "policy": "checkpoint",
        "task_mode": "fixed_target",
        "action_mode": "rate",
        "max_steps": 256,
        "spawn_radius": 0.5,
        "init_vel_radius": 0.20,
        "init_omega_radius": 0.0,
        "room_size": 0.0,
        "num_rays": 0,
    },
    "room-nav": {
        "policy": "pd",
        "task_mode": "indoor_nav",
        "action_mode": "velocity",
        "max_steps": 512,
        "spawn_radius": 1.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 2.0,
        "num_rays": 16,
        "obstacle_mode": "none",
    },
    "doorway-nav": {
        "policy": "pd",
        "task_mode": "indoor_nav",
        "action_mode": "velocity",
        "max_steps": 512,
        "spawn_radius": 1.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 2.0,
        "num_rays": 16,
        "obstacle_mode": "doorway",
    },
    "maze-nav": {
        "policy": "pd",
        "task_mode": "indoor_nav",
        "action_mode": "velocity",
        "max_steps": 768,
        "spawn_radius": 1.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 2.0,
        "num_rays": 24,
        "obstacle_mode": "maze",
    },
    "route-open": {
        "policy": "pd",
        "task_mode": "indoor_route",
        "action_mode": "velocity",
        "action_repeat": 1,
        "max_steps": 1024,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "none",
        "ray_length": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 3,
    },
    "route-reference": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "motor",
        "action_repeat": 1,
        "max_steps": 1024,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 0,
        "obstacle_mode": "none",
        "ray_length": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 3,
        "route_reward_mode": "reference",
        "route_walls": 0,
        "route_direct_obs": 1,
        "rnn_name": "Recurrent",
        "action_std": 0.1,
    },
    "route-walls": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "motor",
        "action_repeat": 1,
        "max_steps": 1024,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 0,
        "obstacle_mode": "none",
        "ray_length": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 3,
        "route_reward_mode": "reference",
        "route_walls": 1,
        "route_direct_obs": 1,
        "route_wall_obs": 1,
        "rnn_name": "Recurrent",
        "action_std": 0.1,
    },
    "route-random": {
        "policy": "pd",
        "task_mode": "indoor_route",
        "action_mode": "velocity",
        "action_repeat": 1,
        "max_steps": 1024,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "random_blocks",
        "ray_length": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 3,
    },
    "route-mixed": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "velocity",
        "action_repeat": 1,
        "max_steps": 1024,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "mixed",
        "ray_length": 4.0,
        "max_vel": 0.7,
        "rnn_name": "Recurrent",
        "action_std": 0.1,
        "route_goal_radius": 1.0,
        "route_goal_count": 3,
    },
    "route-center-block": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "motor",
        "action_repeat": 1,
        "max_steps": 512,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "center_block",
        "ray_length": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 1,
        "route_target_z_min": 1.25,
        "route_target_z_max": 1.75,
        "route_reward_mode": "dense",
        "route_walls": 0,
        "route_direct_obs": 1,
        "route_wall_obs": 0,
        "rnn_name": "Recurrent",
        "action_std": 0.2,
    },
    "route-center-block-velocity": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "velocity",
        "action_repeat": 1,
        "max_steps": 512,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "center_block",
        "ray_length": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 1,
        "route_target_mode": "blocked",
        "route_blocked_target_prob": 1.0,
        "route_target_z_min": 1.25,
        "route_target_z_max": 1.75,
        "route_reward_mode": "dense",
        "route_walls": 1,
        "route_direct_obs": 1,
        "route_wall_obs": 1,
        "route_perception_obs": 0,
        "rnn_name": "Recurrent",
        "action_std": 0.2,
    },
    "route-passable-maze-velocity": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "velocity",
        "action_repeat": 1,
        "max_steps": 768,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "maze",
        "ray_length": 4.0,
        "max_vel": 1.0,
        "max_z_vel": 0.25,
        "route_goal_radius": 1.0,
        "route_goal_count": 1,
        "route_target_mode": "blocked",
        "route_blocked_target_prob": 1.0,
        "route_target_z_min": 1.5,
        "route_target_z_max": 1.5,
        "route_reward_mode": "dense",
        "route_walls": 1,
        "route_direct_obs": 0,
        "route_wall_obs": 1,
        "route_perception_obs": 0,
        "rnn_name": "None",
        "action_std": 0.2,
    },
    "route-layout-mixed-velocity": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "velocity",
        "action_repeat": 1,
        "max_steps": 768,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "layout_mixed",
        "ray_length": 4.0,
        "max_vel": 1.0,
        "max_z_vel": 0.25,
        "vel_ctrl_ki_z": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 1,
        "route_target_mode": "blocked",
        "route_blocked_target_prob": 1.0,
        "route_target_z_min": 1.5,
        "route_target_z_max": 1.5,
        "route_reward_mode": "dense",
        "route_walls": 1,
        "route_direct_obs": 0,
        "route_wall_obs": 1,
        "route_perception_obs": 0,
        "route_target_obs_mode": "body",
        "route_path_margin": 0.35,
        "route_arrival_radius": 1.5,
        "route_arrival_bonus": 0.04,
        "route_arrival_vel_scale": 0.02,
        "route_target_wall_margin": 1.0,
        "rnn_name": "None",
        "action_std": 0.2,
    },
    "route-layout-mixed-finaltarget": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "velocity",
        "action_repeat": 1,
        "max_steps": 1536,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "layout_mixed",
        "ray_length": 4.0,
        "max_vel": 1.0,
        "max_z_vel": 0.25,
        "vel_ctrl_ki_z": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 5,
        "route_target_mode": "blocked",
        "route_blocked_target_prob": 1.0,
        "route_target_z_min": 1.5,
        "route_target_z_max": 1.5,
        "route_reward_mode": "dense",
        "route_walls": 1,
        "route_direct_obs": 0,
        "route_extra_obs": 1,
        "route_wall_obs": 1,
        "route_perception_obs": 0,
        "route_target_obs_mode": "body",
        "route_path_margin": 0.35,
        "route_arrival_radius": 1.5,
        "route_arrival_bonus": 0.04,
        "route_arrival_vel_scale": 0.02,
        "route_target_wall_margin": 1.0,
        "rnn_name": "None",
        "action_std": 0.2,
    },
    "route-layout-mixed-direct": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "velocity",
        "action_repeat": 1,
        "max_steps": 768,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "layout_mixed",
        "ray_length": 4.0,
        "max_vel": 1.0,
        "max_z_vel": 0.25,
        "vel_ctrl_ki_z": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 3,
        "route_target_mode": "blocked",
        "route_blocked_target_prob": 1.0,
        "route_target_z_min": 1.5,
        "route_target_z_max": 1.5,
        "route_reward_mode": "dense",
        "route_walls": 1,
        "route_direct_obs": 1,
        "route_extra_obs": 0,
        "route_wall_obs": 1,
        "route_perception_obs": 0,
        "route_target_obs_mode": "body",
        "route_path_margin": 0.35,
        "route_arrival_radius": 1.5,
        "route_arrival_bonus": 0.04,
        "route_arrival_vel_scale": 0.02,
        "route_target_wall_margin": 1.0,
        "rnn_name": "None",
        "action_std": 0.2,
    },
    "route-layout-mixed-direct-geodesic": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "velocity",
        "action_repeat": 1,
        "max_steps": 768,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "layout_mixed",
        "ray_length": 4.0,
        "max_vel": 1.0,
        "max_z_vel": 0.25,
        "vel_ctrl_ki_z": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 1,
        "route_target_mode": "blocked",
        "route_blocked_target_prob": 1.0,
        "route_target_z_min": 1.5,
        "route_target_z_max": 1.5,
        "route_reward_mode": "dense",
        "route_walls": 1,
        "route_direct_obs": 1,
        "route_extra_obs": 0,
        "route_wall_obs": 1,
        "route_perception_obs": 0,
        "route_target_obs_mode": "body",
        "route_path_margin": 0.35,
        "route_clearance_scale": 0.70,
        "route_arrival_radius": 1.5,
        "route_arrival_bonus": 0.04,
        "route_arrival_vel_scale": 0.02,
        "route_target_wall_margin": 1.0,
        "rnn_name": "None",
        "action_std": 0.2,
    },
    "route-layout-mixed-swiftlite-rate": {
        "policy": "checkpoint",
        "task_mode": "indoor_route",
        "action_mode": "rate",
        "action_repeat": 1,
        "max_steps": 768,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "layout_mixed",
        "ray_length": 4.0,
        "max_rate": 4.0,
        "route_goal_radius": 1.0,
        "route_goal_count": 1,
        "route_target_mode": "blocked",
        "route_blocked_target_prob": 1.0,
        "route_target_z_min": 1.5,
        "route_target_z_max": 1.5,
        "route_reward_mode": "dense",
        "route_walls": 1,
        "route_direct_obs": 0,
        "route_extra_obs": 1,
        "route_wall_obs": 1,
        "route_perception_obs": 0,
        "route_target_obs_mode": "body",
        "route_path_margin": 0.35,
        "route_arrival_radius": 1.5,
        "route_arrival_bonus": 0.04,
        "route_arrival_vel_scale": 0.02,
        "route_target_wall_margin": 1.0,
        "rnn_name": "None",
        "action_history_obs": 1,
        "action_std": 0.25,
    },
    "explore-random": {
        "policy": "ray",
        "task_mode": "indoor_explore",
        "action_mode": "velocity",
        "max_steps": 768,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "random_blocks",
        "ray_length": 4.0,
        "explore_goal_fraction": 0.85,
    },
    "explore-mixed": {
        "policy": "checkpoint",
        "task_mode": "indoor_explore",
        "action_mode": "velocity",
        "max_steps": 768,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "mixed",
        "ray_length": 4.0,
        "explore_goal_fraction": 0.85,
        "max_vel": 0.7,
        "rnn_name": "Recurrent",
        "action_std": 0.1,
    },
    "explore-open": {
        "policy": "checkpoint",
        "task_mode": "indoor_explore",
        "action_mode": "velocity",
        "max_steps": 768,
        "spawn_radius": 2.5,
        "init_vel_radius": 0.0,
        "init_omega_radius": 0.0,
        "room_size": 3.0,
        "num_rays": 24,
        "obstacle_mode": "none",
        "ray_length": 4.0,
        "explore_goal_fraction": 0.85,
        "max_vel": 0.7,
        "rnn_name": "Recurrent",
        "action_std": 0.1,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render quadrotor eval")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="room-nav",
        help="Maintained render scenario",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--policy", choices=("auto", "checkpoint", "pd", "zero", "ray", "raypd"), default="auto")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--action-mode", default=None)
    parser.add_argument("--max-rate", type=float, default=None)
    parser.add_argument("--room-size", type=float, default=None)
    parser.add_argument("--num-rays", type=int, default=None)
    parser.add_argument("--ray-length", type=float, default=None)
    parser.add_argument("--obstacle-block-prob", type=float, default=None)
    parser.add_argument("--obs-noise", type=float, default=0.0)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--action-delay", type=int, default=0)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--inertia-scale", type=float, default=0.0)
    parser.add_argument("--thrust-scale", type=float, default=1.0)
    parser.add_argument("--torque-scale", type=float, default=0.0)
    parser.add_argument("--drag-scale", type=float, default=1.0)
    parser.add_argument("--motor-tau-scale", type=float, default=1.0)
    parser.add_argument("--max-vel", type=float, default=None)
    parser.add_argument("--max-z-vel", type=float, default=None)
    parser.add_argument("--vel-ctrl-kp-vel", type=float, default=0.0)
    parser.add_argument("--vel-ctrl-ki-z", type=float, default=0.0)
    parser.add_argument("--vel-ctrl-int-limit-z", type=float, default=0.0)
    parser.add_argument("--vel-ctrl-max-z-acc", type=float, default=0.0)
    parser.add_argument("--explore-goal-fraction", type=float, default=None)
    parser.add_argument("--route-goal-radius", type=float, default=None)
    parser.add_argument("--route-goal-count", type=int, default=None)
    parser.add_argument("--route-target-z-min", type=float, default=None)
    parser.add_argument("--route-target-z-max", type=float, default=None)
    parser.add_argument("--route-reward-mode", default=None)
    parser.add_argument("--route-target-mode", default=None)
    parser.add_argument("--route-blocked-target-prob", type=float, default=None)
    parser.add_argument("--route-reference-clearance-radius", type=float, default=None)
    parser.add_argument("--route-reference-clearance-scale", type=float, default=None)
    parser.add_argument("--route-path-margin", type=float, default=None)
    parser.add_argument("--route-arrival-radius", type=float, default=None)
    parser.add_argument("--route-arrival-bonus", type=float, default=None)
    parser.add_argument("--route-arrival-vel-scale", type=float, default=None)
    parser.add_argument("--route-arrival-toward-scale", type=float, default=None)
    parser.add_argument("--route-target-wall-margin", type=float, default=None)
    parser.add_argument("--route-reach-bonus", type=float, default=None)
    parser.add_argument("--route-walls", type=int, default=None)
    parser.add_argument("--route-direct-obs", type=int, default=None)
    parser.add_argument("--route-extra-obs", type=int, default=None)
    parser.add_argument("--route-wall-obs", type=int, default=None)
    parser.add_argument("--route-perception-obs", type=int, default=None)
    parser.add_argument("--route-target-obs-mode", default=None)
    parser.add_argument("--action-history-obs", type=int, default=None)
    parser.add_argument("--action-std", type=float, default=None)
    parser.add_argument("--action-selection", choices=("mean", "sample"), default="mean")
    parser.add_argument("--rnn-name", default=None)
    return parser.parse_args()


def main():
    cli = parse_args()
    scenario = SCENARIOS[cli.scenario]
    policy = scenario["policy"] if cli.policy == "auto" else cli.policy

    args = SimpleNamespace(
        render=cli.render,
        device="cpu",
        seed=cli.seed,
        episodes=cli.episodes,
        num_envs=1,
        max_steps=scenario["max_steps"] if cli.max_steps is None else cli.max_steps,
        action_repeat=scenario.get("action_repeat", 10)
        if cli.action_repeat is None
        else cli.action_repeat,
        max_rate=scenario.get("max_rate", 1.0) if cli.max_rate is None else cli.max_rate,
        max_vel=scenario.get("max_vel", 1.0) if cli.max_vel is None else cli.max_vel,
        max_z_vel=scenario.get("max_z_vel", 0.75) if cli.max_z_vel is None else cli.max_z_vel,
        spawn_radius=scenario["spawn_radius"],
        spawn_height=1.5,
        init_vel_radius=scenario["init_vel_radius"],
        init_omega_radius=scenario["init_omega_radius"],
        room_size=scenario["room_size"] if cli.room_size is None else cli.room_size,
        ceiling_height=3.0,
        num_rays=scenario["num_rays"] if cli.num_rays is None else cli.num_rays,
        ray_length=scenario.get("ray_length", 3.0) if cli.ray_length is None else cli.ray_length,
        mask_rays=False,
        mask_route_perception_obs=False,
        hover_radius=0.25 if scenario["task_mode"] == "indoor_nav" else 0.2,
        obstacle_mode=scenario.get("obstacle_mode", "none"),
        obstacle_block_prob=scenario.get("obstacle_block_prob", 0.5)
        if cli.obstacle_block_prob is None
        else cli.obstacle_block_prob,
        obs_noise=cli.obs_noise,
        action_noise=cli.action_noise,
        action_delay=cli.action_delay,
        action_history_obs=scenario.get("action_history_obs", 0)
        if cli.action_history_obs is None
        else cli.action_history_obs,
        mass_scale=cli.mass_scale,
        inertia_scale=cli.inertia_scale,
        thrust_scale=cli.thrust_scale,
        torque_scale=cli.torque_scale,
        drag_scale=cli.drag_scale,
        motor_tau_scale=cli.motor_tau_scale,
        vel_ctrl_kp_vel=cli.vel_ctrl_kp_vel,
        vel_ctrl_ki_z=cli.vel_ctrl_ki_z,
        vel_ctrl_int_limit_z=cli.vel_ctrl_int_limit_z,
        vel_ctrl_max_z_acc=cli.vel_ctrl_max_z_acc,
        explore_cell_size=0.25,
        explore_goal_fraction=scenario.get("explore_goal_fraction", 0.85)
        if cli.explore_goal_fraction is None
        else cli.explore_goal_fraction,
        route_goal_radius=scenario.get("route_goal_radius", 0.3)
        if cli.route_goal_radius is None
        else cli.route_goal_radius,
        route_goal_count=scenario.get("route_goal_count", 3)
        if cli.route_goal_count is None
        else cli.route_goal_count,
        route_target_z_min=scenario.get("route_target_z_min", 0.0)
        if cli.route_target_z_min is None
        else cli.route_target_z_min,
        route_target_z_max=scenario.get("route_target_z_max", 0.0)
        if cli.route_target_z_max is None
        else cli.route_target_z_max,
        route_step_cost=0.004,
        route_proximity_reward=0.02,
        route_dist_scale=0.01,
        route_progress_scale=25.0,
        route_crash_reward=-24.0,
        route_crash_goal_scale=4.0,
        route_crash_dist_scale=1.5,
        route_clearance_radius=0.45,
        route_clearance_scale=0.35,
        route_reference_clearance_radius=scenario.get("route_reference_clearance_radius", 0.45)
        if cli.route_reference_clearance_radius is None
        else cli.route_reference_clearance_radius,
        route_reference_clearance_scale=scenario.get("route_reference_clearance_scale", 0.0)
        if cli.route_reference_clearance_scale is None
        else cli.route_reference_clearance_scale,
        route_path_margin=scenario.get("route_path_margin", 0.0)
        if cli.route_path_margin is None
        else cli.route_path_margin,
        route_vel_scale=0.04,
        route_arrival_radius=scenario.get("route_arrival_radius", 0.0)
        if cli.route_arrival_radius is None
        else cli.route_arrival_radius,
        route_arrival_bonus=scenario.get("route_arrival_bonus", 0.0)
        if cli.route_arrival_bonus is None
        else cli.route_arrival_bonus,
        route_arrival_vel_scale=scenario.get("route_arrival_vel_scale", 0.0)
        if cli.route_arrival_vel_scale is None
        else cli.route_arrival_vel_scale,
        route_arrival_toward_scale=scenario.get("route_arrival_toward_scale", 0.0)
        if cli.route_arrival_toward_scale is None
        else cli.route_arrival_toward_scale,
        route_target_wall_margin=scenario.get("route_target_wall_margin", 0.0)
        if cli.route_target_wall_margin is None
        else cli.route_target_wall_margin,
        route_goal_speed=2.0,
        route_goal_bonus=1.0,
        route_reach_bonus=scenario.get("route_reach_bonus", 5.0)
        if cli.route_reach_bonus is None
        else cli.route_reach_bonus,
        route_reward_mode=scenario.get("route_reward_mode", "dense")
        if cli.route_reward_mode is None
        else cli.route_reward_mode,
        route_target_mode=scenario.get("route_target_mode", "random")
        if cli.route_target_mode is None
        else cli.route_target_mode,
        route_blocked_target_prob=scenario.get("route_blocked_target_prob", 1.0)
        if cli.route_blocked_target_prob is None
        else cli.route_blocked_target_prob,
        route_walls=scenario.get("route_walls", 1)
        if cli.route_walls is None
        else cli.route_walls,
        route_direct_obs=scenario.get("route_direct_obs", 0)
        if cli.route_direct_obs is None
        else cli.route_direct_obs,
        route_extra_obs=scenario.get("route_extra_obs", 0)
        if cli.route_extra_obs is None
        else cli.route_extra_obs,
        route_wall_obs=scenario.get("route_wall_obs", 0)
        if cli.route_wall_obs is None
        else cli.route_wall_obs,
        route_perception_obs=scenario.get("route_perception_obs", 0)
        if cli.route_perception_obs is None
        else cli.route_perception_obs,
        route_target_obs_mode=scenario.get("route_target_obs_mode", "legacy")
        if cli.route_target_obs_mode is None
        else cli.route_target_obs_mode,
        task_mode=scenario["task_mode"],
        action_mode=scenario["action_mode"] if cli.action_mode is None else cli.action_mode,
        hidden_size=128,
        action_std=scenario.get("action_std", 0.05) if cli.action_std is None else cli.action_std,
        action_selection=cli.action_selection,
        rnn_name=scenario.get("rnn_name") if cli.rnn_name is None else cli.rnn_name,
        rnn_input_size=128,
        rnn_hidden_size=128,
        pd_kp_pos=2.0,
        pd_max_vel=1.0,
        pd_kp_vel=6.0,
        pd_max_tilt_acc=GRAVITY * 0.364,
        pd_max_z_acc=GRAVITY * 0.5,
        pd_kp_att=20.0,
        pd_kd_att=5.0,
        pd_avoid_radius=1.0,
        pd_avoid_gain=1.0,
    )

    if policy == "pd":
        result = evaluate("pd", "pd", args)
    elif policy == "zero":
        result = evaluate("zero", "zero", args)
    elif policy == "ray":
        result = evaluate("ray", "ray", args)
    elif policy == "raypd":
        result = evaluate("raypd", "raypd", args)
    else:
        checkpoint = Path(cli.checkpoint)
        result = evaluate(f"{checkpoint.parent.name}/{checkpoint.name}", str(checkpoint), args)

    print_table([result])


if __name__ == "__main__":
    main()
