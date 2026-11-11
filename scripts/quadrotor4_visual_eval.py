#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(PARENT_ROOT))

from pufferlib.ocean.quadrotor import Quadrotor


STRESS_PROFILES = {
    "direct_one_goal": {},
    "direct_goals3": {"route_goal_count": 3, "max_steps": 1024},
    "direct_goals5": {"route_goal_count": 5, "max_steps": 1536},
    "direct_combo": {
        "route_goal_count": 3,
        "max_steps": 1024,
        "route_target_z_min": 1.0,
        "route_target_z_max": 2.0,
        "mass_scale_min": 0.85,
        "mass_scale_max": 1.15,
        "thrust_scale_min": 0.9,
        "thrust_scale_max": 1.1,
        "drag_scale_min": 0.75,
        "drag_scale_max": 1.5,
        "motor_tau_scale_min": 0.75,
        "motor_tau_scale_max": 1.5,
    },
}


class Policy(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size, num_layers):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.encoder = nn.Linear(obs_size, hidden_size)
        self.decoder = nn.Module()
        self.decoder.decoder_mean = nn.Linear(hidden_size, action_size)
        self.decoder.decoder_logstd = nn.Parameter(torch.zeros(1, action_size))
        self.decoder.value_function = nn.Linear(hidden_size, 1)
        self.network = MinGRU(hidden_size, num_layers)

    def initial_state(self, batch_size, device):
        return self.network.initial_state(batch_size, device)

    def forward_eval(self, obs, state):
        hidden = self.encoder.encoder(obs.view(obs.shape[0], -1).float())
        hidden, state = self.network.forward_eval(hidden, state)
        mean = self.decoder.decoder_mean(hidden)
        logstd = self.decoder.decoder_logstd.expand_as(mean)
        dist = torch.distributions.Normal(mean, torch.exp(logstd))
        value = self.decoder.value_function(hidden)
        return dist, value, state


class MinGRU(nn.Module):
    def __init__(self, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, 3 * hidden_size, bias=False)
            for _ in range(num_layers)
        ])

    def _g(self, x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def _highway(self, x, out, proj):
        gate = proj.sigmoid()
        return gate * out + (1.0 - gate) * x

    def initial_state(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),)

    def forward_eval(self, hidden, state):
        state = state[0]
        hidden = hidden.unsqueeze(1)
        state_out = []
        for idx, layer in enumerate(self.layers):
            candidate, gate, proj = layer(hidden).chunk(3, dim=-1)
            out = torch.lerp(state[idx : idx + 1].transpose(0, 1), self._g(candidate), gate.sigmoid())
            hidden = self._highway(hidden, out, proj)
            state_out.append(out[:, -1:])
        return hidden.squeeze(1), (torch.stack(state_out, 0).squeeze(2),)


def create_policy(obs_size, action_size, parsed):
    if parsed.network != "MinGRU":
        raise ValueError("quadrotor4_visual_eval currently supports native MinGRU checkpoints")
    policy = Policy(obs_size, action_size, parsed.hidden_size, parsed.num_layers)
    state = torch.load(parsed.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {key.replace("module.", ""): value for key, value in state.items()}
    policy.load_state_dict(state, strict=False)
    policy.eval()
    return policy


def reset_done_state(state, done):
    if not isinstance(state, tuple) or not np.any(done):
        return state
    out = []
    done_t = torch.as_tensor(done, dtype=torch.bool)
    for value in state:
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == done_t.shape[0]:
            value = value.clone()
            value[:, done_t] = 0
        out.append(value)
    return tuple(out)


def make_env(parsed):
    profile = dict(STRESS_PROFILES[parsed.stress_profile])
    return Quadrotor(
        num_envs=1,
        max_steps=profile.get("max_steps", parsed.max_steps),
        action_repeat=1,
        max_rate=1.0,
        max_vel=1.0,
        max_z_vel=0.25,
        vel_ctrl_kp_vel=0.0,
        vel_ctrl_ki_z=4.0,
        vel_ctrl_int_limit_z=0.0,
        vel_ctrl_max_z_acc=0.0,
        spawn_radius=2.5,
        spawn_height=1.5,
        init_vel_radius=0.0,
        init_omega_radius=0.0,
        room_size=3.0,
        ceiling_height=3.0,
        num_rays=24,
        ray_length=4.0,
        hover_radius=0.2,
        obstacle_mode="layout_mixed",
        obstacle_block_prob=0.5,
        obs_noise=0.0,
        action_noise=0.0,
        action_delay=0,
        action_history_obs=1,
        mass_scale=1.0,
        inertia_scale=0.0,
        thrust_scale=1.0,
        torque_scale=0.0,
        drag_scale=1.0,
        motor_tau_scale=1.0,
        mass_scale_min=profile.get("mass_scale_min", 0.0),
        mass_scale_max=profile.get("mass_scale_max", 0.0),
        thrust_scale_min=profile.get("thrust_scale_min", 0.0),
        thrust_scale_max=profile.get("thrust_scale_max", 0.0),
        drag_scale_min=profile.get("drag_scale_min", 0.0),
        drag_scale_max=profile.get("drag_scale_max", 0.0),
        motor_tau_scale_min=profile.get("motor_tau_scale_min", 0.0),
        motor_tau_scale_max=profile.get("motor_tau_scale_max", 0.0),
        explore_cell_size=0.25,
        explore_goal_fraction=0.85,
        route_goal_radius=1.0,
        route_goal_count=profile.get("route_goal_count", 1),
        route_goal_count_min=0,
        route_step_cost=0.004,
        route_proximity_reward=0.02,
        route_dist_scale=0.01,
        route_progress_scale=25.0,
        route_crash_reward=-24.0,
        route_crash_goal_scale=4.0,
        route_crash_dist_scale=1.5,
        route_clearance_radius=0.45,
        route_clearance_scale=0.35,
        route_reference_clearance_radius=0.45,
        route_reference_clearance_scale=0.0,
        route_path_margin=0.35,
        route_vel_scale=0.04,
        route_arrival_radius=1.5,
        route_arrival_bonus=0.04,
        route_arrival_vel_scale=0.02,
        route_arrival_toward_scale=0.0,
        route_target_wall_margin=1.0,
        route_goal_speed=2.0,
        route_goal_bonus=1.0,
        route_reach_bonus=5.0,
        route_target_z_min=profile.get("route_target_z_min", 1.5),
        route_target_z_max=profile.get("route_target_z_max", 1.5),
        route_reward_mode="dense",
        route_target_mode="blocked",
        route_blocked_target_prob=1.0,
        route_walls=1,
        route_direct_obs=1,
        route_extra_obs=0,
        route_wall_obs=1,
        route_perception_obs=0,
        route_target_obs_mode="body",
        task_mode="indoor_route",
        action_mode="velocity",
        report_interval=1,
        render_mode="human" if parsed.render else None,
        seed=parsed.seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Visual eval for native PufferLib 4 quadrotor checkpoints.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="experiments/qdr145/asym_mixed15_from_qdr144_s88/qdr135_asym_ppo_000005.bin",
    )
    parser.add_argument("--stress-profile", choices=sorted(STRESS_PROFILES), default="direct_goals3")
    parser.add_argument("--network", choices=("MLP", "MinGRU", "GRU", "LSTM"), default="MinGRU")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--max-steps", type=int, default=768)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parsed = parser.parse_args()

    env = make_env(parsed)
    obs, _ = env.reset(seed=parsed.seed)
    policy = create_policy(obs.shape[1], env.single_action_space.shape[0], parsed)
    state = policy.initial_state(env.num_agents, torch.device("cpu"))
    totals = {}
    completed = 0

    try:
        while completed < parsed.episodes:
            if parsed.render:
                env.render()
            with torch.no_grad():
                dist, _, state = policy.forward_eval(torch.as_tensor(obs, dtype=torch.float32), state)
                action = dist.mean.clamp(-1.0, 1.0).cpu().numpy()
            obs, _, terminals, truncations, infos = env.step(action)
            done = np.logical_or(terminals, truncations)
            state = reset_done_state(state, done)
            for info in infos:
                if not info:
                    continue
                completed += 1
                for key, value in info.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        totals[key] = totals.get(key, 0.0) + float(value)
                print(
                    f"episode={completed} success={float(info.get('success', 0.0)):.0f} "
                    f"obstacle={float(info.get('obstacle_hit', 0.0)):.0f} "
                    f"timeout={float(info.get('timeout', 0.0)):.0f} "
                    f"goals={float(info.get('route_goals', 0.0)):.0f}",
                    flush=True,
                )

        if completed:
            success = totals.get("success", 0.0) / completed
            obstacle = totals.get("obstacle_hit", 0.0) / completed
            timeout = totals.get("timeout", 0.0) / completed
            print(
                f"summary n={completed} success={success:.4f} obstacle={obstacle:.4f} timeout={timeout:.4f}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
