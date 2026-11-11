#!/usr/bin/env python3
import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from quadrotor4_frontier_target_distill import (  # noqa: E402
    FRONTIER_TARGET_MODES,
    OBSTACLE_MODES,
    action_policy_to_actions as explore_action_policy_to_actions,
    deployable_input as explore_deployable_input,
    load_resized_output_state,
    target_to_actions as explore_target_to_actions,
)


class _CudaPtr:
    def __init__(self, ptr, shape, dtype):
        typestr = {
            torch.uint8: "|u1",
            torch.float32: "<f4",
        }[dtype]
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": typestr,
            "version": 2,
        }


def _cpu_tensor(ptr, shape, dtype):
    ctype = {
        torch.uint8: ctypes.c_uint8,
        torch.float32: ctypes.c_float,
    }[dtype]
    n = 1
    for size in shape:
        n *= size
    arr = (ctype * n).from_address(ptr)
    return torch.frombuffer(arr, dtype=dtype).reshape(shape)


def load_quadrotor_config():
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        from pufferlib.pufferl import load_config

        return load_config("quadrotor")
    finally:
        sys.argv = old_argv


def make_vec_config(parsed, privileged):
    args = load_quadrotor_config()
    args["vec"]["total_agents"] = parsed.total_agents
    args["vec"]["num_buffers"] = parsed.num_buffers
    args["vec"]["num_threads"] = parsed.num_threads

    env = args["env"]
    env.update(
        {
            "seed": parsed.seed,
            "max_steps": parsed.max_steps,
            "action_repeat": parsed.action_repeat,
            "action_mode": parsed.action_mode,
            "max_vel": parsed.max_vel,
            "max_z_vel": parsed.max_z_vel,
            "max_rate": parsed.max_rate,
            "dynamics_profile": parsed.dynamics_profile,
            "physics_dt": parsed.physics_dt,
            "physics_substeps": parsed.physics_substeps,
            "vel_ctrl_ki_z": parsed.vel_ctrl_ki_z,
            "spawn_radius": parsed.spawn_radius,
            "spawn_radius_min": parsed.spawn_radius_min,
            "spawn_height": parsed.spawn_height,
            "init_vel_radius": parsed.init_vel_radius,
            "init_omega_radius": parsed.init_omega_radius,
            "room_size": parsed.room_size,
            "ceiling_height": parsed.ceiling_height,
            "num_rays": parsed.num_rays,
            "ray_length": parsed.ray_length,
            "route_wall_obs": 1,
            "action_history_obs": 1,
        }
    )

    if parsed.task == "explore":
        env.update(
            {
                "task_mode": 3,
                "obstacle_mode": OBSTACLE_MODES[parsed.mode],
                "explore_cell_size": parsed.cell_size,
                "explore_goal_fraction": parsed.explore_goal_fraction,
                "explore_memory_obs": parsed.explore_memory_obs,
                "explore_frontier_target_mode": FRONTIER_TARGET_MODES[
                    parsed.explore_frontier_target_mode
                ],
                "explore_reachable_coverage": int(parsed.explore_reachable_coverage),
                "explore_frontier_progress_scale": parsed.explore_frontier_progress_scale,
            }
        )
    else:
        target_z_min = parsed.target_z if parsed.target_z_min is None else parsed.target_z_min
        target_z_max = parsed.target_z if parsed.target_z_max is None else parsed.target_z_max
        env.update(
            {
                "task_mode": 4,
                "obstacle_mode": parsed.obstacle_mode,
                "route_goal_radius": parsed.route_goal_radius,
                "route_goal_count": parsed.route_goal_count,
                "route_goal_count_min": parsed.route_goal_count_min,
                "route_path_margin": parsed.route_path_margin,
                "route_target_mode": parsed.route_target_mode,
                "route_blocked_target_prob": parsed.route_blocked_target_prob,
                "route_walls": parsed.route_walls,
                "route_target_obs_mode": parsed.route_target_obs_mode,
                "route_direct_obs": 0 if privileged else 1,
                "route_privileged_obs": 1 if privileged else 0,
                "route_extra_obs": 0,
                "route_perception_obs": 0,
                "route_target_z_min": target_z_min,
                "route_target_z_max": target_z_max,
                "route_reward_mode": parsed.route_reward_mode,
                "route_step_cost": parsed.route_step_cost,
                "route_progress_scale": parsed.route_progress_scale,
                "route_clearance_radius": parsed.route_clearance_radius,
                "route_clearance_scale": parsed.route_clearance_scale,
                "route_vel_scale": parsed.route_vel_scale,
                "route_arrival_radius": parsed.route_arrival_radius,
                "route_arrival_bonus": parsed.route_arrival_bonus,
                "route_arrival_vel_scale": parsed.route_arrival_vel_scale,
                "route_arrival_toward_scale": parsed.route_arrival_toward_scale,
                "route_goal_speed": parsed.route_goal_speed,
                "mass_scale_min": parsed.mass_scale_min,
                "mass_scale_max": parsed.mass_scale_max,
                "thrust_scale_min": parsed.thrust_scale_min,
                "thrust_scale_max": parsed.thrust_scale_max,
                "drag_scale_min": parsed.drag_scale_min,
                "drag_scale_max": parsed.drag_scale_max,
                "motor_tau_scale_min": parsed.motor_tau_scale_min,
                "motor_tau_scale_max": parsed.motor_tau_scale_max,
            }
        )
    return args


def tensor_views(vec, device):
    obs_dtype = {
        "ByteTensor": torch.uint8,
        "FloatTensor": torch.float32,
    }[vec.obs_dtype]
    if device.type == "cuda":
        obs = torch.as_tensor(
            _CudaPtr(vec.gpu_obs_ptr, (vec.total_agents, vec.obs_size), obs_dtype)
        )
        rewards = torch.as_tensor(
            _CudaPtr(vec.gpu_rewards_ptr, (vec.total_agents,), torch.float32)
        )
        terminals = torch.as_tensor(
            _CudaPtr(vec.gpu_terminals_ptr, (vec.total_agents,), torch.float32)
        )
    else:
        obs = _cpu_tensor(vec.obs_ptr, (vec.total_agents, vec.obs_size), obs_dtype)
        rewards = _cpu_tensor(vec.rewards_ptr, (vec.total_agents,), torch.float32)
        terminals = _cpu_tensor(vec.terminals_ptr, (vec.total_agents,), torch.float32)
    return obs, rewards, terminals


def step_vec(vec, actions, device):
    actions = actions.detach().to(dtype=torch.float32).contiguous()
    if device.type == "cuda":
        vec.gpu_step(actions.data_ptr())
    else:
        vec.cpu_step(actions.data_ptr())


def detach_state(state):
    if not isinstance(state, tuple):
        return state
    return tuple(value.detach() if torch.is_tensor(value) else value for value in state)


def clone_state(state):
    if not isinstance(state, tuple):
        return state
    return tuple(value.detach().clone() if torch.is_tensor(value) else value for value in state)


def reset_done_state(state, done):
    if not isinstance(state, tuple) or not torch.is_tensor(done) or not done.any():
        return state
    out = []
    for value in state:
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == done.shape[0]:
            value = value.clone()
            value[:, done] = 0
        out.append(value)
    return tuple(out)


def accumulate_logs(accum, logs):
    n = float(logs.get("n", 0.0))
    if n <= 0:
        return accum
    if accum is None:
        accum = {"n": 0.0, "sums": {}}
    for key, value in logs.items():
        if key == "n":
            continue
        accum["sums"][key] = accum["sums"].get(key, 0.0) + float(value) * n
    accum["n"] += n
    return accum


def finalize_logs(accum):
    if accum is None or accum["n"] <= 0:
        return {}
    out = {"n": accum["n"]}
    out.update({key: value / accum["n"] for key, value in accum["sums"].items()})
    return out


def print_metrics(prefix, metrics):
    keys = (
        "n",
        "success",
        "score",
        "crashed",
        "obstacle_hit",
        "wall_hit",
        "ground_hit",
        "timeout",
        "coverage",
        "explored_cells",
        "new_cell_rate",
        "route_goals",
        "min_clearance",
        "path_efficiency",
        "action_delta",
    )
    parts = [prefix]
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.6g}")
    print(" ".join(parts), flush=True)


def normal_logprob(dist, actions):
    return dist.log_prob(actions.view(-1, actions.shape[-1])).sum(-1)


def normal_entropy(dist):
    return dist.entropy().sum(-1)


def finite_tensor(x):
    return torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=-1.0)


def actor_input(obs, parsed):
    if parsed.task == "explore":
        return finite_tensor(explore_deployable_input(obs, parsed))
    return finite_tensor(obs)


def critic_input(obs):
    return finite_tensor(obs)


def policy_to_env_action(policy_action, raw_obs, parsed):
    policy_action = policy_action.float().clamp(-1.0, 1.0)
    if parsed.task != "explore":
        return policy_action
    raw_obs = finite_tensor(raw_obs)
    if parsed.explore_target_policy:
        return explore_target_to_actions(policy_action, parsed, raw_obs)
    return explore_action_policy_to_actions(policy_action, parsed, raw_obs)


def actor_action_sizes(vec, parsed):
    if parsed.task == "explore" and parsed.explore_target_policy:
        return [1] * parsed.explore_target_outputs
    return list(vec.act_sizes)


def save_actor(actor, path, obs_size, act_sizes, parsed):
    if parsed.task == "explore" and parsed.explore_target_policy:
        data = {
            "input_size": int(obs_size),
            "num_atns": len(act_sizes),
            "act_sizes": list(act_sizes),
            "state_dict": actor.state_dict(),
            "policy": {
                "network": parsed.network,
                "hidden_size": parsed.hidden_size,
                "num_layers": parsed.num_layers,
                "action_std": parsed.action_std,
            },
            "config": {
                "task_mode": 3,
                "explore_memory_obs": parsed.explore_memory_obs,
                "explore_frontier_target_mode": parsed.explore_frontier_target_mode,
                "explore_reachable_coverage": parsed.explore_reachable_coverage,
                "deploy_patch_obs": parsed.deploy_patch_obs,
                "deploy_coverage_obs": parsed.deploy_coverage_obs,
                "predict_frontier_sectors": parsed.explore_target_outputs >= 8,
                "predict_frontier_vector": parsed.explore_target_outputs >= 11,
                "use_pred_vector_fallback": parsed.use_pred_vector_fallback,
                "pred_vector_fallback_primary_threshold": parsed.pred_vector_fallback_primary_threshold,
                "pred_vector_fallback_min_coverage": parsed.pred_vector_fallback_min_coverage,
                "pred_vector_fallback_min_dist": parsed.pred_vector_fallback_min_dist,
                "use_pred_recovery_gate": parsed.use_pred_recovery_gate,
                "pred_recovery_gate_threshold": parsed.pred_recovery_gate_threshold,
                "pred_recovery_gate_min_coverage": parsed.pred_recovery_gate_min_coverage,
                "pred_recovery_gate_min_dist": parsed.pred_recovery_gate_min_dist,
                "safety_cone_ray_stop": parsed.safety_cone_ray_stop,
                "safety_cone_deg": parsed.safety_cone_deg,
                "safety_slow_ray": parsed.safety_slow_ray,
            },
        }
    else:
        data = actor.state_dict()
    torch.save(data, path)


def teacher_actions_from_privileged_obs(obs, parsed):
    unit = obs[:, 17:20].float()
    dist = obs[:, 20:21].float()
    closing = obs[:, 21:22].float()

    xy_norm = torch.linalg.vector_norm(unit[:, :2], dim=1, keepdim=True).clamp_min(1e-6)
    xy_unit = unit[:, :2] / xy_norm
    speed = torch.clamp(parsed.teacher_speed_gain * dist, 0.0, parsed.teacher_max_speed)
    speed = torch.where(dist > parsed.teacher_slow_radius, speed, speed * 0.35)

    actions = torch.zeros((obs.shape[0], 4), dtype=torch.float32, device=obs.device)
    actions[:, 0:2] = xy_unit * speed
    actions[:, 2:3] = torch.clamp(
        parsed.teacher_z_gain * unit[:, 2:3] - parsed.teacher_z_damp * closing,
        -parsed.teacher_max_z,
        parsed.teacher_max_z,
    )
    return torch.clamp(actions, -1.0, 1.0)


def append_recovery_windows(replay_obs, replay_priv, obs, priv_obs, rewards, dones, parsed):
    if parsed.recovery_replay_weight <= 0.0 or parsed.recovery_replay_capacity <= 0:
        return 0

    failed = torch.logical_and(dones > 0.5, rewards < parsed.recovery_failure_reward_threshold)
    hits = failed.nonzero(as_tuple=False)
    added = 0
    window = parsed.recovery_replay_window
    for env_id, step_id in hits.detach().cpu().tolist():
        start = step_id + 1 - window
        if start < 0:
            continue
        replay_obs.append(obs[env_id, start : step_id + 1].detach().cpu())
        replay_priv.append(priv_obs[env_id, start : step_id + 1].detach().cpu())
        added += 1

    overflow = len(replay_obs) - parsed.recovery_replay_capacity
    if overflow > 0:
        del replay_obs[:overflow]
        del replay_priv[:overflow]
    return added


def recovery_replay_is_active(parsed, update, replay_size):
    if parsed.recovery_replay_weight <= 0.0 or parsed.recovery_replay_steps <= 0:
        return False
    if replay_size < parsed.recovery_replay_min_size:
        return False
    if update < parsed.recovery_replay_start_update:
        return False
    if parsed.recovery_replay_end_update > 0 and update > parsed.recovery_replay_end_update:
        return False
    return True


def sample_recovery_batch(replay_obs, replay_priv, parsed, device):
    size = min(parsed.recovery_replay_batch_size, len(replay_obs))
    ids = torch.randint(0, len(replay_obs), (size,)).tolist()
    obs = torch.stack([replay_obs[idx] for idx in ids], dim=0).to(device)
    priv = torch.stack([replay_priv[idx] for idx in ids], dim=0).to(device)
    return obs, priv


def run_recovery_replay_steps(parsed, actor, anchor, actor_optim, replay_obs, replay_priv, device):
    actor.train()
    totals = {"recovery_loss": 0.0, "recovery_anchor": 0.0}
    for _ in range(parsed.recovery_replay_steps):
        obs, priv = sample_recovery_batch(replay_obs, replay_priv, parsed, device)
        recovery_priv = priv
        if parsed.recovery_replay_tail > 0:
            recovery_priv = priv[:, -parsed.recovery_replay_tail :]

        targets = teacher_actions_from_privileged_obs(
            recovery_priv.reshape(-1, recovery_priv.shape[-1]),
            parsed,
        ).reshape(recovery_priv.shape[0], recovery_priv.shape[1], -1)

        dist, _ = actor(obs)
        pred_full = dist.mean.reshape(obs.shape[0], obs.shape[1], -1)
        pred = (
            pred_full[:, -parsed.recovery_replay_tail :]
            if parsed.recovery_replay_tail > 0
            else pred_full
        )
        recovery_loss = F.mse_loss(pred, targets)

        anchor_loss = torch.zeros((), device=device)
        if anchor is not None and parsed.recovery_replay_anchor_weight > 0.0:
            with torch.no_grad():
                anchor_dist, _ = anchor(obs)
                anchor_mean = anchor_dist.mean.reshape(pred_full.shape)
            anchor_loss = F.mse_loss(pred_full, anchor_mean)

        loss = (
            parsed.recovery_replay_weight * recovery_loss
            + parsed.recovery_replay_anchor_weight * anchor_loss
        )
        actor_optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), parsed.max_grad_norm)
        actor_optim.step()

        totals["recovery_loss"] += float(recovery_loss.detach().cpu())
        totals["recovery_anchor"] += float(anchor_loss.detach().cpu())

    return {key: value / parsed.recovery_replay_steps for key, value in totals.items()}


def create_actor(obs_size, act_sizes, parsed, device):
    import pufferlib.models

    encoder = pufferlib.models.DefaultEncoder(obs_size, parsed.hidden_size)
    decoder = pufferlib.models.DefaultDecoder(act_sizes, parsed.hidden_size)
    network_cls = getattr(pufferlib.models, parsed.network)
    network = network_cls(hidden_size=parsed.hidden_size, num_layers=parsed.num_layers)
    if hasattr(decoder, "decoder_logstd"):
        decoder.decoder_logstd.data.fill_(float(np.log(parsed.action_std)))
    return pufferlib.models.Policy(encoder, decoder, network).to(device)


class Critic(nn.Module):
    def __init__(self, obs_size, hidden_size, network_name, num_layers):
        super().__init__()
        import pufferlib.models

        self.encoder = pufferlib.models.DefaultEncoder(obs_size, hidden_size)
        network_cls = getattr(pufferlib.models, network_name)
        self.network = network_cls(hidden_size=hidden_size, num_layers=num_layers)
        self.value = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size, device):
        return self.network.initial_state(batch_size, device)

    def forward_eval(self, obs, state):
        h = self.encoder(obs)
        h, state = self.network.forward_eval(h, state)
        return self.value(h).squeeze(-1), state

    def forward(self, obs):
        batch, horizon = obs.shape[:2]
        h = self.encoder(obs.reshape(batch * horizon, -1))
        h = self.network.forward_train(h.reshape(batch, horizon, -1))
        return self.value(h.reshape(batch * horizon, -1)).reshape(batch, horizon)


def load_actor(
    actor,
    checkpoint,
    device,
    action_std=None,
    resize_input=False,
    resize_output=False,
):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {key.replace("module.", ""): value for key, value in state.items()}
    if resize_input or resize_output:
        copied, resized = load_resized_output_state(
            actor,
            state,
            resize_input=resize_input,
            resize_output=resize_output,
        )
        print(
            f"loaded checkpoint {checkpoint} copied={len(copied)} resized={','.join(resized) or 'none'}",
            flush=True,
        )
    else:
        actor.load_state_dict(state, strict=False)
    if action_std is not None and hasattr(actor.decoder, "decoder_logstd"):
        actor.decoder.decoder_logstd.data.fill_(float(np.log(action_std)))


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(rewards.shape[0], device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        next_values = next_value if t == rewards.shape[1] - 1 else values[:, t + 1]
        next_nonterminal = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_values * next_nonterminal - values[:, t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[:, t] = lastgaelam
    return advantages, advantages + values


def actor_sequence_dist(actor, obs, start_state, dones):
    means = []
    stds = []
    state = clone_state(start_state)
    for t in range(obs.shape[1]):
        dist, _, state = actor.forward_eval(obs[:, t], state)
        means.append(dist.mean)
        stds.append(dist.stddev)
        state = reset_done_state(state, dones[:, t] > 0.5)
    mean = torch.stack(means, dim=1)
    std = torch.stack(stds, dim=1)
    return torch.distributions.Normal(
        mean.reshape(-1, mean.shape[-1]),
        std.reshape(-1, std.shape[-1]),
    )


def critic_sequence_values(critic, obs, start_state, dones):
    values = []
    state = clone_state(start_state)
    for t in range(obs.shape[1]):
        value, state = critic.forward_eval(obs[:, t], state)
        values.append(value)
        state = reset_done_state(state, dones[:, t] > 0.5)
    return torch.stack(values, dim=1)


def deterministic_eval(parsed, actor):
    from pufferlib import _C

    args = make_vec_config(parsed, privileged=False)
    device = next(actor.parameters()).device
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, _, terminals = tensor_views(vec, device)
    vec.reset()
    state = actor.initial_state(vec.total_agents, device)
    accum = None
    try:
        for step in range(1, parsed.max_eval_steps + 1):
            with torch.no_grad():
                x = actor_input(obs, parsed)
                dist, _, state = actor.forward_eval(x, state)
                policy_action = dist.mean.clamp(-1.0, 1.0)
                actions = policy_to_env_action(policy_action, obs, parsed)
            step_vec(vec, actions, device)
            done = terminals > 0.5
            if done.any():
                state = reset_done_state(state, done)
            state = detach_state(state)
            if step % parsed.log_interval == 0:
                accum = accumulate_logs(accum, vec.log())
                if finalize_logs(accum).get("n", 0.0) >= parsed.episodes:
                    break
        accum = accumulate_logs(accum, vec.log())
        return finalize_logs(accum)
    finally:
        vec.close()


def run_train(parsed):
    from pufferlib import _C

    actor_args = make_vec_config(parsed, privileged=False)
    critic_args = make_vec_config(parsed, privileged=True)
    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    actor_vec = _C.create_vec(actor_args, 0 if device.type == "cpu" else 1)
    critic_vec = _C.create_vec(critic_args, 0 if device.type == "cpu" else 1)
    actor_obs, rewards, actor_terms = tensor_views(actor_vec, device)
    critic_obs, _, critic_terms = tensor_views(critic_vec, device)
    actor_vec.reset()
    critic_vec.reset()

    if actor_vec.obs_size != critic_vec.obs_size:
        raise RuntimeError(
            f"actor/critic obs size mismatch: {actor_vec.obs_size} vs {critic_vec.obs_size}"
        )

    act_sizes = actor_action_sizes(actor_vec, parsed)
    actor = create_actor(actor_vec.obs_size, act_sizes, parsed, device)
    if parsed.load_actor:
        load_actor(
            actor,
            parsed.load_actor,
            device,
            parsed.load_action_std,
            resize_input=parsed.resize_input_head,
            resize_output=parsed.resize_output_head,
        )
    anchor = None
    needs_anchor = parsed.kl_coef > 0.0 or parsed.recovery_replay_anchor_weight > 0.0
    if needs_anchor and parsed.load_actor:
        anchor = create_actor(
            actor_vec.obs_size,
            actor_action_sizes(actor_vec, parsed),
            parsed,
            device,
        )
        load_actor(
            anchor,
            parsed.load_actor,
            device,
            parsed.load_action_std,
            resize_input=parsed.resize_input_head,
            resize_output=parsed.resize_output_head,
        )
        anchor.eval()
        for param in anchor.parameters():
            param.requires_grad_(False)
    critic = Critic(
        critic_vec.obs_size,
        parsed.hidden_size,
        parsed.critic_network,
        parsed.critic_num_layers,
    ).to(device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=parsed.actor_lr)
    recovery_optim = (
        torch.optim.Adam(actor.parameters(), lr=parsed.recovery_replay_lr)
        if parsed.recovery_replay_lr > 0.0
        else actor_optim
    )
    critic_optim = torch.optim.Adam(critic.parameters(), lr=parsed.critic_lr)
    actor_state = actor.initial_state(actor_vec.total_agents, device)
    critic_state = critic.initial_state(critic_vec.total_agents, device)
    recovery_obs_replay = []
    recovery_priv_replay = []

    try:
        for update in range(1, parsed.updates + 1):
            if parsed.reset_state_each_rollout:
                actor_state = actor.initial_state(actor_vec.total_agents, device)
                critic_state = critic.initial_state(critic_vec.total_agents, device)

            obs_buf = []
            priv_obs_buf = []
            action_buf = []
            logprob_buf = []
            reward_buf = []
            done_buf = []
            value_buf = []
            rollout_actor_state = clone_state(actor_state)
            rollout_critic_state = clone_state(critic_state)

            actor.eval()
            critic.eval()
            for _ in range(parsed.horizon):
                with torch.no_grad():
                    x = actor_input(actor_obs, parsed)
                    dist, _, actor_state = actor.forward_eval(x, actor_state)
                    value, critic_state = critic.forward_eval(
                        critic_input(critic_obs),
                        critic_state,
                    )
                    raw_action = dist.sample()
                    policy_action = raw_action.clamp(-1.0, 1.0)
                    env_action = policy_to_env_action(policy_action, actor_obs, parsed)
                    logprob = normal_logprob(dist, policy_action)

                obs_buf.append(x.clone())
                priv_obs_buf.append(critic_input(critic_obs).clone())
                action_buf.append(policy_action.clone())
                logprob_buf.append(logprob.clone())
                value_buf.append(value.clone())

                step_vec(actor_vec, env_action, device)
                step_vec(critic_vec, env_action, device)
                done = torch.logical_or(actor_terms > 0.5, critic_terms > 0.5)
                reward_buf.append(rewards.float().clone())
                done_buf.append(done.float().clone())
                if done.any():
                    actor_state = reset_done_state(actor_state, done)
                    critic_state = reset_done_state(critic_state, done)
                actor_state = detach_state(actor_state)
                critic_state = detach_state(critic_state)

            with torch.no_grad():
                next_value, _ = critic.forward_eval(
                    critic_input(critic_obs),
                    clone_state(critic_state),
                )

            obs = torch.stack(obs_buf, dim=1)
            priv_obs = torch.stack(priv_obs_buf, dim=1)
            actions = torch.stack(action_buf, dim=1)
            old_logprobs = torch.stack(logprob_buf, dim=1)
            rewards_t = torch.stack(reward_buf, dim=1)
            dones = torch.stack(done_buf, dim=1)
            values = torch.stack(value_buf, dim=1)
            recovery_added = append_recovery_windows(
                recovery_obs_replay,
                recovery_priv_replay,
                obs,
                priv_obs,
                rewards_t,
                dones,
                parsed,
            )
            advantages, returns = compute_gae(
                rewards_t,
                dones,
                values,
                next_value,
                parsed.gamma,
                parsed.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor.train()
            critic.train()
            last = {}
            for _ in range(parsed.ppo_epochs):
                new_values = critic_sequence_values(
                    critic,
                    priv_obs,
                    rollout_critic_state,
                    dones,
                )
                value_loss = 0.5 * (new_values - returns).pow(2).mean()
                critic_optim.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), parsed.max_grad_norm)
                critic_optim.step()

                if update <= parsed.value_warmup_updates:
                    last = {"value_loss": float(value_loss.detach().cpu())}
                    continue

                dist = actor_sequence_dist(actor, obs, rollout_actor_state, dones)
                flat_actions = actions.reshape(-1, actions.shape[-1])
                new_logprob = normal_logprob(dist, flat_actions).reshape(old_logprobs.shape)
                entropy = normal_entropy(dist).mean()
                logratio = new_logprob - old_logprobs
                ratio = logratio.exp()
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio,
                    1.0 - parsed.clip_coef,
                    1.0 + parsed.clip_coef,
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                approx_kl = ((ratio - 1.0) - logratio).mean()
                kl_loss = torch.zeros((), device=device)
                if anchor is not None:
                    with torch.no_grad():
                        anchor_start = anchor.initial_state(obs.shape[0], device)
                        anchor_dist = actor_sequence_dist(anchor, obs, anchor_start, dones)
                    kl_loss = torch.distributions.kl_divergence(anchor_dist, dist).sum(-1).mean()
                actor_loss = policy_loss - parsed.ent_coef * entropy + parsed.kl_coef * kl_loss

                actor_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), parsed.max_grad_norm)
                actor_optim.step()
                recovery_metrics = {}
                if recovery_replay_is_active(parsed, update, len(recovery_obs_replay)):
                    recovery_metrics = run_recovery_replay_steps(
                        parsed,
                        actor,
                        anchor,
                        recovery_optim,
                        recovery_obs_replay,
                        recovery_priv_replay,
                        device,
                    )
                last = {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "approx_kl": float(approx_kl.detach().cpu()),
                    "anchor_kl": float(kl_loss.detach().cpu()),
                    "recovery_added": float(recovery_added),
                    "recovery_size": float(len(recovery_obs_replay)),
                    **recovery_metrics,
                }

            if update % parsed.print_interval == 0 or update == parsed.updates:
                metrics = actor_vec.log()
                fields = " ".join(f"{key}={value:.6g}" for key, value in last.items())
                print(
                    f"update={update} steps={update * parsed.horizon * actor_vec.total_agents} "
                    f"success={float(metrics.get('success', 0.0)):.4f} "
                    f"obstacle={float(metrics.get('obstacle_hit', 0.0)):.4f} {fields}",
                    flush=True,
                )

            if parsed.output and (
                update == parsed.updates
                or (parsed.save_interval > 0 and update % parsed.save_interval == 0)
            ):
                output = Path(parsed.output)
                if output.suffix:
                    path = output
                else:
                    path = output / f"{parsed.checkpoint_prefix}_{update:06d}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_actor(actor, path, actor_vec.obs_size, act_sizes, parsed)
                print(f"wrote {path}", flush=True)

    finally:
        actor_vec.close()
        critic_vec.close()

    if parsed.eval_after:
        metrics = deterministic_eval(parsed, actor)
        print_metrics("eval", metrics)
        return metrics
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Asymmetric PPO for quadrotor: deployable actor obs, privileged critic obs."
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--task", choices=("route", "explore"), default="route")
    parser.add_argument("--total-agents", type=int, default=1024)
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=768)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--action-mode", type=int, default=1)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--max-z-vel", type=float, default=0.25)
    parser.add_argument("--max-rate", type=float, default=1.0)
    parser.add_argument("--dynamics-profile", type=int, default=0)
    parser.add_argument("--physics-dt", type=float, default=0.05)
    parser.add_argument("--physics-substeps", type=int, default=1)
    parser.add_argument("--vel-ctrl-ki-z", type=float, default=4.0)
    parser.add_argument("--spawn-radius", type=float, default=2.5)
    parser.add_argument("--spawn-radius-min", type=float, default=2.5)
    parser.add_argument("--spawn-height", type=float, default=1.5)
    parser.add_argument("--init-vel-radius", type=float, default=0.0)
    parser.add_argument("--init-omega-radius", type=float, default=0.0)
    parser.add_argument("--room-size", type=float, default=3.0)
    parser.add_argument("--ceiling-height", type=float, default=3.0)
    parser.add_argument("--num-rays", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=4.0)
    parser.add_argument("--obstacle-mode", type=int, default=7)
    parser.add_argument("--mode", choices=tuple(OBSTACLE_MODES), default="procedural_indoor")
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--explore-goal-fraction", type=float, default=0.85)
    parser.add_argument("--explore-memory-obs", type=int, default=3)
    parser.add_argument(
        "--explore-frontier-target-mode",
        choices=tuple(FRONTIER_TARGET_MODES),
        default="nearest",
    )
    parser.add_argument("--explore-reachable-coverage", action="store_true")
    parser.add_argument("--explore-frontier-progress-scale", type=float, default=0.0)
    parser.add_argument("--deploy-patch-obs", action="store_true")
    parser.add_argument("--deploy-coverage-obs", action="store_true")
    parser.add_argument("--explore-target-policy", action="store_true")
    parser.add_argument("--explore-target-outputs", type=int, default=11)
    parser.add_argument("--resize-input-head", action="store_true")
    parser.add_argument("--resize-output-head", action="store_true")
    parser.add_argument("--speed-fraction", type=float, default=0.55)
    parser.add_argument("--yaw-fraction", type=float, default=0.35)
    parser.add_argument("--exists-threshold", type=float, default=0.2)
    parser.add_argument("--safety-min-ray-stop", type=float, default=0.0)
    parser.add_argument("--safety-action-ray-stop", type=float, default=0.0)
    parser.add_argument("--safety-cone-ray-stop", type=float, default=0.0)
    parser.add_argument("--safety-cone-deg", type=float, default=45.0)
    parser.add_argument("--safety-slow-ray", type=float, default=0.0)
    parser.add_argument("--action-policy-safety", action="store_true")
    parser.add_argument("--use-pred-vector-fallback", action="store_true")
    parser.add_argument("--pred-vector-fallback-primary-threshold", type=float, default=0.2)
    parser.add_argument("--pred-vector-fallback-min-coverage", type=float, default=0.0)
    parser.add_argument("--pred-vector-fallback-min-dist", type=float, default=0.02)
    parser.add_argument("--use-pred-recovery-gate", action="store_true")
    parser.add_argument("--pred-recovery-gate-threshold", type=float, default=0.5)
    parser.add_argument("--pred-recovery-gate-min-coverage", type=float, default=0.25)
    parser.add_argument("--pred-recovery-gate-min-dist", type=float, default=0.02)
    parser.add_argument("--route-goal-radius", type=float, default=1.0)
    parser.add_argument("--route-goal-count", type=int, default=1)
    parser.add_argument("--route-goal-count-min", type=int, default=0)
    parser.add_argument("--route-path-margin", type=float, default=0.35)
    parser.add_argument("--route-target-mode", type=int, default=1)
    parser.add_argument("--route-blocked-target-prob", type=float, default=1.0)
    parser.add_argument("--route-walls", type=int, default=1)
    parser.add_argument("--route-target-obs-mode", type=int, default=1)
    parser.add_argument("--route-reward-mode", type=int, default=0)
    parser.add_argument("--route-step-cost", type=float, default=0.004)
    parser.add_argument("--route-progress-scale", type=float, default=25.0)
    parser.add_argument("--route-clearance-radius", type=float, default=0.45)
    parser.add_argument("--route-clearance-scale", type=float, default=0.35)
    parser.add_argument("--route-vel-scale", type=float, default=0.04)
    parser.add_argument("--route-arrival-radius", type=float, default=1.5)
    parser.add_argument("--route-arrival-bonus", type=float, default=0.04)
    parser.add_argument("--route-arrival-vel-scale", type=float, default=0.02)
    parser.add_argument("--route-arrival-toward-scale", type=float, default=0.0)
    parser.add_argument("--route-goal-speed", type=float, default=2.0)
    parser.add_argument("--target-z", type=float, default=1.5)
    parser.add_argument("--target-z-min", type=float, default=None)
    parser.add_argument("--target-z-max", type=float, default=None)
    parser.add_argument("--mass-scale-min", type=float, default=0.0)
    parser.add_argument("--mass-scale-max", type=float, default=0.0)
    parser.add_argument("--thrust-scale-min", type=float, default=0.0)
    parser.add_argument("--thrust-scale-max", type=float, default=0.0)
    parser.add_argument("--drag-scale-min", type=float, default=0.0)
    parser.add_argument("--drag-scale-max", type=float, default=0.0)
    parser.add_argument("--motor-tau-scale-min", type=float, default=0.0)
    parser.add_argument("--motor-tau-scale-max", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--network", choices=("MLP", "MinGRU", "GRU", "LSTM"), default="MinGRU")
    parser.add_argument("--critic-network", choices=("MLP", "MinGRU", "GRU", "LSTM"), default="MinGRU")
    parser.add_argument("--critic-num-layers", type=int, default=1)
    parser.add_argument("--action-std", type=float, default=0.05)
    parser.add_argument("--load-actor")
    parser.add_argument("--load-action-std", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-prefix", default="qdr135_asym_ppo")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.05)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--recovery-replay-weight", type=float, default=0.0)
    parser.add_argument("--recovery-replay-lr", type=float, default=0.0)
    parser.add_argument("--recovery-replay-anchor-weight", type=float, default=0.0)
    parser.add_argument("--recovery-replay-window", type=int, default=24)
    parser.add_argument("--recovery-replay-tail", type=int, default=0)
    parser.add_argument("--recovery-replay-capacity", type=int, default=4096)
    parser.add_argument("--recovery-replay-min-size", type=int, default=1)
    parser.add_argument("--recovery-replay-batch-size", type=int, default=256)
    parser.add_argument("--recovery-replay-steps", type=int, default=1)
    parser.add_argument("--recovery-replay-start-update", type=int, default=1)
    parser.add_argument("--recovery-replay-end-update", type=int, default=0)
    parser.add_argument("--recovery-failure-reward-threshold", type=float, default=-1.0)
    parser.add_argument("--teacher-speed-gain", type=float, default=2.5)
    parser.add_argument("--teacher-max-speed", type=float, default=0.85)
    parser.add_argument("--teacher-slow-radius", type=float, default=0.12)
    parser.add_argument("--teacher-z-gain", type=float, default=0.8)
    parser.add_argument("--teacher-z-damp", type=float, default=0.15)
    parser.add_argument("--teacher-max-z", type=float, default=0.6)
    parser.add_argument("--value-warmup-updates", type=int, default=0)
    parser.add_argument("--reset-state-each-rollout", action="store_true", default=True)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--print-interval", type=int, default=5)
    parser.add_argument("--eval-after", action="store_true")
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--max-eval-steps", type=int, default=30000)
    parser.add_argument("--log-interval", type=int, default=64)
    return parser.parse_args()


def main():
    run_train(parse_args())


if __name__ == "__main__":
    main()
