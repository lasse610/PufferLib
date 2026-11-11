#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import quadrotor4_asym_ppo as asym

BASE_OBS_SIZE = 26
WALL_OBS_SIZE = 6


def local_clearance_m(obs, parsed):
    values = []
    ray_start = BASE_OBS_SIZE
    if parsed.num_rays > 0:
        rays = obs[:, ray_start : ray_start + parsed.num_rays].float()
        values.append(rays.min(dim=1, keepdim=True).values * parsed.ray_length)

    wall_start = ray_start + parsed.num_rays
    if parsed.route_wall_obs:
        wall = obs[:, wall_start : wall_start + WALL_OBS_SIZE].float()
        if wall.shape[1] == WALL_OBS_SIZE:
            xy = wall[:, :4].min(dim=1, keepdim=True).values * (2.0 * parsed.room_size)
            z = wall[:, 4:].min(dim=1, keepdim=True).values * parsed.ceiling_height
            values.append(torch.minimum(xy, z))

    if not values:
        return torch.full((obs.shape[0], 1), parsed.clearance_slow_start_m, device=obs.device)
    return torch.cat(values, dim=1).min(dim=1, keepdim=True).values


def clearance_speed_scale(obs, parsed):
    if parsed.clearance_speed_anchor <= 0.0:
        return torch.ones((obs.shape[0], 1), dtype=torch.float32, device=obs.device)
    denom = max(parsed.clearance_slow_start_m - parsed.clearance_stop_m, 1e-6)
    scale = (local_clearance_m(obs, parsed) - parsed.clearance_stop_m) / denom
    scale = scale.clamp(parsed.clearance_min_speed_scale, 1.0)
    return (1.0 - parsed.clearance_speed_anchor) + parsed.clearance_speed_anchor * scale


def teacher_actions_from_privileged_obs(obs, parsed):
    unit = obs[:, 17:20].float()
    dist = obs[:, 20:21].float()
    closing = obs[:, 21:22].float()

    xy_norm = torch.linalg.vector_norm(unit[:, :2], dim=1, keepdim=True).clamp_min(1e-6)
    xy_unit = unit[:, :2] / xy_norm
    speed = torch.clamp(parsed.teacher_speed_gain * dist, 0.0, parsed.teacher_max_speed)
    speed = torch.where(dist > parsed.teacher_slow_radius, speed, speed * 0.35)
    speed = speed * clearance_speed_scale(obs, parsed)

    actions = torch.zeros((obs.shape[0], 4), dtype=torch.float32, device=obs.device)
    actions[:, 0:2] = xy_unit * speed
    actions[:, 2:3] = torch.clamp(
        parsed.teacher_z_gain * unit[:, 2:3] - parsed.teacher_z_damp * closing,
        -parsed.teacher_max_z,
        parsed.teacher_max_z,
    )
    return torch.clamp(actions, -1.0, 1.0)


def collect_windows(parsed, actor, device):
    from pufferlib import _C

    actor_args = asym.make_vec_config(parsed, privileged=False)
    critic_args = asym.make_vec_config(parsed, privileged=True)
    actor_vec = _C.create_vec(actor_args, 0 if device.type == "cpu" else 1)
    critic_vec = _C.create_vec(critic_args, 0 if device.type == "cpu" else 1)
    actor_obs, rewards, actor_terms = asym.tensor_views(actor_vec, device)
    critic_obs, _, critic_terms = asym.tensor_views(critic_vec, device)
    actor_vec.reset()
    critic_vec.reset()

    actor.eval()
    state = actor.initial_state(actor_vec.total_agents, device)
    obs_ring = []
    priv_ring = []
    failure_obs = []
    failure_priv = []
    anchor_obs = []
    steps = 0
    accum = None

    try:
        while (
            len(failure_obs) < parsed.max_failure_windows
            and steps < parsed.max_collect_steps
        ):
            obs_ring.append(actor_obs.float().detach().cpu().clone())
            priv_ring.append(critic_obs.float().detach().cpu().clone())
            if len(obs_ring) > parsed.window:
                obs_ring.pop(0)
                priv_ring.pop(0)

            with torch.no_grad():
                dist, _, state = actor.forward_eval(actor_obs.float(), state)
                actions = dist.mean.clamp(-1.0, 1.0)

            asym.step_vec(actor_vec, actions, device)
            asym.step_vec(critic_vec, actions, device)
            done = torch.logical_or(actor_terms > 0.5, critic_terms > 0.5)
            failed = torch.logical_and(done, rewards < parsed.failure_reward_threshold)

            if len(obs_ring) == parsed.window and steps % parsed.anchor_sample_interval == 0:
                ids = torch.randperm(actor_vec.total_agents, device=device)[: parsed.anchor_samples_per_tick]
                for env_id in ids.detach().cpu().tolist():
                    anchor_obs.append(torch.stack([frame[env_id] for frame in obs_ring], dim=0))
                    if len(anchor_obs) >= parsed.max_anchor_windows:
                        break

            if failed.any() and len(obs_ring) == parsed.window:
                for env_id in failed.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                    failure_obs.append(torch.stack([frame[env_id] for frame in obs_ring], dim=0))
                    failure_priv.append(torch.stack([frame[env_id] for frame in priv_ring], dim=0))
                    if len(failure_obs) >= parsed.max_failure_windows:
                        break

            if done.any():
                state = asym.reset_done_state(state, done)
            state = asym.detach_state(state)
            steps += 1
            if steps % parsed.log_interval == 0:
                accum = asym.accumulate_logs(accum, actor_vec.log())

        accum = asym.accumulate_logs(accum, actor_vec.log())
        metrics = asym.finalize_logs(accum)
    finally:
        actor_vec.close()
        critic_vec.close()

    if not failure_obs:
        raise RuntimeError("No failure windows collected; increase max_collect_steps or total_agents")
    return (
        torch.stack(failure_obs, dim=0),
        torch.stack(failure_priv, dim=0),
        torch.stack(anchor_obs, dim=0) if anchor_obs else torch.empty(0),
        metrics,
        steps,
    )


def train_recovery(parsed, actor, anchor, failure_obs, failure_priv, anchor_obs, device):
    actor.train()
    anchor.eval()
    opt = torch.optim.Adam(actor.parameters(), lr=parsed.learning_rate)
    n = failure_obs.shape[0]
    anchor_n = anchor_obs.shape[0]
    last = {}

    for epoch in range(1, parsed.epochs + 1):
        order = torch.randperm(n)
        for start in range(0, n, parsed.batch_size):
            idx = order[start : start + parsed.batch_size]
            obs = failure_obs[idx].to(device)
            priv = failure_priv[idx].to(device)
            recovery_obs = obs
            recovery_priv = priv
            if parsed.recovery_tail > 0:
                recovery_obs = obs[:, -parsed.recovery_tail :]
                recovery_priv = priv[:, -parsed.recovery_tail :]
            targets = teacher_actions_from_privileged_obs(
                recovery_priv.reshape(-1, recovery_priv.shape[-1]),
                parsed,
            ).reshape(recovery_obs.shape[0], recovery_obs.shape[1], -1)

            dist, _ = actor(obs)
            pred_full = dist.mean.reshape(obs.shape[0], obs.shape[1], -1)
            pred = pred_full[:, -parsed.recovery_tail :] if parsed.recovery_tail > 0 else pred_full
            recovery_loss = F.mse_loss(pred, targets)

            with torch.no_grad():
                anchor_dist, _ = anchor(obs)
                anchor_mean = anchor_dist.mean.reshape(pred_full.shape)
            anchor_loss = F.mse_loss(pred_full, anchor_mean)

            general_loss = torch.zeros((), device=device)
            if anchor_n > 0 and parsed.general_anchor_weight > 0.0:
                gen_idx = torch.randint(0, anchor_n, (min(parsed.batch_size, anchor_n),))
                gen = anchor_obs[gen_idx].to(device)
                gen_dist, _ = actor(gen)
                with torch.no_grad():
                    gen_anchor_dist, _ = anchor(gen)
                general_loss = F.mse_loss(gen_dist.mean, gen_anchor_dist.mean)

            loss = (
                parsed.recovery_weight * recovery_loss
                + parsed.failure_anchor_weight * anchor_loss
                + parsed.general_anchor_weight * general_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), parsed.max_grad_norm)
            opt.step()

            last = {
                "loss": float(loss.detach().cpu()),
                "recovery": float(recovery_loss.detach().cpu()),
                "fail_anchor": float(anchor_loss.detach().cpu()),
                "general_anchor": float(general_loss.detach().cpu()),
            }

        if epoch % parsed.print_interval == 0 or epoch == parsed.epochs:
            fields = " ".join(f"{key}={value:.6g}" for key, value in last.items())
            print(f"epoch={epoch} {fields}", flush=True)


def run(parsed):
    from pufferlib import _C

    random.seed(parsed.seed)
    np.random.seed(parsed.seed)
    torch.manual_seed(parsed.seed)
    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")

    probe_args = asym.make_vec_config(parsed, privileged=False)
    probe_vec = _C.create_vec(probe_args, 0 if device.type == "cpu" else 1)
    try:
        actor = asym.create_actor(probe_vec.obs_size, probe_vec.act_sizes, parsed, device)
        anchor = asym.create_actor(probe_vec.obs_size, probe_vec.act_sizes, parsed, device)
    finally:
        probe_vec.close()

    asym.load_actor(actor, parsed.load_actor, device, parsed.load_action_std)
    asym.load_actor(anchor, parsed.load_actor, device, parsed.load_action_std)
    for param in anchor.parameters():
        param.requires_grad_(False)

    failure_obs, failure_priv, anchor_obs, collect_metrics, steps = collect_windows(parsed, actor, device)
    print(
        f"collected failure_windows={failure_obs.shape[0]} "
        f"anchor_windows={anchor_obs.shape[0]} steps={steps}",
        flush=True,
    )
    asym.print_metrics("collect", collect_metrics)

    train_recovery(parsed, actor, anchor, failure_obs, failure_priv, anchor_obs, device)

    output = Path(parsed.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), output)
    print(f"wrote {output}", flush=True)

    if parsed.eval_after:
        metrics = asym.deterministic_eval(parsed, actor)
        asym.print_metrics("eval", metrics)


def parse_args():
    parser = argparse.ArgumentParser(
        description="QDR-136 near-failure recovery training from privileged labels."
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--total-agents", type=int, default=2048)
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=768)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--max-z-vel", type=float, default=0.25)
    parser.add_argument("--max-rate", type=float, default=1.0)
    parser.add_argument("--vel-ctrl-ki-z", type=float, default=4.0)
    parser.add_argument("--spawn-radius", type=float, default=2.5)
    parser.add_argument("--spawn-radius-min", type=float, default=2.5)
    parser.add_argument("--spawn-height", type=float, default=1.5)
    parser.add_argument("--room-size", type=float, default=3.0)
    parser.add_argument("--ceiling-height", type=float, default=3.0)
    parser.add_argument("--num-rays", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=4.0)
    parser.add_argument("--obstacle-mode", type=int, default=7)
    parser.add_argument("--route-goal-radius", type=float, default=1.0)
    parser.add_argument("--route-goal-count", type=int, default=1)
    parser.add_argument("--route-goal-count-min", type=int, default=0)
    parser.add_argument("--route-path-margin", type=float, default=0.35)
    parser.add_argument("--route-target-mode", type=int, default=1)
    parser.add_argument("--route-blocked-target-prob", type=float, default=1.0)
    parser.add_argument("--target-z", type=float, default=1.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--network", choices=("MLP", "MinGRU", "GRU", "LSTM"), default="MinGRU")
    parser.add_argument("--action-std", type=float, default=0.05)
    parser.add_argument("--load-actor", required=True)
    parser.add_argument("--load-action-std", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--max-failure-windows", type=int, default=4096)
    parser.add_argument("--max-anchor-windows", type=int, default=4096)
    parser.add_argument("--max-collect-steps", type=int, default=10000)
    parser.add_argument("--failure-reward-threshold", type=float, default=-1.0)
    parser.add_argument("--anchor-sample-interval", type=int, default=8)
    parser.add_argument("--anchor-samples-per-tick", type=int, default=8)
    parser.add_argument("--recovery-tail", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--recovery-weight", type=float, default=1.0)
    parser.add_argument("--failure-anchor-weight", type=float, default=0.2)
    parser.add_argument("--general-anchor-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--teacher-speed-gain", type=float, default=2.5)
    parser.add_argument("--teacher-max-speed", type=float, default=0.85)
    parser.add_argument("--teacher-slow-radius", type=float, default=0.12)
    parser.add_argument("--teacher-z-gain", type=float, default=0.8)
    parser.add_argument("--teacher-z-damp", type=float, default=0.15)
    parser.add_argument("--teacher-max-z", type=float, default=0.6)
    parser.add_argument("--route-wall-obs", type=int, choices=(0, 1), default=1)
    parser.add_argument("--clearance-speed-anchor", type=float, default=0.0)
    parser.add_argument("--clearance-slow-start-m", type=float, default=0.75)
    parser.add_argument("--clearance-stop-m", type=float, default=0.22)
    parser.add_argument("--clearance-min-speed-scale", type=float, default=0.35)
    parser.add_argument("--print-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=64)
    parser.add_argument("--eval-after", action="store_true")
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--max-eval-steps", type=int, default=30000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
