#!/usr/bin/env python3
import argparse
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import quadrotor4_asym_ppo as asym


@dataclass
class HardState:
    step: int
    env_id: int
    actor_snapshot: bytes
    critic_snapshot: bytes
    actor_state: tuple
    min_ray_m: float
    target_dist_obs: float


def clone_state(state):
    if not isinstance(state, tuple):
        return state
    return tuple(value.detach().clone() if torch.is_tensor(value) else value for value in state)


def outcome_from_rollout(records, failure_reward_threshold):
    for idx, record in enumerate(records, start=1):
        if record["terminal"] > 0.5:
            failed = record["reward"] < failure_reward_threshold
            return {
                "terminal": 1,
                "failed": int(failed),
                "steps": idx,
                "reward": record["reward"],
            }
    return {"terminal": 0, "failed": 0, "steps": len(records), "reward": records[-1]["reward"]}


def collect_hard_states(parsed, actor_vec, critic_vec, actor_obs, critic_obs,
        rewards, terminals, actor, device):
    actor.eval()
    state = actor.initial_state(actor_vec.total_agents, device)
    ring = deque(maxlen=parsed.rewind + 1)
    hard_states = []

    for step in range(1, parsed.max_collect_steps + 1):
        ring.append(
            {
                "step": step,
                "actor_snapshots": [
                    actor_vec.snapshot(env_id)
                    for env_id in range(actor_vec.total_agents)
                ],
                "critic_snapshots": [
                    critic_vec.snapshot(env_id)
                    for env_id in range(critic_vec.total_agents)
                ],
                "actor_obs": actor_obs.detach().cpu().clone(),
                "state": clone_state(state),
            }
        )

        with torch.no_grad():
            dist, _, state = actor.forward_eval(actor_obs.float(), state)
            actions = dist.mean.clamp(-1.0, 1.0)
        asym.step_vec(actor_vec, actions, device)
        asym.step_vec(critic_vec, actions, device)

        done = terminals > 0.5
        failed = torch.logical_and(done, rewards < parsed.failure_reward_threshold)
        if failed.any() and len(ring) > parsed.rewind:
            source = list(ring)[-(parsed.rewind + 1)]
            for env_id in failed.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                hard_states.append(
                    HardState(
                        step=source["step"],
                        env_id=env_id,
                        actor_snapshot=source["actor_snapshots"][env_id],
                        critic_snapshot=source["critic_snapshots"][env_id],
                        actor_state=clone_state(source["state"]),
                        min_ray_m=float(
                            source["actor_obs"][
                                env_id,
                                26 : 26 + parsed.num_rays,
                            ].min().item() * parsed.ray_length
                        ),
                        target_dist_obs=float(
                            source["actor_obs"][env_id, 20].item()
                        ),
                    )
                )
                if len(hard_states) >= parsed.hard_states:
                    return hard_states, step

        if done.any():
            state = asym.reset_done_state(state, done)
        state = asym.detach_state(state)

    return hard_states, parsed.max_collect_steps


def restore_pair(actor_vec, critic_vec, hard_state):
    actor_vec.restore(hard_state.env_id, hard_state.actor_snapshot)
    critic_vec.restore(hard_state.env_id, hard_state.critic_snapshot)


def run_actor_replay(parsed, actor_vec, critic_vec, actor_obs, rewards,
        terminals, actor, hard_state, device):
    restore_pair(actor_vec, critic_vec, hard_state)
    state = clone_state(hard_state.actor_state)
    records = []
    actor.eval()
    for _ in range(parsed.replay_horizon):
        with torch.no_grad():
            dist, _, state = actor.forward_eval(actor_obs.float(), state)
            actions = dist.mean.clamp(-1.0, 1.0)
        asym.step_vec(actor_vec, actions, device)
        asym.step_vec(critic_vec, actions, device)
        records.append(
            {
                "reward": float(rewards[hard_state.env_id].detach().cpu().item()),
                "terminal": float(terminals[hard_state.env_id].detach().cpu().item()),
            }
        )
        if records[-1]["terminal"] > 0.5:
            break
        done = terminals > 0.5
        if done.any():
            state = asym.reset_done_state(state, done)
        state = asym.detach_state(state)
    return records


def run_teacher_replay(parsed, actor_vec, critic_vec, critic_obs, rewards,
        terminals, hard_state, device, actor_obs):
    restore_pair(actor_vec, critic_vec, hard_state)
    records = []
    search_action = None
    if parsed.teacher_mode == "rollout_search":
        search_action = rollout_search_action(
            parsed, actor_vec, critic_vec, actor_obs, critic_obs,
            rewards, terminals, hard_state.env_id, device,
        )

    for step in range(parsed.replay_horizon):
        with torch.no_grad():
            actions = teacher_actions(parsed, actor_obs.float(), critic_obs.float())
            if search_action is not None and step < parsed.search_commit_steps:
                actions[hard_state.env_id] = search_action
        asym.step_vec(actor_vec, actions, device)
        asym.step_vec(critic_vec, actions, device)
        records.append(
            {
                "reward": float(rewards[hard_state.env_id].detach().cpu().item()),
                "terminal": float(terminals[hard_state.env_id].detach().cpu().item()),
            }
        )
        if records[-1]["terminal"] > 0.5:
            break
    return records


def teacher_actions(parsed, actor_obs, critic_obs):
    if parsed.teacher_mode == "waypoint":
        return asym.teacher_actions_from_privileged_obs(critic_obs, parsed)
    if parsed.teacher_mode == "rollout_search":
        return asym.teacher_actions_from_privileged_obs(critic_obs, parsed)
    if parsed.teacher_mode != "ray_avoid":
        raise ValueError(f"unknown teacher mode {parsed.teacher_mode!r}")

    unit = critic_obs[:, 17:20].float()
    dist = critic_obs[:, 20:21].float()
    closing = critic_obs[:, 21:22].float()
    xy_norm = torch.linalg.vector_norm(unit[:, :2], dim=1, keepdim=True).clamp_min(1e-6)
    desired = unit[:, :2] / xy_norm

    ray_start = 26
    rays = actor_obs[:, ray_start : ray_start + parsed.num_rays].float()
    ray_dists = rays * parsed.ray_length
    angles = torch.arange(parsed.num_rays, device=actor_obs.device, dtype=torch.float32)
    angles = angles * (2.0 * math.pi / max(parsed.num_rays, 1))
    ray_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    danger = ((parsed.avoid_radius - ray_dists) / max(parsed.avoid_radius, 1e-6)).clamp(0.0, 1.0)
    weights = danger.pow(parsed.avoid_power)
    repel = -torch.matmul(weights, ray_dirs)
    steer = desired + parsed.avoid_gain * repel
    steer = steer / torch.linalg.vector_norm(steer, dim=1, keepdim=True).clamp_min(1e-6)

    clearance = ray_dists.min(dim=1, keepdim=True).values
    slow = ((clearance - parsed.avoid_stop_radius) /
            max(parsed.avoid_slow_radius - parsed.avoid_stop_radius, 1e-6))
    slow = slow.clamp(parsed.avoid_min_speed_scale, 1.0)
    speed = torch.clamp(parsed.teacher_speed_gain * dist, 0.0, parsed.teacher_max_speed)
    speed = torch.where(dist > parsed.teacher_slow_radius, speed, speed * 0.35)
    speed = speed * slow

    actions = torch.zeros((actor_obs.shape[0], 4), dtype=torch.float32, device=actor_obs.device)
    actions[:, 0:2] = steer * speed
    actions[:, 2:3] = torch.clamp(
        parsed.teacher_z_gain * unit[:, 2:3] - parsed.teacher_z_damp * closing,
        -parsed.teacher_max_z,
        parsed.teacher_max_z,
    )
    return torch.clamp(actions, -1.0, 1.0)


def candidate_actions(parsed, critic_obs, env_id, device):
    unit = critic_obs[env_id, 17:20].float()
    dist = critic_obs[env_id, 20:21].float()
    closing = critic_obs[env_id, 21:22].float()
    z_action = torch.clamp(
        parsed.teacher_z_gain * unit[2:3] - parsed.teacher_z_damp * closing,
        -parsed.teacher_max_z,
        parsed.teacher_max_z,
    )[0]

    actions = []
    for speed in parsed.search_speeds:
        for idx in range(parsed.search_directions):
            angle = 2.0 * math.pi * idx / max(parsed.search_directions, 1)
            actions.append([speed * math.cos(angle), speed * math.sin(angle), float(z_action), 0.0])

    xy_norm = torch.linalg.vector_norm(unit[:2]).clamp_min(1e-6)
    target_xy = unit[:2] / xy_norm
    target_speed = min(
        parsed.teacher_max_speed,
        float((parsed.teacher_speed_gain * dist).item()),
    )
    actions.append([
        float(target_xy[0].item() * target_speed),
        float(target_xy[1].item() * target_speed),
        float(z_action),
        0.0,
    ])
    actions.append([0.0, 0.0, float(z_action), 0.0])
    return torch.tensor(actions, dtype=torch.float32, device=device)


def score_search_rollout(parsed, actor_vec, critic_vec, actor_obs, critic_obs,
        rewards, terminals, env_id, action, actor_snapshot, critic_snapshot, device):
    actor_vec.restore(env_id, actor_snapshot)
    critic_vec.restore(env_id, critic_snapshot)
    score = 0.0
    for step in range(parsed.search_horizon):
        actions = teacher_actions(parsed, actor_obs.float(), critic_obs.float())
        actions[env_id] = action
        asym.step_vec(actor_vec, actions, device)
        asym.step_vec(critic_vec, actions, device)

        reward = float(rewards[env_id].detach().cpu().item())
        terminal = float(terminals[env_id].detach().cpu().item())
        ray_min = float(
            actor_obs[
                env_id,
                26 : 26 + parsed.num_rays,
            ].detach().cpu().min().item() * parsed.ray_length
        )
        score += reward + parsed.search_clearance_weight * ray_min
        if terminal > 0.5:
            if reward < parsed.failure_reward_threshold:
                score -= parsed.search_crash_penalty
            else:
                score += parsed.search_success_bonus
            score -= parsed.search_step_penalty * step
            break
    return score


def rollout_search_action(parsed, actor_vec, critic_vec, actor_obs, critic_obs,
        rewards, terminals, env_id, device):
    actor_snapshot = actor_vec.snapshot(env_id)
    critic_snapshot = critic_vec.snapshot(env_id)
    actions = candidate_actions(parsed, critic_obs, env_id, device)
    best_score = None
    best_action = actions[0].clone()
    for action in actions:
        score = score_search_rollout(
            parsed, actor_vec, critic_vec, actor_obs, critic_obs,
            rewards, terminals, env_id, action, actor_snapshot,
            critic_snapshot, device,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_action = action.clone()

    actor_vec.restore(env_id, actor_snapshot)
    critic_vec.restore(env_id, critic_snapshot)
    return best_action


def summarize_outcomes(name, outcomes):
    n = max(len(outcomes), 1)
    failed = sum(outcome["failed"] for outcome in outcomes)
    terminal = sum(outcome["terminal"] for outcome in outcomes)
    open_count = len(outcomes) - terminal
    mean_steps = sum(outcome["steps"] for outcome in outcomes) / n
    mean_terminal_reward = (
        sum(outcome["reward"] for outcome in outcomes if outcome["terminal"]) /
        max(terminal, 1)
    )
    print(
        f"{name} n={len(outcomes)} failed={failed} "
        f"failed_rate={failed / n:.6g} terminal={terminal} "
        f"open={open_count} mean_steps={mean_steps:.6g} "
        f"mean_terminal_reward={mean_terminal_reward:.6g}",
        flush=True,
    )


def summarize_hard_states(hard_states):
    min_rays = sorted(state.min_ray_m for state in hard_states)
    target_dists = [state.target_dist_obs for state in hard_states]
    n = max(len(hard_states), 1)
    def pct(values, q):
        idx = min(max(int(round(q * (len(values) - 1))), 0), len(values) - 1)
        return values[idx]

    print(
        f"hard_state_geometry n={len(hard_states)} "
        f"min_ray_mean={sum(min_rays) / n:.6g} "
        f"min_ray_p10={pct(min_rays, 0.10):.6g} "
        f"min_ray_p50={pct(min_rays, 0.50):.6g} "
        f"target_dist_obs_mean={sum(target_dists) / n:.6g}",
        flush=True,
    )


def run(parsed):
    from pufferlib import _C

    device = torch.device("cpu")
    actor_args = asym.make_vec_config(parsed, privileged=False)
    critic_args = asym.make_vec_config(parsed, privileged=True)
    actor_vec = _C.create_vec(actor_args, 0)
    critic_vec = _C.create_vec(critic_args, 0)
    actor_obs, rewards, terminals = asym.tensor_views(actor_vec, device)
    critic_obs, _, _ = asym.tensor_views(critic_vec, device)
    actor_vec.reset()
    critic_vec.reset()

    try:
        actor = asym.create_actor(actor_vec.obs_size, actor_vec.act_sizes, parsed, device)
        asym.load_actor(actor, parsed.checkpoint, device, parsed.load_action_std)
        hard_states, collect_steps = collect_hard_states(
            parsed, actor_vec, critic_vec, actor_obs, critic_obs,
            rewards, terminals, actor, device,
        )
        print(
            f"collected hard_states={len(hard_states)} "
            f"collect_steps={collect_steps} rewind={parsed.rewind}",
            flush=True,
        )
        if len(hard_states) < parsed.hard_states:
            raise RuntimeError("insufficient hard states collected")
        summarize_hard_states(hard_states)

        actor_outcomes = []
        teacher_outcomes = []
        for hard_state in hard_states:
            actor_records = run_actor_replay(
                parsed, actor_vec, critic_vec, actor_obs, rewards,
                terminals, actor, hard_state, device,
            )
            teacher_records = run_teacher_replay(
                parsed, actor_vec, critic_vec, critic_obs, rewards,
                terminals, hard_state, device, actor_obs,
            )
            actor_outcomes.append(
                outcome_from_rollout(actor_records, parsed.failure_reward_threshold)
            )
            teacher_outcomes.append(
                outcome_from_rollout(teacher_records, parsed.failure_reward_threshold)
            )

        summarize_outcomes("actor", actor_outcomes)
        summarize_outcomes("teacher", teacher_outcomes)
        recovered = sum(
            1 for actor_o, teacher_o in zip(actor_outcomes, teacher_outcomes)
            if actor_o["failed"] and not teacher_o["failed"]
        )
        print(f"teacher_rescued={recovered}", flush=True)
        print(f"teacher_rescue_rate={recovered / max(len(hard_states), 1):.6g}", flush=True)
    finally:
        actor_vec.close()
        critic_vec.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare QDR actor and privileged teacher on identical resettable hard states."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="experiments/qdr145/asym_mixed15_from_qdr144_s88/qdr135_asym_ppo_000005.bin",
    )
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--total-agents", type=int, default=64)
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1024)
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
    parser.add_argument("--route-goal-count", type=int, default=3)
    parser.add_argument("--route-goal-count-min", type=int, default=0)
    parser.add_argument("--route-path-margin", type=float, default=0.35)
    parser.add_argument("--route-target-mode", type=int, default=1)
    parser.add_argument("--route-blocked-target-prob", type=float, default=1.0)
    parser.add_argument("--target-z", type=float, default=1.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--network", choices=("MLP", "MinGRU", "GRU", "LSTM"), default="MinGRU")
    parser.add_argument("--action-std", type=float, default=0.05)
    parser.add_argument("--load-action-std", type=float, default=0.05)
    parser.add_argument("--teacher-speed-gain", type=float, default=2.5)
    parser.add_argument("--teacher-max-speed", type=float, default=0.85)
    parser.add_argument("--teacher-slow-radius", type=float, default=0.12)
    parser.add_argument("--teacher-z-gain", type=float, default=0.8)
    parser.add_argument("--teacher-z-damp", type=float, default=0.15)
    parser.add_argument("--teacher-max-z", type=float, default=0.6)
    parser.add_argument(
        "--teacher-mode",
        choices=("waypoint", "ray_avoid", "rollout_search"),
        default="waypoint",
    )
    parser.add_argument("--avoid-radius", type=float, default=0.9)
    parser.add_argument("--avoid-stop-radius", type=float, default=0.22)
    parser.add_argument("--avoid-slow-radius", type=float, default=0.9)
    parser.add_argument("--avoid-min-speed-scale", type=float, default=0.2)
    parser.add_argument("--avoid-gain", type=float, default=1.0)
    parser.add_argument("--avoid-power", type=float, default=2.0)
    parser.add_argument("--search-directions", type=int, default=16)
    parser.add_argument("--search-speeds", type=float, nargs="+", default=(0.25, 0.5, 0.85))
    parser.add_argument("--search-horizon", type=int, default=24)
    parser.add_argument("--search-commit-steps", type=int, default=24)
    parser.add_argument("--search-clearance-weight", type=float, default=0.02)
    parser.add_argument("--search-crash-penalty", type=float, default=100.0)
    parser.add_argument("--search-success-bonus", type=float, default=20.0)
    parser.add_argument("--search-step-penalty", type=float, default=0.0)
    parser.add_argument("--hard-states", type=int, default=128)
    parser.add_argument("--rewind", type=int, default=8)
    parser.add_argument("--replay-horizon", type=int, default=128)
    parser.add_argument("--max-collect-steps", type=int, default=8192)
    parser.add_argument("--failure-reward-threshold", type=float, default=-1.0)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
