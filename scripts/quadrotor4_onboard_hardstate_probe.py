#!/usr/bin/env python3
import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quadrotor4_frontier_target_distill import (  # noqa: E402
    OBSTACLE_MODES,
    _combined_safety_scale,
    add_common,
    make_vec_config,
    step_vec,
    target_labels,
    target_to_actions,
    tensor_views,
)


@dataclass
class StallState:
    step: int
    env_id: int
    episode_step: int
    since_progress: int
    snapshot: bytes
    coverage: float
    min_ray_m: float
    target_dist: float
    target_exists: float
    action_speed: float


def _make_vec(parsed, device, total_agents):
    from pufferlib import _C

    run = copy.copy(parsed)
    run.total_agents = total_agents
    args = make_vec_config(run)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, rewards, terminals = tensor_views(vec, device)
    vec.reset()
    return vec, obs, rewards, terminals


def _ray_stats(obs, parsed):
    if parsed.num_rays <= 0 or obs.shape[1] < 26 + parsed.num_rays:
        return torch.zeros((obs.shape[0],), dtype=torch.float32, device=obs.device)
    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    return rays.min(dim=1).values


def _done_outcome(reward, terminal, parsed):
    if not terminal:
        return "open"
    if reward > parsed.success_reward_threshold:
        return "success"
    if reward < parsed.crash_reward_threshold:
        return "crash"
    return "timeout"


def _baseline_actions(obs, parsed):
    return target_to_actions(target_labels(obs, parsed), parsed, obs)


def _unit_xy(xy):
    norm = torch.linalg.vector_norm(xy, dim=1, keepdim=True)
    return xy / norm.clamp_min(1e-6)


def _candidate_dirs(obs, parsed):
    target = target_labels(obs, parsed)
    target_dir = target[:, 0:2].float().clamp(-1.0, 1.0)
    base = _unit_xy(target_dir)
    if torch.linalg.vector_norm(target_dir, dim=1).max() < 1e-6:
        base = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=obs.device)

    offsets = torch.tensor(
        [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 135, -135, 180],
        dtype=torch.float32,
        device=obs.device,
    ) * (math.pi / 180.0)
    cos_o = torch.cos(offsets)
    sin_o = torch.sin(offsets)
    rotated = torch.stack(
        (
            base[:, 0:1] * cos_o[None, :] - base[:, 1:2] * sin_o[None, :],
            base[:, 0:1] * sin_o[None, :] + base[:, 1:2] * cos_o[None, :],
        ),
        dim=2,
    ).reshape(-1, 2)

    if parsed.num_rays <= 0 or obs.shape[1] < 26 + parsed.num_rays:
        return rotated

    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    topk = min(parsed.mpc_ray_dirs, parsed.num_rays)
    best = rays[0].topk(topk).indices
    angles = best.float() * (2.0 * math.pi / float(parsed.num_rays))
    ray_dirs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    return torch.cat((rotated, ray_dirs), dim=0)


def _candidate_actions(obs, parsed):
    base = _baseline_actions(obs, parsed)
    dirs = _candidate_dirs(obs, parsed)
    speed_fracs = torch.tensor(
        parsed.mpc_speed_fractions,
        dtype=torch.float32,
        device=obs.device,
    )
    num_dirs = dirs.shape[0]
    dirs = dirs[:, None, :].expand(-1, speed_fracs.numel(), -1).reshape(-1, 2)
    speeds = (
        parsed.speed_fraction * speed_fracs[None, :].expand(num_dirs, -1)
    ).reshape(-1, 1)

    actions = torch.zeros((dirs.shape[0], 4), dtype=torch.float32, device=obs.device)
    scale = _combined_safety_scale(
        obs.expand(dirs.shape[0], -1),
        dirs,
        parsed,
    )[:, None]
    actions[:, 0:2] = dirs * speeds * scale
    actions[:, 3:4] = (dirs[:, 1:2] * parsed.yaw_fraction).clamp(-1.0, 1.0)
    return torch.cat((base, actions.clamp(-1.0, 1.0)), dim=0)


def _score_candidate(parsed, start_coverage, records, min_clearance):
    if not records:
        return -1.0e9

    final = records[-1]
    coverage_gain = final["coverage"] - start_coverage
    best_gain = max(item["coverage"] for item in records) - start_coverage
    score = parsed.mpc_final_coverage_weight * coverage_gain
    score += parsed.mpc_best_coverage_weight * best_gain
    score += parsed.mpc_clearance_weight * min_clearance
    score -= parsed.mpc_step_penalty * len(records)

    if final["outcome"] == "success":
        score += parsed.mpc_success_bonus
    elif final["outcome"] == "crash":
        score -= parsed.mpc_crash_penalty
    elif final["outcome"] == "timeout":
        score -= parsed.mpc_timeout_penalty
    return score


def _roll_constant_action(parsed, vec, obs, rewards, terminals, action, device, steps):
    records = []
    min_clearance = float("inf")
    for _ in range(steps):
        step_vec(vec, action, device)
        coverage = float(obs[0, 17].detach().cpu())
        reward = float(rewards[0].detach().cpu())
        terminal = float(terminals[0].detach().cpu()) > 0.5
        min_clearance = min(
            min_clearance,
            float(_ray_stats(obs[0:1], parsed)[0].detach().cpu()),
        )
        records.append({
            "coverage": coverage,
            "reward": reward,
            "terminal": terminal,
            "outcome": _done_outcome(reward, terminal, parsed),
        })
        if terminal:
            break
    return records, min_clearance


def choose_mpc_action(parsed, vec, obs, rewards, terminals, device):
    start_snapshot = vec.snapshot(0)
    start_coverage = float(obs[0, 17].detach().cpu())
    candidates = _candidate_actions(obs[0:1], parsed)

    best_score = -1.0e18
    best_action = candidates[0:1].detach().clone()
    for idx in range(candidates.shape[0]):
        vec.restore(0, start_snapshot)
        action = candidates[idx : idx + 1].detach().clone()
        records, min_clearance = _roll_constant_action(
            parsed,
            vec,
            obs,
            rewards,
            terminals,
            action,
            device,
            parsed.mpc_rollout_steps,
        )
        score = _score_candidate(parsed, start_coverage, records, min_clearance)
        if score > best_score:
            best_score = score
            best_action = action

    vec.restore(0, start_snapshot)
    return best_action


def collect_stall_states(parsed, device):
    vec, obs, rewards, terminals = _make_vec(parsed, device, parsed.total_agents)
    episode_step = torch.zeros((vec.total_agents,), dtype=torch.int64, device=device)
    last_progress_step = torch.zeros_like(episode_step)
    best_coverage = torch.zeros((vec.total_agents,), dtype=torch.float32, device=device)
    captured_this_episode = torch.zeros((vec.total_agents,), dtype=torch.bool, device=device)
    hard_states = []

    try:
        for step in range(1, parsed.collect_steps + 1):
            with torch.no_grad():
                target = target_labels(obs, parsed)
                actions = target_to_actions(target, parsed, obs)
                min_ray = _ray_stats(obs, parsed)
                speed = torch.linalg.vector_norm(actions[:, 0:2].float(), dim=1)

            since_progress = episode_step - last_progress_step
            capture = (
                (since_progress >= parsed.stall_steps)
                & (min_ray <= parsed.stall_clearance)
                & (target[:, 3].float() > parsed.exists_threshold)
                & ~captured_this_episode
            )
            for env_id in capture.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                hard_states.append(
                    StallState(
                        step=step,
                        env_id=env_id,
                        episode_step=int(episode_step[env_id].detach().cpu()),
                        since_progress=int(since_progress[env_id].detach().cpu()),
                        snapshot=vec.snapshot(env_id),
                        coverage=float(obs[env_id, 17].detach().cpu()),
                        min_ray_m=float(min_ray[env_id].detach().cpu()),
                        target_dist=float(target[env_id, 2].detach().cpu()),
                        target_exists=float(target[env_id, 3].detach().cpu()),
                        action_speed=float(speed[env_id].detach().cpu()),
                    )
                )
                captured_this_episode[env_id] = True
                if len(hard_states) >= parsed.hard_states:
                    return hard_states, step

            step_vec(vec, actions, device)
            episode_step += 1

            coverage = obs[:, 17].float()
            progressed = coverage > best_coverage + parsed.stall_delta
            best_coverage = torch.where(progressed, coverage, best_coverage)
            last_progress_step = torch.where(progressed, episode_step, last_progress_step)

            done = terminals > 0.5
            if done.any():
                episode_step = torch.where(done, torch.zeros_like(episode_step), episode_step)
                last_progress_step = torch.where(
                    done, torch.zeros_like(last_progress_step), last_progress_step
                )
                best_coverage = torch.where(done, torch.zeros_like(best_coverage), best_coverage)
                captured_this_episode = torch.where(
                    done, torch.zeros_like(captured_this_episode), captured_this_episode
                )
    finally:
        vec.close()
    return hard_states, parsed.collect_steps


def replay_controller(parsed, hard_states, controller, device):
    vec, obs, rewards, terminals = _make_vec(parsed, device, 1)
    outcomes = []
    try:
        for hard_state in hard_states:
            vec.restore(0, hard_state.snapshot)
            start_coverage = float(obs[0, 17].detach().cpu())
            best_coverage = start_coverage
            min_clearance = float("inf")
            committed_action = None
            committed_steps_left = 0
            final = {
                "outcome": "open",
                "steps": 0,
                "coverage": start_coverage,
                "coverage_gain": 0.0,
                "best_gain": 0.0,
                "min_clearance": 0.0,
            }
            for step in range(1, parsed.replay_horizon + 1):
                with torch.no_grad():
                    if controller == "baseline":
                        actions = _baseline_actions(obs[0:1], parsed)
                    elif controller == "mpc":
                        if committed_action is None or committed_steps_left <= 0:
                            committed_action = choose_mpc_action(
                                parsed, vec, obs, rewards, terminals, device
                            )
                            committed_steps_left = parsed.mpc_commit_steps
                        actions = committed_action
                    else:
                        raise ValueError(controller)
                step_vec(vec, actions, device)
                if controller == "mpc":
                    committed_steps_left -= 1
                coverage = float(obs[0, 17].detach().cpu())
                best_coverage = max(best_coverage, coverage)
                min_clearance = min(
                    min_clearance,
                    float(_ray_stats(obs[0:1], parsed)[0].detach().cpu()),
                )
                reward = float(rewards[0].detach().cpu())
                terminal = float(terminals[0].detach().cpu()) > 0.5
                outcome = _done_outcome(reward, terminal, parsed)
                final = {
                    "outcome": outcome,
                    "steps": step,
                    "coverage": coverage,
                    "coverage_gain": coverage - start_coverage,
                    "best_gain": best_coverage - start_coverage,
                    "min_clearance": min_clearance,
                }
                if terminal:
                    break
            outcomes.append(final)
    finally:
        vec.close()
    return outcomes


def summarize_states(hard_states, collect_steps):
    def mean(values):
        return sum(values) / max(len(values), 1)

    print(f"collected_stall_states n={len(hard_states)} steps={collect_steps}", flush=True)
    if not hard_states:
        return
    print(f"hard.coverage_mean {mean([s.coverage for s in hard_states]):.6g}", flush=True)
    print(f"hard.min_ray_mean {mean([s.min_ray_m for s in hard_states]):.6g}", flush=True)
    print(f"hard.since_progress_mean {mean([s.since_progress for s in hard_states]):.6g}", flush=True)
    print(f"hard.target_dist_mean {mean([s.target_dist for s in hard_states]):.6g}", flush=True)
    print(f"hard.action_speed_mean {mean([s.action_speed for s in hard_states]):.6g}", flush=True)


def summarize_outcomes(name, outcomes):
    counts = {"success": 0, "crash": 0, "timeout": 0, "open": 0}
    for item in outcomes:
        counts[item["outcome"]] += 1
    n = max(len(outcomes), 1)
    mean_steps = sum(item["steps"] for item in outcomes) / n
    mean_cov = sum(item["coverage"] for item in outcomes) / n
    mean_gain = sum(item["coverage_gain"] for item in outcomes) / n
    mean_best_gain = sum(item["best_gain"] for item in outcomes) / n
    mean_clearance = sum(item["min_clearance"] for item in outcomes) / n
    print(
        f"{name} n={len(outcomes)} "
        f"success={counts['success']} success_rate={counts['success'] / n:.6g} "
        f"crash={counts['crash']} crash_rate={counts['crash'] / n:.6g} "
        f"timeout={counts['timeout']} timeout_rate={counts['timeout'] / n:.6g} "
        f"open={counts['open']} open_rate={counts['open'] / n:.6g} "
        f"mean_steps={mean_steps:.6g} mean_coverage={mean_cov:.6g} "
        f"mean_gain={mean_gain:.6g} mean_best_gain={mean_best_gain:.6g} "
        f"mean_min_clearance={mean_clearance:.6g}",
        flush=True,
    )


def run(parsed):
    from pufferlib import _C

    parsed.obstacle_mode = OBSTACLE_MODES[parsed.mode]
    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    hard_states, collect_steps = collect_stall_states(parsed, device)
    summarize_states(hard_states, collect_steps)
    if not hard_states:
        return

    baseline = replay_controller(parsed, hard_states, "baseline", device)
    summarize_outcomes("baseline_replay", baseline)
    if parsed.run_mpc:
        mpc = replay_controller(parsed, hard_states, "mpc", device)
        summarize_outcomes("mpc_replay", mpc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture active onboard-planner wall-stall states and replay them under stronger local controllers."
    )
    add_common(parser)
    parser.add_argument("--hard-states", type=int, default=16)
    parser.add_argument("--collect-steps", type=int, default=20000)
    parser.add_argument("--stall-steps", type=int, default=90)
    parser.add_argument("--stall-delta", type=float, default=0.002)
    parser.add_argument("--stall-clearance", type=float, default=0.35)
    parser.add_argument("--replay-horizon", type=int, default=512)
    parser.add_argument("--run-mpc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mpc-rollout-steps", type=int, default=12)
    parser.add_argument("--mpc-commit-steps", type=int, default=1)
    parser.add_argument("--mpc-ray-dirs", type=int, default=6)
    parser.add_argument("--mpc-final-coverage-weight", type=float, default=250.0)
    parser.add_argument("--mpc-best-coverage-weight", type=float, default=100.0)
    parser.add_argument("--mpc-clearance-weight", type=float, default=0.20)
    parser.add_argument("--mpc-step-penalty", type=float, default=0.01)
    parser.add_argument("--mpc-success-bonus", type=float, default=1000.0)
    parser.add_argument("--mpc-crash-penalty", type=float, default=1000.0)
    parser.add_argument("--mpc-timeout-penalty", type=float, default=100.0)
    parser.add_argument(
        "--mpc-speed-fractions",
        type=lambda value: tuple(float(item) for item in value.split(",")),
        default=(1.0, 0.65, 0.35),
    )
    parser.add_argument("--crash-reward-threshold", type=float, default=-5.0)
    parser.add_argument("--success-reward-threshold", type=float, default=5.0)
    parser.set_defaults(
        total_agents=64,
        num_buffers=1,
        num_threads=8,
        mode="procedural_indoor_hard_p1",
        explore_frontier_target_mode="onboard",
        explore_reachable_coverage=True,
        room_size=6.0,
        max_steps=4096,
        max_eval_steps=60000,
        speed_fraction=0.35,
        spawn_radius_min=0.0,
        explore_goal_fraction=0.98,
        safety_cone_ray_stop=0.25,
        safety_cone_deg=60.0,
        safety_slow_ray=0.70,
    )
    parsed = parser.parse_args()
    if parsed.hard_states <= 0:
        parser.error("--hard-states must be positive")
    if parsed.mpc_rollout_steps <= 0:
        parser.error("--mpc-rollout-steps must be positive")
    if parsed.mpc_commit_steps <= 0:
        parser.error("--mpc-commit-steps must be positive")
    if not parsed.mpc_speed_fractions:
        parser.error("--mpc-speed-fractions must contain at least one value")
    return parsed


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
