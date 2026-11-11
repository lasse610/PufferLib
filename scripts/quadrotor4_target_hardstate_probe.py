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

from quadrotor4_frontier_target_distill import (  # noqa: E402
    OBSTACLE_MODES,
    add_common,
    deployable_input,
    load_policy,
    make_vec_config,
    pred_recovery_gate_mask,
    pred_vector_fallback_mask,
    reset_done_state,
    step_vec,
    target_labels,
    target_to_actions,
    tensor_views,
)
from quadrotor4_privileged_distill import clone_detached_state, detach_state  # noqa: E402


@dataclass
class HardState:
    step: int
    env_id: int
    snapshot: bytes
    policy_state: tuple
    coverage: float
    min_ray_m: float
    pred_oracle_angle_deg: float
    pred_ray_m: float
    oracle_ray_m: float
    pred_exists: float
    pred_exists_value: float
    oracle_exists: float
    pred_vector_dist: float
    oracle_vector_dist: float
    sector_mse: float
    sector_top_match: float
    pred_sector_max: float
    oracle_sector_max: float
    vector_angle_deg: float
    vector_dist_error: float
    pred_vector_fallback: float
    pred_recovery_gate: float
    pred_recovery_gate_value: float
    patch_mse: float
    action_speed: float


def _unit(target):
    direction = target[:, 0:2].clamp(-1.0, 1.0)
    return direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-6)


def _direction_ray(obs, direction, parsed):
    if parsed.num_rays <= 0:
        return torch.zeros((obs.shape[0],), dtype=torch.float32, device=obs.device)
    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    angle = torch.atan2(direction[:, 1], direction[:, 0]).remainder(2.0 * math.pi)
    idx = torch.round(angle * (parsed.num_rays / (2.0 * math.pi))).long()
    idx = idx.remainder(parsed.num_rays)
    return rays.gather(1, idx[:, None]).squeeze(1)


def _features(obs, pred_target, oracle_target, action, parsed):
    pred_unit = _unit(pred_target)
    oracle_unit = _unit(oracle_target)
    cos = (pred_unit * oracle_unit).sum(dim=1).clamp(-1.0, 1.0)
    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    min_ray = rays.min(dim=1).values if parsed.num_rays > 0 else torch.zeros_like(cos)
    sector_mse = torch.zeros_like(cos)
    sector_top_match = torch.zeros_like(cos)
    pred_sector_max = torch.zeros_like(cos)
    oracle_sector_max = torch.zeros_like(cos)
    if pred_target.shape[1] >= 8 and oracle_target.shape[1] >= 8:
        pred_sector = pred_target[:, 4:8].clamp(0.0, 1.0)
        oracle_sector = oracle_target[:, 4:8].clamp(0.0, 1.0)
        sector_mse = ((pred_sector - oracle_sector) ** 2).mean(dim=1)
        pred_sector_max, pred_sector_idx = pred_sector.max(dim=1)
        oracle_sector_max, oracle_sector_idx = oracle_sector.max(dim=1)
        valid = oracle_sector_max >= parsed.sector_match_threshold
        sector_top_match = (
            (pred_sector_idx == oracle_sector_idx).float() * valid.float()
        )
    vector_angle = torch.zeros_like(cos)
    vector_dist_error = torch.zeros_like(cos)
    pred_vector_dist = torch.zeros_like(cos)
    oracle_vector_dist = torch.zeros_like(cos)
    if pred_target.shape[1] >= 11 and oracle_target.shape[1] >= 11:
        pred_vec = pred_target[:, 8:10].clamp(-1.0, 1.0)
        oracle_vec = oracle_target[:, 8:10].clamp(-1.0, 1.0)
        pred_vec = pred_vec / torch.linalg.vector_norm(pred_vec, dim=1, keepdim=True).clamp_min(1e-6)
        oracle_vec = oracle_vec / torch.linalg.vector_norm(oracle_vec, dim=1, keepdim=True).clamp_min(1e-6)
        vec_cos = (pred_vec * oracle_vec).sum(dim=1).clamp(-1.0, 1.0)
        vector_angle = torch.rad2deg(torch.acos(vec_cos))
        valid = oracle_target[:, 10] > 1e-5
        vector_angle = vector_angle * valid.float()
        vector_dist_error = (pred_target[:, 10].clamp(0.0, 1.0) - oracle_target[:, 10]).abs()
        vector_dist_error = vector_dist_error * valid.float()
        pred_vector_dist = pred_target[:, 10].clamp(0.0, 1.0)
        oracle_vector_dist = oracle_target[:, 10].clamp(0.0, 1.0)
    pred_vector_fallback = pred_vector_fallback_mask(pred_target, parsed, obs).float().squeeze(1)
    pred_recovery_gate = pred_recovery_gate_mask(pred_target, parsed, obs).float().squeeze(1)
    pred_recovery_gate_value = torch.zeros_like(cos)
    if pred_target.shape[1] >= 12:
        pred_recovery_gate_value = torch.sigmoid(pred_target[:, 11])
    patch_mse = torch.zeros_like(cos)
    if pred_target.shape[1] >= 27 and oracle_target.shape[1] >= 27:
        patch_mse = (
            (pred_target[:, 11:27].clamp(0.0, 1.0) - oracle_target[:, 11:27]) ** 2
        ).mean(dim=1)
    return {
        "coverage": obs[:, 17].float(),
        "min_ray_m": min_ray,
        "pred_oracle_angle_deg": torch.rad2deg(torch.acos(cos)),
        "pred_ray_m": _direction_ray(obs, pred_unit, parsed),
        "oracle_ray_m": _direction_ray(obs, oracle_unit, parsed),
        "pred_exists": (pred_target[:, 3] > parsed.exists_threshold).float(),
        "pred_exists_value": pred_target[:, 3].float(),
        "oracle_exists": (oracle_target[:, 3] > parsed.exists_threshold).float(),
        "pred_vector_dist": pred_vector_dist,
        "oracle_vector_dist": oracle_vector_dist,
        "sector_mse": sector_mse,
        "sector_top_match": sector_top_match,
        "pred_sector_max": pred_sector_max,
        "oracle_sector_max": oracle_sector_max,
        "vector_angle_deg": vector_angle,
        "vector_dist_error": vector_dist_error,
        "pred_vector_fallback": pred_vector_fallback,
        "pred_recovery_gate": pred_recovery_gate,
        "pred_recovery_gate_value": pred_recovery_gate_value,
        "patch_mse": patch_mse,
        "action_speed": torch.linalg.vector_norm(action[:, 0:2], dim=1),
    }


def _done_groups(rewards, terminals, parsed):
    done = terminals > 0.5
    crash = done & (rewards < parsed.crash_reward_threshold)
    success = done & (rewards > parsed.success_reward_threshold)
    timeout = done & ~(crash | success)
    return done, crash, success, timeout


def collect_timeout_states(parsed, vec, obs, rewards, terminals, policy, device):
    policy.eval()
    state = policy.initial_state(vec.total_agents, device)
    ring = deque(maxlen=parsed.rewind + 1)
    hard_states = []

    for step in range(1, parsed.max_collect_steps + 1):
        with torch.no_grad():
            x = deployable_input(obs, parsed)
            logits, _, next_state = policy.forward_eval(x, state)
            pred_target = logits.mean
            oracle_target = target_labels(obs, parsed)
            actions = target_to_actions(pred_target, parsed, obs)
            features = _features(obs, pred_target, oracle_target, actions, parsed)

        ring.append(
            {
                "step": step,
                "snapshots": [vec.snapshot(env_id) for env_id in range(vec.total_agents)],
                "state": clone_detached_state(state),
                "features": {key: value.detach().cpu().clone() for key, value in features.items()},
            }
        )

        state = next_state
        step_vec(vec, actions, device)
        done, _, _, timeout = _done_groups(rewards, terminals, parsed)
        if timeout.any() and len(ring) > parsed.rewind:
            source = list(ring)[-(parsed.rewind + 1)]
            for env_id in timeout.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                hard_states.append(
                    HardState(
                        step=source["step"],
                        env_id=env_id,
                        snapshot=source["snapshots"][env_id],
                        policy_state=clone_detached_state(source["state"]),
                        coverage=float(source["features"]["coverage"][env_id].item()),
                        min_ray_m=float(source["features"]["min_ray_m"][env_id].item()),
                        pred_oracle_angle_deg=float(
                            source["features"]["pred_oracle_angle_deg"][env_id].item()
                        ),
                        pred_ray_m=float(source["features"]["pred_ray_m"][env_id].item()),
                        oracle_ray_m=float(source["features"]["oracle_ray_m"][env_id].item()),
                        pred_exists=float(source["features"]["pred_exists"][env_id].item()),
                        pred_exists_value=float(
                            source["features"]["pred_exists_value"][env_id].item()
                        ),
                        oracle_exists=float(source["features"]["oracle_exists"][env_id].item()),
                        pred_vector_dist=float(
                            source["features"]["pred_vector_dist"][env_id].item()
                        ),
                        oracle_vector_dist=float(
                            source["features"]["oracle_vector_dist"][env_id].item()
                        ),
                        sector_mse=float(source["features"]["sector_mse"][env_id].item()),
                        sector_top_match=float(
                            source["features"]["sector_top_match"][env_id].item()
                        ),
                        pred_sector_max=float(
                            source["features"]["pred_sector_max"][env_id].item()
                        ),
                        oracle_sector_max=float(
                            source["features"]["oracle_sector_max"][env_id].item()
                        ),
                        vector_angle_deg=float(
                            source["features"]["vector_angle_deg"][env_id].item()
                        ),
                        vector_dist_error=float(
                            source["features"]["vector_dist_error"][env_id].item()
                        ),
                        pred_vector_fallback=float(
                            source["features"]["pred_vector_fallback"][env_id].item()
                        ),
                        pred_recovery_gate=float(
                            source["features"]["pred_recovery_gate"][env_id].item()
                        ),
                        pred_recovery_gate_value=float(
                            source["features"]["pred_recovery_gate_value"][env_id].item()
                        ),
                        patch_mse=float(source["features"]["patch_mse"][env_id].item()),
                        action_speed=float(source["features"]["action_speed"][env_id].item()),
                    )
                )
                if len(hard_states) >= parsed.hard_states:
                    return hard_states, step

        if done.any():
            state = reset_done_state(state, done)
        state = detach_state(state)

    return hard_states, parsed.max_collect_steps


def rollout_student(parsed, vec, obs, rewards, terminals, policy, hard_state, device):
    vec.restore(hard_state.env_id, hard_state.snapshot)
    state = clone_detached_state(hard_state.policy_state)
    records = []
    policy.eval()
    for _ in range(parsed.replay_horizon):
        with torch.no_grad():
            logits, _, state = policy.forward_eval(deployable_input(obs, parsed), state)
            actions = target_to_actions(logits.mean, parsed, obs)
        coverage = float(obs[hard_state.env_id, 17].detach().cpu().item())
        step_vec(vec, actions, device)
        records.append(_record(parsed, rewards, terminals, hard_state.env_id, coverage))
        if records[-1]["terminal"]:
            break
        done = terminals > 0.5
        if done.any():
            state = reset_done_state(state, done)
        state = detach_state(state)
    return records


def rollout_oracle(parsed, vec, obs, rewards, terminals, hard_state, device):
    vec.restore(hard_state.env_id, hard_state.snapshot)
    records = []
    for _ in range(parsed.replay_horizon):
        with torch.no_grad():
            actions = target_to_actions(target_labels(obs, parsed), parsed, obs)
        coverage = float(obs[hard_state.env_id, 17].detach().cpu().item())
        step_vec(vec, actions, device)
        records.append(_record(parsed, rewards, terminals, hard_state.env_id, coverage))
        if records[-1]["terminal"]:
            break
    return records


def _record(parsed, rewards, terminals, env_id, coverage):
    terminal = float(terminals[env_id].detach().cpu().item()) > 0.5
    reward = float(rewards[env_id].detach().cpu().item())
    if terminal and reward > parsed.success_reward_threshold:
        outcome = "success"
    elif terminal and reward < parsed.crash_reward_threshold:
        outcome = "crash"
    elif terminal:
        outcome = "timeout"
    else:
        outcome = "open"
    return {
        "terminal": terminal,
        "reward": reward,
        "coverage": coverage,
        "outcome": outcome,
    }


def outcome(records):
    if not records:
        return {"outcome": "open", "steps": 0, "coverage": 0.0, "reward": 0.0}
    last = records[-1]
    return {
        "outcome": last["outcome"],
        "steps": len(records),
        "coverage": last["coverage"],
        "reward": last["reward"],
    }


def summarize_hard_states(hard_states):
    def mean(values):
        return sum(values) / max(len(values), 1)

    print(f"hard_states n={len(hard_states)}")
    if not hard_states:
        return
    print(f"hard.coverage_mean {mean([s.coverage for s in hard_states]):.6g}")
    print(f"hard.min_ray_mean {mean([s.min_ray_m for s in hard_states]):.6g}")
    print(f"hard.angle_mean {mean([s.pred_oracle_angle_deg for s in hard_states]):.6g}")
    print(f"hard.pred_ray_mean {mean([s.pred_ray_m for s in hard_states]):.6g}")
    print(f"hard.oracle_ray_mean {mean([s.oracle_ray_m for s in hard_states]):.6g}")
    print(f"hard.pred_exists_mean {mean([s.pred_exists for s in hard_states]):.6g}")
    print(f"hard.pred_exists_value_mean {mean([s.pred_exists_value for s in hard_states]):.6g}")
    print(f"hard.oracle_exists_mean {mean([s.oracle_exists for s in hard_states]):.6g}")
    print(f"hard.pred_vector_dist_mean {mean([s.pred_vector_dist for s in hard_states]):.6g}")
    print(f"hard.oracle_vector_dist_mean {mean([s.oracle_vector_dist for s in hard_states]):.6g}")
    print(f"hard.sector_mse_mean {mean([s.sector_mse for s in hard_states]):.6g}")
    print(f"hard.sector_top_match_mean {mean([s.sector_top_match for s in hard_states]):.6g}")
    print(f"hard.pred_sector_max_mean {mean([s.pred_sector_max for s in hard_states]):.6g}")
    print(f"hard.oracle_sector_max_mean {mean([s.oracle_sector_max for s in hard_states]):.6g}")
    print(f"hard.vector_angle_mean {mean([s.vector_angle_deg for s in hard_states]):.6g}")
    print(f"hard.vector_dist_error_mean {mean([s.vector_dist_error for s in hard_states]):.6g}")
    print(f"hard.pred_vector_fallback_mean {mean([s.pred_vector_fallback for s in hard_states]):.6g}")
    print(f"hard.pred_recovery_gate_mean {mean([s.pred_recovery_gate for s in hard_states]):.6g}")
    print(
        f"hard.pred_recovery_gate_value_mean "
        f"{mean([s.pred_recovery_gate_value for s in hard_states]):.6g}"
    )
    print(f"hard.patch_mse_mean {mean([s.patch_mse for s in hard_states]):.6g}")
    print(f"hard.action_speed_mean {mean([s.action_speed for s in hard_states]):.6g}")


def summarize_outcomes(name, outcomes):
    counts = {"success": 0, "crash": 0, "timeout": 0, "open": 0}
    for item in outcomes:
        counts[item["outcome"]] += 1
    n = max(len(outcomes), 1)
    mean_steps = sum(item["steps"] for item in outcomes) / n
    mean_coverage = sum(item["coverage"] for item in outcomes) / n
    print(
        f"{name} n={len(outcomes)} success={counts['success']} "
        f"success_rate={counts['success'] / n:.6g} crash={counts['crash']} "
        f"crash_rate={counts['crash'] / n:.6g} timeout={counts['timeout']} "
        f"timeout_rate={counts['timeout'] / n:.6g} open={counts['open']} "
        f"open_rate={counts['open'] / n:.6g} mean_steps={mean_steps:.6g} "
        f"mean_coverage={mean_coverage:.6g}",
        flush=True,
    )


def run(parsed):
    from pufferlib import _C

    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    args = make_vec_config(parsed)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, rewards, terminals = tensor_views(vec, device)
    vec.reset()
    try:
        policy = load_policy(parsed.checkpoint, vec, parsed, device)
        hard_states, collect_steps = collect_timeout_states(
            parsed, vec, obs, rewards, terminals, policy, device
        )
        print(f"checkpoint {parsed.checkpoint}")
        print(f"mode {parsed.mode}")
        print(f"collect_steps {collect_steps}")
        summarize_hard_states(hard_states)

        student_outcomes = []
        oracle_outcomes = []
        for hard_state in hard_states:
            student_outcomes.append(
                outcome(rollout_student(parsed, vec, obs, rewards, terminals, policy, hard_state, device))
            )
            oracle_outcomes.append(
                outcome(rollout_oracle(parsed, vec, obs, rewards, terminals, hard_state, device))
            )

        summarize_outcomes("student_replay", student_outcomes)
        summarize_outcomes("oracle_replay", oracle_outcomes)
    finally:
        vec.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay exact procedural exploration timeout windows under student and oracle control."
    )
    parser.add_argument("checkpoint")
    add_common(parser)
    parser.add_argument("--hard-states", type=int, default=128)
    parser.add_argument("--rewind", type=int, default=256)
    parser.add_argument("--replay-horizon", type=int, default=512)
    parser.add_argument("--max-collect-steps", type=int, default=8192)
    parser.add_argument("--crash-reward-threshold", type=float, default=-5.0)
    parser.add_argument("--success-reward-threshold", type=float, default=5.0)
    parser.add_argument("--sector-match-threshold", type=float, default=0.15)
    parsed = parser.parse_args()
    parsed.obstacle_mode = OBSTACLE_MODES[parsed.mode]
    return parsed


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
