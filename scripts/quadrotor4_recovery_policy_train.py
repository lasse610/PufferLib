#!/usr/bin/env python3
import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quadrotor4_frontier_target_distill import (  # noqa: E402
    OBSTACLE_MODES,
    RecoveryPolicyNet,
    StatefulRecoveryController,
    add_common,
    apply_eval_env_config_from_checkpoint,
    deployable_input,
    load_policy,
    make_vec_config,
    recovery_policy_features,
    reset_done_state,
    run_eval,
    step_vec,
    target_to_actions,
    tensor_views,
)
from quadrotor4_onboard_hardstate_probe import (  # noqa: E402
    _candidate_actions,
    _roll_constant_action,
    _score_candidate,
)
from quadrotor4_privileged_distill import detach_state  # noqa: E402


def _make_vec(parsed, device, total_agents=None):
    from pufferlib import _C

    run = copy.copy(parsed)
    if total_agents is not None:
        run.total_agents = int(total_agents)
    args = make_vec_config(run)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, rewards, terminals = tensor_views(vec, device)
    vec.reset()
    return vec, obs, rewards, terminals


def choose_search_action_scalar(parsed, vec, obs, rewards, terminals, device):
    start_snapshot = vec.snapshot(0)
    start_coverage = float(obs[0, 17].detach().cpu())
    candidates = _candidate_actions(obs[0:1], parsed)

    best = {
        "score": -1.0e18,
        "action": None,
        "best_gain": 0.0,
        "min_clearance": 0.0,
        "outcome": "open",
    }
    try:
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
            if score <= best["score"]:
                continue
            best_gain = max(item["coverage"] for item in records) - start_coverage
            best.update(
                score=score,
                action=action,
                best_gain=best_gain,
                min_clearance=min_clearance,
                outcome=records[-1]["outcome"] if records else "open",
            )
    finally:
        vec.restore(0, start_snapshot)

    return best["action"], best


def choose_search_action_batched(parsed, vec, obs, rewards, terminals, device):
    start_snapshot = vec.snapshot(0)
    start_coverage = float(obs[0, 17].detach().cpu())
    candidates = _candidate_actions(obs[0:1], parsed).detach()
    n = min(candidates.shape[0], vec.total_agents)
    candidates = candidates[:n].to(device=device)

    for env_id in range(n):
        vec.restore(env_id, start_snapshot)

    actions = torch.zeros(
        (vec.total_agents, 4),
        dtype=torch.float32,
        device=device,
    )
    active = torch.ones((n,), dtype=torch.bool, device=device)
    final_coverage = torch.full(
        (n,), start_coverage, dtype=torch.float32, device=device
    )
    best_coverage = final_coverage.clone()
    min_clearance = torch.full(
        (n,),
        float(getattr(parsed, "ray_length", 4.0)),
        dtype=torch.float32,
        device=device,
    )
    steps = torch.zeros((n,), dtype=torch.float32, device=device)
    outcome = torch.zeros((n,), dtype=torch.int64, device=device)

    for _ in range(parsed.mpc_rollout_steps):
        actions.zero_()
        actions[:n] = candidates
        actions[:n] = torch.where(active[:, None], actions[:n], torch.zeros_like(actions[:n]))
        step_vec(vec, actions, device)

        was_active = active.clone()
        steps += was_active.float()
        terminal = (terminals[:n] > 0.5) & was_active
        reward = rewards[:n].float()
        crash = terminal & (reward < parsed.crash_reward_threshold)
        success = terminal & (reward > parsed.success_reward_threshold)
        timeout = terminal & ~(crash | success)
        outcome = torch.where(crash, torch.full_like(outcome, 2), outcome)
        outcome = torch.where(success, torch.full_like(outcome, 1), outcome)
        outcome = torch.where(timeout, torch.full_like(outcome, 3), outcome)

        still_active = was_active & ~terminal
        coverage = obs[:n, 17].float()
        final_coverage = torch.where(still_active, coverage, final_coverage)
        best_coverage = torch.where(
            still_active,
            torch.maximum(best_coverage, coverage),
            best_coverage,
        )
        if parsed.num_rays > 0:
            rays = obs[:n, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
            ray_min = rays.min(dim=1).values
            min_clearance = torch.where(
                still_active,
                torch.minimum(min_clearance, ray_min),
                min_clearance,
            )
        active = still_active
        if not bool(active.any().detach().cpu()):
            break

    coverage_gain = final_coverage - start_coverage
    best_gain = best_coverage - start_coverage
    score = parsed.mpc_final_coverage_weight * coverage_gain
    score += parsed.mpc_best_coverage_weight * best_gain
    score += parsed.mpc_clearance_weight * min_clearance
    score -= parsed.mpc_step_penalty * steps
    score = torch.where(
        outcome == 1,
        score + parsed.mpc_success_bonus,
        score,
    )
    score = torch.where(
        outcome == 2,
        score - parsed.mpc_crash_penalty,
        score,
    )
    score = torch.where(
        outcome == 3,
        score - parsed.mpc_timeout_penalty,
        score,
    )
    best_idx = int(score.argmax().detach().cpu())
    vec.restore(0, start_snapshot)
    best_outcome = ["open", "success", "crash", "timeout"][
        int(outcome[best_idx].detach().cpu())
    ]
    best = {
        "score": float(score[best_idx].detach().cpu()),
        "action": candidates[best_idx : best_idx + 1].detach().clone(),
        "best_gain": float(best_gain[best_idx].detach().cpu()),
        "min_clearance": float(min_clearance[best_idx].detach().cpu()),
        "outcome": best_outcome,
    }
    return best["action"], best


def choose_search_action(parsed, vec, obs, rewards, terminals, device):
    if getattr(parsed, "search_vectorized", True) and vec.total_agents > 1:
        return choose_search_action_batched(parsed, vec, obs, rewards, terminals, device)
    return choose_search_action_scalar(parsed, vec, obs, rewards, terminals, device)


def choose_search_label(parsed, vec, obs, rewards, terminals, device):
    action, best = choose_search_action(parsed, vec, obs, rewards, terminals, device)

    if best["action"] is None:
        return None, best
    if best["outcome"] == "crash":
        return None, best
    if best["best_gain"] < parsed.search_min_best_gain:
        return None, best
    if best["min_clearance"] < parsed.search_min_clearance:
        return None, best
    return best["action"], best


def choose_trajectory_label(parsed, vec, obs, rewards, terminals, device):
    start_snapshot = vec.snapshot(0)
    start_coverage = float(obs[0, 17].detach().cpu())
    first_action = None
    best_gain = 0.0
    min_clearance = float("inf")
    outcome = "open"

    try:
        for _ in range(parsed.trajectory_steps):
            action, search_info = choose_search_action(
                parsed,
                vec,
                obs,
                rewards,
                terminals,
                device,
            )
            if action is None:
                break
            if first_action is None:
                first_action = action.detach().clone()
            step_vec(vec, action, device)
            if parsed.num_rays > 0:
                rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
                min_clearance = min(
                    min_clearance,
                    float(rays.min(dim=1).values[0].detach().cpu()),
                )
            coverage = float(obs[0, 17].detach().cpu())
            best_gain = max(best_gain, coverage - start_coverage)
            terminal = float(terminals[0].detach().cpu()) > 0.5
            reward = float(rewards[0].detach().cpu())
            if terminal:
                if reward > parsed.success_reward_threshold:
                    outcome = "success"
                elif reward < parsed.crash_reward_threshold:
                    outcome = "crash"
                else:
                    outcome = "timeout"
                break
            if best_gain >= parsed.trajectory_stop_gain:
                break
    finally:
        vec.restore(0, start_snapshot)

    info = {
        "best_gain": best_gain,
        "min_clearance": min_clearance if min_clearance != float("inf") else 0.0,
        "outcome": outcome,
    }
    if first_action is None:
        return None, info
    if outcome == "crash":
        return None, info
    if best_gain < parsed.trajectory_min_best_gain:
        return None, info
    if info["min_clearance"] < parsed.trajectory_min_clearance:
        return None, info
    return first_action, info


def collect_recovery_samples(parsed, policy, vec, obs, terminals, device):
    policy.eval()
    teacher = StatefulRecoveryController(parsed)
    search_vec = None
    search_obs = None
    search_rewards = None
    search_terminals = None
    search_device = torch.device("cpu")
    if parsed.label_source in ("search", "trajectory"):
        search_vec, search_obs, search_rewards, search_terminals = _make_vec(
            parsed,
            search_device,
            total_agents=parsed.search_agents,
        )
    state = policy.initial_state(vec.total_agents, device)
    xs = []
    ys = []
    deltas = []
    min_rays = []
    collect_count = 0
    trigger_count = 0
    filtered_search_count = 0
    try:
        for step in range(1, parsed.collect_steps + 1):
            with torch.no_grad():
                x = deployable_input(obs, parsed)
                logits, _, state = policy.forward_eval(x, state)
                target = logits.mean
                base_actions = target_to_actions(target, parsed, obs)
                teacher_actions = teacher.actions(target, obs)
                features = recovery_policy_features(target, obs, parsed, controller=teacher)

                rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
                min_ray = rays.min(dim=1).values if parsed.num_rays > 0 else torch.zeros(
                    (obs.shape[0],), dtype=torch.float32, device=device
                )
                exists = target[:, 3].float() > parsed.exists_threshold
                recover = exists & (teacher.recovery_timer > 0)
                delta = torch.linalg.vector_norm(
                    teacher_actions[:, 0:2] - base_actions[:, 0:2],
                    dim=1,
                )
                mask = (
                    recover
                    & (delta >= parsed.min_action_delta)
                    & (min_ray <= parsed.recovery_near_wall)
                )

                trigger_count += int(mask.sum().detach().cpu())
                for env_id in mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                    label_action = teacher_actions[env_id : env_id + 1]
                    if parsed.label_source in ("search", "trajectory"):
                        search_vec.restore(0, vec.snapshot(env_id))
                        if parsed.label_source == "trajectory":
                            label_action, search_info = choose_trajectory_label(
                                parsed,
                                search_vec,
                                search_obs,
                                search_rewards,
                                search_terminals,
                            search_device,
                        )
                        else:
                            label_action, search_info = choose_search_label(
                                parsed,
                                search_vec,
                                search_obs,
                                search_rewards,
                                search_terminals,
                                search_device,
                            )
                        if label_action is None:
                            filtered_search_count += 1
                            continue
                    label_action = label_action.to(device=device)
                    label_delta = torch.linalg.vector_norm(
                        label_action[:, 0:2] - base_actions[env_id : env_id + 1, 0:2],
                        dim=1,
                    )[0]
                    if float(label_delta.detach().cpu()) < parsed.min_action_delta:
                        continue
                    xs.append(features[env_id].detach().cpu())
                    ys.append(label_action[0].detach().cpu())
                    deltas.append(float(label_delta.detach().cpu()))
                    min_rays.append(float(min_ray[env_id].detach().cpu()))
                    collect_count += 1
                    if collect_count >= parsed.max_samples:
                        break
                if collect_count >= parsed.max_samples:
                    break

            step_vec(vec, base_actions, device)
            done = terminals > 0.5
            if done.any():
                state = reset_done_state(state, done)
                teacher.reset_done(done)
            state = detach_state(state)

            if step % parsed.log_interval == 0:
                print(
                    f"collect_step={step} samples={collect_count} "
                    f"triggered={trigger_count} filtered={filtered_search_count}",
                    flush=True,
                )
    finally:
        if search_vec is not None:
            search_vec.close()

    if not xs:
        raise RuntimeError("No recovery samples collected; lower --min-action-delta or collect longer")

    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    print(
        f"dataset source={parsed.label_source} samples={x.shape[0]} input_size={x.shape[1]} "
        f"delta_mean={float(np.mean(deltas)):.6g} "
        f"min_ray_mean={float(np.mean(min_rays)):.6g} "
        f"filtered={filtered_search_count}",
        flush=True,
    )
    return x, y


def train_recovery_policy(parsed, x, y, device):
    model = RecoveryPolicyNet(x.shape[1], parsed.recovery_hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
    )
    x = x.to(device)
    y = y.to(device)
    n = x.shape[0]
    last_loss = 0.0

    for epoch in range(1, parsed.epochs + 1):
        order = torch.randperm(n, device=device)
        for start in range(0, n, parsed.batch_size):
            idx = order[start : start + parsed.batch_size]
            pred = model(x[idx])
            loss = F.mse_loss(pred, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), parsed.max_grad_norm)
            optimizer.step()
            last_loss = float(loss.detach().cpu())

        if epoch % parsed.print_interval == 0 or epoch == parsed.epochs:
            with torch.no_grad():
                train_l1 = (model(x) - y).abs().mean()
            print(
                f"epoch={epoch} loss={last_loss:.6g} train_l1={float(train_l1.detach().cpu()):.6g}",
                flush=True,
            )
    return model


def run(parsed):
    from pufferlib import _C

    random.seed(parsed.seed)
    np.random.seed(parsed.seed)
    torch.manual_seed(parsed.seed)
    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    parsed.obstacle_mode = OBSTACLE_MODES[parsed.mode]
    apply_eval_env_config_from_checkpoint(parsed.checkpoint, parsed)

    vec, obs, _, terminals = _make_vec(parsed, device)
    try:
        policy = load_policy(parsed.checkpoint, vec, parsed, device)
        x, y = collect_recovery_samples(parsed, policy, vec, obs, terminals, device)
    finally:
        vec.close()

    model = train_recovery_policy(parsed, x, y, device)
    output = Path(parsed.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": x.shape[1],
            "hidden_size": parsed.recovery_hidden_size,
            "config": {
                "train_mode": "learned_recovery_sidecar",
                "checkpoint": parsed.checkpoint,
                "mode": parsed.mode,
                "seed": parsed.seed,
                "samples": x.shape[0],
                "label_source": parsed.label_source,
                "min_action_delta": parsed.min_action_delta,
                "recovery_stall_steps": parsed.recovery_stall_steps,
                "recovery_duration_steps": parsed.recovery_duration_steps,
                "recovery_near_wall": parsed.recovery_near_wall,
            },
        },
        output,
    )
    print(f"wrote {output}", flush=True)

    if parsed.eval_after:
        eval_parsed = copy.copy(parsed)
        eval_parsed.recovery_policy = str(output)
        return run_eval(eval_parsed, policy=policy)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small deployable local recovery sidecar from doorway stall states."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    add_common(parser)
    parser.add_argument("--collect-steps", type=int, default=20000)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--min-action-delta", type=float, default=0.05)
    parser.add_argument(
        "--label-source",
        choices=("teacher", "search", "trajectory"),
        default="teacher",
    )
    parser.add_argument("--recovery-hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--print-interval", type=int, default=5)
    parser.add_argument("--eval-after", action="store_true")
    parser.add_argument("--search-agents", type=int, default=64)
    parser.add_argument(
        "--search-vectorized",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--search-min-best-gain", type=float, default=0.002)
    parser.add_argument("--search-min-clearance", type=float, default=0.30)
    parser.add_argument("--trajectory-steps", type=int, default=64)
    parser.add_argument("--trajectory-stop-gain", type=float, default=0.01)
    parser.add_argument("--trajectory-min-best-gain", type=float, default=0.005)
    parser.add_argument("--trajectory-min-clearance", type=float, default=0.28)
    parser.set_defaults(
        mode="procedural_indoor_hard_p1",
        total_agents=128,
        num_buffers=1,
        num_threads=8,
        max_steps=2048,
        max_eval_steps=50000,
        room_size=6.0,
        spawn_radius_min=0.0,
        explore_goal_fraction=0.95,
        speed_fraction=0.50,
        explore_onboard_path_lookahead=2,
        safety_slide=True,
        safety_cone_ray_stop=0.30,
        safety_cone_deg=60.0,
        safety_slow_ray=0.80,
        recovery_stall_steps=96,
        recovery_duration_steps=64,
        recovery_near_wall=0.36,
        drain_initial_episodes=True,
    )
    parsed = parser.parse_args()
    if parsed.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if parsed.collect_steps <= 0:
        parser.error("--collect-steps must be positive")
    if not parsed.mpc_speed_fractions:
        parser.error("--mpc-speed-fractions must contain at least one value")
    if parsed.trajectory_steps <= 0:
        parser.error("--trajectory-steps must be positive")
    if parsed.search_agents <= 0:
        parser.error("--search-agents must be positive")
    return parsed


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
