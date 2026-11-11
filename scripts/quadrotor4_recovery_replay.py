#!/usr/bin/env python3
import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quadrotor4_frontier_target_distill import (  # noqa: E402
    OBSTACLE_MODES,
    RecoveryPolicyNet,
    StatefulRecoveryController,
    LearnedRecoveryController,
    _combined_safety_scale,
    _slide_safety_actions,
    add_common,
    apply_eval_env_config_from_checkpoint,
    deployable_input,
    load_policy,
    load_recovery_policy,
    make_vec_config,
    recovery_policy_features,
    reset_done_state,
    run_eval,
    step_vec,
    target_to_actions,
    tensor_views,
)
from quadrotor4_privileged_distill import clone_detached_state, detach_state  # noqa: E402


@dataclass
class RecoveryHardState:
    step: int
    env_id: int
    snapshot: bytes
    policy_state: tuple
    coverage: float
    best_coverage: float
    min_ray_m: float
    since_progress: int
    recovery_timer: int
    recovery_side: float
    target_exists: float
    action_speed: float


def _make_vec(parsed, device, total_agents):
    from pufferlib import _C

    run = copy.copy(parsed)
    run.total_agents = int(total_agents)
    args = make_vec_config(run)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, rewards, terminals = tensor_views(vec, device)
    vec.reset()
    return vec, obs, rewards, terminals


def _state_for_batch(hard_states, device):
    template = hard_states[0].policy_state
    if not isinstance(template, tuple):
        return template

    out = []
    for slot, value in enumerate(template):
        if not torch.is_tensor(value):
            out.append(value)
            continue
        pieces = []
        for hard_state in hard_states:
            source = hard_state.policy_state[slot]
            pieces.append(source[:, hard_state.env_id].to(device))
        out.append(torch.stack(pieces, dim=1).detach())
    return tuple(out)


def _restore_batch(vec, hard_states):
    for env_id, hard_state in enumerate(hard_states):
        vec.restore(env_id, hard_state.snapshot)


def _seed_controller(controller, obs, hard_states):
    controller._ensure(obs)
    device = obs.device
    for env_id, hard_state in enumerate(hard_states):
        controller.best_coverage[env_id] = float(hard_state.best_coverage)
        controller.since_progress[env_id] = int(hard_state.since_progress)
        controller.recovery_timer[env_id] = int(hard_state.recovery_timer)
        controller.recovery_side[env_id] = float(hard_state.recovery_side)
    if len(hard_states) < obs.shape[0]:
        start = len(hard_states)
        controller.best_coverage[start:] = obs[start:, 17].float().to(device)
        controller.since_progress[start:] = 0
        controller.recovery_timer[start:] = 0
        controller.recovery_side[start:] = 1.0


def _min_ray(obs, parsed):
    if parsed.num_rays <= 0:
        return torch.zeros((obs.shape[0],), dtype=torch.float32, device=obs.device)
    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    return rays.min(dim=1).values


def _outcome_from_done(reward, terminal, parsed):
    if not terminal:
        return "open"
    if reward > parsed.success_reward_threshold:
        return "success"
    if reward < parsed.crash_reward_threshold:
        return "crash"
    return "timeout"


def summarize_hard_states(hard_states):
    def mean(values):
        return sum(values) / max(len(values), 1)

    print(f"recovery_hard_states n={len(hard_states)}", flush=True)
    if not hard_states:
        return
    print(f"hard.coverage_mean {mean([s.coverage for s in hard_states]):.6g}", flush=True)
    print(f"hard.min_ray_mean {mean([s.min_ray_m for s in hard_states]):.6g}", flush=True)
    print(
        f"hard.since_progress_mean {mean([float(s.since_progress) for s in hard_states]):.6g}",
        flush=True,
    )
    print(
        f"hard.recovery_timer_mean {mean([float(s.recovery_timer) for s in hard_states]):.6g}",
        flush=True,
    )
    print(f"hard.action_speed_mean {mean([s.action_speed for s in hard_states]):.6g}", flush=True)


def collect_recovery_hard_states(parsed, policy, vec, obs, rewards, terminals, device):
    policy.eval()
    controller = StatefulRecoveryController(parsed)
    state = policy.initial_state(vec.total_agents, device)
    hard_states = []
    last_capture = torch.full((vec.total_agents,), -10**9, dtype=torch.long, device=device)

    for step in range(1, parsed.collect_steps + 1):
        with torch.no_grad():
            logits, _, next_state = policy.forward_eval(deployable_input(obs, parsed), state)
            target = logits.mean
            base_actions = target_to_actions(target, parsed, obs)
            recovery_actions = controller.actions(target, obs)
            min_ray = _min_ray(obs, parsed)
            exists = target[:, 3].float() > parsed.exists_threshold
            recovering = exists & (controller.recovery_timer > 0)
            cooled = (step - last_capture) >= parsed.capture_cooldown
            mask = recovering & cooled & (min_ray <= parsed.recovery_near_wall)

            for env_id in mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                hard_states.append(
                    RecoveryHardState(
                        step=step,
                        env_id=env_id,
                        snapshot=vec.snapshot(env_id),
                        policy_state=clone_detached_state(state),
                        coverage=float(obs[env_id, 17].detach().cpu()),
                        best_coverage=float(controller.best_coverage[env_id].detach().cpu()),
                        min_ray_m=float(min_ray[env_id].detach().cpu()),
                        since_progress=int(controller.since_progress[env_id].detach().cpu()),
                        recovery_timer=int(controller.recovery_timer[env_id].detach().cpu()),
                        recovery_side=float(controller.recovery_side[env_id].detach().cpu()),
                        target_exists=float(target[env_id, 3].detach().cpu()),
                        action_speed=float(
                            torch.linalg.vector_norm(
                                base_actions[env_id : env_id + 1, 0:2], dim=1
                            )[0].detach().cpu()
                        ),
                    )
                )
                last_capture[env_id] = step
                if len(hard_states) >= parsed.hard_states:
                    return hard_states, step

            actions = recovery_actions if parsed.collect_with_stateful_recovery else base_actions

        state = next_state
        step_vec(vec, actions, device)
        done = terminals > 0.5
        if done.any():
            state = reset_done_state(state, done)
            controller.reset_done(done)
        state = detach_state(state)

        if step % parsed.log_interval == 0:
            print(f"collect_step={step} hard_states={len(hard_states)}", flush=True)

    return hard_states, parsed.collect_steps


def _wrap_recovery_actions(parsed, pred_actions, default_actions, obs):
    pred_actions = pred_actions.clone()
    if parsed.num_rays > 0 and parsed.safety_cone_ray_stop > 0.0:
        if getattr(parsed, "safety_slide", False):
            pred_actions = _slide_safety_actions(pred_actions, obs, parsed)
        else:
            speed = torch.linalg.vector_norm(pred_actions[:, 0:2], dim=1, keepdim=True)
            unit = pred_actions[:, 0:2] / speed.clamp_min(1e-6)
            scale = _combined_safety_scale(obs, unit, parsed)
            pred_actions[:, 0:2] = pred_actions[:, 0:2] * scale[:, None]
    pred_actions[:, 2:3] = default_actions[:, 2:3]
    xy_speed = torch.linalg.vector_norm(pred_actions[:, 0:2], dim=1, keepdim=True)
    unit_y = torch.where(
        xy_speed > 1e-6,
        pred_actions[:, 1:2] / xy_speed.clamp_min(1e-6),
        default_actions[:, 3:4] / max(float(parsed.yaw_fraction), 1e-6),
    )
    pred_actions[:, 3:4] = (unit_y * parsed.yaw_fraction).clamp(-1.0, 1.0)
    return pred_actions.clamp(-1.0, 1.0)


def _make_controller(name, parsed, device):
    if name == "base":
        return None
    if name == "stateful":
        return StatefulRecoveryController(parsed)
    if name.startswith("learned:"):
        path = name.split(":", 1)[1]
        return LearnedRecoveryController(parsed, load_recovery_policy(path, device))
    raise ValueError(f"unknown controller {name!r}")


def _rollout_chunk(parsed, policy, hard_states, device, controller_name, model=None):
    vec, obs, rewards, terminals = _make_vec(parsed, device, len(hard_states))
    try:
        _restore_batch(vec, hard_states)
        state = _state_for_batch(hard_states, device)
        controller = _make_controller(controller_name, parsed, device) if model is None else StatefulRecoveryController(parsed)
        if controller is not None:
            _seed_controller(controller, obs, hard_states)

        active = torch.ones((len(hard_states),), dtype=torch.bool, device=device)
        start_coverage = torch.tensor([s.coverage for s in hard_states], dtype=torch.float32, device=device)
        final_coverage = obs[:, 17].float().clone()
        best_coverage = final_coverage.clone()
        min_clearance = _min_ray(obs, parsed)
        steps = torch.zeros((len(hard_states),), dtype=torch.float32, device=device)
        outcomes = ["open" for _ in hard_states]

        for _ in range(parsed.replay_horizon):
            with torch.no_grad():
                logits, _, state = policy.forward_eval(deployable_input(obs, parsed), state)
                target = logits.mean
                default_actions = target_to_actions(target, parsed, obs)
                if model is not None:
                    _ = controller.actions(target, obs)
                    features = recovery_policy_features(
                        target,
                        obs,
                        parsed,
                        controller=controller,
                        input_size=getattr(model, "input_size", None),
                    )
                    pred_actions = model(features)
                    actions = _wrap_recovery_actions(parsed, pred_actions, default_actions, obs)
                elif controller is not None:
                    actions = controller.actions(target, obs)
                else:
                    actions = default_actions
                actions = torch.where(active[:, None], actions, torch.zeros_like(actions))

            step_vec(vec, actions, device)
            was_active = active.clone()
            steps += was_active.float()
            terminal = (terminals > 0.5) & was_active
            reward = rewards.float()
            for env_id in terminal.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                outcomes[env_id] = _outcome_from_done(
                    float(reward[env_id].detach().cpu()),
                    True,
                    parsed,
                )
            active = was_active & ~terminal
            coverage = obs[:, 17].float()
            final_coverage = torch.where(was_active, coverage, final_coverage)
            best_coverage = torch.where(was_active, torch.maximum(best_coverage, coverage), best_coverage)
            min_clearance = torch.where(was_active, torch.minimum(min_clearance, _min_ray(obs, parsed)), min_clearance)
            if terminal.any():
                state = reset_done_state(state, terminal)
                if controller is not None:
                    controller.reset_done(terminal)
            state = detach_state(state)
            if not bool(active.any().detach().cpu()):
                break

        return [
            {
                "outcome": outcomes[idx],
                "steps": float(steps[idx].detach().cpu()),
                "coverage": float(final_coverage[idx].detach().cpu()),
                "coverage_gain": float((final_coverage[idx] - start_coverage[idx]).detach().cpu()),
                "best_gain": float((best_coverage[idx] - start_coverage[idx]).detach().cpu()),
                "min_clearance": float(min_clearance[idx].detach().cpu()),
            }
            for idx in range(len(hard_states))
        ]
    finally:
        vec.close()


def summarize_outcomes(name, outcomes):
    counts = {"success": 0, "crash": 0, "timeout": 0, "open": 0}
    for item in outcomes:
        counts[item["outcome"]] += 1
    n = max(len(outcomes), 1)
    mean_steps = sum(item["steps"] for item in outcomes) / n
    mean_cov = sum(item["coverage"] for item in outcomes) / n
    mean_gain = sum(item["coverage_gain"] for item in outcomes) / n
    mean_best = sum(item["best_gain"] for item in outcomes) / n
    mean_clearance = sum(item["min_clearance"] for item in outcomes) / n
    print(
        f"{name} n={len(outcomes)} success={counts['success']} "
        f"success_rate={counts['success'] / n:.6g} crash={counts['crash']} "
        f"crash_rate={counts['crash'] / n:.6g} timeout={counts['timeout']} "
        f"timeout_rate={counts['timeout'] / n:.6g} open={counts['open']} "
        f"open_rate={counts['open'] / n:.6g} mean_steps={mean_steps:.6g} "
        f"mean_coverage={mean_cov:.6g} mean_gain={mean_gain:.6g} "
        f"mean_best_gain={mean_best:.6g} mean_min_clearance={mean_clearance:.6g}",
        flush=True,
    )


def evaluate_replay(parsed, policy, hard_states, device, controller_name, model=None, label_suffix=""):
    outcomes = []
    for start in range(0, len(hard_states), parsed.replay_agents):
        chunk = hard_states[start : start + parsed.replay_agents]
        outcomes.extend(_rollout_chunk(parsed, policy, chunk, device, controller_name, model=model))
    label = controller_name if model is None else "trained_recovery"
    label = f"{label}{label_suffix}"
    summarize_outcomes(f"replay_{label}", outcomes)
    return outcomes


def _discounted_returns(rewards, masks, gamma):
    out = torch.zeros_like(rewards)
    running = torch.zeros((rewards.shape[1],), dtype=rewards.dtype, device=rewards.device)
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * running * masks[t]
        out[t] = running
    return out


def _train_rollout(parsed, policy, model, hard_states, device):
    vec, obs, rewards, terminals = _make_vec(parsed, device, len(hard_states))
    try:
        _restore_batch(vec, hard_states)
        state = _state_for_batch(hard_states, device)
        controller = StatefulRecoveryController(parsed)
        _seed_controller(controller, obs, hard_states)
        active = torch.ones((len(hard_states),), dtype=torch.bool, device=device)

        logps = []
        rewards_out = []
        masks = []
        for _ in range(parsed.train_horizon):
            with torch.no_grad():
                coverage_before = obs[:, 17].float().clone()
                min_ray_before = _min_ray(obs, parsed)
                logits, _, state = policy.forward_eval(deployable_input(obs, parsed), state)
                target = logits.mean
                default_actions = target_to_actions(target, parsed, obs)
                _ = controller.actions(target, obs)
                recover = (target[:, 3].float() > parsed.exists_threshold) & (controller.recovery_timer > 0)

            features = recovery_policy_features(
                target.detach(),
                obs,
                parsed,
                controller=controller,
                input_size=getattr(model, "input_size", None),
            )
            mean_actions = model(features)
            dist = torch.distributions.Normal(mean_actions[:, 0:2], parsed.pg_action_std)
            sampled_xy = dist.sample().clamp(-1.0, 1.0)
            logp = dist.log_prob(sampled_xy).sum(dim=1)
            pred_actions = mean_actions.clone()
            pred_actions[:, 0:2] = sampled_xy
            sampled_actions = _wrap_recovery_actions(parsed, pred_actions, default_actions, obs)
            actions = torch.where(recover[:, None], sampled_actions, default_actions)
            actions = torch.where(active[:, None], actions, torch.zeros_like(actions))

            step_vec(vec, actions.detach(), device)
            terminal = (terminals > 0.5) & active
            coverage_after = obs[:, 17].float()
            min_ray_after = _min_ray(obs, parsed)
            gain = (coverage_after - coverage_before).clamp(-0.02, 0.02)
            reward = parsed.pg_coverage_weight * gain
            reward = reward + parsed.pg_clearance_weight * (
                min_ray_after - min_ray_before
            ).clamp(-0.20, 0.20)
            danger = (parsed.safety_cone_ray_stop - min_ray_after).clamp_min(0.0)
            reward = reward - parsed.pg_danger_weight * danger
            reward = reward - parsed.pg_step_penalty
            crash = terminal & (rewards.float() < parsed.crash_reward_threshold)
            success = terminal & (rewards.float() > parsed.success_reward_threshold)
            timeout = terminal & ~(crash | success)
            reward = torch.where(success, reward + parsed.pg_success_bonus, reward)
            reward = torch.where(crash, reward - parsed.pg_crash_penalty, reward)
            reward = torch.where(timeout, reward - parsed.pg_timeout_penalty, reward)

            train_mask = active & recover
            logps.append(logp)
            rewards_out.append(torch.where(active, reward, torch.zeros_like(reward)))
            masks.append(train_mask.float())

            active = active & ~terminal
            if terminal.any():
                state = reset_done_state(state, terminal)
                controller.reset_done(terminal)
            state = detach_state(state)
            if not bool(active.any().detach().cpu()):
                break

        logps_t = torch.stack(logps, dim=0)
        rewards_t = torch.stack(rewards_out, dim=0)
        masks_t = torch.stack(masks, dim=0)
        returns = _discounted_returns(rewards_t, masks_t, parsed.pg_gamma)
        valid = masks_t > 0.5
        if valid.any():
            adv = returns.detach()
            adv_valid = adv[valid]
            adv = (adv - adv_valid.mean()) / adv_valid.std(unbiased=False).clamp_min(1e-6)
            loss = -((logps_t * adv * masks_t).sum() / masks_t.sum().clamp_min(1.0))
        else:
            loss = (model(features).sum() * 0.0)
        return loss, {
            "reward": float((rewards_t * masks_t).sum().detach().cpu() / masks_t.sum().clamp_min(1.0).detach().cpu()),
            "valid": float(masks_t.mean().detach().cpu()),
        }
    finally:
        vec.close()


def train_recovery_pg(parsed, policy, hard_states, device):
    probe_vec, probe_obs, _, _ = _make_vec(parsed, device, 1)
    try:
        _restore_batch(probe_vec, hard_states[:1])
        controller = StatefulRecoveryController(parsed)
        _seed_controller(controller, probe_obs, hard_states[:1])
        with torch.no_grad():
            state = _state_for_batch(hard_states[:1], device)
            logits, _, _ = policy.forward_eval(deployable_input(probe_obs, parsed), state)
            input_size = recovery_policy_features(logits.mean, probe_obs, parsed, controller=controller).shape[1]
    finally:
        probe_vec.close()

    model = RecoveryPolicyNet(input_size, parsed.recovery_hidden_size).to(device)
    model.input_size = input_size
    if parsed.bc_warmup_epochs > 0:
        warmup_recovery_bc(parsed, policy, model, hard_states, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parsed.pg_learning_rate,
        weight_decay=parsed.pg_weight_decay,
    )
    rng = random.Random(parsed.seed)
    last_loss = 0.0
    for update in range(1, parsed.pg_updates + 1):
        batch = [hard_states[rng.randrange(len(hard_states))] for _ in range(parsed.pg_batch_size)]
        loss, info = _train_rollout(parsed, policy, model, batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parsed.max_grad_norm)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if update % parsed.print_interval == 0 or update == parsed.pg_updates:
            print(
                f"pg_update={update} loss={last_loss:.6g} "
                f"reward={info['reward']:.6g} valid_frac={info['valid']:.6g}",
                flush=True,
            )
    return model


def warmup_recovery_bc(parsed, policy, model, hard_states, device):
    xs = []
    ys = []
    for start in range(0, len(hard_states), parsed.replay_agents):
        chunk = hard_states[start : start + parsed.replay_agents]
        vec, obs, _, _ = _make_vec(parsed, device, len(chunk))
        try:
            _restore_batch(vec, chunk)
            state = _state_for_batch(chunk, device)
            controller = StatefulRecoveryController(parsed)
            _seed_controller(controller, obs, chunk)
            with torch.no_grad():
                logits, _, _ = policy.forward_eval(deployable_input(obs, parsed), state)
                target = logits.mean
                base_actions = target_to_actions(target, parsed, obs)
                stateful_actions = controller.actions(target, obs)
                features = recovery_policy_features(
                    target,
                    obs,
                    parsed,
                    controller=controller,
                    input_size=getattr(model, "input_size", None),
                )
                labels = base_actions if parsed.bc_warmup_target == "base" else stateful_actions
            xs.append(features.detach().cpu())
            ys.append(labels.detach().cpu())
        finally:
            vec.close()

    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parsed.bc_learning_rate,
        weight_decay=parsed.pg_weight_decay,
    )
    last_l1 = 0.0
    for epoch in range(1, parsed.bc_warmup_epochs + 1):
        order = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], parsed.pg_batch_size):
            idx = order[start : start + parsed.pg_batch_size]
            pred = model(x[idx])
            loss = torch.nn.functional.mse_loss(pred, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), parsed.max_grad_norm)
            optimizer.step()
        if epoch % parsed.print_interval == 0 or epoch == parsed.bc_warmup_epochs:
            with torch.no_grad():
                last_l1 = float((model(x) - y).abs().mean().detach().cpu())
            print(
                f"bc_warmup_epoch={epoch} target={parsed.bc_warmup_target} "
                f"train_l1={last_l1:.6g}",
                flush=True,
            )


def run(parsed):
    from pufferlib import _C

    random.seed(parsed.seed)
    np.random.seed(parsed.seed)
    torch.manual_seed(parsed.seed)
    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    parsed.obstacle_mode = OBSTACLE_MODES[parsed.mode]
    if not parsed.ignore_checkpoint_env_config:
        apply_eval_env_config_from_checkpoint(parsed.checkpoint, parsed)

    collect_vec, obs, rewards, terminals = _make_vec(parsed, device, parsed.total_agents)
    try:
        policy = load_policy(parsed.checkpoint, collect_vec, parsed, device)
        hard_states, collect_steps = collect_recovery_hard_states(
            parsed, policy, collect_vec, obs, rewards, terminals, device
        )
    finally:
        collect_vec.close()

    print(f"checkpoint {parsed.checkpoint}", flush=True)
    print(f"mode {parsed.mode}", flush=True)
    print(f"collect_steps {collect_steps}", flush=True)
    summarize_hard_states(hard_states)
    if not hard_states:
        raise RuntimeError("No recovery hard states collected; lower trigger thresholds or collect longer")

    controller_names = [item.strip() for item in parsed.controllers.split(",") if item.strip()]
    for path in parsed.learned_recovery_policy:
        controller_names.append(f"learned:{path}")
    original_horizon = parsed.replay_horizon
    for name in controller_names:
        evaluate_replay(parsed, policy, hard_states, device, name)

    extra_horizons = [
        int(item)
        for item in str(parsed.extra_replay_horizons).split(",")
        if item.strip()
    ]
    for horizon in extra_horizons:
        parsed.replay_horizon = horizon
        print(f"extra_replay_horizon {horizon}", flush=True)
        for name in controller_names:
            evaluate_replay(parsed, policy, hard_states, device, name, label_suffix=f"_h{horizon}")
    parsed.replay_horizon = original_horizon

    trained = None
    if parsed.pg_updates > 0:
        trained = train_recovery_pg(parsed, policy, hard_states, device)
        evaluate_replay(parsed, policy, hard_states, device, "pg", model=trained)
        if parsed.output:
            output = Path(parsed.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": trained.state_dict(),
                    "input_size": trained.input_size,
                    "hidden_size": parsed.recovery_hidden_size,
                    "config": {
                        "train_mode": "resettable_recovery_pg",
                        "checkpoint": parsed.checkpoint,
                        "mode": parsed.mode,
                        "seed": parsed.seed,
                        "hard_states": len(hard_states),
                        "pg_updates": parsed.pg_updates,
                        "pg_batch_size": parsed.pg_batch_size,
                        "pg_horizon": parsed.train_horizon,
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
        description="Collect and replay exact doorway recovery states for closed-loop recovery diagnostics."
    )
    parser.add_argument("checkpoint")
    add_common(parser)
    parser.add_argument("--output")
    parser.add_argument("--collect-steps", type=int, default=12000)
    parser.add_argument("--hard-states", type=int, default=64)
    parser.add_argument("--capture-cooldown", type=int, default=48)
    parser.add_argument("--collect-with-stateful-recovery", action="store_true")
    parser.add_argument("--replay-agents", type=int, default=64)
    parser.add_argument("--replay-horizon", type=int, default=256)
    parser.add_argument("--extra-replay-horizons", default="")
    parser.add_argument("--controllers", default="base,stateful")
    parser.add_argument("--learned-recovery-policy", action="append", default=[])
    parser.add_argument("--crash-reward-threshold", type=float, default=-5.0)
    parser.add_argument("--success-reward-threshold", type=float, default=5.0)
    parser.add_argument("--pg-updates", type=int, default=0)
    parser.add_argument("--pg-batch-size", type=int, default=32)
    parser.add_argument("--train-horizon", type=int, default=96)
    parser.add_argument("--pg-action-std", type=float, default=0.20)
    parser.add_argument("--pg-learning-rate", type=float, default=0.0003)
    parser.add_argument("--pg-weight-decay", type=float, default=0.0)
    parser.add_argument("--pg-gamma", type=float, default=0.98)
    parser.add_argument("--pg-coverage-weight", type=float, default=300.0)
    parser.add_argument("--pg-clearance-weight", type=float, default=0.40)
    parser.add_argument("--pg-danger-weight", type=float, default=4.0)
    parser.add_argument("--pg-step-penalty", type=float, default=0.003)
    parser.add_argument("--pg-success-bonus", type=float, default=2.0)
    parser.add_argument("--pg-crash-penalty", type=float, default=4.0)
    parser.add_argument("--pg-timeout-penalty", type=float, default=0.5)
    parser.add_argument("--recovery-hidden-size", type=int, default=128)
    parser.add_argument("--bc-warmup-epochs", type=int, default=0)
    parser.add_argument("--bc-warmup-target", choices=("base", "stateful"), default="base")
    parser.add_argument("--bc-learning-rate", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--eval-after", action="store_true")
    parsed = parser.parse_args()
    parsed.obstacle_mode = OBSTACLE_MODES[parsed.mode]
    return parsed


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
