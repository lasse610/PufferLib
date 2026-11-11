#!/usr/bin/env python3
import argparse
import copy
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quadrotor4_frontier_target_distill import (  # noqa: E402
    OBSTACLE_MODES,
    _combined_safety_scale,
    add_common,
    deployable_input,
    false_negative_exists_loss,
    frontier_sector_loss,
    frontier_vector_loss,
    load_policy,
    make_vec_config,
    predict_sequence,
    reset_done_state,
    run_eval,
    step_vec,
    target_labels,
    target_loss,
    target_to_actions,
    tensor_views,
)
from quadrotor4_privileged_distill import clone_detached_state, detach_state  # noqa: E402
from quadrotor4_target_hardstate_probe import (  # noqa: E402
    _features,
    collect_timeout_states,
    outcome,
    rollout_oracle,
    summarize_hard_states,
    summarize_outcomes,
)


@dataclass
class BurninHardState:
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
    burnin_snapshot: bytes
    burnin_actions: torch.Tensor


def _done_groups(rewards, terminals, parsed):
    done = terminals > 0.5
    crash = done & (rewards < parsed.crash_reward_threshold)
    success = done & (rewards > parsed.success_reward_threshold)
    timeout = done & ~(crash | success)
    return done, crash, success, timeout


def _state_for_hard_batch(hard_states, device):
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


def _make_vec(parsed, device, total_agents, mode=None):
    from pufferlib import _C

    run_parsed = copy.copy(parsed)
    run_parsed.total_agents = total_agents
    if mode is not None:
        run_parsed.mode = mode
        run_parsed.obstacle_mode = OBSTACLE_MODES[mode]
    args = make_vec_config(run_parsed)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, rewards, terminals = tensor_views(vec, device)
    vec.reset()
    return vec, obs, rewards, terminals


def _collect_recoverable_hard_states(parsed, policy, device):
    vec, obs, rewards, terminals = _make_vec(parsed, device, parsed.total_agents)
    try:
        hard_states, collect_steps = collect_timeout_states(
            parsed, vec, obs, rewards, terminals, policy, device
        )
        print(f"collected_timeout_states n={len(hard_states)} steps={collect_steps}", flush=True)
        summarize_hard_states(hard_states)

        recoverable = []
        outcomes = []
        for hard_state in hard_states:
            item = outcome(rollout_oracle(parsed, vec, obs, rewards, terminals, hard_state, device))
            outcomes.append(item)
            if item["outcome"] == "success":
                recoverable.append(hard_state)
        summarize_outcomes("oracle_filter", outcomes)
    finally:
        vec.close()

    if len(recoverable) < parsed.min_recoverable_states:
        raise RuntimeError(
            f"only {len(recoverable)} oracle-recoverable hard states; "
            f"need at least {parsed.min_recoverable_states}"
        )
    print(f"recoverable_hard_states n={len(recoverable)}", flush=True)
    return recoverable


def _collect_burnin_timeout_states(parsed, vec, obs, rewards, terminals, policy, device):
    policy.eval()
    state = policy.initial_state(vec.total_agents, device)
    ring = deque(maxlen=parsed.rewind + parsed.burnin_steps + 1)
    hard_states = []

    for step in range(1, parsed.max_collect_steps + 1):
        with torch.no_grad():
            x = deployable_input(obs, parsed)
            logits, _, next_state = policy.forward_eval(x, state)
            pred_target = logits.mean
            oracle_target = target_labels(obs, parsed)
            actions = target_to_actions(pred_target, parsed, obs)
            features = _features(obs, pred_target, oracle_target, actions, parsed)

        ring.append({
            "step": step,
            "snapshots": [vec.snapshot(env_id) for env_id in range(vec.total_agents)],
            "state": clone_detached_state(state),
            "actions": actions.detach().cpu().clone(),
            "features": {key: value.detach().cpu().clone() for key, value in features.items()},
        })

        state = next_state
        step_vec(vec, actions, device)
        done, _, _, timeout = _done_groups(rewards, terminals, parsed)
        if timeout.any() and len(ring) > parsed.rewind + parsed.burnin_steps:
            entries = list(ring)
            hard_entry = entries[-(parsed.rewind + 1)]
            burnin_entry = entries[-(parsed.rewind + parsed.burnin_steps + 1)]
            action_entries = entries[
                -(parsed.rewind + parsed.burnin_steps + 1):-(parsed.rewind + 1)
            ]
            for env_id in timeout.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                hard_states.append(
                    BurninHardState(
                        step=hard_entry["step"],
                        env_id=env_id,
                        snapshot=hard_entry["snapshots"][env_id],
                        policy_state=clone_detached_state(hard_entry["state"]),
                        coverage=float(hard_entry["features"]["coverage"][env_id].item()),
                        min_ray_m=float(hard_entry["features"]["min_ray_m"][env_id].item()),
                        pred_oracle_angle_deg=float(
                            hard_entry["features"]["pred_oracle_angle_deg"][env_id].item()
                        ),
                        pred_ray_m=float(hard_entry["features"]["pred_ray_m"][env_id].item()),
                        oracle_ray_m=float(hard_entry["features"]["oracle_ray_m"][env_id].item()),
                        pred_exists=float(hard_entry["features"]["pred_exists"][env_id].item()),
                        pred_exists_value=float(
                            hard_entry["features"]["pred_exists_value"][env_id].item()
                        ),
                        oracle_exists=float(hard_entry["features"]["oracle_exists"][env_id].item()),
                        pred_vector_dist=float(
                            hard_entry["features"]["pred_vector_dist"][env_id].item()
                        ),
                        oracle_vector_dist=float(
                            hard_entry["features"]["oracle_vector_dist"][env_id].item()
                        ),
                        sector_mse=float(hard_entry["features"]["sector_mse"][env_id].item()),
                        sector_top_match=float(
                            hard_entry["features"]["sector_top_match"][env_id].item()
                        ),
                        pred_sector_max=float(
                            hard_entry["features"]["pred_sector_max"][env_id].item()
                        ),
                        oracle_sector_max=float(
                            hard_entry["features"]["oracle_sector_max"][env_id].item()
                        ),
                        vector_angle_deg=float(
                            hard_entry["features"]["vector_angle_deg"][env_id].item()
                        ),
                        vector_dist_error=float(
                            hard_entry["features"]["vector_dist_error"][env_id].item()
                        ),
                        pred_vector_fallback=float(
                            hard_entry["features"]["pred_vector_fallback"][env_id].item()
                        ),
                        pred_recovery_gate=float(
                            hard_entry["features"]["pred_recovery_gate"][env_id].item()
                        ),
                        pred_recovery_gate_value=float(
                            hard_entry["features"]["pred_recovery_gate_value"][env_id].item()
                        ),
                        patch_mse=float(hard_entry["features"]["patch_mse"][env_id].item()),
                        action_speed=float(hard_entry["features"]["action_speed"][env_id].item()),
                        burnin_snapshot=burnin_entry["snapshots"][env_id],
                        burnin_actions=torch.stack(
                            [entry["actions"][env_id] for entry in action_entries],
                            dim=0,
                        ),
                    )
                )
                if len(hard_states) >= parsed.hard_states:
                    return hard_states, step

        if done.any():
            state = reset_done_state(state, done)
        state = detach_state(state)

    return hard_states, parsed.max_collect_steps


def _collect_training_hard_states(parsed, policy, device):
    vec, obs, rewards, terminals = _make_vec(parsed, device, parsed.total_agents)
    try:
        if parsed.burnin_steps > 0:
            hard_states, collect_steps = _collect_burnin_timeout_states(
                parsed, vec, obs, rewards, terminals, policy, device
            )
        else:
            hard_states, collect_steps = collect_timeout_states(
                parsed, vec, obs, rewards, terminals, policy, device
            )
        print(f"collected_timeout_states n={len(hard_states)} steps={collect_steps}", flush=True)
        summarize_hard_states(hard_states)

        recoverable = []
        outcomes = []
        for hard_state in hard_states:
            item = outcome(rollout_oracle(parsed, vec, obs, rewards, terminals, hard_state, device))
            outcomes.append(item)
            if item["outcome"] == "success":
                recoverable.append(hard_state)
        summarize_outcomes("oracle_filter", outcomes)
    finally:
        vec.close()

    if len(recoverable) < parsed.min_recoverable_states:
        raise RuntimeError(
            f"only {len(recoverable)} oracle-recoverable hard states; "
            f"need at least {parsed.min_recoverable_states}"
        )
    print(f"recoverable_hard_states n={len(recoverable)}", flush=True)
    return recoverable


def _restore_batch(vec, hard_states):
    for env_id, hard_state in enumerate(hard_states):
        vec.restore(env_id, hard_state.snapshot)


def _burnin_batch(parsed, policy, vec, obs, terminals, hard_states, device):
    for env_id, hard_state in enumerate(hard_states):
        vec.restore(env_id, hard_state.burnin_snapshot)

    state = policy.initial_state(vec.total_agents, device)
    for burnin_idx in range(parsed.burnin_steps):
        with torch.no_grad():
            _, _, state = policy.forward_eval(deployable_input(obs, parsed), state)
            actions = torch.stack(
                [hard_state.burnin_actions[burnin_idx] for hard_state in hard_states],
                dim=0,
            ).to(device=device, dtype=torch.float32)
        step_vec(vec, actions, device)
        done = terminals > 0.5
        if done.any():
            state = reset_done_state(state, done)
        state = detach_state(state)
    return clone_detached_state(state)


def _soft_target_to_actions(target, parsed, obs=None):
    target = target.float()
    direction = target[:, 0:2].clamp(-1.0, 1.0)
    unit = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
    exists = target[:, 3:4].clamp(0.0, 1.0)
    speed = parsed.speed_fraction * exists

    has_pred_speed = target.shape[1] > 4 and getattr(parsed, "use_pred_speed_scale", False)
    if has_pred_speed:
        speed = speed * target[:, 4:5].clamp(0.0, 1.0)

    if (
        obs is not None
        and parsed.num_rays > 0
        and (not has_pred_speed or getattr(parsed, "external_safety_with_pred_speed", False))
    ):
        speed = speed * _combined_safety_scale(obs, unit, parsed)[:, None]

    actions = torch.zeros((target.shape[0], 4), dtype=torch.float32, device=target.device)
    actions[:, 0:2] = unit * speed
    actions[:, 3:4] = (unit[:, 1:2] * parsed.yaw_fraction).clamp(-1.0, 1.0)
    return actions.clamp(-1.0, 1.0)


def _soft_sequence_actions(target, obs, parsed):
    batch, steps, width = target.shape
    flat_target = target.reshape(batch * steps, width)
    flat_obs = obs.reshape(batch * steps, obs.shape[-1])
    return _soft_target_to_actions(flat_target, parsed, flat_obs).reshape(batch, steps, 4)


def _action_imitation_loss(pred_actions, target_actions, weights=None):
    loss = ((pred_actions - target_actions) ** 2).sum(dim=-1)
    if weights is not None:
        weights = weights.float()
        weights = weights / weights.mean().clamp_min(1e-6)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)
    return loss.mean()


def _rollout_hard_batch(
    parsed,
    policy,
    vec,
    obs,
    terminals,
    hard_states,
    device,
    update,
    round_update,
    round_updates,
):
    chosen = [
        hard_states[(update * parsed.replay_agents + idx) % len(hard_states)]
        for idx in range(parsed.replay_agents)
    ]
    if parsed.burnin_steps > 0:
        rollout_state = _burnin_batch(parsed, policy, vec, obs, terminals, chosen, device)
        start_state = clone_detached_state(rollout_state)
    else:
        _restore_batch(vec, chosen)
        rollout_state = _state_for_hard_batch(chosen, device)
        start_state = _state_for_hard_batch(chosen, device)

    inputs = []
    targets = []
    action_targets = []
    dones = []
    if round_update <= parsed.teacher_rollout_updates:
        student_prob = 0.0
    else:
        denom = max(1, round_updates - parsed.teacher_rollout_updates)
        frac = (round_update - parsed.teacher_rollout_updates) / denom
        student_prob = min(parsed.max_student_rollout_prob, frac * parsed.max_student_rollout_prob)

    for _ in range(parsed.seq_len):
        x_t = deployable_input(obs, parsed).detach()
        y_t = target_labels(obs, parsed).detach()
        inputs.append(x_t)
        targets.append(y_t)

        with torch.no_grad():
            logits, _, rollout_state = policy.forward_eval(x_t, rollout_state)
            student_target = logits.mean.detach()
            y_drive = y_t[:, : student_target.shape[1]]
            oracle_actions = target_to_actions(y_drive, parsed, obs)
            if student_prob <= 0.0:
                drive_target = y_drive
            else:
                use_student = (
                    torch.rand((vec.total_agents, 1), device=device) < student_prob
                )
                drive_target = torch.where(use_student, student_target, y_drive)
            actions = target_to_actions(drive_target, parsed, obs)

        action_targets.append(oracle_actions.detach())
        step_vec(vec, actions, device)
        done = terminals > 0.5
        dones.append(done.clone())
        if done.any():
            rollout_state = reset_done_state(rollout_state, done)
        rollout_state = detach_state(rollout_state)

    return (
        torch.stack(inputs, dim=1),
        torch.stack(targets, dim=1),
        torch.stack(action_targets, dim=1),
        torch.stack(dones, dim=1),
        start_state,
        student_prob,
    )


def run(parsed):
    from pufferlib import _C

    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    vec0, _, _, _ = _make_vec(parsed, device, parsed.total_agents)
    try:
        policy = load_policy(parsed.init_checkpoint, vec0, parsed, device)
        anchor_policy = None
        if parsed.behavior_anchor_weight > 0.0:
            anchor_policy = load_policy(parsed.init_checkpoint, vec0, copy.copy(parsed), device)
            anchor_policy.eval()
            for param in anchor_policy.parameters():
                param.requires_grad_(False)
    finally:
        vec0.close()

    replay_vec, replay_obs, _, replay_terminals = _make_vec(
        parsed, device, parsed.replay_agents
    )
    anchor_vec = None
    anchor_obs = None
    anchor_terminals = None
    anchor_state = None
    policy_anchor_state = None
    if anchor_policy is not None:
        anchor_vec, anchor_obs, _, anchor_terminals = _make_vec(
            parsed,
            device,
            parsed.behavior_anchor_agents,
            mode=parsed.behavior_anchor_mode,
        )
        anchor_state = anchor_policy.initial_state(anchor_vec.total_agents, device)
        policy_anchor_state = policy.initial_state(anchor_vec.total_agents, device)
        print(
            f"behavior_anchor mode={parsed.behavior_anchor_mode} "
            f"agents={parsed.behavior_anchor_agents} weight={parsed.behavior_anchor_weight}",
            flush=True,
        )
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=parsed.learning_rate)

    last_loss = 0.0
    last_false_negative_loss = 0.0
    last_sector_loss = 0.0
    last_vector_loss = 0.0
    last_action_loss = 0.0
    last_anchor_loss = 0.0
    last_positive_frac = 0.0
    last_student_prob = 0.0
    recoverable_count = 0
    global_update = 0
    try:
        rounds = max(1, parsed.replay_rounds)
        base_updates = parsed.seq_updates // rounds
        extra_updates = parsed.seq_updates % rounds
        for replay_round in range(1, rounds + 1):
            policy.eval()
            hard_states = _collect_training_hard_states(parsed, policy, device)
            recoverable_count = len(hard_states)
            policy.train()
            round_updates = base_updates + (1 if replay_round <= extra_updates else 0)
            for round_update in range(1, round_updates + 1):
                global_update += 1
                x, target, action_target, done_seq, start_state, student_prob = _rollout_hard_batch(
                    parsed,
                    policy,
                    replay_vec,
                    replay_obs,
                    replay_terminals,
                    hard_states,
                    device,
                    global_update,
                    round_update,
                    round_updates,
                )
                pred = predict_sequence(policy, x, start_state, done_seq)
                target_main = target[..., : pred.shape[-1]]
                weights = None
                if parsed.positive_only:
                    weights = (target_main[..., 3] > parsed.exists_threshold).float()
                    last_positive_frac = float(weights.mean().detach().cpu())
                else:
                    last_positive_frac = 1.0
                loss = parsed.replay_loss_weight * target_loss(
                    pred, target_main, parsed, weights=weights
                )

                false_negative_loss = torch.zeros((), dtype=torch.float32, device=device)
                if parsed.false_negative_weight > 0.0:
                    false_negative_loss = false_negative_exists_loss(
                        pred,
                        target_main,
                        parsed,
                        coverage=x[..., 17].float(),
                        weights=weights,
                    )
                    loss = loss + parsed.false_negative_weight * false_negative_loss

                sector_loss = torch.zeros((), dtype=torch.float32, device=device)
                if parsed.predict_frontier_sectors and parsed.frontier_sector_loss_weight > 0.0:
                    sector_loss = frontier_sector_loss(pred, target_main, weights=weights)
                    loss = loss + parsed.frontier_sector_loss_weight * sector_loss

                vector_loss = torch.zeros((), dtype=torch.float32, device=device)
                if parsed.predict_frontier_vector and parsed.frontier_vector_loss_weight > 0.0:
                    vector_loss = frontier_vector_loss(pred, target_main, parsed, weights=weights)
                    loss = loss + parsed.frontier_vector_loss_weight * vector_loss

                action_loss = torch.zeros((), dtype=torch.float32, device=device)
                if parsed.action_loss_weight > 0.0:
                    pred_actions = _soft_sequence_actions(pred, x, parsed)
                    action_loss = _action_imitation_loss(pred_actions, action_target, weights=weights)
                    loss = loss + parsed.action_loss_weight * action_loss

                anchor_loss = torch.zeros((), dtype=torch.float32, device=device)
                if anchor_policy is not None:
                    x_anchor = deployable_input(anchor_obs, parsed).detach()
                    with torch.no_grad():
                        anchor_logits, _, anchor_state = anchor_policy.forward_eval(
                            x_anchor, anchor_state
                        )
                        anchor_target = anchor_logits.mean.detach()
                    pred_anchor_logits, _, policy_anchor_state = policy.forward_eval(
                        x_anchor, policy_anchor_state
                    )
                    anchor_loss = F.mse_loss(pred_anchor_logits.mean, anchor_target)
                    loss = loss + parsed.behavior_anchor_weight * anchor_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), parsed.max_grad_norm)
                optimizer.step()

                if anchor_policy is not None:
                    with torch.no_grad():
                        anchor_actions = target_to_actions(anchor_target, parsed, anchor_obs)
                    step_vec(anchor_vec, anchor_actions, device)
                    done = anchor_terminals > 0.5
                    if done.any():
                        anchor_state = reset_done_state(anchor_state, done)
                        policy_anchor_state = reset_done_state(policy_anchor_state, done)
                    anchor_state = detach_state(anchor_state)
                    policy_anchor_state = detach_state(policy_anchor_state)

                last_loss = float(loss.detach().cpu())
                last_false_negative_loss = float(false_negative_loss.detach().cpu())
                last_sector_loss = float(sector_loss.detach().cpu())
                last_vector_loss = float(vector_loss.detach().cpu())
                last_action_loss = float(action_loss.detach().cpu())
                last_anchor_loss = float(anchor_loss.detach().cpu())
                last_student_prob = student_prob

                if global_update % parsed.print_interval == 0 or global_update == parsed.seq_updates:
                    logs = replay_vec.log()
                    print(
                        f"hard_replay_update={global_update} round={replay_round}/{rounds} "
                        f"round_update={round_update}/{round_updates} "
                        f"loss={last_loss:.6g} "
                        f"fn_loss={last_false_negative_loss:.6g} "
                        f"sector_loss={last_sector_loss:.6g} "
                        f"vector_loss={last_vector_loss:.6g} "
                        f"action_loss={last_action_loss:.6g} "
                        f"anchor_loss={last_anchor_loss:.6g} "
                        f"positive_frac={last_positive_frac:.4f} "
                        f"student_prob={last_student_prob:.3f} "
                        f"success={float(logs.get('success', 0.0)):.4f} "
                        f"coverage={float(logs.get('coverage', 0.0)):.4f} "
                        f"obstacle={float(logs.get('obstacle_hit', 0.0)):.4f} "
                        f"timeout={float(logs.get('timeout', 0.0)):.4f}",
                        flush=True,
                    )
    finally:
        replay_vec.close()
        if anchor_vec is not None:
            anchor_vec.close()

    output = Path(parsed.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "input_size": replay_obs.shape[1],
            "num_atns": len(policy.decoder.nvec),
            "act_sizes": list(policy.decoder.nvec),
            "config": {
                "task_mode": 3,
                "obstacle_mode": parsed.obstacle_mode,
                "train_mode": "hardstate_replay_target_distill",
                "init_checkpoint": parsed.init_checkpoint,
                "deploy_patch_obs": parsed.deploy_patch_obs,
                "deploy_coverage_obs": parsed.deploy_coverage_obs,
                "explore_frontier_target_mode": parsed.explore_frontier_target_mode,
                "explore_reachable_coverage": parsed.explore_reachable_coverage,
                "recoverable_hard_states": recoverable_count,
                "seq_len": parsed.seq_len,
                "seq_updates": parsed.seq_updates,
                "replay_rounds": parsed.replay_rounds,
                "burnin_steps": parsed.burnin_steps,
                "teacher_rollout_updates": parsed.teacher_rollout_updates,
                "max_student_rollout_prob": parsed.max_student_rollout_prob,
                "replay_loss_weight": parsed.replay_loss_weight,
                "positive_only": parsed.positive_only,
                "false_negative_weight": parsed.false_negative_weight,
                "false_negative_margin": parsed.false_negative_margin,
                "false_negative_min_coverage": parsed.false_negative_min_coverage,
                "behavior_anchor_weight": parsed.behavior_anchor_weight,
                "behavior_anchor_mode": parsed.behavior_anchor_mode,
                "behavior_anchor_agents": parsed.behavior_anchor_agents,
                "predict_frontier_sectors": parsed.predict_frontier_sectors,
                "predict_frontier_vector": parsed.predict_frontier_vector,
                "frontier_sector_loss_weight": parsed.frontier_sector_loss_weight,
                "frontier_vector_loss_weight": parsed.frontier_vector_loss_weight,
                "action_loss_weight": parsed.action_loss_weight,
                "absent_fallback": parsed.absent_fallback,
                "absent_fallback_min_ray": parsed.absent_fallback_min_ray,
                "absent_fallback_angle_penalty": parsed.absent_fallback_angle_penalty,
                "use_pred_vector_fallback": parsed.use_pred_vector_fallback,
                "pred_vector_fallback_primary_threshold": parsed.pred_vector_fallback_primary_threshold,
                "pred_vector_fallback_min_coverage": parsed.pred_vector_fallback_min_coverage,
                "pred_vector_fallback_min_dist": parsed.pred_vector_fallback_min_dist,
                "safety_cone_ray_stop": parsed.safety_cone_ray_stop,
                "safety_cone_deg": parsed.safety_cone_deg,
                "safety_slow_ray": parsed.safety_slow_ray,
            },
            "policy": {
                "network": parsed.network,
                "hidden_size": parsed.hidden_size,
                "num_layers": parsed.num_layers,
                "action_std": parsed.action_std,
            },
            "final_loss": last_loss,
        },
        output,
    )
    print(f"wrote {output} final_loss={last_loss:.6g}", flush=True)

    if parsed.eval_after:
        parsed.checkpoint = str(output)
        return run_eval(parsed)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the target policy from resettable oracle-recoverable timeout-window states."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    add_common(parser)
    parser.add_argument("--replay-agents", type=int, default=64)
    parser.add_argument("--hard-states", type=int, default=256)
    parser.add_argument("--min-recoverable-states", type=int, default=32)
    parser.add_argument("--rewind", type=int, default=256)
    parser.add_argument("--replay-horizon", type=int, default=512)
    parser.add_argument("--max-collect-steps", type=int, default=16384)
    parser.add_argument("--crash-reward-threshold", type=float, default=-5.0)
    parser.add_argument("--success-reward-threshold", type=float, default=5.0)
    parser.add_argument("--sector-match-threshold", type=float, default=0.15)
    parser.add_argument("--seq-updates", type=int, default=800)
    parser.add_argument("--replay-rounds", type=int, default=1)
    parser.add_argument("--burnin-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--teacher-rollout-updates", type=int, default=80)
    parser.add_argument("--max-student-rollout-prob", type=float, default=0.5)
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument("--eval-after", action="store_true")
    parser.add_argument("--replay-loss-weight", type=float, default=1.0)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--false-negative-weight", type=float, default=0.0)
    parser.add_argument("--false-negative-margin", type=float, default=0.80)
    parser.add_argument("--false-negative-min-coverage", type=float, default=0.0)
    parser.add_argument("--behavior-anchor-weight", type=float, default=0.0)
    parser.add_argument("--behavior-anchor-agents", type=int, default=64)
    parser.add_argument(
        "--behavior-anchor-mode",
        choices=tuple(OBSTACLE_MODES),
        default="procedural_indoor",
    )
    parser.add_argument("--dist-loss-weight", type=float, default=0.5)
    parser.add_argument("--exists-loss-weight", type=float, default=0.5)
    parser.add_argument("--negative-exists-weight", type=float, default=1.0)
    parser.add_argument("--false-positive-weight", type=float, default=0.0)
    parser.add_argument("--false-positive-margin", type=float, default=0.1)
    parser.add_argument("--absent-direction-weight", type=float, default=0.0)
    parser.add_argument("--speed-loss-weight", type=float, default=1.0)
    parser.add_argument("--predict-frontier-sectors", action="store_true")
    parser.add_argument("--frontier-sector-loss-weight", type=float, default=4.0)
    parser.add_argument("--predict-frontier-vector", action="store_true")
    parser.add_argument("--frontier-vector-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.0)
    parsed = parser.parse_args()
    parsed.obstacle_mode = OBSTACLE_MODES[parsed.mode]
    if parsed.predict_frontier_vector and not parsed.predict_frontier_sectors:
        parser.error("--predict-frontier-vector requires --predict-frontier-sectors")
    return parsed


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
