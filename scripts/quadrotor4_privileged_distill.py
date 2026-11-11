#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


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
    env["seed"] = parsed.seed
    env["max_steps"] = parsed.max_steps
    env["action_repeat"] = parsed.action_repeat
    env["task_mode"] = 4
    env["action_mode"] = 1
    env["max_vel"] = parsed.max_vel
    env["max_z_vel"] = parsed.max_z_vel
    env["max_rate"] = parsed.max_rate
    env["vel_ctrl_ki_z"] = parsed.vel_ctrl_ki_z
    env["spawn_radius"] = parsed.spawn_radius
    env["spawn_radius_min"] = parsed.spawn_radius_min
    env["spawn_height"] = parsed.spawn_height
    env["room_size"] = parsed.room_size
    env["ceiling_height"] = parsed.ceiling_height
    env["num_rays"] = parsed.num_rays
    env["ray_length"] = parsed.ray_length
    env["obstacle_mode"] = parsed.obstacle_mode
    env["route_goal_radius"] = parsed.route_goal_radius
    env["route_goal_count"] = parsed.route_goal_count
    env["route_goal_count_min"] = parsed.route_goal_count_min
    env["route_path_margin"] = parsed.route_path_margin
    env["route_target_mode"] = parsed.route_target_mode
    env["route_blocked_target_prob"] = parsed.route_blocked_target_prob
    env["route_walls"] = 1
    env["route_wall_obs"] = 1
    env["route_target_obs_mode"] = 1
    env["route_direct_obs"] = 0 if privileged else 1
    env["action_history_obs"] = 1 if parsed.native_action_history_obs else 0
    env["route_extra_obs"] = 0
    env["route_perception_obs"] = parsed.route_perception_obs
    env["route_target_z_min"] = parsed.target_z
    env["route_target_z_max"] = parsed.target_z
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


def create_policy(input_size, act_sizes, parsed, device):
    import pufferlib.models

    encoder = pufferlib.models.DefaultEncoder(input_size, parsed.hidden_size)
    decoder = pufferlib.models.DefaultDecoder(act_sizes, parsed.hidden_size)
    network_cls = getattr(pufferlib.models, parsed.network)
    network = network_cls(
        hidden_size=parsed.hidden_size,
        num_layers=parsed.num_layers,
    )
    if hasattr(decoder, "decoder_logstd"):
        decoder.decoder_logstd.data.fill_(float(np.log(parsed.action_std)))
    return pufferlib.models.Policy(encoder, decoder, network).to(device)


def detach_state(state):
    if not isinstance(state, tuple):
        return state
    out = []
    for value in state:
        if torch.is_tensor(value):
            out.append(value.detach())
        else:
            out.append(value)
    return tuple(out)


def clone_detached_state(state):
    if not isinstance(state, tuple):
        return state
    out = []
    for value in state:
        if torch.is_tensor(value):
            out.append(value.detach().clone())
        else:
            out.append(value)
    return tuple(out)


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


def student_input(obs, prev_action, parsed):
    if parsed.native_action_history_obs:
        return obs.float()
    return torch.cat([obs.float(), prev_action.float()], dim=1)


def predict_sequence_eval_path(policy, x, state, dones):
    preds = []
    for t in range(x.shape[1]):
        logits, _, state = policy.forward_eval(x[:, t], state)
        preds.append(logits.mean)
        state = reset_done_state(state, dones[:, t])
    return torch.stack(preds, dim=1)


def step_vec(vec, actions, device):
    actions = actions.detach().to(dtype=torch.float32).contiguous()
    if device.type == "cuda":
        vec.gpu_step(actions.data_ptr())
    else:
        vec.cpu_step(actions.data_ptr())


def accumulate_logs(accum, logs):
    n = float(logs.get("n", 0.0))
    if n <= 0:
        return accum
    if accum is None:
        accum = {"n": 0.0, "sums": {}}
    for key, value in logs.items():
        value = float(value)
        if key == "n":
            continue
        accum["sums"][key] = accum["sums"].get(key, 0.0) + value * n
    accum["n"] += n
    return accum


def finalize_logs(accum):
    if accum is None or accum["n"] <= 0:
        return {}
    out = {"n": accum["n"]}
    for key, value in accum["sums"].items():
        out[key] = value / accum["n"]
    return out


def print_metrics(metrics):
    keys = (
        "n",
        "success",
        "score",
        "crashed",
        "obstacle_hit",
        "wall_hit",
        "ground_hit",
        "timeout",
        "route_goals",
        "min_clearance",
        "path_efficiency",
        "action_delta",
    )
    for key in keys:
        if key in metrics:
            print(f"{key} {metrics[key]:.6g}")


def run_eval(parsed, policy=None, checkpoint=None):
    from pufferlib import _C

    use_teacher = parsed.eval_policy == "teacher"
    args = make_vec_config(parsed, privileged=use_teacher)
    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, _, terminals = tensor_views(vec, device)
    vec.reset()

    prev_action = torch.zeros((vec.total_agents, vec.num_atns), device=device)
    if policy is None and checkpoint is not None:
        data = torch.load(checkpoint, map_location=device, weights_only=False)
        input_size = int(data["input_size"])
        policy_meta = data.get("policy", {})
        parsed.network = policy_meta.get("network", parsed.network)
        parsed.hidden_size = int(policy_meta.get("hidden_size", parsed.hidden_size))
        parsed.num_layers = int(policy_meta.get("num_layers", parsed.num_layers))
        parsed.action_std = float(policy_meta.get("action_std", parsed.action_std))
        policy = create_policy(input_size, vec.act_sizes, parsed, device)
        policy.load_state_dict(data["state_dict"])
    if policy is not None:
        policy.eval()
        state = policy.initial_state(vec.total_agents, device)
    else:
        state = None

    accum = None
    steps = 0
    try:
        while steps < parsed.max_eval_steps:
            with torch.no_grad():
                if use_teacher:
                    actions = teacher_actions_from_privileged_obs(obs, parsed)
                else:
                    logits, _, state = policy.forward_eval(student_input(obs, prev_action, parsed), state)
                    actions = logits.mean
            step_vec(vec, actions, device)
            done = terminals > 0.5
            prev_action = actions.detach()
            if done.any():
                prev_action = prev_action.clone()
                prev_action[done] = 0.0
                state = reset_done_state(state, done)
            state = detach_state(state)
            steps += 1
            if steps % parsed.log_interval == 0:
                accum = accumulate_logs(accum, vec.log())
                metrics = finalize_logs(accum)
                if metrics.get("n", 0.0) >= parsed.episodes:
                    break
        accum = accumulate_logs(accum, vec.log())
        metrics = finalize_logs(accum)
    finally:
        vec.close()

    label = parsed.eval_policy if checkpoint is None else checkpoint
    print(f"eval {label}")
    print_metrics(metrics)
    return metrics


def run_train(parsed):
    from pufferlib import _C

    teacher_args = make_vec_config(parsed, privileged=True)
    student_args = make_vec_config(parsed, privileged=False)
    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    teacher_vec = _C.create_vec(teacher_args, 0 if device.type == "cpu" else 1)
    student_vec = _C.create_vec(student_args, 0 if device.type == "cpu" else 1)
    teacher_obs, _, teacher_terms = tensor_views(teacher_vec, device)
    student_obs, _, student_terms = tensor_views(student_vec, device)
    teacher_vec.reset()
    student_vec.reset()

    if teacher_vec.obs_size != student_vec.obs_size:
        raise RuntimeError(
            f"teacher/student obs size mismatch: {teacher_vec.obs_size} vs {student_vec.obs_size}"
        )

    input_size = student_vec.obs_size + (0 if parsed.native_action_history_obs else student_vec.num_atns)
    policy = create_policy(input_size, student_vec.act_sizes, parsed, device)
    policy.train()
    state = policy.initial_state(student_vec.total_agents, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=parsed.learning_rate)
    prev_action = torch.zeros((student_vec.total_agents, student_vec.num_atns), device=device)
    last_loss = 0.0

    try:
        for update in range(1, parsed.updates + 1):
            teacher_actions = teacher_actions_from_privileged_obs(teacher_obs, parsed)
            logits, _, state = policy.forward_eval(student_input(student_obs, prev_action, parsed), state)
            loss = F.mse_loss(logits.mean, teacher_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), parsed.max_grad_norm)
            optimizer.step()
            last_loss = float(loss.detach().cpu())

            with torch.no_grad():
                student_actions = logits.mean.detach().clamp(-1.0, 1.0)
                if update <= parsed.teacher_rollout_updates:
                    actions = teacher_actions
                    student_prob = 0.0
                else:
                    denom = max(1, parsed.updates - parsed.teacher_rollout_updates)
                    frac = (update - parsed.teacher_rollout_updates) / denom
                    student_prob = min(parsed.max_student_rollout_prob, frac * parsed.max_student_rollout_prob)
                    if parsed.max_student_rollout_prob <= 0.0:
                        actions = teacher_actions
                    else:
                        mask = torch.rand((student_vec.total_agents, 1), device=device) < student_prob
                        actions = torch.where(mask, student_actions, teacher_actions)

            step_vec(teacher_vec, actions, device)
            step_vec(student_vec, actions, device)
            done = torch.logical_or(teacher_terms > 0.5, student_terms > 0.5)
            prev_action = actions.detach()
            if done.any():
                prev_action = prev_action.clone()
                prev_action[done] = 0.0
                state = reset_done_state(state, done)
            state = detach_state(state)

            if update % parsed.print_interval == 0 or update == parsed.updates:
                teacher_logs = teacher_vec.log()
                student_logs = student_vec.log()
                print(
                    f"update={update} loss={last_loss:.6g} "
                    f"student_prob={student_prob:.3f} "
                    f"teacher_success={float(teacher_logs.get('success', 0.0)):.4f} "
                    f"student_success={float(student_logs.get('success', 0.0)):.4f} "
                    f"student_obstacle={float(student_logs.get('obstacle_hit', 0.0)):.4f}",
                    flush=True,
                )

        output = Path(parsed.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "input_size": input_size,
                "base_obs_size": student_vec.obs_size,
                "num_atns": student_vec.num_atns,
                "act_sizes": list(student_vec.act_sizes),
                "prev_action_size": 0 if parsed.native_action_history_obs else student_vec.num_atns,
                "config": {
                    "route_direct_obs": 1,
                    "teacher_route_direct_obs": 0,
                    "route_target_obs_mode": 1,
                    "route_wall_obs": 1,
                    "num_rays": parsed.num_rays,
                    "route_perception_obs": parsed.route_perception_obs,
                    "native_action_history_obs": parsed.native_action_history_obs,
                },
                "policy": {
                    "network": parsed.network,
                    "hidden_size": parsed.hidden_size,
                    "num_layers": parsed.num_layers,
                    "action_std": parsed.action_std,
                },
            },
            output,
        )
        print(f"wrote {output} final_loss={last_loss:.6g}", flush=True)
    finally:
        teacher_vec.close()
        student_vec.close()

    if parsed.eval_after:
        parsed.eval_policy = "student"
        return run_eval(parsed, policy=policy)
    return {}


def run_train_seq(parsed):
    from pufferlib import _C

    teacher_args = make_vec_config(parsed, privileged=True)
    student_args = make_vec_config(parsed, privileged=False)
    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    teacher_vec = _C.create_vec(teacher_args, 0 if device.type == "cpu" else 1)
    student_vec = _C.create_vec(student_args, 0 if device.type == "cpu" else 1)
    teacher_obs, _, teacher_terms = tensor_views(teacher_vec, device)
    student_obs, _, student_terms = tensor_views(student_vec, device)

    if teacher_vec.obs_size != student_vec.obs_size:
        raise RuntimeError(
            f"teacher/student obs size mismatch: {teacher_vec.obs_size} vs {student_vec.obs_size}"
        )

    input_size = student_vec.obs_size + (0 if parsed.native_action_history_obs else student_vec.num_atns)
    policy = create_policy(input_size, student_vec.act_sizes, parsed, device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=parsed.learning_rate)
    last_loss = 0.0
    last_mask_frac = 0.0
    teacher_vec.reset()
    student_vec.reset()
    prev_action = torch.zeros((student_vec.total_agents, student_vec.num_atns), device=device)
    rollout_state = policy.initial_state(student_vec.total_agents, device)

    try:
        for update in range(1, parsed.seq_updates + 1):
            if parsed.reset_each_seq:
                teacher_vec.reset()
                student_vec.reset()
                prev_action = torch.zeros((student_vec.total_agents, student_vec.num_atns), device=device)
                rollout_state = policy.initial_state(student_vec.total_agents, device)
                active = torch.ones((student_vec.total_agents,), dtype=torch.bool, device=device)
            start_state = clone_detached_state(rollout_state)

            inputs = []
            targets = []
            masks = []
            dones = []
            if update <= parsed.teacher_rollout_updates:
                student_prob = 0.0
            else:
                denom = max(1, parsed.seq_updates - parsed.teacher_rollout_updates)
                frac = (update - parsed.teacher_rollout_updates) / denom
                student_prob = min(parsed.max_student_rollout_prob, frac * parsed.max_student_rollout_prob)

            for _ in range(parsed.seq_len):
                x_t = student_input(student_obs, prev_action, parsed).detach()
                y_t = teacher_actions_from_privileged_obs(teacher_obs, parsed).detach()
                inputs.append(x_t.clone())
                targets.append(y_t.clone())
                if parsed.reset_each_seq:
                    masks.append(active.clone())

                with torch.no_grad():
                    logits, _, rollout_state = policy.forward_eval(x_t, rollout_state)
                    student_actions = logits.mean.detach().clamp(-1.0, 1.0)
                    if student_prob <= 0.0:
                        actions = y_t
                    else:
                        use_student = torch.rand(
                            (student_vec.total_agents, 1), device=device
                        ) < student_prob
                        actions = torch.where(use_student, student_actions, y_t)

                step_vec(teacher_vec, actions, device)
                step_vec(student_vec, actions, device)
                done = torch.logical_or(teacher_terms > 0.5, student_terms > 0.5)
                dones.append(done.clone())
                prev_action = actions.detach()
                if done.any():
                    prev_action = prev_action.clone()
                    prev_action[done] = 0.0
                    rollout_state = reset_done_state(rollout_state, done)
                rollout_state = detach_state(rollout_state)
                if parsed.reset_each_seq:
                    active = torch.logical_and(active, torch.logical_not(done))

            x = torch.stack(inputs, dim=1)
            target = torch.stack(targets, dim=1)
            if parsed.reset_each_seq:
                mask = torch.stack(masks, dim=1).unsqueeze(-1)
                logits, _ = policy(x)
                pred = logits.mean.reshape(target.shape)
                per_action = (pred - target).pow(2)
                denom = mask.sum() * per_action.shape[-1]
                if denom <= 0:
                    continue
                loss = (per_action * mask).sum() / denom
                last_mask_frac = float(mask.float().mean().detach().cpu())
            else:
                done_seq = torch.stack(dones, dim=1)
                pred = predict_sequence_eval_path(policy, x, start_state, done_seq)
                loss = F.mse_loss(pred, target)
                last_mask_frac = 1.0

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), parsed.max_grad_norm)
            optimizer.step()
            last_loss = float(loss.detach().cpu())

            if update % parsed.print_interval == 0 or update == parsed.seq_updates:
                teacher_logs = teacher_vec.log()
                student_logs = student_vec.log()
                print(
                    f"seq_update={update} loss={last_loss:.6g} "
                    f"mask_frac={last_mask_frac:.3f} "
                    f"student_prob={student_prob:.3f} "
                    f"teacher_success={float(teacher_logs.get('success', 0.0)):.4f} "
                    f"student_success={float(student_logs.get('success', 0.0)):.4f} "
                    f"student_obstacle={float(student_logs.get('obstacle_hit', 0.0)):.4f}",
                    flush=True,
                )

        output = Path(parsed.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "input_size": input_size,
                "base_obs_size": student_vec.obs_size,
                "num_atns": student_vec.num_atns,
                "act_sizes": list(student_vec.act_sizes),
                "prev_action_size": 0 if parsed.native_action_history_obs else student_vec.num_atns,
                "config": {
                    "route_direct_obs": 1,
                    "teacher_route_direct_obs": 0,
                    "route_target_obs_mode": 1,
                    "route_wall_obs": 1,
                    "num_rays": parsed.num_rays,
                    "route_perception_obs": parsed.route_perception_obs,
                    "native_action_history_obs": parsed.native_action_history_obs,
                    "train_mode": "sequence_distill",
                    "seq_len": parsed.seq_len,
                    "reset_each_seq": parsed.reset_each_seq,
                },
                "policy": {
                    "network": parsed.network,
                    "hidden_size": parsed.hidden_size,
                    "num_layers": parsed.num_layers,
                    "action_std": parsed.action_std,
                },
            },
            output,
        )
        print(
            f"wrote {output} final_loss={last_loss:.6g} mask_frac={last_mask_frac:.3f}",
            flush=True,
        )
    finally:
        teacher_vec.close()
        student_vec.close()

    if parsed.eval_after:
        parsed.eval_policy = "student"
        return run_eval(parsed, policy=policy)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="QDR-130 privileged waypoint teacher -> direct-observation student distillation"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--seed", type=int, default=88)
        p.add_argument("--total-agents", type=int, default=4096)
        p.add_argument("--num-buffers", type=int, default=2)
        p.add_argument("--num-threads", type=int, default=16)
        p.add_argument("--max-steps", type=int, default=768)
        p.add_argument("--action-repeat", type=int, default=1)
        p.add_argument("--max-vel", type=float, default=1.0)
        p.add_argument("--max-z-vel", type=float, default=0.25)
        p.add_argument("--max-rate", type=float, default=1.0)
        p.add_argument("--vel-ctrl-ki-z", type=float, default=4.0)
        p.add_argument("--spawn-radius", type=float, default=2.5)
        p.add_argument("--spawn-radius-min", type=float, default=2.5)
        p.add_argument("--spawn-height", type=float, default=1.5)
        p.add_argument("--room-size", type=float, default=3.0)
        p.add_argument("--ceiling-height", type=float, default=3.0)
        p.add_argument("--num-rays", type=int, default=24)
        p.add_argument("--ray-length", type=float, default=4.0)
        p.add_argument("--obstacle-mode", type=int, default=7)
        p.add_argument("--route-goal-radius", type=float, default=1.0)
        p.add_argument("--route-goal-count", type=int, default=1)
        p.add_argument("--route-goal-count-min", type=int, default=0)
        p.add_argument("--route-path-margin", type=float, default=0.35)
        p.add_argument("--route-target-mode", type=int, default=1)
        p.add_argument("--route-blocked-target-prob", type=float, default=1.0)
        p.add_argument("--route-perception-obs", type=int, choices=(0, 1, 2, 3), default=0)
        p.add_argument("--native-action-history-obs", action="store_true")
        p.add_argument("--target-z", type=float, default=1.5)
        p.add_argument("--hidden-size", type=int, default=128)
        p.add_argument("--num-layers", type=int, default=1)
        p.add_argument("--network", choices=("MLP", "MinGRU", "GRU", "LSTM"), default="MLP")
        p.add_argument("--action-std", type=float, default=0.2)
        p.add_argument("--teacher-speed-gain", type=float, default=2.5)
        p.add_argument("--teacher-max-speed", type=float, default=0.85)
        p.add_argument("--teacher-slow-radius", type=float, default=0.12)
        p.add_argument("--teacher-z-gain", type=float, default=0.8)
        p.add_argument("--teacher-z-damp", type=float, default=0.15)
        p.add_argument("--teacher-max-z", type=float, default=0.6)
        p.add_argument("--log-interval", type=int, default=64)

    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--output", required=True)
    train.add_argument("--updates", type=int, default=12000)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--teacher-rollout-updates", type=int, default=1000)
    train.add_argument("--max-student-rollout-prob", type=float, default=0.5)
    train.add_argument("--print-interval", type=int, default=500)
    train.add_argument("--eval-after", action="store_true")
    train.add_argument("--episodes", type=int, default=4096)
    train.add_argument("--max-eval-steps", type=int, default=20000)

    train_seq = sub.add_parser("train-seq")
    add_common(train_seq)
    train_seq.add_argument("--output", required=True)
    train_seq.add_argument("--seq-updates", type=int, default=400)
    train_seq.add_argument("--seq-len", type=int, default=64)
    train_seq.add_argument("--learning-rate", type=float, default=3e-4)
    train_seq.add_argument("--max-grad-norm", type=float, default=1.0)
    train_seq.add_argument("--teacher-rollout-updates", type=int, default=40)
    train_seq.add_argument("--max-student-rollout-prob", type=float, default=1.0)
    train_seq.add_argument("--print-interval", type=int, default=20)
    train_seq.add_argument("--reset-each-seq", action="store_true")
    train_seq.add_argument("--eval-after", action="store_true")
    train_seq.add_argument("--episodes", type=int, default=4096)
    train_seq.add_argument("--max-eval-steps", type=int, default=20000)

    eval_p = sub.add_parser("eval")
    add_common(eval_p)
    eval_p.add_argument("checkpoint", nargs="?")
    eval_p.add_argument("--eval-policy", choices=("student", "teacher"), default="student")
    eval_p.add_argument("--episodes", type=int, default=4096)
    eval_p.add_argument("--max-eval-steps", type=int, default=20000)

    return parser.parse_args()


def main():
    parsed = parse_args()
    if parsed.cmd == "train":
        run_train(parsed)
    elif parsed.cmd == "train-seq":
        run_train_seq(parsed)
    elif parsed.cmd == "eval":
        if parsed.eval_policy == "student" and not parsed.checkpoint:
            raise SystemExit("student eval requires a checkpoint")
        run_eval(parsed, checkpoint=parsed.checkpoint)
    else:
        raise SystemExit(f"unknown command {parsed.cmd}")


if __name__ == "__main__":
    main()
