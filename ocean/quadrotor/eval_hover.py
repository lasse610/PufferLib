import argparse
import json
from pathlib import Path

import numpy as np
import torch

import pufferlib.ocean.torch as ocean_torch
from pufferlib.ocean.quadrotor import Quadrotor

GRAVITY = 9.81
VEL_OBS_SCALE = 1.0
MIN_OFFSET_OBS_SCALE = 0.1
TARGET_OBS_START = 14
RAY_OBS_START = 26
WALL_OBS_SIZE = 6
ROUTE_EXTRA_OBS_SIZE = 8
ROUTE_PERCEPTION_OBS_SIZE = 12
ROUTE_COMPACT_PERCEPTION_OBS_SIZE = 8
ROUTE_AFFORDANCE_OBS_SIZE = 12


def route_perception_obs_size(mode):
    if int(mode) == 1:
        return ROUTE_PERCEPTION_OBS_SIZE
    if int(mode) == 2:
        return ROUTE_COMPACT_PERCEPTION_OBS_SIZE
    if int(mode) == 3:
        return ROUTE_AFFORDANCE_OBS_SIZE
    return 0


def make_env(args):
    return Quadrotor(
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        action_repeat=args.action_repeat,
        max_rate=args.max_rate,
        max_vel=args.max_vel,
        max_z_vel=args.max_z_vel,
        vel_ctrl_kp_vel=getattr(args, "vel_ctrl_kp_vel", 0.0),
        vel_ctrl_ki_z=getattr(args, "vel_ctrl_ki_z", 0.0),
        vel_ctrl_int_limit_z=getattr(args, "vel_ctrl_int_limit_z", 0.0),
        vel_ctrl_max_z_acc=getattr(args, "vel_ctrl_max_z_acc", 0.0),
        spawn_radius=args.spawn_radius,
        spawn_height=args.spawn_height,
        init_vel_radius=args.init_vel_radius,
        init_omega_radius=args.init_omega_radius,
        room_size=args.room_size,
        ceiling_height=args.ceiling_height,
        num_rays=args.num_rays,
        ray_length=args.ray_length,
        hover_radius=args.hover_radius,
        obstacle_mode=args.obstacle_mode,
        obstacle_block_prob=args.obstacle_block_prob,
        obs_noise=args.obs_noise,
        action_noise=args.action_noise,
        action_delay=args.action_delay,
        action_delay_min=getattr(args, "action_delay_min", -1),
        action_delay_max=getattr(args, "action_delay_max", -1),
        action_history_obs=getattr(args, "action_history_obs", 0),
        mass_scale=args.mass_scale,
        inertia_scale=args.inertia_scale,
        thrust_scale=args.thrust_scale,
        torque_scale=args.torque_scale,
        drag_scale=args.drag_scale,
        motor_tau_scale=args.motor_tau_scale,
        mass_scale_min=getattr(args, "mass_scale_min", 0.0),
        mass_scale_max=getattr(args, "mass_scale_max", 0.0),
        inertia_scale_min=getattr(args, "inertia_scale_min", 0.0),
        inertia_scale_max=getattr(args, "inertia_scale_max", 0.0),
        thrust_scale_min=getattr(args, "thrust_scale_min", 0.0),
        thrust_scale_max=getattr(args, "thrust_scale_max", 0.0),
        torque_scale_min=getattr(args, "torque_scale_min", 0.0),
        torque_scale_max=getattr(args, "torque_scale_max", 0.0),
        drag_scale_min=getattr(args, "drag_scale_min", 0.0),
        drag_scale_max=getattr(args, "drag_scale_max", 0.0),
        motor_tau_scale_min=getattr(args, "motor_tau_scale_min", 0.0),
        motor_tau_scale_max=getattr(args, "motor_tau_scale_max", 0.0),
        explore_cell_size=args.explore_cell_size,
        explore_goal_fraction=args.explore_goal_fraction,
        route_goal_radius=args.route_goal_radius,
        route_goal_count=args.route_goal_count,
        route_goal_count_min=getattr(args, "route_goal_count_min", 0),
        route_step_cost=args.route_step_cost,
        route_proximity_reward=args.route_proximity_reward,
        route_dist_scale=args.route_dist_scale,
        route_progress_scale=args.route_progress_scale,
        route_crash_reward=args.route_crash_reward,
        route_crash_goal_scale=args.route_crash_goal_scale,
        route_crash_dist_scale=args.route_crash_dist_scale,
        route_clearance_radius=args.route_clearance_radius,
        route_clearance_scale=args.route_clearance_scale,
        route_reference_clearance_radius=args.route_reference_clearance_radius,
        route_reference_clearance_scale=args.route_reference_clearance_scale,
        route_path_margin=args.route_path_margin,
        route_vel_scale=args.route_vel_scale,
        route_arrival_radius=args.route_arrival_radius,
        route_arrival_bonus=args.route_arrival_bonus,
        route_arrival_vel_scale=args.route_arrival_vel_scale,
        route_arrival_toward_scale=args.route_arrival_toward_scale,
        route_target_wall_margin=args.route_target_wall_margin,
        route_goal_speed=args.route_goal_speed,
        route_goal_bonus=args.route_goal_bonus,
        route_reach_bonus=args.route_reach_bonus,
        route_target_z_min=getattr(args, "route_target_z_min", 0.0),
        route_target_z_max=getattr(args, "route_target_z_max", 0.0),
        route_reward_mode=args.route_reward_mode,
        route_target_mode=args.route_target_mode,
        route_blocked_target_prob=getattr(args, "route_blocked_target_prob", 1.0),
        route_walls=args.route_walls,
        route_direct_obs=args.route_direct_obs,
        route_extra_obs=args.route_extra_obs,
        route_wall_obs=args.route_wall_obs,
        route_perception_obs=getattr(args, "route_perception_obs", 0),
        route_target_obs_mode=getattr(args, "route_target_obs_mode", "legacy"),
        task_mode=args.task_mode,
        action_mode=args.action_mode,
        report_interval=1,
        render_mode="human" if args.render else None,
        seed=args.seed,
    )


def rnn_name(args):
    name = getattr(args, "rnn_name", None)
    if name in (None, "", "None", "none", "null"):
        return None
    return name


def load_policy(env, checkpoint, args):
    torch.manual_seed(args.seed)
    policy_cls = getattr(ocean_torch, getattr(args, "policy_name", "Quadrotor"))
    base_policy = policy_cls(
        env,
        hidden_size=args.hidden_size,
        action_std=args.action_std,
    )
    name = rnn_name(args)
    if name is None:
        policy = base_policy
    else:
        rnn_cls = getattr(ocean_torch, name)
        policy = rnn_cls(
            env,
            base_policy,
            input_size=args.rnn_input_size,
            hidden_size=args.rnn_hidden_size,
        )
    policy = policy.to(args.device)
    policy.eval()

    if checkpoint is None:
        return policy

    state_dict = torch.load(checkpoint, map_location=args.device)
    if "optimizer_state_dict" in state_dict:
        raise ValueError(
            f"{checkpoint} looks like trainer_state.pt. Pass a model_*.pt checkpoint."
        )

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    return policy


def initial_policy_state(args):
    if rnn_name(args) is None:
        return {}
    return {"lstm_h": None, "lstm_c": None}


def reset_done_policy_state(state, dones):
    if not state or state.get("lstm_h") is None:
        return
    done = torch.as_tensor(dones, dtype=torch.bool, device=state["lstm_h"].device)
    if not torch.any(done):
        return
    state["lstm_h"][done] = 0
    state["lstm_c"][done] = 0


def policy_distribution_stats(policy):
    module = getattr(policy, "policy", policy)
    logstd = getattr(module, "decoder_logstd", None)
    if logstd is None:
        return {}

    std = torch.exp(logstd.detach()).cpu().reshape(-1).numpy()
    stats = {
        "policy_std_mean": float(std.mean()),
        "policy_std_min": float(std.min()),
        "policy_std_max": float(std.max()),
    }
    for idx, value in enumerate(std):
        stats[f"policy_std_{idx}"] = float(value)
    return stats


def policy_action(policy, obs, env, args, state, selection=None):
    selection = selection or getattr(args, "action_selection", "mean")
    with torch.no_grad():
        obs = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
        logits, _ = policy.forward_eval(obs, state)
        if isinstance(logits, torch.distributions.Normal):
            action = logits.sample() if selection == "sample" else logits.mean
        elif isinstance(logits, (list, tuple)):
            if selection == "sample":
                action = torch.stack(
                    [torch.distributions.Categorical(logits=x).sample() for x in logits],
                    dim=-1,
                )
            else:
                action = torch.stack([x.argmax(dim=-1) for x in logits], dim=-1)
        else:
            if selection == "sample":
                action = torch.distributions.Categorical(logits=logits).sample()
            else:
                action = logits.argmax(dim=-1)

    action = action.cpu().numpy()
    return np.clip(
        action,
        env.single_action_space.low,
        env.single_action_space.high,
    ).astype(np.float32)


def deterministic_action(policy, obs, env, args, state):
    return policy_action(policy, obs, env, args, state, selection="mean")


def zero_action(env):
    return np.zeros((env.num_agents, env.single_action_space.shape[0]), dtype=np.float32)


def ray_reactive_action(obs, env, args, step):
    action = np.zeros((env.num_agents, env.single_action_space.shape[0]), dtype=np.float32)
    if args.action_mode not in ("velocity", "vel", 1) or args.num_rays <= 0:
        return action

    obs = np.asarray(obs, dtype=np.float32)
    rays = obs[:, RAY_OBS_START:RAY_OBS_START + args.num_rays]
    angles = np.arange(args.num_rays, dtype=np.float32) * (2.0 * np.pi / args.num_rays)
    ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    desired = np.zeros((obs.shape[0], 2), dtype=np.float32)
    desired[:, 0] = 0.35
    desired[:, 1] = 0.15 * np.sin(0.09 * step)

    distances = rays * args.ray_length
    avoid_radius = min(max(1.25, 0.35 * args.ray_length), args.ray_length)
    weights = np.clip((avoid_radius - distances) / avoid_radius, 0.0, 1.0) ** 2
    desired -= weights @ ray_dirs * 1.8

    norm = np.linalg.norm(desired, axis=1, keepdims=True)
    desired = np.where(norm > 1.0, desired / np.maximum(norm, 1e-6), desired)
    action[:, :2] = desired
    action[:, 2] = np.clip(2.0 * obs[:, TARGET_OBS_START + 2], -0.3, 0.3)
    action[:, 3] = 0.25
    return np.clip(action, env.single_action_space.low, env.single_action_space.high)


def target_offset_scale(args):
    if args.task_mode in ("indoor_nav", "nav", "indoor_route", "route", 2, 4) and args.room_size > 0.0:
        return max(2.0 * args.room_size, MIN_OFFSET_OBS_SCALE)

    return max(args.spawn_radius, args.hover_radius, MIN_OFFSET_OBS_SCALE)


def route_target_obs_mode(args):
    value = getattr(args, "route_target_obs_mode", "legacy")
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("drone", "native_drone", "puffer_drone", "drone_hover"):
            return "drone"
        if value in ("body", "body_frame", "local"):
            return "body"
        if value in ("minimal", "body_minimal", "unit"):
            return "minimal"
        return "legacy"
    if int(value) == 3:
        return "drone"
    if int(value) == 1:
        return "body"
    if int(value) == 2:
        return "minimal"
    return "legacy"


def body_to_world(quat, vec):
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    vx = vec[:, 0]
    vy = vec[:, 1]
    vz = vec[:, 2]

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    out = np.empty_like(vec)
    out[:, 0] = vx + w * tx + (y * tz - z * ty)
    out[:, 1] = vy + w * ty + (z * tx - x * tz)
    out[:, 2] = vz + w * tz + (x * ty - y * tx)
    return out


def world_to_body(quat, vec):
    conj = quat.copy()
    conj[:, 1:4] *= -1.0
    return body_to_world(conj, vec)


def target_offsets(obs, quat, args):
    mode = route_target_obs_mode(args)
    scale = target_offset_scale(args)
    ceiling = max(args.ceiling_height, MIN_OFFSET_OBS_SCALE)

    if mode == "body":
        body_offset = np.empty((obs.shape[0], 3), dtype=np.float32)
        body_offset[:, 0:2] = obs[:, TARGET_OBS_START:TARGET_OBS_START + 2] * scale
        body_offset[:, 2] = obs[:, TARGET_OBS_START + 2] * ceiling
        world_offset = body_to_world(quat, body_offset)
        if obs.shape[1] > TARGET_OBS_START + 11:
            world_offset[:, 2] = 0.5 * (
                obs[:, TARGET_OBS_START + 10] -
                obs[:, TARGET_OBS_START + 11]
            ) * ceiling
        return world_offset, body_offset

    if mode == "minimal":
        dist = obs[:, TARGET_OBS_START + 3] * scale
        body_offset = obs[:, TARGET_OBS_START:TARGET_OBS_START + 3] * dist[:, None]
        body_offset[:, 2] = obs[:, TARGET_OBS_START + 4] * ceiling
        return body_to_world(quat, body_offset), body_offset

    world_offset = np.empty((obs.shape[0], 3), dtype=np.float32)
    world_offset[:, 0] = obs[:, TARGET_OBS_START] * 30.0
    world_offset[:, 1] = obs[:, TARGET_OBS_START + 1] * 30.0
    world_offset[:, 2] = obs[:, TARGET_OBS_START + 2] * 10.0
    return world_offset, world_to_body(quat, world_offset)


def target_velocity(obs, quat, args):
    target_offset, _ = target_offsets(obs, quat, args)
    return np.clip(
        args.pd_kp_pos * target_offset,
        -args.pd_max_vel,
        args.pd_max_vel,
    )


def pd_velocity_action(obs, env, args):
    obs = np.asarray(obs, dtype=np.float32)
    quat = obs[:, 6:10]
    offset_world, offset_body = target_offsets(obs, quat, args)
    vel_des_world = np.clip(
        args.pd_kp_pos * offset_world,
        -args.pd_max_vel,
        args.pd_max_vel,
    )
    vel_des_body = np.clip(
        args.pd_kp_pos * offset_body,
        -args.pd_max_vel,
        args.pd_max_vel,
    )

    action = np.zeros((env.num_agents, env.single_action_space.shape[0]), dtype=np.float32)
    action[:, 0] = vel_des_body[:, 0] / max(args.max_vel, 1e-6)
    action[:, 1] = vel_des_body[:, 1] / max(args.max_vel, 1e-6)
    action[:, 2] = vel_des_world[:, 2] / max(args.max_z_vel, 1e-6)
    return np.clip(action, env.single_action_space.low, env.single_action_space.high)


def ray_pd_action(obs, env, args):
    action = pd_hover_action(obs, env, args)
    if args.action_mode not in ("velocity", "vel", 1) or args.num_rays <= 0:
        return action

    obs = np.asarray(obs, dtype=np.float32)
    rays = obs[:, RAY_OBS_START:RAY_OBS_START + args.num_rays]
    angles = np.arange(args.num_rays, dtype=np.float32) * (2.0 * np.pi / args.num_rays)
    ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    distances = rays * args.ray_length
    avoid_radius = min(max(args.pd_avoid_radius, 0.05), args.ray_length)
    weights = np.clip((avoid_radius - distances) / avoid_radius, 0.0, 1.0) ** 2
    action[:, :2] -= args.pd_avoid_gain * (weights @ ray_dirs)

    xy_norm = np.linalg.norm(action[:, :2], axis=1, keepdims=True)
    action[:, :2] = np.where(
        xy_norm > 1.0,
        action[:, :2] / np.maximum(xy_norm, 1e-6),
        action[:, :2],
    )
    return np.clip(action, env.single_action_space.low, env.single_action_space.high)


def rate_action_from_acceleration(acc_des, quat, omega, args):
    acc_des = acc_des.copy()
    acc_des[:, 0:2] = np.clip(
        acc_des[:, 0:2],
        -args.pd_max_tilt_acc,
        args.pd_max_tilt_acc,
    )
    z_extra = np.clip(acc_des[:, 2], -args.pd_max_z_acc, args.pd_max_z_acc)
    acc_des[:, 2] = GRAVITY + z_extra

    acc_norm = np.linalg.norm(acc_des, axis=1)
    acc_norm = np.maximum(acc_norm, 0.01)
    zb_des = acc_des / acc_norm[:, None]

    xc = np.zeros_like(zb_des)
    xc[:, 0] = 1.0
    yb_des = np.cross(zb_des, xc)
    yb_norm = np.linalg.norm(yb_des, axis=1)
    yb_norm = np.maximum(yb_norm, 1e-6)
    yb_des /= yb_norm[:, None]
    xb_des = np.cross(yb_des, zb_des)

    bx_body = np.zeros_like(zb_des)
    by_body = np.zeros_like(zb_des)
    bz_body = np.zeros_like(zb_des)
    bx_body[:, 0] = 1.0
    by_body[:, 1] = 1.0
    bz_body[:, 2] = 1.0

    xb_cur = body_to_world(quat, bx_body)
    yb_cur = body_to_world(quat, by_body)
    zb_cur = body_to_world(quat, bz_body)

    e_r = np.empty_like(zb_des)
    e_r[:, 0] = 0.5 * (
        np.sum(yb_cur * zb_des, axis=1) - np.sum(zb_cur * yb_des, axis=1)
    )
    e_r[:, 1] = 0.5 * (
        np.sum(zb_cur * xb_des, axis=1) - np.sum(xb_cur * zb_des, axis=1)
    )
    e_r[:, 2] = 0.5 * (
        np.sum(xb_cur * yb_des, axis=1) - np.sum(yb_cur * xb_des, axis=1)
    )

    return acc_norm, -args.pd_kp_att * e_r - args.pd_kd_att * omega


def pd_hover_action(obs, env, args):
    if args.action_mode in ("velocity", "vel", 1):
        return pd_velocity_action(obs, env, args)

    obs = np.asarray(obs, dtype=np.float32)
    velocity = obs[:, 0:3] * VEL_OBS_SCALE
    quat = obs[:, 6:10]
    omega_scale = max(args.max_rate, 1.0)
    omega = obs[:, 10:13] * omega_scale

    vel_des = target_velocity(obs, quat, args)
    acc_des = args.pd_kp_vel * (vel_des - velocity)
    acc_norm, rate_des = rate_action_from_acceleration(acc_des, quat, omega, args)

    action = np.zeros((env.num_agents, env.single_action_space.shape[0]), dtype=np.float32)
    action[:, 0] = acc_norm / GRAVITY - 1.0
    action[:, 1:4] = rate_des / args.max_rate
    return np.clip(action, env.single_action_space.low, env.single_action_space.high)


def update_metrics(total, info):
    n = float(info.get("n", 0.0))
    if n <= 0:
        return

    total["n"] += n
    for key, value in info.items():
        if key == "n":
            continue
        total[key] = total.get(key, 0.0) + float(value) * n


def finalize_metrics(total):
    n = total.pop("n", 0.0)
    if n <= 0:
        return {"n": 0.0}

    metrics = {key: value / n for key, value in total.items()}
    metrics["n"] = n
    return metrics


def evaluate(label, checkpoint, args):
    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    policy = None if checkpoint in ("zero", "pd", "ray", "raypd") else load_policy(env, checkpoint, args)
    policy_stats = {} if policy is None else policy_distribution_stats(policy)
    state = initial_policy_state(args)
    total = {"n": 0.0}
    target_episodes = args.episodes * args.num_envs
    max_steps = args.max_steps * args.episodes * 4

    for _ in range(max_steps):
        if args.render:
            env.render()

        action_obs = obs
        if args.mask_rays and args.num_rays > 0:
            action_obs = obs.copy()
            action_obs[:, RAY_OBS_START:RAY_OBS_START + args.num_rays] = 1.0
        perception_obs_size = route_perception_obs_size(
            getattr(args, "route_perception_obs", 0)
        )
        if getattr(args, "mask_route_perception_obs", False) and perception_obs_size:
            if action_obs is obs:
                action_obs = obs.copy()
            start = RAY_OBS_START + args.num_rays
            if args.route_wall_obs:
                start += WALL_OBS_SIZE
            if getattr(args, "route_extra_obs", 0):
                start += ROUTE_EXTRA_OBS_SIZE
            action_obs[:, start:start + perception_obs_size] = 0.0

        if checkpoint == "zero":
            actions = zero_action(env)
        elif checkpoint == "ray":
            actions = ray_reactive_action(action_obs, env, args, _)
        elif checkpoint == "pd":
            actions = pd_hover_action(action_obs, env, args)
        elif checkpoint == "raypd":
            actions = ray_pd_action(action_obs, env, args)
        else:
            actions = policy_action(policy, action_obs, env, args, state)

        obs, _, terminals, truncations, infos = env.step(actions)
        reset_done_policy_state(state, np.logical_or(terminals, truncations))
        for info in infos:
            update_metrics(total, info)

        if total["n"] >= target_episodes:
            break

    metrics = finalize_metrics(total)
    metrics["label"] = label
    metrics["action_selection"] = args.action_selection
    metrics.update(policy_stats)
    env.close()
    return metrics


def print_table(results):
    fields = [
        "label",
        "n",
        "perf",
        "success",
        "crashed",
        "avg_dist",
        "avg_vel",
        "avg_tilt",
        "max_dist",
        "final_dist",
        "final_z_error",
        "final_vel",
        "settle_time",
        "settled",
        "hover_rate",
        "timeout",
        "obstacle_hit",
        "wall_hit",
        "ground_hit",
        "min_clearance",
        "path_length",
        "path_efficiency",
        "action_saturation",
        "action_delta",
        "action_selection",
        "policy_std_mean",
        "policy_std_min",
        "policy_std_max",
        "mass_scale",
        "thrust_scale",
        "drag_scale",
        "motor_tau_scale",
        "coverage",
        "explored_cells",
        "new_cell_rate",
        "route_goals",
        "score",
    ]
    widths = {field: len(field) for field in fields}
    rows = []
    for result in results:
        row = {}
        for field in fields:
            value = result.get(field, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            row[field] = str(value)
            widths[field] = max(widths[field], len(row[field]))
        rows.append(row)

    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        print("  ".join(row[field].ljust(widths[field]) for field in fields))


def parse_args():
    parser = argparse.ArgumentParser(description="Quadrotor policy evaluation")
    parser.add_argument("checkpoints", nargs="*", help="model_*.pt checkpoints to compare")
    parser.add_argument("--include-zero", action="store_true", help="Evaluate zero-action baseline")
    parser.add_argument("--include-untrained", action="store_true", help="Evaluate deterministic untrained policy")
    parser.add_argument("--include-pd", action="store_true", help="Evaluate a deterministic PD baseline")
    parser.add_argument("--include-ray", action="store_true", help="Evaluate a simple ray-reactive diagnostic policy")
    parser.add_argument("--include-raypd", action="store_true", help="Evaluate target PD with ray-based obstacle repulsion")
    parser.add_argument("--mask-rays", action="store_true", help="Set ray observations to no-hit before policy inference")
    parser.add_argument("--mask-route-perception-obs", action="store_true", help="Zero target-aligned route perception observations before policy inference")
    parser.add_argument("--action-selection", choices=("mean", "sample"), default="mean", help="Use mean actions or sample from the policy distribution")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    parser.add_argument("--render", action="store_true", help="Render the first vector env while evaluating")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--max-rate", type=float, default=1.0)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--max-z-vel", type=float, default=0.75)
    parser.add_argument("--vel-ctrl-kp-vel", type=float, default=0.0)
    parser.add_argument("--vel-ctrl-ki-z", type=float, default=0.0)
    parser.add_argument("--vel-ctrl-int-limit-z", type=float, default=0.0)
    parser.add_argument("--vel-ctrl-max-z-acc", type=float, default=0.0)
    parser.add_argument("--spawn-radius", type=float, default=0.0)
    parser.add_argument("--spawn-height", type=float, default=1.5)
    parser.add_argument("--init-vel-radius", type=float, default=0.05)
    parser.add_argument("--init-omega-radius", type=float, default=0.0)
    parser.add_argument("--room-size", type=float, default=0.0)
    parser.add_argument("--ceiling-height", type=float, default=3.0)
    parser.add_argument("--num-rays", type=int, default=0)
    parser.add_argument("--ray-length", type=float, default=3.0)
    parser.add_argument("--hover-radius", type=float, default=0.2)
    parser.add_argument("--obstacle-mode", default="none")
    parser.add_argument("--obstacle-block-prob", type=float, default=0.5)
    parser.add_argument("--obs-noise", type=float, default=0.0)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--action-delay", type=int, default=0)
    parser.add_argument("--action-delay-min", type=int, default=-1)
    parser.add_argument("--action-delay-max", type=int, default=-1)
    parser.add_argument("--action-history-obs", type=int, default=0)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--inertia-scale", type=float, default=0.0)
    parser.add_argument("--thrust-scale", type=float, default=1.0)
    parser.add_argument("--torque-scale", type=float, default=0.0)
    parser.add_argument("--drag-scale", type=float, default=1.0)
    parser.add_argument("--motor-tau-scale", type=float, default=1.0)
    parser.add_argument("--mass-scale-min", type=float, default=0.0)
    parser.add_argument("--mass-scale-max", type=float, default=0.0)
    parser.add_argument("--inertia-scale-min", type=float, default=0.0)
    parser.add_argument("--inertia-scale-max", type=float, default=0.0)
    parser.add_argument("--thrust-scale-min", type=float, default=0.0)
    parser.add_argument("--thrust-scale-max", type=float, default=0.0)
    parser.add_argument("--torque-scale-min", type=float, default=0.0)
    parser.add_argument("--torque-scale-max", type=float, default=0.0)
    parser.add_argument("--drag-scale-min", type=float, default=0.0)
    parser.add_argument("--drag-scale-max", type=float, default=0.0)
    parser.add_argument("--motor-tau-scale-min", type=float, default=0.0)
    parser.add_argument("--motor-tau-scale-max", type=float, default=0.0)
    parser.add_argument("--explore-cell-size", type=float, default=0.25)
    parser.add_argument("--explore-goal-fraction", type=float, default=0.85)
    parser.add_argument("--route-goal-radius", type=float, default=0.3)
    parser.add_argument("--route-goal-count", type=int, default=3)
    parser.add_argument("--route-goal-count-min", type=int, default=0)
    parser.add_argument("--route-step-cost", type=float, default=0.004)
    parser.add_argument("--route-proximity-reward", type=float, default=0.02)
    parser.add_argument("--route-dist-scale", type=float, default=0.01)
    parser.add_argument("--route-progress-scale", type=float, default=25.0)
    parser.add_argument("--route-crash-reward", type=float, default=-24.0)
    parser.add_argument("--route-crash-goal-scale", type=float, default=4.0)
    parser.add_argument("--route-crash-dist-scale", type=float, default=1.5)
    parser.add_argument("--route-clearance-radius", type=float, default=0.45)
    parser.add_argument("--route-clearance-scale", type=float, default=0.35)
    parser.add_argument("--route-reference-clearance-radius", type=float, default=0.45)
    parser.add_argument("--route-reference-clearance-scale", type=float, default=0.0)
    parser.add_argument("--route-path-margin", type=float, default=0.0)
    parser.add_argument("--route-vel-scale", type=float, default=0.04)
    parser.add_argument("--route-arrival-radius", type=float, default=0.0)
    parser.add_argument("--route-arrival-bonus", type=float, default=0.0)
    parser.add_argument("--route-arrival-vel-scale", type=float, default=0.0)
    parser.add_argument("--route-arrival-toward-scale", type=float, default=0.0)
    parser.add_argument("--route-target-wall-margin", type=float, default=0.0)
    parser.add_argument("--route-goal-speed", type=float, default=2.0)
    parser.add_argument("--route-goal-bonus", type=float, default=1.0)
    parser.add_argument("--route-reach-bonus", type=float, default=5.0)
    parser.add_argument("--route-target-z-min", type=float, default=0.0)
    parser.add_argument("--route-target-z-max", type=float, default=0.0)
    parser.add_argument("--route-reward-mode", default="dense")
    parser.add_argument("--route-target-mode", default="random")
    parser.add_argument("--route-blocked-target-prob", type=float, default=1.0)
    parser.add_argument("--route-walls", type=int, default=1)
    parser.add_argument("--route-direct-obs", type=int, default=0)
    parser.add_argument("--route-extra-obs", type=int, default=0)
    parser.add_argument("--route-wall-obs", type=int, default=0)
    parser.add_argument("--route-perception-obs", type=int, default=0)
    parser.add_argument("--route-target-obs-mode", default="legacy")
    parser.add_argument("--task-mode", default="hold_hover")
    parser.add_argument("--action-mode", default="rate")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--action-std", type=float, default=0.05)
    parser.add_argument("--policy-name", default="Quadrotor")
    parser.add_argument("--rnn-name", default=None)
    parser.add_argument("--rnn-input-size", type=int, default=128)
    parser.add_argument("--rnn-hidden-size", type=int, default=128)
    parser.add_argument("--pd-kp-pos", type=float, default=2.0)
    parser.add_argument("--pd-max-vel", type=float, default=3.0)
    parser.add_argument("--pd-kp-vel", type=float, default=6.0)
    parser.add_argument("--pd-max-tilt-acc", type=float, default=GRAVITY * 0.364)
    parser.add_argument("--pd-max-z-acc", type=float, default=GRAVITY * 0.5)
    parser.add_argument("--pd-kp-att", type=float, default=20.0)
    parser.add_argument("--pd-kd-att", type=float, default=5.0)
    parser.add_argument("--pd-avoid-radius", type=float, default=1.0)
    parser.add_argument("--pd-avoid-gain", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    results = []

    if args.include_zero or not args.checkpoints:
        results.append(evaluate("zero", "zero", args))

    if args.include_ray:
        results.append(evaluate("ray", "ray", args))

    if args.include_raypd:
        results.append(evaluate("raypd", "raypd", args))

    if args.include_untrained:
        results.append(evaluate("untrained", None, args))

    if args.include_pd:
        results.append(evaluate("pd", "pd", args))

    for checkpoint in args.checkpoints:
        path = Path(checkpoint)
        results.append(evaluate(f"{path.parent.name}/{path.name}", str(path), args))

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
