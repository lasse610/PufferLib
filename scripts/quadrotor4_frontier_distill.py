#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quadrotor4_frontier_teacher_probe import (  # noqa: E402
    bfs_path,
    distance_grid,
    nearest_cell,
    pose_from_obs,
    world_to_body,
)
from quadrotor4_maze_frontier_probe import build_world, precompute_visibility  # noqa: E402
from quadrotor4_privileged_distill import (  # noqa: E402
    accumulate_logs,
    clone_detached_state,
    create_policy,
    detach_state,
    finalize_logs,
    print_metrics,
    reset_done_state,
    step_vec,
    tensor_views,
)


OBSTACLE_MODES = {
    "center_block": 1,
    "doorway": 2,
    "corridor": 3,
    "maze": 4,
    "random_blocks": 5,
    "layout_mixed": 7,
    "procedural_indoor": 8,
    "procedural_indoor_p0": 9,
    "procedural_indoor_p1": 10,
    "procedural_indoor_p2": 11,
    "procedural_indoor_p3": 12,
    "procedural_indoor_hard": 13,
    "procedural_indoor_hard_p0": 14,
    "procedural_indoor_hard_p1": 15,
    "procedural_indoor_hard_p2": 16,
    "procedural_indoor_hard_p3": 17,
    "procedural_indoor_tight": 18,
    "procedural_indoor_tight_p0": 19,
    "procedural_indoor_tight_p1": 20,
    "procedural_indoor_tight_p2": 21,
    "procedural_indoor_tight_p3": 22,
    "procedural_house": 23,
    "procedural_house_p0": 24,
    "procedural_house_p1": 25,
    "procedural_house_p2": 26,
    "procedural_house_p3": 27,
    "procedural_house_multizone": 28,
    "procedural_house_multizone_p0": 29,
    "procedural_house_multizone_p1": 30,
    "procedural_house_multizone_p2": 31,
    "procedural_house_multizone_p3": 32,
}


def load_quadrotor_config():
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        from pufferlib.pufferl import load_config
        return load_config("quadrotor")
    finally:
        sys.argv = old_argv


def make_vec_config(parsed):
    args = load_quadrotor_config()
    args["reset_state"] = False
    args["vec"]["total_agents"] = parsed.total_agents
    args["vec"]["num_buffers"] = parsed.num_buffers
    args["vec"]["num_threads"] = parsed.num_threads
    args["train"]["horizon"] = parsed.seq_len

    env = args["env"]
    env.update({
        "seed": parsed.seed,
        "task_mode": 3,
        "action_mode": 1,
        "max_steps": parsed.max_steps,
        "action_repeat": parsed.action_repeat,
        "max_vel": parsed.max_vel,
        "max_z_vel": parsed.max_z_vel,
        "vel_ctrl_ki_z": parsed.vel_ctrl_ki_z,
        "spawn_radius": parsed.spawn_radius,
        "spawn_radius_min": parsed.spawn_radius_min,
        "spawn_height": parsed.spawn_height,
        "room_size": parsed.room_size,
        "ceiling_height": parsed.ceiling_height,
        "num_rays": parsed.num_rays,
        "ray_length": parsed.ray_length,
        "obstacle_mode": parsed.obstacle_mode,
        "explore_cell_size": parsed.cell_size,
        "explore_goal_fraction": parsed.explore_goal_fraction,
        "route_wall_obs": 1,
        "action_history_obs": parsed.action_history_obs,
    })
    return args


class FrontierTeacher:
    def __init__(self, parsed, num_agents):
        self.parsed = parsed
        world = build_world(parsed)
        boxes, coords, xx, yy, valid_cells, drone_cells = world
        poses, masks = precompute_visibility(parsed, boxes, xx, yy, valid_cells, drone_cells)
        self.boxes = boxes
        self.coords = coords
        self.xx = xx
        self.yy = yy
        self.valid_cells = valid_cells
        self.drone_cells = drone_cells
        self.poses = poses
        self.masks = masks
        self.pose_to_index = {pose: idx for idx, pose in enumerate(poses)}
        self.valid_count = max(int(valid_cells.sum()), 1)
        self.mask_flat = np.stack([mask.reshape(-1) for mask in masks], axis=0)
        self.dist_to_poses = self._precompute_pose_distances()
        self.reset(num_agents)

    def _precompute_pose_distances(self):
        distances = np.full((len(self.poses), len(self.poses)), -1, dtype=np.int16)
        for idx, pose in enumerate(self.poses):
            grid = distance_grid(pose, self.drone_cells)
            for other_idx, other in enumerate(self.poses):
                distances[idx, other_idx] = int(grid[other])
        return distances

    def reset(self, num_agents=None):
        if num_agents is not None:
            self.num_agents = num_agents
        self.seen = [np.zeros_like(self.valid_cells, dtype=bool) for _ in range(self.num_agents)]
        self.paths = [[] for _ in range(self.num_agents)]
        self.reached = np.zeros(self.num_agents, dtype=np.int32)
        self.stale = np.zeros(self.num_agents, dtype=np.int32)

    def reset_done(self, done):
        for idx in np.flatnonzero(done):
            self.seen[idx][:] = False
            self.paths[idx] = []
            self.reached[idx] = 0
            self.stale[idx] = 0

    def actions(self, obs):
        parsed = self.parsed
        xy, z = pose_from_obs(obs, parsed)
        quat = obs[:, 6:10]
        desired_world = np.zeros((obs.shape[0], 3), dtype=np.float32)

        for idx in range(obs.shape[0]):
            cell = nearest_cell(
                float(xy[idx, 0]),
                float(xy[idx, 1]),
                self.xx,
                self.yy,
                self.drone_cells,
                parsed.room_size,
                parsed.cell_size,
            )
            self.seen[idx] |= self.masks[self.pose_to_index[cell]]
            coverage = float(self.seen[idx].sum() / self.valid_count)
            if coverage >= parsed.explore_goal_fraction and parsed.hold_after_mirror_success:
                self.paths[idx] = []
                continue

            path = self.paths[idx]
            while path:
                target_xy = np.array([self.xx[path[0]], self.yy[path[0]]], dtype=np.float32)
                if np.linalg.norm(xy[idx] - target_xy) >= parsed.waypoint_radius:
                    break
                path.pop(0)
                self.reached[idx] += 1

            if not path:
                path = self.select_path(cell, self.seen[idx])
                if path and path[0] == cell:
                    path = path[1:]
                self.paths[idx] = path

            if not path:
                self.stale[idx] += 1
                continue

            target = np.array([self.xx[path[0]], self.yy[path[0]]], dtype=np.float32)
            delta = target - xy[idx]
            dist = float(np.linalg.norm(delta))
            if dist > 1e-6:
                speed = min(parsed.teacher_max_vel, parsed.kp_xy * dist)
                desired_world[idx, 0:2] = delta / dist * speed
            desired_world[idx, 2] = np.clip(
                parsed.kp_z * (parsed.spawn_height - float(z[idx, 0])),
                -parsed.max_z_vel,
                parsed.max_z_vel,
            )

        desired_body = world_to_body(quat, desired_world)
        actions = np.zeros((obs.shape[0], 4), dtype=np.float32)
        actions[:, 0] = np.clip(desired_body[:, 0] / parsed.max_vel, -1.0, 1.0)
        actions[:, 1] = np.clip(desired_body[:, 1] / parsed.max_vel, -1.0, 1.0)
        actions[:, 2] = np.clip(desired_world[:, 2] / parsed.max_z_vel, -1.0, 1.0)
        return actions

    def select_path(self, cell, seen):
        current_idx = self.pose_to_index[cell]
        unseen = np.logical_not(seen.reshape(-1))
        new_counts = np.logical_and(self.mask_flat, unseen).sum(axis=1)
        distances = self.dist_to_poses[current_idx]
        valid = np.logical_and(new_counts > 0, distances >= 0)
        if not valid.any():
            return []
        scores = np.full(len(self.poses), -np.inf, dtype=np.float32)
        scores[valid] = new_counts[valid] / np.maximum(distances[valid], 1)
        best_idx = int(np.argmax(scores))
        return bfs_path(cell, self.poses[best_idx], self.drone_cells)


def obs_to_numpy(obs):
    if obs.device.type == "cuda":
        return obs.detach().cpu().numpy().copy()
    return obs.detach().numpy().copy()


def teacher_action_tensor(teacher, obs, device):
    return torch.from_numpy(teacher.actions(obs_to_numpy(obs))).to(device=device)


def load_checkpoint(path, vec, parsed, device):
    data = torch.load(path, map_location=device, weights_only=False)
    policy_meta = data.get("policy", {})
    parsed.network = policy_meta.get("network", parsed.network)
    parsed.hidden_size = int(policy_meta.get("hidden_size", parsed.hidden_size))
    parsed.num_layers = int(policy_meta.get("num_layers", parsed.num_layers))
    parsed.action_std = float(policy_meta.get("action_std", parsed.action_std))
    input_size = int(data.get("input_size", vec.obs_size))
    policy = create_policy(input_size, vec.act_sizes, parsed, device)
    policy.load_state_dict(data["state_dict"])
    return policy, data


def run_eval(parsed, policy=None, checkpoint=None):
    from pufferlib import _C

    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    args = make_vec_config(parsed)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, _, terminals = tensor_views(vec, device)
    vec.reset()

    teacher = FrontierTeacher(parsed, vec.total_agents) if parsed.eval_policy == "teacher" else None
    if parsed.eval_policy == "student" and policy is None:
        if checkpoint is None:
            raise RuntimeError("student eval requires a checkpoint")
        policy, _ = load_checkpoint(checkpoint, vec, parsed, device)

    if policy is not None:
        policy.eval()
        state = policy.initial_state(vec.total_agents, device)
    else:
        state = None

    accum = None
    try:
        for step in range(1, parsed.max_eval_steps + 1):
            with torch.no_grad():
                if teacher is not None:
                    actions = teacher_action_tensor(teacher, obs, device)
                else:
                    logits, _, state = policy.forward_eval(obs.float(), state)
                    actions = logits.mean.clamp(-1.0, 1.0)

            step_vec(vec, actions, device)
            done = terminals > 0.5
            if teacher is not None and done.any():
                teacher.reset_done(done.detach().cpu().numpy())
            if state is not None:
                state = reset_done_state(state, done)
                state = detach_state(state)

            if step % parsed.log_interval == 0:
                accum = accumulate_logs(accum, vec.log())
                if finalize_logs(accum).get("n", 0.0) >= parsed.episodes:
                    break
        accum = accumulate_logs(accum, vec.log())
    finally:
        vec.close()

    label = parsed.eval_policy if checkpoint is None else checkpoint
    print(f"frontier_eval {label}")
    metrics = finalize_logs(accum)
    print_metrics(metrics)
    return metrics


def predict_sequence(policy, x, start_state, dones):
    preds = []
    state = start_state
    for t in range(x.shape[1]):
        logits, _, state = policy.forward_eval(x[:, t], state)
        preds.append(logits.mean)
        state = reset_done_state(state, dones[:, t])
    return torch.stack(preds, dim=1)


def run_train(parsed):
    from pufferlib import _C

    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    args = make_vec_config(parsed)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, _, terminals = tensor_views(vec, device)
    vec.reset()

    teacher = FrontierTeacher(parsed, vec.total_agents)
    policy = create_policy(vec.obs_size, vec.act_sizes, parsed, device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=parsed.learning_rate)
    rollout_state = policy.initial_state(vec.total_agents, device)

    last_loss = 0.0
    last_mask_frac = 1.0
    last_student_prob = 0.0
    try:
        for update in range(1, parsed.seq_updates + 1):
            if parsed.reset_each_seq:
                vec.reset()
                teacher.reset(vec.total_agents)
                rollout_state = policy.initial_state(vec.total_agents, device)
                active = torch.ones((vec.total_agents,), dtype=torch.bool, device=device)
            else:
                active = None

            start_state = clone_detached_state(rollout_state)
            inputs = []
            targets = []
            dones = []
            masks = []

            if update <= parsed.teacher_rollout_updates:
                student_prob = 0.0
            else:
                denom = max(1, parsed.seq_updates - parsed.teacher_rollout_updates)
                frac = (update - parsed.teacher_rollout_updates) / denom
                student_prob = min(parsed.max_student_rollout_prob, frac * parsed.max_student_rollout_prob)
            last_student_prob = student_prob

            for _ in range(parsed.seq_len):
                x_t = obs.float().detach().clone()
                y_t = teacher_action_tensor(teacher, obs, device).detach()
                inputs.append(x_t)
                targets.append(y_t)
                if active is not None:
                    masks.append(active.clone())

                with torch.no_grad():
                    logits, _, rollout_state = policy.forward_eval(x_t, rollout_state)
                    student_actions = logits.mean.detach().clamp(-1.0, 1.0)
                    if student_prob <= 0:
                        actions = y_t
                    else:
                        use_student = torch.rand((vec.total_agents, 1), device=device) < student_prob
                        actions = torch.where(use_student, student_actions, y_t)

                step_vec(vec, actions, device)
                done = terminals > 0.5
                dones.append(done.clone())
                if done.any():
                    teacher.reset_done(done.detach().cpu().numpy())
                    rollout_state = reset_done_state(rollout_state, done)
                rollout_state = detach_state(rollout_state)
                if active is not None:
                    active = torch.logical_and(active, torch.logical_not(done))

            x = torch.stack(inputs, dim=1)
            target = torch.stack(targets, dim=1)
            done_seq = torch.stack(dones, dim=1)
            pred = predict_sequence(policy, x, start_state, done_seq)
            per_action = (pred - target).pow(2)
            if masks:
                mask = torch.stack(masks, dim=1).unsqueeze(-1)
                denom = mask.sum() * per_action.shape[-1]
                if denom <= 0:
                    continue
                loss = (per_action * mask).sum() / denom
                last_mask_frac = float(mask.float().mean().detach().cpu())
            else:
                loss = F.mse_loss(pred, target)
                last_mask_frac = 1.0

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), parsed.max_grad_norm)
            optimizer.step()
            last_loss = float(loss.detach().cpu())

            if update % parsed.print_interval == 0 or update == parsed.seq_updates:
                logs = vec.log()
                print(
                    f"seq_update={update} loss={last_loss:.6g} "
                    f"mask_frac={last_mask_frac:.3f} "
                    f"student_prob={last_student_prob:.3f} "
                    f"success={float(logs.get('success', 0.0)):.4f} "
                    f"coverage={float(logs.get('coverage', 0.0)):.4f} "
                    f"obstacle={float(logs.get('obstacle_hit', 0.0)):.4f} "
                    f"timeout={float(logs.get('timeout', 0.0)):.4f}",
                    flush=True,
                )

        output = Path(parsed.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "input_size": vec.obs_size,
                "base_obs_size": vec.obs_size,
                "num_atns": vec.num_atns,
                "act_sizes": list(vec.act_sizes),
                "prev_action_size": 0,
                "config": {
                    "task_mode": 3,
                    "action_mode": 1,
                    "obstacle_mode": parsed.obstacle_mode,
                    "explore_goal_fraction": parsed.explore_goal_fraction,
                    "route_wall_obs": 1,
                    "action_history_obs": parsed.action_history_obs,
                    "num_rays": parsed.num_rays,
                    "ray_length": parsed.ray_length,
                    "train_mode": "frontier_teacher_sequence_distill",
                    "seq_len": parsed.seq_len,
                    "teacher_rollout_updates": parsed.teacher_rollout_updates,
                    "max_student_rollout_prob": parsed.max_student_rollout_prob,
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
        print(
            f"wrote {output} final_loss={last_loss:.6g} "
            f"student_prob={last_student_prob:.3f}",
            flush=True,
        )
    finally:
        vec.close()

    if parsed.eval_after:
        parsed.eval_policy = "student"
        return run_eval(parsed, policy=policy)
    return {}


def add_common(parser):
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--total-agents", type=int, default=512)
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--max-eval-steps", type=int, default=20000)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--max-vel", type=float, default=0.7)
    parser.add_argument("--teacher-max-vel", type=float, default=0.5)
    parser.add_argument("--max-z-vel", type=float, default=0.25)
    parser.add_argument("--vel-ctrl-ki-z", type=float, default=4.0)
    parser.add_argument("--spawn-radius", type=float, default=2.5)
    parser.add_argument("--spawn-radius-min", type=float, default=0.0)
    parser.add_argument("--spawn-height", type=float, default=1.5)
    parser.add_argument("--room-size", type=float, default=3.0)
    parser.add_argument("--ceiling-height", type=float, default=3.0)
    parser.add_argument("--num-rays", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=4.0)
    parser.add_argument("--action-history-obs", type=int, choices=(0, 1), default=1)
    parser.add_argument("--mode", choices=tuple(OBSTACLE_MODES), default="maze")
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--coverage-margin", type=float, default=0.30)
    parser.add_argument("--drone-radius", type=float, default=0.30)
    parser.add_argument("--explore-goal-fraction", type=float, default=0.85)
    parser.add_argument("--kp-xy", type=float, default=1.0)
    parser.add_argument("--kp-z", type=float, default=1.5)
    parser.add_argument("--waypoint-radius", type=float, default=0.16)
    parser.add_argument("--hold-after-mirror-success", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--network", choices=("MLP", "MinGRU", "GRU", "LSTM"), default="MinGRU")
    parser.add_argument("--action-std", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--log-interval", type=int, default=64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frontier-teacher distillation for deployable quadrotor exploration observations."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--output", required=True)
    train.add_argument("--seq-updates", type=int, default=400)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--teacher-rollout-updates", type=int, default=40)
    train.add_argument("--max-student-rollout-prob", type=float, default=0.5)
    train.add_argument("--print-interval", type=int, default=20)
    train.add_argument("--reset-each-seq", action="store_true")
    train.add_argument("--eval-after", action="store_true")

    teacher_eval = sub.add_parser("teacher-eval")
    add_common(teacher_eval)
    teacher_eval.set_defaults(eval_policy="teacher")

    eval_p = sub.add_parser("eval")
    add_common(eval_p)
    eval_p.add_argument("checkpoint")
    eval_p.add_argument("--eval-policy", choices=("student", "teacher"), default="student")

    parsed = parser.parse_args()
    parsed.obstacle_mode = OBSTACLE_MODES[parsed.mode]
    return parsed


def main():
    parsed = parse_args()
    if parsed.cmd == "train":
        run_train(parsed)
    elif parsed.cmd in ("teacher-eval", "eval"):
        checkpoint = getattr(parsed, "checkpoint", None)
        run_eval(parsed, checkpoint=checkpoint)
    else:
        raise SystemExit(f"unknown command {parsed.cmd}")


if __name__ == "__main__":
    main()
