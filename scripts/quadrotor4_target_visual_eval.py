#!/usr/bin/env python3
import argparse
import copy
import itertools
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quadrotor4_frontier_target_distill import (  # noqa: E402
    OBSTACLE_MODES,
    _barrier_steering_actions,
    _direction_cone_ray,
    _direction_ray,
    _slide_safety_actions,
    add_common,
    accumulate_logs,
    apply_eval_env_config_from_checkpoint,
    deployable_input,
    detach_state,
    finalize_logs,
    graph_candidate_target_from_logits,
    load_policy,
    make_vec_config,
    make_recovery_controller,
    reset_done_state,
    step_vec,
    target_labels,
    target_to_actions,
    tensor_views,
)


def _fmt(value):
    return f"{float(value):.4f}"


def _linear_safety_scale(clearance, stop, slow):
    if stop <= 0.0:
        return 1.0
    if slow > stop:
        return max(0.0, min(1.0, (clearance - stop) / (slow - stop)))
    return 1.0 if clearance >= stop else 0.0


def make_visual_args(parsed):
    run = copy.copy(parsed)
    run.obstacle_mode = OBSTACLE_MODES[run.mode]
    run.total_agents = max(1, int(run.total_agents))
    return run


def choose_device(parsed):
    if parsed.gpu:
        from pufferlib import _C

        if not _C.gpu:
            raise RuntimeError("This pufferlib._C build does not have GPU support")
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_log_near_wall_stall(
    *,
    step,
    episode_step,
    last_progress_step,
    last_log_step,
    obs,
    target,
    actions,
    parsed,
):
    if (
        parsed.wall_stall_debug_steps <= 0
        or parsed.num_rays <= 0
        or obs.shape[1] < 26 + parsed.num_rays
    ):
        return last_log_step, False

    since_progress = episode_step - last_progress_step
    if since_progress < parsed.wall_stall_debug_steps:
        return last_log_step, False
    if step - last_log_step < parsed.wall_stall_log_interval:
        return last_log_step, False

    sample = obs[0:1].detach()
    rays = sample[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    min_ray = rays.min(dim=1).values
    if float(min_ray[0].detach().cpu()) > parsed.wall_stall_clearance:
        return last_log_step, False

    target_sample = target[0:1].detach()
    actions_sample = actions[0:1].detach()
    target_dir = target_sample[:, 0:2].float().clamp(-1.0, 1.0)
    target_unit = target_dir / torch.linalg.vector_norm(
        target_dir, dim=1, keepdim=True
    ).clamp_min(1e-6)
    action_xy = actions_sample[:, 0:2].float()
    action_speed = torch.linalg.vector_norm(action_xy, dim=1)
    action_unit = action_xy / action_speed[:, None].clamp_min(1e-6)
    target_ray = _direction_ray(sample, target_unit, parsed)
    target_cone = _direction_cone_ray(sample, target_unit, parsed)
    action_ray = _direction_ray(sample, action_unit, parsed)
    action_cone = _direction_cone_ray(sample, action_unit, parsed)
    raw_actions = target_to_actions(target_sample, parsed, sample, use_safety=False)
    raw_xy = raw_actions[:, 0:2].float()
    raw_speed = torch.linalg.vector_norm(raw_xy, dim=1)
    raw_unit = raw_xy / raw_speed[:, None].clamp_min(1e-6)
    raw_ray = _direction_ray(sample, raw_unit, parsed)
    raw_cone = _direction_cone_ray(sample, raw_unit, parsed)
    barrier_actions = (
        _barrier_steering_actions(raw_actions, sample, raw_unit, raw_speed[:, None], parsed)
        if getattr(parsed, "safety_barrier_actions", False)
        else raw_actions
    )
    barrier_xy = barrier_actions[:, 0:2].float()
    barrier_speed = torch.linalg.vector_norm(barrier_xy, dim=1)
    barrier_unit = barrier_xy / barrier_speed[:, None].clamp_min(1e-6)
    barrier_ray = _direction_ray(sample, barrier_unit, parsed)
    barrier_cone = _direction_cone_ray(sample, barrier_unit, parsed)
    barrier_progress = (barrier_xy * raw_unit).sum(dim=1)
    barrier_turn_cos = (barrier_unit * raw_unit).sum(dim=1).clamp(-1.0, 1.0)
    barrier_turn_deg = torch.rad2deg(torch.acos(barrier_turn_cos))
    slide_actions = (
        _slide_safety_actions(barrier_actions, sample, parsed, raw_unit)
        if getattr(parsed, "safety_slide", False)
        else barrier_actions
    )
    slide_xy = slide_actions[:, 0:2].float()
    slide_speed = torch.linalg.vector_norm(slide_xy, dim=1)
    slide_unit = slide_xy / slide_speed[:, None].clamp_min(1e-6)
    slide_progress = (slide_xy * raw_unit).sum(dim=1)
    slide_turn_cos = (slide_unit * raw_unit).sum(dim=1).clamp(-1.0, 1.0)
    slide_turn_deg = torch.rad2deg(torch.acos(slide_turn_cos))
    action_progress = (action_xy * raw_unit).sum(dim=1)
    raw_turn_cos = (action_unit * raw_unit).sum(dim=1).clamp(-1.0, 1.0)
    raw_turn_deg = torch.rad2deg(torch.acos(raw_turn_cos))
    exists = target_sample[:, 3].float()
    onboard_dir = sample[:, 14:16].float().clamp(-1.0, 1.0)
    onboard_unit = onboard_dir / torch.linalg.vector_norm(
        onboard_dir, dim=1, keepdim=True
    ).clamp_min(1e-6)
    align = (target_unit * onboard_unit).sum(dim=1).clamp(-1.0, 1.0)
    angle_err = torch.rad2deg(torch.acos(align))
    vel_body = sample[:, 0:2].float()
    vel_speed = torch.linalg.vector_norm(vel_body, dim=1)
    base_speed = parsed.speed_fraction if float(exists[0].detach().cpu()) > parsed.exists_threshold else 0.0
    safety_scale = (
        float(action_speed[0].detach().cpu()) / max(base_speed, 1e-6)
        if base_speed > 0.0
        else 0.0
    )
    dist_scale = max(2.0 * parsed.room_size, 1e-6)
    cell_size = max(parsed.cell_size, 1e-6)
    target_dist_obs = float(target_sample[0, 2].detach().cpu())
    onboard_dist_obs = float(sample[0, 16].detach().cpu())
    target_dist_m = target_dist_obs * dist_scale
    onboard_dist_m = onboard_dist_obs * dist_scale
    raw_speed_value = float(raw_speed[0].detach().cpu())
    target_cone_value = float(target_cone[0].detach().cpu())
    target_cone_scale = _linear_safety_scale(
        target_cone_value,
        parsed.safety_cone_ray_stop,
        parsed.safety_slow_ray,
    )
    progress_ratio = (
        float(action_progress[0].detach().cpu()) / max(raw_speed_value, 1e-6)
        if raw_speed_value > 0.0
        else 0.0
    )
    speed_ratio = (
        float(action_speed[0].detach().cpu()) / max(raw_speed_value, 1e-6)
        if raw_speed_value > 0.0
        else 0.0
    )
    barrier_speed_value = float(barrier_speed[0].detach().cpu())
    barrier_progress_ratio = (
        float(barrier_progress[0].detach().cpu()) / max(raw_speed_value, 1e-6)
        if raw_speed_value > 0.0
        else 0.0
    )
    slide_progress_ratio = (
        float(slide_progress[0].detach().cpu()) / max(raw_speed_value, 1e-6)
        if raw_speed_value > 0.0
        else 0.0
    )
    print(
        "near_wall_stall "
        f"step={episode_step} since_progress={since_progress} "
        f"cov={float(sample[0, 17].detach().cpu()):.4f} "
        f"min_ray={float(min_ray[0].detach().cpu()):.3f} "
        f"front_ray={float(rays[0, 0].detach().cpu()):.3f} "
        f"target=({float(target_sample[0, 0].detach().cpu()):+.2f},"
        f"{float(target_sample[0, 1].detach().cpu()):+.2f}) "
        f"target_dist={target_dist_obs:.3f} "
        f"target_dist_m={target_dist_m:.3f} "
        f"target_cell={target_dist_m / cell_size:.2f} "
        f"exists={float(exists[0].detach().cpu()):.3f} "
        f"onboard=({float(onboard_dir[0, 0].detach().cpu()):+.2f},"
        f"{float(onboard_dir[0, 1].detach().cpu()):+.2f}) "
        f"onboard_dist={onboard_dist_obs:.3f} "
        f"onboard_dist_m={onboard_dist_m:.3f} "
        f"onboard_cell={onboard_dist_m / cell_size:.2f} "
        f"onboard_exists={float(sample[0, 18].detach().cpu()):.3f} "
        f"angle_err={float(angle_err[0].detach().cpu()):.1f} "
        f"target_ray={float(target_ray[0].detach().cpu()):.3f} "
        f"target_cone={target_cone_value:.3f} "
        f"target_cone_scale={target_cone_scale:.3f} "
        f"raw_speed={raw_speed_value:.3f} "
        f"raw_ray={float(raw_ray[0].detach().cpu()):.3f} "
        f"raw_cone={float(raw_cone[0].detach().cpu()):.3f} "
        f"barrier_speed={barrier_speed_value:.3f} "
        f"barrier_progress_ratio={barrier_progress_ratio:.3f} "
        f"barrier_turn_deg={float(barrier_turn_deg[0].detach().cpu()):.1f} "
        f"barrier_ray={float(barrier_ray[0].detach().cpu()):.3f} "
        f"barrier_cone={float(barrier_cone[0].detach().cpu()):.3f} "
        f"slide_speed={float(slide_speed[0].detach().cpu()):.3f} "
        f"slide_progress_ratio={slide_progress_ratio:.3f} "
        f"slide_turn_deg={float(slide_turn_deg[0].detach().cpu()):.1f} "
        f"action=({float(actions_sample[0, 0].detach().cpu()):+.2f},"
        f"{float(actions_sample[0, 1].detach().cpu()):+.2f}) "
        f"action_speed={float(action_speed[0].detach().cpu()):.3f} "
        f"speed_ratio={speed_ratio:.3f} "
        f"progress_ratio={progress_ratio:.3f} "
        f"raw_turn_deg={float(raw_turn_deg[0].detach().cpu()):.1f} "
        f"vel=({float(vel_body[0, 0].detach().cpu()):+.2f},"
        f"{float(vel_body[0, 1].detach().cpu()):+.2f}) "
        f"vel_speed={float(vel_speed[0].detach().cpu()):.3f} "
        f"action_ray={float(action_ray[0].detach().cpu()):.3f} "
        f"cone_ray={float(action_cone[0].detach().cpu()):.3f} "
        f"safety_scale={safety_scale:.3f}",
        flush=True,
    )
    return step, True


def run_visual(parsed):
    from pufferlib import _C

    parsed = make_visual_args(parsed)
    device = choose_device(parsed)
    if parsed.controller == "student" and not parsed.ignore_checkpoint_env_config:
        apply_eval_env_config_from_checkpoint(parsed.checkpoint, parsed)
    args = make_vec_config(parsed)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, rewards, terminals = tensor_views(vec, device)
    vec.reset()

    policy = None
    state = None
    if parsed.controller == "student":
        policy = load_policy(parsed.checkpoint, vec, parsed, device)
        policy.eval()
        state = policy.initial_state(vec.total_agents, device)

    accum = None
    completed = 0.0
    episode_step = 0
    best_coverage = 0.0
    last_progress_step = 0
    last_wall_stall_log_step = -10**9
    wall_stall_logs = 0
    action_controller = make_recovery_controller(parsed, device)

    def reset_visual_episode(reason):
        nonlocal episode_step, best_coverage, last_progress_step, state, completed
        print(
            f"visual_restart reason={reason} step={episode_step} "
            f"coverage={best_coverage:.4f}",
            flush=True,
        )
        vec.reset()
        episode_step = 0
        best_coverage = 0.0
        last_progress_step = 0
        if policy is not None:
            state = policy.initial_state(vec.total_agents, device)
        if action_controller is not None:
            action_controller.best_coverage = None
        if parsed.count_early_restarts:
            completed += 1.0

    try:
        steps = (
            itertools.count(1)
            if parsed.episodes == 0
            else range(1, parsed.max_eval_steps + 1)
        )
        for step in steps:
            if parsed.render and step % parsed.render_every == 0:
                vec.render(0)

            with torch.no_grad():
                if parsed.controller == "oracle":
                    target = target_labels(obs, parsed)
                else:
                    x = deployable_input(obs, parsed)
                    logits, _, state = policy.forward_eval(x, state)
                    target = (
                        graph_candidate_target_from_logits(logits.mean, obs, parsed)
                        if getattr(parsed, "predict_graph_candidate", False)
                        else logits.mean
                    )
                actions = (
                    action_controller.actions(target, obs)
                    if action_controller is not None
                    else target_to_actions(target, parsed, obs)
                )

            last_wall_stall_log_step, logged_wall_stall = maybe_log_near_wall_stall(
                step=step,
                episode_step=episode_step,
                last_progress_step=last_progress_step,
                last_log_step=last_wall_stall_log_step,
                obs=obs,
                target=target,
                actions=actions,
                parsed=parsed,
            )
            if logged_wall_stall:
                wall_stall_logs += 1
            step_vec(vec, actions, device)
            episode_step += 1
            done = terminals > 0.5
            if state is not None:
                state = reset_done_state(state, done)
                state = detach_state(state)
            if action_controller is not None:
                action_controller.reset_done(done)

            if done.any():
                logs = vec.log()
                counted = 0.0
                if logs:
                    n = float(logs.get("n", 0.0))
                    counted = n
                    completed += n
                    accum = accumulate_logs(accum, logs)
                    print(
                        "episodes={episodes:.0f} "
                        "success={success} coverage={coverage} "
                        "obstacle={obstacle} timeout={timeout} "
                        "new_cell_rate={new_cell_rate}".format(
                            episodes=completed,
                            success=_fmt(logs.get("success", 0.0)),
                            coverage=_fmt(logs.get("coverage", 0.0)),
                            obstacle=_fmt(logs.get("obstacle_hit", 0.0)),
                            timeout=_fmt(logs.get("timeout", 0.0)),
                            new_cell_rate=_fmt(logs.get("new_cell_rate", 0.0)),
                        ),
                        flush=True,
                    )
                if counted <= 0.0:
                    completed += float(done.sum().detach().cpu())
                episode_step = 0
                best_coverage = 0.0
                last_progress_step = 0
                if parsed.episodes > 0 and completed >= parsed.episodes:
                    break
                continue

            if parsed.restart_on_stall_steps > 0 and obs.shape[1] > 17:
                coverage = float(obs[0, 17].detach().cpu())
                if coverage > best_coverage + parsed.restart_on_stall_delta:
                    best_coverage = coverage
                    last_progress_step = episode_step
                elif episode_step - last_progress_step >= parsed.restart_on_stall_steps:
                    reset_visual_episode("coverage_stall")
                    if parsed.episodes > 0 and completed >= parsed.episodes:
                        break

        accum = accumulate_logs(accum, vec.log())
    finally:
        vec.close()

    metrics = finalize_logs(accum)
    if metrics:
        summary = (
            "summary "
            f"n={metrics.get('n', 0.0):.0f} "
            f"success={_fmt(metrics.get('success', 0.0))} "
            f"coverage={_fmt(metrics.get('coverage', 0.0))} "
            f"obstacle={_fmt(metrics.get('obstacle_hit', 0.0))} "
            f"timeout={_fmt(metrics.get('timeout', 0.0))} "
            f"new_cell_rate={_fmt(metrics.get('new_cell_rate', 0.0))}"
        )
        if "region_tracker_regions" in metrics:
            summary += (
                f" region_tracker_regions={_fmt(metrics.get('region_tracker_regions', 0.0))}"
                f" region_tracker_active_age={_fmt(metrics.get('region_tracker_active_age', 0.0))}"
                f" region_tracker_since_progress={_fmt(metrics.get('region_tracker_since_progress', 0.0))}"
                f" region_tracker_switch_rate={_fmt(metrics.get('region_tracker_switch_rate', 0.0))}"
                f" region_tracker_stale_rate={_fmt(metrics.get('region_tracker_stale_rate', 0.0))}"
            )
        if "frontier_graph_nodes" in metrics:
            summary += (
                f" frontier_graph_nodes={_fmt(metrics.get('frontier_graph_nodes', 0.0))}"
                f" frontier_graph_unknown_cells={_fmt(metrics.get('frontier_graph_unknown_cells', 0.0))}"
                f" frontier_graph_best_unknown_cells={_fmt(metrics.get('frontier_graph_best_unknown_cells', 0.0))}"
                f" frontier_graph_active_rate={_fmt(metrics.get('frontier_graph_active_rate', 0.0))}"
            )
        print(summary, flush=True)
    print(f"wall_stall_logs={wall_stall_logs}", flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual eval for frontier target-distill quadrotor checkpoints."
    )
    parser.add_argument("checkpoint", nargs="?", default="")
    parser.add_argument("--controller", choices=("student", "oracle"), default="student")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--restart-on-stall-steps", type=int, default=300)
    parser.add_argument("--restart-on-stall-delta", type=float, default=0.002)
    parser.add_argument("--count-early-restarts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wall-stall-debug-steps", type=int, default=90)
    parser.add_argument("--wall-stall-clearance", type=float, default=0.35)
    parser.add_argument("--wall-stall-log-interval", type=int, default=60)
    parser.add_argument("--gpu", action="store_true")
    add_common(parser)
    parser.set_defaults(
        episodes=0,
        total_agents=1,
        num_buffers=1,
        num_threads=1,
        max_steps=2048,
        max_eval_steps=100000,
        speed_fraction=0.35,
        spawn_radius_min=0.0,
        explore_goal_fraction=0.95,
        safety_cone_ray_stop=0.25,
        safety_cone_deg=60.0,
        safety_slow_ray=0.70,
    )
    parsed = parser.parse_args()
    if parsed.render_every < 1:
        parser.error("--render-every must be >= 1")
    if parsed.episodes < 0:
        parser.error("--episodes must be >= 0; use 0 to run visual eval until the window is closed")
    if not parsed.render and parsed.episodes == 0:
        parser.error("--episodes 0 is only valid with rendering enabled")
    if parsed.controller == "student" and not parsed.checkpoint:
        parser.error("checkpoint is required for --controller student")
    return parsed


def main():
    parsed = parse_args()
    run_visual(parsed)


if __name__ == "__main__":
    main()
