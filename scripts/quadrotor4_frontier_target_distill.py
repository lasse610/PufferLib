#!/usr/bin/env python3
import argparse
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quadrotor4_privileged_distill import (  # noqa: E402
    accumulate_logs,
    clone_detached_state,
    create_policy as create_default_policy,
    detach_state,
    finalize_logs,
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

FRONTIER_TARGET_MODES = {
    "nearest": 0,
    "gain": 1,
    "unexplored": 2,
    "onboard": 3,
    "region": 4,
}

BASE_DEPLOYABLE_OBS_SIZE = 60
FRONTIER_PATCH_SIZE = 16
FRONTIER_PATCH_START = BASE_DEPLOYABLE_OBS_SIZE
FRONTIER_PATCH_END = FRONTIER_PATCH_START + FRONTIER_PATCH_SIZE
FAR_MAP_START = 174
FAR_MAP_SIDE = 11
FAR_MAP_CHANNELS = 2
FAR_MAP_SIZE = FAR_MAP_CHANNELS * FAR_MAP_SIDE * FAR_MAP_SIDE
GRAPH_CANDIDATE_COUNT = 4
GRAPH_CANDIDATE_FEATURES = 10
GRAPH_CANDIDATE_OBS_SIZE = GRAPH_CANDIDATE_COUNT * GRAPH_CANDIDATE_FEATURES
DEPTH_CAMERA_DEFAULT_WIDTH = 16
DEPTH_CAMERA_DEFAULT_HEIGHT = 12
DEPTH_CAMERA_MAX_WIDTH = 20
DEPTH_CAMERA_MAX_HEIGHT = 15
DEPTH_CAMERA_STRIDE = DEPTH_CAMERA_MAX_WIDTH
DEPTH_CAMERA_OBS_SIZE = DEPTH_CAMERA_STRIDE * DEPTH_CAMERA_MAX_HEIGHT
DEPTH_CAMERA_START = BASE_DEPLOYABLE_OBS_SIZE
DEPTH_CAMERA_END = DEPTH_CAMERA_START + DEPTH_CAMERA_OBS_SIZE


class ResidualMapEncoder(torch.nn.Module):
    def __init__(self, obs_size, hidden_size, map_start=FAR_MAP_START, map_side=FAR_MAP_SIDE):
        super().__init__()
        self.encoder = torch.nn.Linear(obs_size, hidden_size)
        self.map_start = int(map_start)
        self.map_side = int(map_side)
        cells = self.map_side * self.map_side
        self.map_net = torch.nn.Sequential(
            torch.nn.Conv2d(FAR_MAP_CHANNELS, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * cells, hidden_size),
        )
        torch.nn.init.zeros_(self.map_net[-1].weight)
        torch.nn.init.zeros_(self.map_net[-1].bias)

    def forward(self, observations):
        flat = observations.view(observations.shape[0], -1).float()
        hidden = self.encoder(flat)
        end = self.map_start + FAR_MAP_CHANNELS * self.map_side * self.map_side
        if flat.shape[1] < end:
            return hidden
        map_obs = flat[:, self.map_start:end].reshape(
            flat.shape[0],
            FAR_MAP_CHANNELS,
            self.map_side,
            self.map_side,
        )
        return hidden + self.map_net(map_obs)


def create_policy(input_size, act_sizes, parsed, device):
    if not getattr(parsed, "map_sidecar_encoder", False):
        return create_default_policy(input_size, act_sizes, parsed, device)

    import pufferlib.models

    encoder = ResidualMapEncoder(
        input_size,
        parsed.hidden_size,
        getattr(parsed, "far_map_start", FAR_MAP_START),
        getattr(parsed, "far_map_side", FAR_MAP_SIDE),
    )
    decoder = pufferlib.models.DefaultDecoder(act_sizes, parsed.hidden_size)
    network_cls = getattr(pufferlib.models, parsed.network)
    network = network_cls(
        hidden_size=parsed.hidden_size,
        num_layers=parsed.num_layers,
    )
    if hasattr(decoder, "decoder_logstd"):
        decoder.decoder_logstd.data.fill_(float(math.log(parsed.action_std)))
    return pufferlib.models.Policy(encoder, decoder, network).to(device)


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

    args["env"].update({
        "seed": parsed.seed,
        "task_mode": 3,
        "action_mode": 1,
        "max_steps": parsed.max_steps,
        "action_repeat": parsed.action_repeat,
        "max_vel": parsed.max_vel,
        "max_z_vel": parsed.max_z_vel,
        "vel_ctrl_kp_vel": parsed.vel_ctrl_kp_vel,
        "vel_ctrl_ki_z": parsed.vel_ctrl_ki_z,
        "spawn_radius": parsed.spawn_radius,
        "spawn_radius_min": parsed.spawn_radius_min,
        "spawn_height": parsed.spawn_height,
        "room_size": parsed.room_size,
        "ceiling_height": parsed.ceiling_height,
        "num_rays": parsed.num_rays,
        "ray_length": parsed.ray_length,
        "camera_depth_obs": int(parsed.camera_depth_obs or parsed.deploy_camera_depth_obs),
        "camera_depth_width": parsed.camera_depth_width,
        "camera_depth_height": parsed.camera_depth_height,
        "camera_depth_fov_x_deg": parsed.camera_depth_fov_x_deg,
        "camera_depth_fov_y_deg": parsed.camera_depth_fov_y_deg,
        "obstacle_mode": parsed.obstacle_mode,
        "explore_cell_size": parsed.cell_size,
        "explore_goal_fraction": parsed.explore_goal_fraction,
        "explore_memory_obs": parsed.explore_memory_obs,
        "explore_frontier_target_mode": FRONTIER_TARGET_MODES[
            parsed.explore_frontier_target_mode
        ],
        "explore_reachable_coverage": int(parsed.explore_reachable_coverage),
        "explore_onboard_commit_steps": parsed.explore_onboard_commit_steps,
        "explore_onboard_path_lookahead": parsed.explore_onboard_path_lookahead,
        "explore_onboard_adaptive_lookahead": int(parsed.explore_onboard_adaptive_lookahead),
        "explore_onboard_adaptive_lookahead_min": parsed.explore_onboard_adaptive_lookahead_min,
        "explore_onboard_adaptive_lookahead_max": parsed.explore_onboard_adaptive_lookahead_max,
        "explore_onboard_adaptive_clearance": parsed.explore_onboard_adaptive_clearance,
        "explore_onboard_adaptive_turn_dot": parsed.explore_onboard_adaptive_turn_dot,
        "explore_onboard_commit_min_clearance": parsed.explore_onboard_commit_min_clearance,
        "explore_onboard_stall_viewpoint_ticks": parsed.explore_onboard_stall_viewpoint_ticks,
        "explore_onboard_stall_viewpoint_max_depth": parsed.explore_onboard_stall_viewpoint_max_depth,
        "explore_onboard_approach_viewpoints": int(parsed.explore_onboard_approach_viewpoints),
        "explore_onboard_approach_max_depth": parsed.explore_onboard_approach_max_depth,
        "explore_onboard_emit_objective_target": int(parsed.explore_onboard_emit_objective_target),
        "explore_onboard_objective_obs": int(parsed.explore_onboard_objective_obs),
        "explore_onboard_region_commit": int(parsed.explore_onboard_region_commit),
        "explore_onboard_region_commit_stall_ticks": parsed.explore_onboard_region_commit_stall_ticks,
        "explore_onboard_persistent_regions": int(parsed.explore_onboard_persistent_regions),
        "explore_onboard_region_tracker": int(parsed.explore_onboard_region_tracker),
        "explore_onboard_clearance_passable": int(parsed.explore_onboard_clearance_passable),
        "explore_onboard_passable_radius": parsed.explore_onboard_passable_radius,
        "explore_onboard_passage_quality": int(parsed.explore_onboard_passage_quality),
        "explore_onboard_passage_weight": parsed.explore_onboard_passage_weight,
        "explore_onboard_clearance_cone_deg": parsed.explore_onboard_clearance_cone_deg,
        "explore_onboard_budget_frontier": int(parsed.explore_onboard_budget_frontier),
        "explore_onboard_budget_gain_weight": parsed.explore_onboard_budget_gain_weight,
        "explore_onboard_budget_cluster_weight": parsed.explore_onboard_budget_cluster_weight,
        "explore_onboard_budget_cell_weight": parsed.explore_onboard_budget_cell_weight,
        "explore_onboard_budget_depth_bias": parsed.explore_onboard_budget_depth_bias,
        "explore_onboard_budget_depth_weight": parsed.explore_onboard_budget_depth_weight,
        "explore_onboard_optimistic_unknown": int(parsed.explore_onboard_optimistic_unknown),
        "explore_onboard_frontier_graph": int(parsed.explore_onboard_frontier_graph),
        "explore_onboard_graph_candidate_obs": int(
            parsed.explore_onboard_graph_candidate_obs
            or parsed.deploy_graph_candidate_obs
        ),
        "explore_onboard_graph_gain_weight": parsed.explore_onboard_graph_gain_weight,
        "explore_onboard_graph_cluster_weight": parsed.explore_onboard_graph_cluster_weight,
        "explore_onboard_graph_depth_weight": parsed.explore_onboard_graph_depth_weight,
        "explore_onboard_graph_active_bonus": parsed.explore_onboard_graph_active_bonus,
        "explore_onboard_graph_stale_penalty": parsed.explore_onboard_graph_stale_penalty,
        "explore_onboard_graph_failure_penalty": parsed.explore_onboard_graph_failure_penalty,
        "explore_onboard_graph_attempt_penalty": parsed.explore_onboard_graph_attempt_penalty,
        "explore_onboard_graph_gain_cap": parsed.explore_onboard_graph_gain_cap,
        "explore_frontier_progress_scale": 0.0,
        "route_wall_obs": 1,
        "action_history_obs": 1,
    })
    return args


def deployable_input(obs, parsed=None):
    x = obs.float().clone()
    coverage = x[:, 17:18].clone() if x.shape[1] > 17 else None
    target_obs = x[:, 14:26].clone() if x.shape[1] >= 26 else None
    graph_candidate_obs = None
    if getattr(parsed, "deploy_graph_candidate_obs", False) and x.shape[1] >= GRAPH_CANDIDATE_OBS_SIZE:
        graph_candidate_obs = x[:, -GRAPH_CANDIDATE_OBS_SIZE:].clone()
    camera_depth_obs = None
    if getattr(parsed, "deploy_camera_depth_obs", False) and x.shape[1] >= DEPTH_CAMERA_END:
        camera_depth_obs = x[:, DEPTH_CAMERA_START:DEPTH_CAMERA_END].clone()
    x[:, 14:26] = 0.0
    if getattr(parsed, "deploy_onboard_target_obs", False):
        mode = getattr(parsed, "explore_frontier_target_mode", "")
        if mode != "onboard":
            raise ValueError("--deploy-onboard-target-obs requires --explore-frontier-target-mode onboard")
        if target_obs is not None:
            x[:, 14:26] = target_obs
    elif coverage is not None and getattr(parsed, "deploy_coverage_obs", False):
        x[:, 17:18] = coverage
    if x.shape[1] > BASE_DEPLOYABLE_OBS_SIZE and not getattr(parsed, "deploy_patch_obs", False):
        x[:, BASE_DEPLOYABLE_OBS_SIZE:] = 0.0
        if graph_candidate_obs is not None:
            x[:, -GRAPH_CANDIDATE_OBS_SIZE:] = graph_candidate_obs
        if camera_depth_obs is not None:
            x[:, DEPTH_CAMERA_START:DEPTH_CAMERA_END] = camera_depth_obs
    return x


def target_labels(obs, parsed=None):
    predict_speed = bool(getattr(parsed, "predict_speed_scale", False))
    predict_frontier_sectors = bool(getattr(parsed, "predict_frontier_sectors", False))
    predict_frontier_vector = bool(getattr(parsed, "predict_frontier_vector", False))
    predict_frontier_patch = bool(getattr(parsed, "predict_frontier_patch", False))
    predict_recovery_gate = bool(getattr(parsed, "predict_recovery_gate", False))
    if predict_speed and (predict_frontier_sectors or predict_frontier_vector or predict_frontier_patch):
        raise ValueError("predict_speed_scale cannot be combined with frontier auxiliaries")
    if predict_frontier_vector and not predict_frontier_sectors:
        raise ValueError("predict_frontier_vector requires predict_frontier_sectors")
    if predict_recovery_gate and not predict_frontier_vector:
        raise ValueError("predict_recovery_gate requires predict_frontier_vector")
    if predict_recovery_gate and predict_frontier_patch:
        raise ValueError("predict_recovery_gate cannot be combined with predict_frontier_patch")
    if predict_frontier_patch and not predict_frontier_vector:
        raise ValueError("predict_frontier_patch requires predict_frontier_vector")
    label_size = (
        11 + FRONTIER_PATCH_SIZE
        if predict_frontier_patch
        else (
            12
            if predict_recovery_gate
            else (11 if predict_frontier_vector else (8 if predict_frontier_sectors else (5 if predict_speed else 4)))
        )
    )
    labels = torch.zeros((obs.shape[0], label_size), dtype=torch.float32, device=obs.device)
    labels[:, 0:2] = obs[:, 14:16].float()
    labels[:, 2] = obs[:, 16].float()
    labels[:, 3] = obs[:, 18].float()
    if parsed is not None and getattr(parsed, "absent_fallback", "none") == "clear_ray":
        apply_clear_ray_absent_fallback(labels, obs, parsed)
    if predict_speed:
        direction = labels[:, 0:2].clamp(-1.0, 1.0)
        unit = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
        labels[:, 4] = _combined_safety_scale(obs, unit, parsed) * labels[:, 3].clamp(0.0, 1.0)
    if predict_frontier_sectors:
        labels[:, 4:8] = obs[:, 22:26].float().clamp(0.0, 1.0)
        apply_sector_fallback_target(labels, obs, parsed)
    if predict_frontier_vector:
        labels[:, 8:10] = obs[:, 19:21].float().clamp(-1.0, 1.0)
        labels[:, 10] = obs[:, 21].float().clamp(0.0, 1.0)
        apply_vector_fallback_target(labels, obs, parsed)
    if predict_frontier_patch and obs.shape[1] >= FRONTIER_PATCH_END:
        labels[:, 11:11 + FRONTIER_PATCH_SIZE] = (
            obs[:, FRONTIER_PATCH_START:FRONTIER_PATCH_END].float().clamp(0.0, 1.0)
        )
    return labels


def graph_candidate_features(obs):
    if obs.shape[1] < GRAPH_CANDIDATE_OBS_SIZE:
        raise ValueError(
            "graph candidate prediction requires graph-candidate observation tail "
            f"({GRAPH_CANDIDATE_OBS_SIZE} floats)"
        )
    return obs[:, -GRAPH_CANDIDATE_OBS_SIZE:].float().reshape(
        obs.shape[0],
        GRAPH_CANDIDATE_COUNT,
        GRAPH_CANDIDATE_FEATURES,
    )


def graph_candidate_target_from_index(obs, index, valid):
    cand = graph_candidate_features(obs)
    row = torch.arange(obs.shape[0], dtype=torch.long, device=obs.device)
    index = index.long().clamp(0, GRAPH_CANDIDATE_COUNT - 1)
    chosen = cand[row, index]
    exists = (chosen[:, 9:10] > 0.5).float() * valid.float().reshape(-1, 1)
    target = torch.zeros((obs.shape[0], 4), dtype=torch.float32, device=obs.device)
    target[:, 0:2] = chosen[:, 0:2].clamp(-1.0, 1.0)
    target[:, 2:3] = chosen[:, 4:5].clamp(0.0, 1.0)
    target[:, 3:4] = exists
    return target


def graph_candidate_target_from_logits(logits, obs, parsed):
    cand = graph_candidate_features(obs)
    exists = cand[:, :, 9] > 0.5
    scores = logits[:, :GRAPH_CANDIDATE_COUNT].float()
    scores = torch.where(exists, scores, torch.full_like(scores, -1.0e9))
    index = scores.argmax(dim=1)
    valid = exists.any(dim=1).float()
    return graph_candidate_target_from_index(obs, index, valid)


def graph_candidate_labels(obs, oracle_target, parsed):
    cand = graph_candidate_features(obs)
    exists = cand[:, :, 9] > 0.5
    any_exists = exists.any(dim=1)
    if getattr(parsed, "graph_candidate_label_source", "privileged") == "selected":
        selected_score = cand[:, :, 8].clamp(0.0, 1.0)
        selected_score = torch.where(
            exists,
            selected_score,
            torch.full_like(selected_score, -1.0e9),
        )
        index = selected_score.argmax(dim=1)
        selected = selected_score.gather(1, index[:, None]).squeeze(1)
        return index, (any_exists & (selected > 0.5)).float()

    oracle_exists = oracle_target[:, 3] > parsed.exists_threshold

    cand_dir = cand[:, :, 2:4].clamp(-1.0, 1.0)
    cand_dir = cand_dir / torch.linalg.vector_norm(
        cand_dir, dim=2, keepdim=True
    ).clamp_min(1e-6)
    oracle_dir = oracle_target[:, 0:2].clamp(-1.0, 1.0)
    oracle_dir = oracle_dir / torch.linalg.vector_norm(
        oracle_dir, dim=1, keepdim=True
    ).clamp_min(1e-6)
    align = (cand_dir * oracle_dir[:, None, :]).sum(dim=2).clamp(-1.0, 1.0)

    obs_rep = obs[:, None, :].expand(-1, GRAPH_CANDIDATE_COUNT, -1)
    clearance = _direction_ray(
        obs_rep.reshape(-1, obs.shape[1]),
        cand_dir.reshape(-1, 2),
        parsed,
    ).reshape(obs.shape[0], GRAPH_CANDIDATE_COUNT)
    clearance = (clearance / max(float(parsed.ray_length), 1e-6)).clamp(0.0, 1.0)

    score = getattr(parsed, "graph_candidate_label_align_weight", 2.0) * align
    score += getattr(parsed, "graph_candidate_label_clearance_weight", 0.40) * clearance
    score += getattr(parsed, "graph_candidate_label_gain_weight", 0.30) * cand[:, :, 5].clamp(0.0, 1.0)
    score += getattr(parsed, "graph_candidate_label_cluster_weight", 0.10) * cand[:, :, 6].clamp(0.0, 1.0)
    score += getattr(parsed, "graph_candidate_label_passage_weight", 0.20) * cand[:, :, 7].clamp(0.0, 1.0)
    score += getattr(parsed, "graph_candidate_label_selected_weight", 0.0) * cand[:, :, 8].clamp(0.0, 1.0)
    score -= getattr(parsed, "graph_candidate_label_depth_weight", 0.10) * cand[:, :, 4].clamp(0.0, 1.0)
    score = torch.where(exists, score, torch.full_like(score, -1.0e9))
    index = score.argmax(dim=1)
    best_align = align.gather(1, index[:, None]).squeeze(1)
    valid = any_exists & oracle_exists
    min_align = getattr(parsed, "graph_candidate_label_min_align", -1.0)
    if min_align > -1.0:
        valid = valid & (best_align >= min_align)
    return index, valid.float()


def graph_candidate_label_targets(obs, oracle_target, parsed):
    index, valid = graph_candidate_labels(obs, oracle_target, parsed)
    return torch.stack((index.float(), valid), dim=1)


def graph_candidate_loss(pred, target, parsed, weights=None):
    logits = pred[..., :GRAPH_CANDIDATE_COUNT]
    labels = target[..., 0].long().clamp(0, GRAPH_CANDIDATE_COUNT - 1)
    valid = target[..., 1].float().clamp(0.0, 1.0)
    if weights is not None:
        weights = weights.float()
        weights = weights / weights.mean().clamp_min(1e-6)
        valid = valid * weights
    ce = F.cross_entropy(
        logits.reshape(-1, GRAPH_CANDIDATE_COUNT),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)
    return (ce * valid).sum() / valid.sum().clamp_min(1.0)


def sector_fallback_mask(obs, parsed):
    if (
        parsed is None
        or not getattr(parsed, "sector_fallback_target", False)
        or not getattr(parsed, "predict_frontier_sectors", False)
    ):
        return torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
    path_absent = obs[:, 18].float() <= parsed.exists_threshold
    coverage_ok = obs[:, 17].float() >= parsed.sector_fallback_min_coverage
    sector_scores = obs[:, 22:26].float().clamp(0.0, 1.0)
    max_score = sector_scores.max(dim=1).values
    return path_absent & coverage_ok & (max_score >= parsed.sector_fallback_threshold)


def apply_sector_fallback_target(labels, obs, parsed):
    fallback = sector_fallback_mask(obs, parsed)
    if not fallback.any():
        return labels

    sector_scores = obs[:, 22:26].float().clamp(0.0, 1.0)
    best_idx = sector_scores.argmax(dim=1)
    directions = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float32,
        device=obs.device,
    )
    labels[fallback, 0:2] = directions[best_idx[fallback]]
    labels[fallback, 2] = parsed.sector_fallback_distance
    labels[fallback, 3] = 1.0
    return labels


def vector_fallback_mask(obs, parsed):
    if (
        parsed is None
        or not getattr(parsed, "vector_fallback_target", False)
        or not getattr(parsed, "predict_frontier_vector", False)
    ):
        return torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
    path_absent = obs[:, 18].float() <= parsed.exists_threshold
    coverage_ok = obs[:, 17].float() >= parsed.vector_fallback_min_coverage
    vector_dist = obs[:, 21].float().clamp(0.0, 1.0)
    return path_absent & coverage_ok & (vector_dist >= parsed.vector_fallback_min_dist)


def apply_vector_fallback_target(labels, obs, parsed):
    fallback = vector_fallback_mask(obs, parsed)
    if not fallback.any():
        return labels

    labels[fallback, 0:2] = obs[fallback, 19:21].float().clamp(-1.0, 1.0)
    labels[fallback, 2] = obs[fallback, 21].float().clamp(0.0, 1.0)
    labels[fallback, 3] = 1.0
    return labels


def apply_clear_ray_absent_fallback(labels, obs, parsed):
    if parsed.num_rays <= 0:
        return labels
    absent = labels[:, 3] <= parsed.exists_threshold
    if not absent.any():
        return labels

    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    angles = torch.arange(parsed.num_rays, dtype=torch.float32, device=obs.device)
    angles = angles * (2.0 * math.pi / float(parsed.num_rays))
    wrapped = torch.atan2(torch.sin(angles), torch.cos(angles)).abs()
    score = rays - parsed.absent_fallback_angle_penalty * wrapped[None, :] / math.pi
    best_idx = score.argmax(dim=1)
    best_angle = angles.gather(0, best_idx)
    best_clearance = rays.gather(1, best_idx[:, None]).squeeze(1)
    fallback_exists = (best_clearance >= parsed.absent_fallback_min_ray).float()

    labels[absent, 0] = torch.cos(best_angle[absent])
    labels[absent, 1] = torch.sin(best_angle[absent])
    labels[absent, 2] = (best_clearance[absent] / max(parsed.ray_length, 1e-6)).clamp(0.0, 1.0)
    labels[absent, 3] = fallback_exists[absent]
    return labels


def apply_objective_drive_gate(target, parsed, obs=None):
    if (
        obs is None
        or not getattr(parsed, "drive_objective_when_close", False)
        or not getattr(parsed, "explore_onboard_objective_obs", False)
        or obs.shape[1] < 22
    ):
        return target

    local_exists = target[:, 3:4] > parsed.exists_threshold
    local_close = target[:, 2:3].clamp(0.0, 1.0) <= parsed.drive_objective_close_dist
    objective_dir = obs[:, 19:21].float().clamp(-1.0, 1.0)
    objective_dist = obs[:, 21:22].float().clamp(0.0, 1.0)
    objective_norm = torch.linalg.vector_norm(objective_dir, dim=1, keepdim=True)
    objective_ready = (
        objective_norm >= 1e-4
    ) & (
        objective_dist >= parsed.drive_objective_min_dist
    )
    gate = local_exists & local_close & objective_ready
    if not gate.any():
        return target

    out = target.clone()
    out[:, 0:2] = torch.where(gate, objective_dir, out[:, 0:2])
    out[:, 2:3] = torch.where(gate, objective_dist, out[:, 2:3])
    out[:, 3:4] = torch.where(gate, torch.ones_like(out[:, 3:4]), out[:, 3:4])
    return out


def _direction_ray(obs, direction, parsed):
    if parsed.num_rays <= 0:
        return torch.zeros((obs.shape[0],), dtype=torch.float32, device=obs.device)
    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    angle = torch.atan2(direction[:, 1], direction[:, 0]).remainder(2.0 * math.pi)
    idx = torch.round(angle * (parsed.num_rays / (2.0 * math.pi))).long()
    idx = idx.remainder(parsed.num_rays)
    return rays.gather(1, idx[:, None]).squeeze(1)


def _direction_cone_ray(obs, direction, parsed):
    if parsed.num_rays <= 0:
        return torch.zeros((obs.shape[0],), dtype=torch.float32, device=obs.device)
    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    action_angle = torch.atan2(direction[:, 1], direction[:, 0]).remainder(2.0 * math.pi)
    ray_angles = torch.arange(parsed.num_rays, dtype=torch.float32, device=obs.device)
    ray_angles = ray_angles * (2.0 * math.pi / float(parsed.num_rays))
    diff = ray_angles[None, :] - action_angle[:, None]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()
    cone = math.radians(max(parsed.safety_cone_deg, 0.0))
    mask = diff <= cone
    masked = torch.where(mask, rays, torch.full_like(rays, parsed.ray_length))
    return masked.min(dim=1).values


def _safety_scale(clearance, stop, slow):
    if stop <= 0.0:
        return torch.ones_like(clearance)
    if slow > stop:
        return ((clearance - stop) / (slow - stop)).clamp(0.0, 1.0)
    return (clearance >= stop).float()


def _combined_safety_scale(obs, unit, parsed):
    scale = torch.ones((obs.shape[0],), dtype=torch.float32, device=obs.device)
    if parsed.num_rays <= 0:
        return scale

    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    if parsed.safety_min_ray_stop > 0.0:
        min_ray = rays.min(dim=1).values
        scale = torch.minimum(
            scale,
            _safety_scale(min_ray, parsed.safety_min_ray_stop, parsed.safety_slow_ray),
        )
    if parsed.safety_action_ray_stop > 0.0:
        action_ray = _direction_ray(obs, unit, parsed)
        scale = torch.minimum(
            scale,
            _safety_scale(action_ray, parsed.safety_action_ray_stop, parsed.safety_slow_ray),
        )
    if parsed.safety_cone_ray_stop > 0.0:
        cone_ray = _direction_cone_ray(obs, unit, parsed)
        scale = torch.minimum(
            scale,
            _safety_scale(cone_ray, parsed.safety_cone_ray_stop, parsed.safety_slow_ray),
        )
    return scale


def _min_ray_speed_cap_actions(actions, obs, parsed):
    if (
        obs is None
        or parsed.num_rays <= 0
        or not getattr(parsed, "safety_min_ray_speed_cap", False)
    ):
        return actions

    stop = getattr(parsed, "safety_min_ray_speed_stop", 0.0)
    slow = getattr(parsed, "safety_min_ray_speed_slow", 0.0)
    if stop <= 0.0 or slow <= stop:
        return actions

    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    min_ray = rays.min(dim=1)
    scale = _safety_scale(min_ray.values, stop, slow)[:, None]
    min_scale = getattr(parsed, "safety_min_ray_speed_min_scale", 0.0)
    if min_scale > 0.0:
        scale = min_scale + (1.0 - min_scale) * scale

    out = actions.clone()
    out[:, 0:2] = out[:, 0:2] * scale.clamp(0.0, 1.0)
    return out


def _passage_speed_cap_actions(actions, obs, parsed):
    if (
        obs is None
        or parsed.num_rays <= 0
        or not getattr(parsed, "safety_passage_speed_cap", False)
    ):
        return actions

    stop = getattr(parsed, "safety_passage_speed_stop", 0.0)
    slow = getattr(parsed, "safety_passage_speed_slow", 0.0)
    if stop <= 0.0 or slow <= stop:
        return actions

    xy = actions[:, 0:2].float()
    speed = torch.linalg.vector_norm(xy, dim=1)
    active = speed > 1e-6
    if not active.any():
        return actions

    unit = xy / speed[:, None].clamp_min(1e-6)
    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    min_ray = rays.min(dim=1).values
    action_ray = _direction_ray(obs, unit, parsed)
    cone_ray = _direction_cone_ray(obs, unit, parsed)
    front = getattr(parsed, "safety_passage_speed_front", 0.80)
    near = getattr(parsed, "safety_passage_speed_near", 0.45)
    side_gap = getattr(parsed, "safety_passage_speed_side_gap", 0.20)
    passage_risk = (
        active
        & (min_ray <= near)
        & (action_ray >= front)
        & (cone_ray >= front)
        & ((torch.minimum(action_ray, cone_ray) - min_ray) >= side_gap)
    )
    if not passage_risk.any():
        return actions

    scale = _safety_scale(min_ray, stop, slow)
    min_scale = getattr(parsed, "safety_passage_speed_min_scale", 0.0)
    if min_scale > 0.0:
        scale = min_scale + (1.0 - min_scale) * scale

    out = actions.clone()
    capped_xy = xy * scale[:, None].clamp(0.0, 1.0)
    out[:, 0:2] = torch.where(passage_risk[:, None], capped_xy, xy)
    return out


def _momentum_speed_cap_actions(actions, obs, parsed):
    if (
        obs is None
        or parsed.num_rays <= 0
        or not getattr(parsed, "safety_momentum_speed_cap", False)
    ):
        return actions

    stop = getattr(parsed, "safety_momentum_speed_stop", 0.0)
    slow = getattr(parsed, "safety_momentum_speed_slow", 0.0)
    if stop <= 0.0 or slow <= stop:
        return actions

    xy = actions[:, 0:2].float()
    action_speed = torch.linalg.vector_norm(xy, dim=1)
    active = action_speed > 1e-6
    if not active.any():
        return actions

    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    min_ray = rays.min(dim=1).values
    vel = obs[:, 0:2].float()
    vel_speed = torch.linalg.vector_norm(vel, dim=1)
    if not (vel_speed > 1e-6).any():
        return actions

    angles = torch.arange(parsed.num_rays, dtype=torch.float32, device=obs.device)
    angles = angles * (2.0 * math.pi / float(parsed.num_rays))
    ray_dirs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    near = getattr(parsed, "safety_momentum_speed_near", 0.45)
    severity = ((near - rays) / max(near - stop, 1e-6)).clamp(0.0, 1.0)
    vel_dot = torch.einsum("nd,rd->nr", vel, ray_dirs).clamp_min(0.0)
    toward_close = (severity * vel_dot).max(dim=1).values
    active = (
        active
        & (min_ray <= near)
        & (vel_speed >= getattr(parsed, "safety_momentum_speed_min_vel", 0.08))
        & (toward_close >= getattr(parsed, "safety_momentum_speed_toward", 0.02))
    )
    if not active.any():
        return actions

    scale = _safety_scale(min_ray, stop, slow)
    min_scale = getattr(parsed, "safety_momentum_speed_min_scale", 0.0)
    if min_scale > 0.0:
        scale = min_scale + (1.0 - min_scale) * scale

    out = actions.clone()
    capped_xy = xy * scale[:, None].clamp(0.0, 1.0)
    out[:, 0:2] = torch.where(active[:, None], capped_xy, xy)
    return out


def _passage_center_actions(actions, obs, progress_unit, parsed):
    if (
        obs is None
        or parsed.num_rays <= 0
        or not getattr(parsed, "safety_passage_center_actions", False)
    ):
        return actions

    xy = actions[:, 0:2].float()
    speed = torch.linalg.vector_norm(xy, dim=1, keepdim=True)
    active = speed[:, 0] > 1e-6
    if not active.any():
        return actions

    ref = progress_unit.float()
    ref = ref / torch.linalg.vector_norm(ref, dim=1, keepdim=True).clamp_min(1e-6)
    right = torch.stack((ref[:, 1], -ref[:, 0]), dim=1)
    left = -right

    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    min_ray = rays.min(dim=1).values
    front_ray = _direction_cone_ray(obs, ref, parsed)
    left_ray = _direction_ray(obs, left, parsed)
    right_ray = _direction_ray(obs, right, parsed)
    side_sum = (left_ray + right_ray).clamp_min(1e-6)
    imbalance = ((right_ray - left_ray) / side_sum).clamp(-1.0, 1.0)

    gain = max(float(getattr(parsed, "safety_passage_center_gain", 0.0)), 0.0)
    max_turn = math.radians(max(float(getattr(parsed, "safety_passage_center_max_turn_deg", 45.0)), 0.0))
    steer = (gain * imbalance).clamp(-math.tan(max_turn), math.tan(max_turn))
    centered = ref + steer[:, None] * right
    centered = centered / torch.linalg.vector_norm(centered, dim=1, keepdim=True).clamp_min(1e-6)

    candidate = actions.clone()
    candidate[:, 0:2] = centered * speed
    action_ray = _direction_ray(obs, centered, parsed)
    cone_ray = _direction_cone_ray(obs, centered, parsed)
    valid = (
        active
        & (min_ray <= getattr(parsed, "safety_passage_center_near", 0.45))
        & (front_ray >= getattr(parsed, "safety_passage_center_front", 0.60))
        & (action_ray >= max(parsed.safety_action_ray_stop, parsed.safety_cone_ray_stop))
        & (cone_ray >= parsed.safety_cone_ray_stop)
        & (imbalance.abs() >= getattr(parsed, "safety_passage_center_min_imbalance", 0.05))
    )
    return torch.where(valid[:, None], candidate, actions)


def _rotate_unit(unit, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return torch.stack(
        (unit[:, 0] * cos_a - unit[:, 1] * sin_a, unit[:, 0] * sin_a + unit[:, 1] * cos_a),
        dim=1,
    )


def _corridor_width_actions(actions, obs, progress_unit, parsed):
    if (
        obs is None
        or parsed.num_rays <= 0
        or not getattr(parsed, "safety_corridor_width_actions", False)
    ):
        return actions

    xy = actions[:, 0:2].float()
    speed = torch.linalg.vector_norm(xy, dim=1, keepdim=True)
    active = speed[:, 0] > 1e-6
    if not active.any():
        return actions

    ref = progress_unit.float()
    ref = ref / torch.linalg.vector_norm(ref, dim=1, keepdim=True).clamp_min(1e-6)
    offsets = torch.tensor(
        [0.0, 10.0, -10.0, 22.0, -22.0, 35.0, -35.0, 50.0, -50.0, 65.0, -65.0],
        dtype=torch.float32,
        device=obs.device,
    )
    max_turn = max(float(getattr(parsed, "safety_corridor_width_max_turn_deg", 55.0)), 0.0)
    offsets = offsets[offsets.abs() <= max_turn] * (math.pi / 180.0)
    speed_scales = torch.tensor([1.0, 0.85, 0.70, 0.55], dtype=torch.float32, device=obs.device)

    cos_o = torch.cos(offsets)[None, :]
    sin_o = torch.sin(offsets)[None, :]
    dx = ref[:, 0:1]
    dy = ref[:, 1:2]
    dirs = torch.stack(
        (dx * cos_o - dy * sin_o, dx * sin_o + dy * cos_o),
        dim=2,
    )
    cand = dirs[:, :, None, :].expand(-1, -1, speed_scales.numel(), -1)
    cand = cand.reshape(obs.shape[0], -1, 2)
    cand_speed_scale = speed_scales.repeat(offsets.numel())[None, :]

    flat = cand.reshape(-1, 2)
    obs_rep = obs[:, None, :].expand(-1, cand.shape[1], -1)
    obs_rep = obs_rep.reshape(-1, obs.shape[1])
    action_ray = _direction_ray(obs_rep, flat, parsed).reshape(obs.shape[0], cand.shape[1])
    cone_ray = _direction_cone_ray(obs_rep, flat, parsed).reshape(obs.shape[0], cand.shape[1])

    side_angle = math.radians(
        max(min(float(getattr(parsed, "safety_corridor_width_side_deg", 80.0)), 120.0), 35.0)
    )
    left = _rotate_unit(flat, side_angle)
    right = _rotate_unit(flat, -side_angle)
    left_ray = _direction_ray(obs_rep, left, parsed).reshape(obs.shape[0], cand.shape[1])
    right_ray = _direction_ray(obs_rep, right, parsed).reshape(obs.shape[0], cand.shape[1])
    side_clearance = torch.minimum(left_ray, right_ray)
    width = left_ray + right_ray
    balance = side_clearance / width.clamp_min(1e-6)

    side_stop = getattr(parsed, "safety_corridor_width_side_stop", 0.24)
    side_slow = getattr(parsed, "safety_corridor_width_side_slow", 0.65)
    side_scale = _safety_scale(side_clearance, side_stop, side_slow)
    min_scale = getattr(parsed, "safety_corridor_width_min_scale", 0.0)
    if min_scale > 0.0:
        side_scale = min_scale + (1.0 - min_scale) * side_scale

    front = torch.minimum(action_ray, cone_ray)
    front_scale = _safety_scale(
        front,
        max(parsed.safety_action_ray_stop, parsed.safety_cone_ray_stop),
        parsed.safety_slow_ray,
    )
    scale = torch.minimum(side_scale, front_scale)
    progress = (cand * ref[:, None, :]).sum(dim=2)
    turn_cost = offsets.abs().repeat_interleave(speed_scales.numel())[None, :] / math.pi
    speed_cost = 1.0 - cand_speed_scale

    side_norm = (side_clearance / max(side_slow, 1e-6)).clamp(0.0, 1.0)
    front_norm = (front / max(parsed.ray_length, 1e-6)).clamp(0.0, 1.0)
    width_norm = (width / max(2.0 * side_slow, 1e-6)).clamp(0.0, 1.0)
    score = 2.30 * progress * cand_speed_scale
    score += 1.45 * side_norm
    score += 0.55 * width_norm
    score += 0.45 * front_norm
    score += 0.25 * balance
    score -= 0.22 * turn_cost
    score -= 0.18 * speed_cost

    progress_floor = getattr(parsed, "safety_corridor_width_progress_floor", 0.10)
    min_front = getattr(parsed, "safety_corridor_width_front", 0.70)
    score = torch.where(progress >= progress_floor, score, torch.full_like(score, -1.0e6))
    score = torch.where(front >= min_front, score, torch.full_like(score, -1.0e6))
    score = torch.where(side_clearance >= side_stop, score, torch.full_like(score, -1.0e6))
    score = torch.where(scale > 0.0, score, torch.full_like(score, -1.0e6))

    best = score.argmax(dim=1)
    row = torch.arange(obs.shape[0], device=obs.device)
    best_dir = cand[row, best]
    best_scale = scale[row, best][:, None]
    best_speed_scale = cand_speed_scale.expand(obs.shape[0], -1)[row, best][:, None]

    current_unit = xy / speed.clamp_min(1e-6)
    current_left = _direction_ray(obs, _rotate_unit(current_unit, side_angle), parsed)
    current_right = _direction_ray(obs, _rotate_unit(current_unit, -side_angle), parsed)
    current_side = torch.minimum(current_left, current_right)
    current_front = torch.minimum(
        _direction_ray(obs, current_unit, parsed),
        _direction_cone_ray(obs, current_unit, parsed),
    )
    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    min_ray = rays.min(dim=1).values
    gate = (
        active
        & (min_ray <= getattr(parsed, "safety_corridor_width_near", 0.55))
        & (current_side <= side_slow)
        & (current_front >= min_front)
    )
    if not gate.any():
        return actions

    out = actions.clone()
    planned_xy = best_dir * speed * best_speed_scale * best_scale
    out[:, 0:2] = torch.where(gate[:, None], planned_xy, out[:, 0:2])
    return out


def _slide_safety_actions(actions, obs, parsed, progress_unit=None):
    if (
        obs is None
        or parsed.num_rays <= 0
        or parsed.safety_cone_ray_stop <= 0.0
        or parsed.safety_slow_ray <= parsed.safety_cone_ray_stop
    ):
        return actions

    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    severity = (
        (parsed.safety_slow_ray - rays)
        / (parsed.safety_slow_ray - parsed.safety_cone_ray_stop)
    ).clamp(0.0, 1.0)
    if not severity.any():
        return actions

    angles = torch.arange(parsed.num_rays, dtype=torch.float32, device=obs.device)
    angles = angles * (2.0 * math.pi / float(parsed.num_rays))
    ray_dirs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    original_xy = actions[:, 0:2].float()
    safe_xy = original_xy
    for _ in range(2):
        speed = torch.linalg.vector_norm(safe_xy, dim=1, keepdim=True)
        unit = safe_xy / speed.clamp_min(1e-6)
        forward = (ray_dirs[None, :, :] * unit[:, None, :]).sum(dim=2).clamp_min(0.0)
        risk, idx = (severity * forward).max(dim=1)
        normal = ray_dirs[idx]
        inward_speed = (safe_xy * normal).sum(dim=1).clamp_min(0.0)
        correction = (risk * inward_speed)[:, None] * normal
        safe_xy = safe_xy - correction

    original_speed = torch.linalg.vector_norm(actions[:, 0:2], dim=1, keepdim=True)
    safe_speed = torch.linalg.vector_norm(safe_xy, dim=1, keepdim=True)
    speed_limit = torch.minimum(torch.ones_like(safe_speed), original_speed / safe_speed.clamp_min(1e-6))
    safe_xy = safe_xy * speed_limit
    if (
        progress_unit is not None
        and getattr(parsed, "safety_slide_progress_aware", False)
    ):
        ref = progress_unit.float()
        ref_norm = torch.linalg.vector_norm(ref, dim=1, keepdim=True)
        ref = ref / ref_norm.clamp_min(1e-6)
        pre_progress = (original_xy * ref).sum(dim=1, keepdim=True)
        post_progress = (safe_xy * ref).sum(dim=1, keepdim=True)
        cone_ray = _direction_cone_ray(obs, ref, parsed)[:, None]
        min_progress = pre_progress * getattr(
            parsed, "safety_slide_progress_fraction", 0.60
        )
        active = (
            (pre_progress > 1e-6)
            & (post_progress < min_progress)
            & (cone_ray >= getattr(parsed, "safety_slide_progress_cone", 0.60))
        )
        denom = (pre_progress - post_progress).clamp_min(1e-6)
        alpha = ((min_progress - post_progress) / denom).clamp(0.0, 1.0)
        candidate = safe_xy + alpha * (original_xy - safe_xy)
        candidate_speed = torch.linalg.vector_norm(candidate, dim=1, keepdim=True)
        candidate = candidate * torch.minimum(
            torch.ones_like(candidate_speed),
            original_speed / candidate_speed.clamp_min(1e-6),
        )
        safe_xy = torch.where(active, candidate, safe_xy)

    out = actions.clone()
    out[:, 0:2] = safe_xy
    return out


def _sampled_safety_actions(actions, obs, parsed):
    if (
        obs is None
        or parsed.num_rays <= 0
        or parsed.safety_cone_ray_stop <= 0.0
        or parsed.safety_slow_ray <= parsed.safety_cone_ray_stop
    ):
        return actions

    xy = actions[:, 0:2].float()
    speed = torch.linalg.vector_norm(xy, dim=1, keepdim=True)
    desired = xy / speed.clamp_min(1e-6)
    active = speed[:, 0] > 1e-6
    if not active.any():
        return actions

    offsets = torch.tensor(
        [0.0, 20.0, -20.0, 40.0, -40.0, 65.0, -65.0, 90.0, -90.0],
        dtype=torch.float32,
        device=obs.device,
    )
    offsets = offsets * (math.pi / 180.0)
    cos_o = torch.cos(offsets)[None, :]
    sin_o = torch.sin(offsets)[None, :]
    dx = desired[:, 0:1]
    dy = desired[:, 1:2]
    cand = torch.stack(
        (dx * cos_o - dy * sin_o, dx * sin_o + dy * cos_o),
        dim=2,
    )

    flat = cand.reshape(-1, 2)
    obs_rep = obs[:, None, :].expand(-1, offsets.numel(), -1)
    obs_rep = obs_rep.reshape(-1, obs.shape[1])
    cone_ray = _direction_cone_ray(obs_rep, flat, parsed).reshape(
        obs.shape[0], offsets.numel()
    )
    scale = _safety_scale(
        cone_ray, parsed.safety_cone_ray_stop, parsed.safety_slow_ray
    )
    desired_scale = scale[:, 0:1]
    progress = (cand * desired[:, None, :]).sum(dim=2)
    clearance = (cone_ray / max(parsed.ray_length, 1e-6)).clamp(0.0, 1.0)
    turn_cost = offsets.abs()[None, :] / (0.5 * math.pi)

    score = 3.00 * progress
    score += 0.70 * scale
    score += 0.20 * clearance
    score -= 0.25 * turn_cost
    score = torch.where(progress >= -0.05, score, torch.full_like(score, -1.0e6))

    best = score.argmax(dim=1)
    best_dir = cand[torch.arange(cand.shape[0], device=obs.device), best]
    best_scale = scale[torch.arange(scale.shape[0], device=obs.device), best]

    out = actions.clone()
    sampled_xy = best_dir * speed * best_scale[:, None]
    scaled_desired_xy = xy * desired_scale
    use_sample = desired_scale < getattr(parsed, "safety_sample_trigger_scale", 0.35)
    out[:, 0:2] = torch.where(use_sample, sampled_xy, scaled_desired_xy)
    out = torch.where(active[:, None], out, actions)
    return out


def _barrier_steering_actions(actions, obs, unit, desired_speed, parsed):
    if (
        obs is None
        or parsed.num_rays <= 0
        or parsed.safety_cone_ray_stop <= 0.0
        or parsed.safety_slow_ray <= parsed.safety_cone_ray_stop
    ):
        return actions

    xy = actions[:, 0:2].float()
    speed = torch.linalg.vector_norm(xy, dim=1, keepdim=True)
    active = speed[:, 0] > 1e-6
    if not active.any():
        return actions

    desired = unit.float()
    offsets = torch.tensor(
        [0.0, 15.0, -15.0, 30.0, -30.0, 50.0, -50.0, 75.0, -75.0, 105.0, -105.0, 135.0, -135.0, 180.0],
        dtype=torch.float32,
        device=obs.device,
    )
    offsets = offsets * (math.pi / 180.0)
    cos_o = torch.cos(offsets)[None, :]
    sin_o = torch.sin(offsets)[None, :]
    dx = desired[:, 0:1]
    dy = desired[:, 1:2]
    cand = torch.stack(
        (dx * cos_o - dy * sin_o, dx * sin_o + dy * cos_o),
        dim=2,
    )

    flat = cand.reshape(-1, 2)
    obs_rep = obs[:, None, :].expand(-1, offsets.numel(), -1)
    obs_rep = obs_rep.reshape(-1, obs.shape[1])
    cone_ray = _direction_cone_ray(obs_rep, flat, parsed).reshape(
        obs.shape[0], offsets.numel()
    )
    action_ray = _direction_ray(obs_rep, flat, parsed).reshape(
        obs.shape[0], offsets.numel()
    )
    action_scale = _safety_scale(
        action_ray, max(parsed.safety_action_ray_stop, parsed.safety_cone_ray_stop), parsed.safety_slow_ray
    )
    cone_scale = _safety_scale(
        cone_ray, parsed.safety_cone_ray_stop, parsed.safety_slow_ray
    )
    scale = torch.minimum(action_scale, cone_scale)

    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    min_ray = rays.min(dim=1).values
    angles = torch.arange(parsed.num_rays, dtype=torch.float32, device=obs.device)
    angles = angles * (2.0 * math.pi / float(parsed.num_rays))
    ray_dirs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    severity = (
        (parsed.safety_slow_ray - rays)
        / (parsed.safety_slow_ray - parsed.safety_cone_ray_stop)
    ).clamp(0.0, 1.0)
    ray_dot = torch.einsum("nkd,rd->nkr", cand, ray_dirs)
    toward_close = (severity[:, None, :] * ray_dot.clamp_min(0.0)).max(dim=2).values
    away_close = (severity[:, None, :] * (-ray_dot).clamp_min(0.0)).max(dim=2).values

    progress = (cand * desired[:, None, :]).sum(dim=2)
    clearance = (torch.minimum(action_ray, cone_ray) / max(parsed.ray_length, 1e-6)).clamp(0.0, 1.0)
    rollout_room = (cone_ray - parsed.safety_cone_ray_stop).clamp_min(0.0)
    rollout_need = desired_speed[:, 0:1].clamp_min(0.05) * parsed.action_repeat * 0.02
    rollout = (rollout_room / rollout_need.clamp_min(1e-6)).clamp(0.0, 1.0)
    turn_cost = offsets.abs()[None, :] / math.pi

    near_wall = min_ray <= getattr(parsed, "safety_barrier_near_wall", 0.36)
    progress_weight = torch.where(
        near_wall[:, None],
        torch.full_like(progress, 0.85),
        torch.full_like(progress, 2.60),
    )
    clearance_weight = torch.where(
        near_wall[:, None],
        torch.full_like(progress, 1.15),
        torch.full_like(progress, 0.35),
    )
    escape_weight = torch.where(
        near_wall[:, None],
        torch.full_like(progress, 2.15),
        torch.full_like(progress, 0.35),
    )
    toward_weight = torch.where(
        near_wall[:, None],
        torch.full_like(progress, 2.25),
        torch.full_like(progress, 0.85),
    )

    score = progress_weight * progress
    score += 0.70 * scale
    score += clearance_weight * clearance
    score += 0.55 * rollout
    score += escape_weight * away_close
    score -= toward_weight * toward_close
    score -= 0.18 * turn_cost

    progress_floor = torch.where(
        near_wall[:, None],
        torch.full_like(progress, -0.75),
        torch.full_like(progress, -0.05),
    )
    score = torch.where(progress >= progress_floor, score, torch.full_like(score, -1.0e6))
    score = torch.where(scale > 0.0, score, torch.full_like(score, -1.0e6))

    best = score.argmax(dim=1)
    best_dir = cand[torch.arange(cand.shape[0], device=obs.device), best]
    best_scale = scale[torch.arange(scale.shape[0], device=obs.device), best]

    out = actions.clone()
    out[:, 0:2] = best_dir * desired_speed * best_scale[:, None]
    out = torch.where(active[:, None], out, actions)
    return out


def _trajectory_steering_actions(actions, obs, unit, desired_speed, parsed):
    if (
        obs is None
        or parsed.num_rays <= 0
        or parsed.safety_cone_ray_stop <= 0.0
        or parsed.safety_slow_ray <= parsed.safety_cone_ray_stop
    ):
        return actions

    xy = actions[:, 0:2].float()
    speed = torch.linalg.vector_norm(xy, dim=1, keepdim=True)
    active = speed[:, 0] > 1e-6
    if not active.any():
        return actions

    desired = unit.float()
    offsets = torch.tensor(
        [0.0, 12.0, -12.0, 25.0, -25.0, 40.0, -40.0, 60.0, -60.0, 85.0, -85.0, 115.0, -115.0, 150.0, -150.0],
        dtype=torch.float32,
        device=obs.device,
    )
    speed_scales = torch.tensor([1.0, 0.75, 0.50], dtype=torch.float32, device=obs.device)
    offsets = offsets * (math.pi / 180.0)
    cos_o = torch.cos(offsets)[None, :]
    sin_o = torch.sin(offsets)[None, :]
    dx = desired[:, 0:1]
    dy = desired[:, 1:2]
    dirs = torch.stack(
        (dx * cos_o - dy * sin_o, dx * sin_o + dy * cos_o),
        dim=2,
    )
    cand = dirs[:, :, None, :].expand(-1, -1, speed_scales.numel(), -1)
    cand = cand.reshape(obs.shape[0], -1, 2)
    cand_speed_scale = speed_scales.repeat(offsets.numel())[None, :]

    flat = cand.reshape(-1, 2)
    obs_rep = obs[:, None, :].expand(-1, cand.shape[1], -1)
    obs_rep = obs_rep.reshape(-1, obs.shape[1])
    action_ray = _direction_ray(obs_rep, flat, parsed).reshape(obs.shape[0], cand.shape[1])
    cone_ray = _direction_cone_ray(obs_rep, flat, parsed).reshape(obs.shape[0], cand.shape[1])
    action_scale = _safety_scale(
        action_ray,
        max(parsed.safety_action_ray_stop, parsed.safety_cone_ray_stop),
        parsed.safety_slow_ray,
    )
    cone_scale = _safety_scale(cone_ray, parsed.safety_cone_ray_stop, parsed.safety_slow_ray)
    scale = torch.minimum(action_scale, cone_scale)

    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    min_ray = rays.min(dim=1).values
    ray_angles = torch.arange(parsed.num_rays, dtype=torch.float32, device=obs.device)
    ray_angles = ray_angles * (2.0 * math.pi / float(parsed.num_rays))
    ray_dirs = torch.stack((torch.cos(ray_angles), torch.sin(ray_angles)), dim=1)
    severity = (
        (parsed.safety_slow_ray - rays)
        / (parsed.safety_slow_ray - parsed.safety_cone_ray_stop)
    ).clamp(0.0, 1.0)
    ray_dot = torch.einsum("nkd,rd->nkr", cand, ray_dirs)
    toward_close = (severity[:, None, :] * ray_dot.clamp_min(0.0)).max(dim=2).values
    away_close = (severity[:, None, :] * (-ray_dot).clamp_min(0.0)).max(dim=2).values

    progress = (cand * desired[:, None, :]).sum(dim=2)
    command_speed = desired_speed * cand_speed_scale
    horizon = max(float(parsed.safety_trajectory_steps), 1.0) * float(parsed.safety_trajectory_dt)
    path_len = command_speed * horizon
    clearance = (torch.minimum(action_ray, cone_ray) / max(parsed.ray_length, 1e-6)).clamp(0.0, 1.0)
    trajectory_margin = torch.minimum(action_ray, cone_ray) - parsed.safety_cone_ray_stop - path_len
    trajectory_fit = (trajectory_margin / path_len.clamp_min(1e-6)).clamp(-1.0, 1.0)
    turn_cost = offsets.abs().repeat_interleave(speed_scales.numel())[None, :] / math.pi
    speed_cost = 1.0 - cand_speed_scale

    near_wall = min_ray <= getattr(parsed, "safety_trajectory_near_wall", 0.42)
    progress_weight = torch.where(
        near_wall[:, None],
        torch.full_like(progress, 0.80),
        torch.full_like(progress, 2.30),
    )
    clearance_weight = torch.where(
        near_wall[:, None],
        torch.full_like(progress, 1.20),
        torch.full_like(progress, 0.35),
    )
    escape_weight = torch.where(
        near_wall[:, None],
        torch.full_like(progress, 1.70),
        torch.full_like(progress, 0.25),
    )

    score = progress_weight * progress * cand_speed_scale
    score += clearance_weight * clearance
    score += 1.05 * trajectory_fit
    score += 0.45 * scale
    score += escape_weight * away_close
    score -= 1.15 * toward_close
    score -= 0.18 * turn_cost
    score -= 0.25 * speed_cost

    progress_floor = torch.where(
        near_wall[:, None],
        torch.full_like(progress, -0.65),
        torch.full_like(progress, -0.08),
    )
    min_fit = getattr(parsed, "safety_trajectory_min_fit", -0.25)
    score = torch.where(progress >= progress_floor, score, torch.full_like(score, -1.0e6))
    score = torch.where(trajectory_fit >= min_fit, score, torch.full_like(score, -1.0e6))
    score = torch.where(scale > 0.0, score, torch.full_like(score, -1.0e6))

    best = score.argmax(dim=1)
    best_dir = cand[torch.arange(cand.shape[0], device=obs.device), best]
    best_speed_scale = cand_speed_scale.expand(obs.shape[0], -1)[
        torch.arange(obs.shape[0], device=obs.device), best
    ][:, None]
    best_scale = scale[torch.arange(scale.shape[0], device=obs.device), best][:, None]

    out = actions.clone()
    direct_scale = scale[:, 0]
    direct_fit = trajectory_fit[:, 0]
    use_trajectory = (
        near_wall
        | (direct_scale < getattr(parsed, "safety_trajectory_trigger_scale", 0.45))
        | (direct_fit < getattr(parsed, "safety_trajectory_direct_min_fit", 0.20))
    )
    planned_xy = best_dir * desired_speed * best_speed_scale * best_scale
    out[:, 0:2] = torch.where(use_trajectory[:, None], planned_xy, out[:, 0:2])
    out = torch.where(active[:, None], out, actions)
    return out


def _progress_speed_floor(actions, obs, unit, desired_speed, parsed):
    floor_scale = getattr(parsed, "safety_progress_floor_scale", 0.0)
    if (
        obs is None
        or parsed.num_rays <= 0
        or floor_scale <= 0.0
        or desired_speed.numel() == 0
    ):
        return actions

    action_ray = _direction_ray(obs, unit, parsed)
    cone_ray = _direction_cone_ray(obs, unit, parsed)
    allow = (
        (desired_speed[:, 0] > 1e-6)
        & (action_ray >= parsed.safety_progress_floor_ray)
        & (cone_ray >= parsed.safety_progress_floor_cone)
    )
    if not allow.any():
        return actions

    current_xy = actions[:, 0:2].float()
    current_speed = torch.linalg.vector_norm(current_xy, dim=1, keepdim=True)
    floor_speed = desired_speed * floor_scale
    use_floor = allow[:, None] & (current_speed < floor_speed)
    out = actions.clone()
    out[:, 0:2] = torch.where(use_floor, unit * floor_speed, out[:, 0:2])
    return out


def pred_vector_fallback_mask(target, parsed, obs=None):
    target = target.float()
    mask = torch.zeros((target.shape[0], 1), dtype=torch.bool, device=target.device)
    if not getattr(parsed, "use_pred_vector_fallback", False):
        return mask
    if target.shape[1] < 11:
        return mask

    primary_threshold = getattr(
        parsed,
        "pred_vector_fallback_primary_threshold",
        getattr(parsed, "exists_threshold", 0.2),
    )
    primary_weak = target[:, 3:4] <= primary_threshold
    vector_dist = target[:, 10:11].clamp(0.0, 1.0)
    vector_ready = vector_dist >= getattr(parsed, "pred_vector_fallback_min_dist", 0.02)

    coverage_ok = torch.ones_like(primary_weak)
    min_coverage = getattr(parsed, "pred_vector_fallback_min_coverage", 0.0)
    if obs is not None and min_coverage > 0.0 and obs.shape[1] > 17:
        coverage_ok = obs[:, 17:18].float() >= min_coverage

    return primary_weak & vector_ready & coverage_ok


def apply_pred_vector_fallback(target, parsed, obs=None):
    mask = pred_vector_fallback_mask(target, parsed, obs)
    if not mask.any():
        return target

    effective = target.clone()
    effective[:, 0:2] = torch.where(mask, target[:, 8:10].clamp(-1.0, 1.0), effective[:, 0:2])
    effective[:, 2:3] = torch.where(mask, target[:, 10:11].clamp(0.0, 1.0), effective[:, 2:3])
    effective[:, 3:4] = torch.where(mask, torch.ones_like(effective[:, 3:4]), effective[:, 3:4])
    return effective


def pred_recovery_gate_mask(target, parsed, obs=None):
    target = target.float()
    mask = torch.zeros((target.shape[0], 1), dtype=torch.bool, device=target.device)
    if not getattr(parsed, "use_pred_recovery_gate", False):
        return mask
    if target.shape[1] < 12:
        return mask

    gate = torch.sigmoid(target[:, 11:12])
    gate_ready = gate >= getattr(parsed, "pred_recovery_gate_threshold", 0.5)
    vector_dist = target[:, 10:11].clamp(0.0, 1.0)
    vector_ready = vector_dist >= getattr(parsed, "pred_recovery_gate_min_dist", 0.02)

    coverage_ok = torch.ones_like(gate_ready)
    min_coverage = getattr(parsed, "pred_recovery_gate_min_coverage", 0.0)
    if obs is not None and min_coverage > 0.0 and obs.shape[1] > 17:
        coverage_ok = obs[:, 17:18].float() >= min_coverage

    return gate_ready & vector_ready & coverage_ok


def apply_pred_recovery_gate(target, parsed, obs=None):
    mask = pred_recovery_gate_mask(target, parsed, obs)
    if not mask.any():
        return target

    effective = target.clone()
    effective[:, 0:2] = torch.where(mask, target[:, 8:10].clamp(-1.0, 1.0), effective[:, 0:2])
    effective[:, 2:3] = torch.where(mask, target[:, 10:11].clamp(0.0, 1.0), effective[:, 2:3])
    effective[:, 3:4] = torch.where(mask, torch.ones_like(effective[:, 3:4]), effective[:, 3:4])
    return effective


def target_to_actions(target, parsed, obs=None, use_safety=True):
    target = target.float()
    target = apply_pred_recovery_gate(target, parsed, obs)
    target = apply_pred_vector_fallback(target, parsed, obs)
    target = apply_objective_drive_gate(target, parsed, obs)
    direction = target[:, 0:2].clamp(-1.0, 1.0)
    norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
    unit = direction / norm
    exists = (target[:, 3:4] > parsed.exists_threshold).float()
    speed = parsed.speed_fraction * exists
    has_pred_speed = target.shape[1] > 4 and getattr(parsed, "use_pred_speed_scale", False)
    if has_pred_speed:
        speed = speed * target[:, 4:5].clamp(0.0, 1.0)

    apply_external_safety = (
        use_safety
        and obs is not None
        and parsed.num_rays > 0
        and (not has_pred_speed or getattr(parsed, "external_safety_with_pred_speed", False))
    )

    actions = torch.zeros((target.shape[0], 4), dtype=torch.float32, device=target.device)
    actions[:, 0:2] = unit * speed
    if apply_external_safety:
        if getattr(parsed, "safety_trajectory_actions", False):
            if getattr(parsed, "safety_barrier_actions", False):
                actions = _barrier_steering_actions(actions, obs, unit, speed, parsed)
            actions = _trajectory_steering_actions(actions, obs, unit, speed, parsed)
            if getattr(parsed, "safety_slide", False):
                actions = _slide_safety_actions(actions, obs, parsed, unit)
        elif getattr(parsed, "safety_barrier_actions", False):
            actions = _barrier_steering_actions(actions, obs, unit, speed, parsed)
            if getattr(parsed, "safety_slide", False):
                actions = _slide_safety_actions(actions, obs, parsed, unit)
        elif getattr(parsed, "safety_sample_actions", False):
            actions = _sampled_safety_actions(actions, obs, parsed)
        elif getattr(parsed, "safety_slide", False):
            actions = _slide_safety_actions(actions, obs, parsed, unit)
        else:
            scale = _combined_safety_scale(obs, unit, parsed)
            actions[:, 0:2] = actions[:, 0:2] * scale[:, None]
        actions = _progress_speed_floor(actions, obs, unit, speed, parsed)
        actions = _corridor_width_actions(actions, obs, unit, parsed)
    actions = _passage_center_actions(actions, obs, unit, parsed)
    actions = _passage_speed_cap_actions(actions, obs, parsed)
    actions = _momentum_speed_cap_actions(actions, obs, parsed)
    actions = _min_ray_speed_cap_actions(actions, obs, parsed)
    final_xy = actions[:, 0:2]
    final_speed = torch.linalg.vector_norm(final_xy, dim=1, keepdim=True)
    final_unit_y = torch.where(
        final_speed > 1e-6,
        final_xy[:, 1:2] / final_speed.clamp_min(1e-6),
        unit[:, 1:2],
    )
    actions[:, 3:4] = (final_unit_y * parsed.yaw_fraction).clamp(-1.0, 1.0)
    return actions.clamp(-1.0, 1.0)


class StatefulRecoveryController:
    def __init__(self, parsed):
        self.parsed = parsed
        self.best_coverage = None
        self.since_progress = None
        self.recovery_timer = None
        self.recovery_side = None

    def _ensure(self, obs):
        n = obs.shape[0]
        device = obs.device
        if self.best_coverage is not None and self.best_coverage.shape[0] == n:
            return
        self.best_coverage = obs[:, 17].float().clone()
        self.since_progress = torch.zeros((n,), dtype=torch.long, device=device)
        self.recovery_timer = torch.zeros((n,), dtype=torch.long, device=device)
        self.recovery_side = torch.ones((n,), dtype=torch.float32, device=device)

    def reset_done(self, done):
        if self.best_coverage is None or done is None:
            return
        done = done.to(device=self.best_coverage.device).bool().reshape(-1)
        if done.numel() != self.best_coverage.numel() or not done.any():
            return
        self.best_coverage[done] = 0.0
        self.since_progress[done] = 0
        self.recovery_timer[done] = 0
        self.recovery_side[done] = 1.0

    def _choose_side(self, obs):
        rays = obs[:, 26 : 26 + self.parsed.num_rays].float() * self.parsed.ray_length
        left_idx = max(0, int(round(self.parsed.num_rays * 0.25))) % self.parsed.num_rays
        right_idx = max(0, int(round(self.parsed.num_rays * 0.75))) % self.parsed.num_rays
        left = rays[:, left_idx]
        right = rays[:, right_idx]
        return torch.where(left >= right, torch.ones_like(left), -torch.ones_like(right))

    def _recovery_actions(self, target, obs, default_actions, recover):
        xy = default_actions[:, 0:2].float()
        default_speed = torch.linalg.vector_norm(xy, dim=1, keepdim=True)
        target_dir = target[:, 0:2].float().clamp(-1.0, 1.0)
        desired = target_dir / torch.linalg.vector_norm(
            target_dir, dim=1, keepdim=True
        ).clamp_min(1e-6)

        side = self.recovery_side[:, None]
        offsets = torch.tensor(
            [35.0, 55.0, 75.0, 95.0, -35.0],
            dtype=torch.float32,
            device=obs.device,
        )
        offsets = offsets[None, :] * side * (math.pi / 180.0)
        cos_o = torch.cos(offsets)
        sin_o = torch.sin(offsets)
        dx = desired[:, 0:1]
        dy = desired[:, 1:2]
        cand = torch.stack(
            (dx * cos_o - dy * sin_o, dx * sin_o + dy * cos_o),
            dim=2,
        )

        flat = cand.reshape(-1, 2)
        obs_rep = obs[:, None, :].expand(-1, cand.shape[1], -1)
        obs_rep = obs_rep.reshape(-1, obs.shape[1])
        cone_ray = _direction_cone_ray(obs_rep, flat, self.parsed).reshape(
            obs.shape[0], cand.shape[1]
        )
        direct_ray = _direction_ray(obs_rep, flat, self.parsed).reshape(
            obs.shape[0], cand.shape[1]
        )
        scale = _safety_scale(
            cone_ray, self.parsed.safety_cone_ray_stop, self.parsed.safety_slow_ray
        )
        progress = (cand * desired[:, None, :]).sum(dim=2)
        clearance = (torch.minimum(cone_ray, direct_ray) / max(self.parsed.ray_length, 1e-6)).clamp(0.0, 1.0)
        rollout_room = (cone_ray - self.parsed.safety_cone_ray_stop).clamp_min(0.0)
        rollout_need = (
            self.parsed.speed_fraction
            * self.parsed.recovery_rollout_steps
            * self.parsed.recovery_rollout_dt
        )
        rollout = (rollout_room / max(rollout_need, 1e-6)).clamp(0.0, 1.0)

        score = 1.25 * clearance
        score += 0.90 * rollout
        score += 0.55 * scale
        score += 0.45 * progress
        score = torch.where(progress >= -0.20, score, torch.full_like(score, -1.0e6))
        best = score.argmax(dim=1)
        best_dir = cand[torch.arange(cand.shape[0], device=obs.device), best]
        best_scale = scale[torch.arange(scale.shape[0], device=obs.device), best]

        recovery_speed = torch.full_like(default_speed, self.parsed.speed_fraction)
        recovery_xy = best_dir * recovery_speed * best_scale[:, None]
        out = default_actions.clone()
        out[:, 0:2] = torch.where(recover[:, None], recovery_xy, out[:, 0:2])
        return out

    def actions(self, target, obs):
        self._ensure(obs)
        coverage = obs[:, 17].float()
        improved = coverage > self.best_coverage + self.parsed.recovery_progress_delta
        self.best_coverage = torch.where(improved, coverage, self.best_coverage)
        self.since_progress = torch.where(
            improved,
            torch.zeros_like(self.since_progress),
            self.since_progress + 1,
        )
        self.recovery_timer = torch.where(
            improved,
            torch.zeros_like(self.recovery_timer),
            torch.clamp(self.recovery_timer - 1, min=0),
        )

        default_actions = target_to_actions(target, self.parsed, obs)
        if self.parsed.num_rays <= 0:
            return default_actions

        rays = obs[:, 26 : 26 + self.parsed.num_rays].float() * self.parsed.ray_length
        min_ray = rays.min(dim=1).values
        exists = target[:, 3].float() > self.parsed.exists_threshold
        trigger = (
            exists
            & (self.since_progress >= self.parsed.recovery_stall_steps)
            & (min_ray <= self.parsed.recovery_near_wall)
        )
        newly_triggered = trigger & (self.recovery_timer <= 0)
        if newly_triggered.any():
            self.recovery_side = torch.where(
                newly_triggered, self._choose_side(obs), self.recovery_side
            )
            self.recovery_timer = torch.where(
                newly_triggered,
                torch.full_like(self.recovery_timer, self.parsed.recovery_duration_steps),
                self.recovery_timer,
            )

        recover = exists & (self.recovery_timer > 0)
        if not recover.any():
            return default_actions
        return self._recovery_actions(target, obs, default_actions, recover).clamp(-1.0, 1.0)


class RecoveryPolicyNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 4),
            torch.nn.Tanh(),
        )
        self.input_size = int(input_size)

    def forward(self, x):
        return self.net(x.float())


def recovery_policy_features(target, obs, parsed, controller=None, input_size=None):
    base_actions = target_to_actions(target, parsed, obs)
    parts = [
        deployable_input(obs, parsed).float(),
        target.float(),
        base_actions.float(),
    ]
    if controller is not None and getattr(controller, "since_progress", None) is not None:
        since_progress = controller.since_progress.to(device=obs.device).float()
        recovery_timer = controller.recovery_timer.to(device=obs.device).float()
        recovery_side = controller.recovery_side.to(device=obs.device).float()
        denom = max(float(getattr(parsed, "recovery_stall_steps", 1)), 1.0)
        duration = max(float(getattr(parsed, "recovery_duration_steps", 1)), 1.0)
        parts.append((since_progress / denom).clamp(0.0, 4.0)[:, None])
        parts.append((recovery_timer / duration).clamp(0.0, 1.0)[:, None])
        parts.append(recovery_side.clamp(-1.0, 1.0)[:, None])
    features = torch.cat(parts, dim=1)
    if input_size is not None and features.shape[1] != input_size:
        if features.shape[1] > input_size:
            return features[:, :input_size]
        pad = torch.zeros(
            (features.shape[0], input_size - features.shape[1]),
            dtype=features.dtype,
            device=features.device,
        )
        return torch.cat((features, pad), dim=1)
    return features


def load_recovery_policy(path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    input_size = int(data["input_size"])
    hidden_size = int(data.get("hidden_size", 128))
    policy = RecoveryPolicyNet(input_size, hidden_size).to(device)
    policy.load_state_dict(data["state_dict"])
    policy.eval()
    return policy


class LearnedRecoveryController(StatefulRecoveryController):
    def __init__(self, parsed, policy):
        super().__init__(parsed)
        self.policy = policy

    def _recovery_actions(self, target, obs, default_actions, recover):
        with torch.no_grad():
            pred_actions = self.policy(
                recovery_policy_features(
                    target,
                    obs,
                    self.parsed,
                    controller=self,
                    input_size=getattr(self.policy, "input_size", None),
                )
            )
            if self.parsed.num_rays > 0 and self.parsed.safety_cone_ray_stop > 0.0:
                if getattr(self.parsed, "safety_slide", False):
                    pred_speed = torch.linalg.vector_norm(
                        pred_actions[:, 0:2], dim=1, keepdim=True
                    )
                    pred_unit = pred_actions[:, 0:2] / pred_speed.clamp_min(1e-6)
                    pred_actions = _slide_safety_actions(
                        pred_actions, obs, self.parsed, pred_unit
                    )
                else:
                    speed = torch.linalg.vector_norm(pred_actions[:, 0:2], dim=1, keepdim=True)
                    unit = pred_actions[:, 0:2] / speed.clamp_min(1e-6)
                    scale = _combined_safety_scale(obs, unit, self.parsed)
                    pred_actions = pred_actions.clone()
                    pred_actions[:, 0:2] = pred_actions[:, 0:2] * scale[:, None]
            pred_actions = pred_actions.clone()
            pred_actions[:, 2:3] = default_actions[:, 2:3]
            xy_speed = torch.linalg.vector_norm(pred_actions[:, 0:2], dim=1, keepdim=True)
            unit_y = torch.where(
                xy_speed > 1e-6,
                pred_actions[:, 1:2] / xy_speed.clamp_min(1e-6),
                default_actions[:, 3:4] / max(float(self.parsed.yaw_fraction), 1e-6),
            )
            pred_actions[:, 3:4] = (unit_y * self.parsed.yaw_fraction).clamp(-1.0, 1.0)
        out = default_actions.clone()
        out[:, 0:4] = torch.where(recover[:, None], pred_actions, out[:, 0:4])
        return out


def make_recovery_controller(parsed, device):
    recovery_policy = getattr(parsed, "recovery_policy", "")
    if recovery_policy:
        return LearnedRecoveryController(
            parsed,
            load_recovery_policy(recovery_policy, device),
        )
    if getattr(parsed, "safety_stateful_recovery", False):
        return StatefulRecoveryController(parsed)
    return None


def action_policy_sample_weights(obs, raw_actions, safe_actions, parsed):
    weight = torch.ones((obs.shape[0],), dtype=torch.float32, device=obs.device)
    correction_weight = getattr(parsed, "action_policy_correction_weight", 0.0)
    if correction_weight > 0.0:
        correction = torch.linalg.vector_norm(
            raw_actions[:, 0:2] - safe_actions[:, 0:2],
            dim=1,
        )
        correction = correction / max(float(parsed.speed_fraction), 1e-6)
        weight = weight + correction_weight * correction.clamp(0.0, 1.0)

    clearance_weight = getattr(parsed, "action_policy_clearance_weight", 0.0)
    if clearance_weight > 0.0 and parsed.num_rays > 0:
        rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
        min_ray = rays.min(dim=1).values
        if parsed.safety_slow_ray > parsed.safety_cone_ray_stop:
            near = (
                (parsed.safety_slow_ray - min_ray)
                / (parsed.safety_slow_ray - parsed.safety_cone_ray_stop)
            ).clamp(0.0, 1.0)
        else:
            near = (min_ray <= parsed.safety_cone_ray_stop).float()
        weight = weight + clearance_weight * near
    return weight


def action_policy_loss(pred, target, weights=None):
    loss = (pred.clamp(-1.0, 1.0) - target) ** 2
    if weights is None:
        return loss.mean()
    weights = weights.float()
    weights = weights / weights.mean().clamp_min(1e-6)
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    return (loss * weights).mean()


def action_policy_to_actions(raw_actions, parsed, obs=None):
    actions = raw_actions.float().clamp(-1.0, 1.0)
    if (
        obs is None
        or not getattr(parsed, "action_policy_safety", False)
        or parsed.num_rays <= 0
    ):
        return actions

    xy = actions[:, 0:2]
    norm = torch.linalg.vector_norm(xy, dim=1, keepdim=True)
    unit = xy / norm.clamp_min(1e-6)
    safe = actions.clone()
    if getattr(parsed, "safety_barrier_actions", False):
        safe = _barrier_steering_actions(safe, obs, unit, norm, parsed)
        if getattr(parsed, "safety_slide", False):
            safe = _slide_safety_actions(safe, obs, parsed, unit)
    elif getattr(parsed, "safety_sample_actions", False):
        safe = _sampled_safety_actions(safe, obs, parsed)
    elif getattr(parsed, "safety_slide", False):
        safe = _slide_safety_actions(safe, obs, parsed, unit)
    else:
        scale = _combined_safety_scale(obs, unit, parsed)
        safe[:, 0:2] = xy * scale[:, None]
    desired_speed = norm.clamp(max=max(float(parsed.speed_fraction), 1.0))
    safe = _progress_speed_floor(safe, obs, unit, desired_speed, parsed)
    safe = _passage_center_actions(safe, obs, unit, parsed)
    safe = _min_ray_speed_cap_actions(safe, obs, parsed)
    return safe.clamp(-1.0, 1.0)


def _unit_target_direction(target):
    direction = target[:, 0:2].clamp(-1.0, 1.0)
    return direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-6)


def hard_state_weights(obs, student_target, oracle_target, student_action, parsed):
    oracle_exists = oracle_target[:, 3] > parsed.exists_threshold
    if parsed.hard_min_coverage > 0.0:
        oracle_exists = oracle_exists & (obs[:, 17].float() >= parsed.hard_min_coverage)

    student_unit = _unit_target_direction(student_target)
    oracle_unit = _unit_target_direction(oracle_target)
    cos = (student_unit * oracle_unit).sum(dim=1).clamp(-1.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(cos))
    angle_bad = angle_deg > parsed.hard_angle_deg

    pred_ray = _direction_ray(obs, student_unit, parsed)
    oracle_ray = _direction_ray(obs, oracle_unit, parsed)
    clearance_bad = (oracle_ray - pred_ray) > parsed.hard_clearance_gap

    action_speed = torch.linalg.vector_norm(student_action[:, 0:2], dim=1)
    if parsed.hard_low_speed > 0.0:
        speed_bad = action_speed < parsed.hard_low_speed
    else:
        speed_bad = torch.zeros_like(oracle_exists)

    hard = oracle_exists & (angle_bad | clearance_bad | speed_bad)
    weights = torch.ones((obs.shape[0],), dtype=torch.float32, device=obs.device)
    return weights + parsed.hard_example_weight * hard.float()


def predict_sequence(policy, x, start_state, dones):
    preds = []
    state = start_state
    for t in range(x.shape[1]):
        logits, _, state = policy.forward_eval(x[:, t], state)
        preds.append(logits.mean)
        state = reset_done_state(state, dones[:, t])
    return torch.stack(preds, dim=1)


def predict_sequence_with_hidden(policy, x, start_state, dones):
    preds = []
    hiddens = []
    state = start_state
    for t in range(x.shape[1]):
        h = policy.encoder(x[:, t])
        h, state = policy.network.forward_eval(h, state)
        logits, _ = policy.decoder(h)
        preds.append(logits.mean)
        hiddens.append(h)
        state = reset_done_state(state, dones[:, t])
    return torch.stack(preds, dim=1), torch.stack(hiddens, dim=1)


def target_loss(pred, target, parsed, weights=None):
    exists = target[..., 3:4].clamp(0.0, 1.0)
    absent = 1.0 - exists
    if weights is not None:
        weights = weights.float()
        weights = weights / weights.mean().clamp_min(1e-6)
        weights = weights[..., None]

        direction_loss = ((pred[..., 0:2] - target[..., 0:2]) ** 2 * exists * weights).sum()
        direction_loss = direction_loss / ((exists * weights).sum().clamp_min(1.0) * 2.0)

        absent_direction_loss = ((pred[..., 0:2]) ** 2 * absent * weights).sum()
        absent_direction_loss = absent_direction_loss / ((absent * weights).sum().clamp_min(1.0) * 2.0)

        dist_loss = ((pred[..., 2:3].clamp(0.0, 1.0) - target[..., 2:3]) ** 2 * weights).sum()
        dist_loss = dist_loss / weights.sum().clamp_min(1.0)

        exists_weight = torch.where(exists > 0.5, 1.0, parsed.negative_exists_weight)
        exists_loss = ((pred[..., 3:4] - target[..., 3:4]) ** 2 * exists_weight * weights).sum()
        exists_loss = exists_loss / weights.sum().clamp_min(1.0)

        false_positive_loss = (
            F.relu(pred[..., 3:4] - parsed.false_positive_margin) ** 2 * absent * weights
        ).sum()
        false_positive_loss = false_positive_loss / (absent * weights).sum().clamp_min(1.0)

        speed_loss = torch.zeros((), dtype=torch.float32, device=pred.device)
        if (
            pred.shape[-1] > 4
            and target.shape[-1] > 4
            and getattr(parsed, "predict_speed_scale", False)
        ):
            speed_loss = ((pred[..., 4:5].clamp(0.0, 1.0) - target[..., 4:5]) ** 2 * weights).sum()
            speed_loss = speed_loss / weights.sum().clamp_min(1.0)

        return (
            direction_loss
            + parsed.absent_direction_weight * absent_direction_loss
            + parsed.dist_loss_weight * dist_loss
            + parsed.exists_loss_weight * exists_loss
            + parsed.false_positive_weight * false_positive_loss
            + parsed.speed_loss_weight * speed_loss
        )

    direction_loss = ((pred[..., 0:2] - target[..., 0:2]) ** 2 * exists).sum()
    direction_loss = direction_loss / (exists.sum().clamp_min(1.0) * 2.0)

    absent_direction_loss = ((pred[..., 0:2]) ** 2 * absent).sum()
    absent_direction_loss = absent_direction_loss / (absent.sum().clamp_min(1.0) * 2.0)

    dist_loss = F.mse_loss(pred[..., 2:3].clamp(0.0, 1.0), target[..., 2:3])

    exists_weight = torch.where(exists > 0.5, 1.0, parsed.negative_exists_weight)
    exists_loss = ((pred[..., 3:4] - target[..., 3:4]) ** 2 * exists_weight).mean()
    false_positive_loss = (F.relu(pred[..., 3:4] - parsed.false_positive_margin) ** 2 * absent).sum()
    false_positive_loss = false_positive_loss / absent.sum().clamp_min(1.0)
    speed_loss = torch.zeros((), dtype=torch.float32, device=pred.device)
    if (
        pred.shape[-1] > 4
        and target.shape[-1] > 4
        and getattr(parsed, "predict_speed_scale", False)
    ):
        speed_loss = F.mse_loss(pred[..., 4:5].clamp(0.0, 1.0), target[..., 4:5])

    return (
        direction_loss
        + parsed.absent_direction_weight * absent_direction_loss
        + parsed.dist_loss_weight * dist_loss
        + parsed.exists_loss_weight * exists_loss
        + parsed.false_positive_weight * false_positive_loss
        + parsed.speed_loss_weight * speed_loss
    )


def false_negative_exists_loss(pred, target, parsed, coverage=None, weights=None):
    exists = (target[..., 3:4] > parsed.exists_threshold).float()
    if coverage is not None and parsed.false_negative_min_coverage > 0.0:
        exists = exists * (coverage[..., None] >= parsed.false_negative_min_coverage).float()
    if weights is not None:
        weights = weights.float()
        weights = weights / weights.mean().clamp_min(1e-6)
        weights = weights[..., None]
    else:
        weights = torch.ones_like(exists)

    loss = F.relu(parsed.false_negative_margin - pred[..., 3:4]) ** 2
    denom = (exists * weights).sum().clamp_min(1.0)
    return (loss * exists * weights).sum() / denom


def frontier_sector_loss(pred, target, weights=None):
    if weights is not None:
        weights = weights.float()
        weights = weights / weights.mean().clamp_min(1e-6)
        weights = weights[..., None]
    else:
        weights = torch.ones_like(target[..., 4:5])
    loss = (pred[..., 4:8].clamp(0.0, 1.0) - target[..., 4:8]) ** 2
    return (loss * weights).sum() / (weights.sum().clamp_min(1.0) * 4.0)


def frontier_vector_loss(pred, target, parsed, weights=None):
    exists = (target[..., 10:11] > 1e-5).float()
    if weights is not None:
        weights = weights.float()
        weights = weights / weights.mean().clamp_min(1e-6)
        weights = weights[..., None]
    else:
        weights = torch.ones_like(exists)

    direction_loss = ((pred[..., 8:10].clamp(-1.0, 1.0) - target[..., 8:10]) ** 2).sum(
        dim=-1,
        keepdim=True,
    )
    dist_pred = pred[..., 10:11]
    if not getattr(parsed, "unclamped_frontier_vector_dist_loss", False):
        dist_pred = dist_pred.clamp(0.0, 1.0)
    dist_loss = (dist_pred - target[..., 10:11]) ** 2
    loss = direction_loss + parsed.frontier_vector_dist_loss_weight * dist_loss
    denom = (exists * weights).sum().clamp_min(1.0)
    return (loss * exists * weights).sum() / denom


def frontier_patch_loss(pred, target, weights=None):
    if weights is not None:
        weights = weights.float()
        weights = weights / weights.mean().clamp_min(1e-6)
        weights = weights[..., None]
    else:
        weights = torch.ones_like(target[..., 11:12])
    loss = (pred[..., 11:27].clamp(0.0, 1.0) - target[..., 11:27]) ** 2
    return (loss * weights).sum() / (weights.sum().clamp_min(1.0) * FRONTIER_PATCH_SIZE)


def frontier_patch_aux_loss(patch_pred, target, weights=None):
    if weights is not None:
        weights = weights.float()
        weights = weights / weights.mean().clamp_min(1e-6)
        weights = weights[..., None]
    else:
        weights = torch.ones_like(target[..., 11:12])
    loss = (patch_pred.clamp(0.0, 1.0) - target[..., 11:27]) ** 2
    return (loss * weights).sum() / (weights.sum().clamp_min(1.0) * FRONTIER_PATCH_SIZE)


def recovery_gate_targets(obs, pred, target, parsed):
    gate = torch.zeros((*pred.shape[:-1], 1), dtype=torch.float32, device=pred.device)
    if pred.shape[-1] < 12 or target.shape[-1] < 11:
        return gate

    coverage_ok = torch.ones_like(gate, dtype=torch.bool)
    min_coverage = getattr(parsed, "recovery_gate_min_coverage", 0.25)
    if obs is not None and min_coverage > 0.0 and obs.shape[-1] > 17:
        coverage_ok = obs[..., 17:18].float() >= min_coverage

    vector_dist = target[..., 10:11].clamp(0.0, 1.0)
    vector_ready = vector_dist >= getattr(parsed, "recovery_gate_min_dist", 0.02)
    oracle_exists = target[..., 3:4] > getattr(parsed, "exists_threshold", 0.2)

    student_unit = _unit_target_direction(pred.reshape(-1, pred.shape[-1])).reshape(
        *pred.shape[:-1],
        2,
    )
    oracle_unit = _unit_target_direction(target.reshape(-1, target.shape[-1])).reshape(
        *target.shape[:-1],
        2,
    )
    vector = target[..., 8:10].clamp(-1.0, 1.0)
    vector_unit = vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-6)

    student_cos = (student_unit * oracle_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    vector_cos = (vector_unit * oracle_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    student_angle = torch.rad2deg(torch.acos(student_cos))
    vector_angle = torch.rad2deg(torch.acos(vector_cos))
    student_bad = student_angle >= getattr(parsed, "recovery_gate_bad_angle_deg", 60.0)
    margin = getattr(parsed, "recovery_gate_margin_deg", 10.0)
    vector_better = (vector_angle + margin) < student_angle

    use_vector = vector_ready & coverage_ok & (~oracle_exists | (student_bad & vector_better))
    return use_vector.float()


def recovery_gate_loss(pred, target, obs, parsed, weights=None):
    labels = recovery_gate_targets(obs, pred.detach(), target, parsed)
    if weights is not None:
        weights = weights.float()
        weights = weights / weights.mean().clamp_min(1e-6)
        weights = weights[..., None]
    else:
        weights = torch.ones_like(labels)
    loss = F.binary_cross_entropy_with_logits(pred[..., 11:12], labels, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0), labels.mean()


def freeze_except_frontier_vector_dist(policy):
    for name, param in policy.named_parameters():
        param.requires_grad = name in (
            "decoder.decoder_mean.weight",
            "decoder.decoder_mean.bias",
        )


def mask_frontier_vector_dist_grads(policy):
    for name, param in policy.named_parameters():
        if param.grad is None:
            continue
        if name == "decoder.decoder_mean.weight":
            keep = param.grad.new_zeros((param.grad.shape[0], 1))
            if keep.shape[0] > 10:
                keep[10] = 1.0
            param.grad.mul_(keep)
        elif name == "decoder.decoder_mean.bias":
            keep = param.grad.new_zeros(param.grad.shape)
            if keep.shape[0] > 10:
                keep[10] = 1.0
            param.grad.mul_(keep)
        else:
            param.grad.zero_()


def freeze_except_recovery_gate(policy):
    for name, param in policy.named_parameters():
        param.requires_grad = name in (
            "decoder.decoder_mean.weight",
            "decoder.decoder_mean.bias",
        )


def mask_recovery_gate_grads(policy):
    for name, param in policy.named_parameters():
        if param.grad is None:
            continue
        if name == "decoder.decoder_mean.weight":
            keep = param.grad.new_zeros((param.grad.shape[0], 1))
            if keep.shape[0] > 11:
                keep[11] = 1.0
            param.grad.mul_(keep)
        elif name == "decoder.decoder_mean.bias":
            keep = param.grad.new_zeros(param.grad.shape)
            if keep.shape[0] > 11:
                keep[11] = 1.0
            param.grad.mul_(keep)
        else:
            param.grad.zero_()


def zero_encoder_input_columns(policy, columns):
    weight = getattr(getattr(policy, "encoder", None), "encoder", None)
    if weight is None or not hasattr(weight, "weight"):
        return []
    changed = []
    with torch.no_grad():
        for column in columns:
            if 0 <= column < weight.weight.shape[1]:
                weight.weight[:, column].zero_()
                changed.append(column)
    return changed


def freeze_except_new_input_columns(policy):
    for name, param in policy.named_parameters():
        param.requires_grad = name == "encoder.encoder.weight"


def mask_new_input_grads(policy, start_column):
    weight = getattr(getattr(policy, "encoder", None), "encoder", None)
    if weight is None or not hasattr(weight, "weight") or weight.weight.grad is None:
        return
    start_column = max(0, min(int(start_column), weight.weight.grad.shape[1]))
    weight.weight.grad[:, :start_column].zero_()


def freeze_except_map_sidecar(policy):
    for name, param in policy.named_parameters():
        param.requires_grad = name.startswith("encoder.map_net.")


def load_resized_output_state(policy, state_dict, resize_input=False, resize_output=True):
    current = policy.state_dict()
    copied = []
    resized = []
    for key, value in state_dict.items():
        if key not in current:
            continue
        if current[key].shape == value.shape:
            current[key] = value
            copied.append(key)
            continue
        if (
            resize_input
            and key == "encoder.encoder.weight"
            and current[key].ndim == value.ndim == 2
            and current[key].shape[0] == value.shape[0]
            and current[key].shape[1] >= value.shape[1]
        ):
            current[key] = current[key].clone()
            current[key].zero_()
            current[key][:, : value.shape[1]] = value
            resized.append(key)
        elif (
            resize_output
            and key in ("decoder.decoder_mean.weight", "decoder.decoder_mean.bias")
            and current[key].ndim == value.ndim
            and current[key].shape[0] >= value.shape[0]
            and current[key].shape[1:] == value.shape[1:]
        ):
            current[key] = current[key].clone()
            current[key][: value.shape[0]] = value
            resized.append(key)
        elif (
            resize_output
            and key == "decoder.decoder_logstd"
            and current[key].ndim == value.ndim == 2
            and current[key].shape[0] == value.shape[0]
            and current[key].shape[1] >= value.shape[1]
        ):
            current[key] = current[key].clone()
            current[key][:, : value.shape[1]] = value
            resized.append(key)
    policy.load_state_dict(current)
    return copied, resized


def load_policy(path, vec, parsed, device):
    data = torch.load(path, map_location=device, weights_only=False)
    meta = data.get("policy", {})
    config = data.get("config", {})
    parsed.network = meta.get("network", parsed.network)
    parsed.hidden_size = int(meta.get("hidden_size", parsed.hidden_size))
    parsed.num_layers = int(meta.get("num_layers", parsed.num_layers))
    parsed.action_std = float(meta.get("action_std", parsed.action_std))
    parsed.use_pred_speed_scale = bool(config.get("predict_speed_scale", parsed.use_pred_speed_scale))
    parsed.predict_frontier_sectors = bool(
        config.get("predict_frontier_sectors", getattr(parsed, "predict_frontier_sectors", False))
    )
    parsed.predict_frontier_vector = bool(
        config.get("predict_frontier_vector", getattr(parsed, "predict_frontier_vector", False))
    )
    parsed.predict_recovery_gate = bool(
        config.get("predict_recovery_gate", getattr(parsed, "predict_recovery_gate", False))
    )
    parsed.predict_frontier_patch = bool(
        config.get("predict_frontier_patch", getattr(parsed, "predict_frontier_patch", False))
    )
    parsed.predict_graph_candidate = bool(
        config.get("predict_graph_candidate", getattr(parsed, "predict_graph_candidate", False))
    )
    parsed.graph_candidate_label_source = config.get(
        "graph_candidate_label_source",
        getattr(parsed, "graph_candidate_label_source", "privileged"),
    )
    parsed.action_policy = bool(
        config.get("action_policy", getattr(parsed, "action_policy", False))
    )
    parsed.action_policy_safety = bool(
        config.get("action_policy_safety", False)
        or getattr(parsed, "action_policy_safety", False)
    )
    parsed.action_policy_correction_weight = float(
        config.get(
            "action_policy_correction_weight",
            getattr(parsed, "action_policy_correction_weight", 0.0),
        )
    )
    parsed.action_policy_clearance_weight = float(
        config.get(
            "action_policy_clearance_weight",
            getattr(parsed, "action_policy_clearance_weight", 0.0),
        )
    )
    parsed.safety_slide = bool(
        config.get("safety_slide", False)
        or getattr(parsed, "safety_slide", False)
    )
    parsed.safety_slide_progress_aware = bool(
        config.get("safety_slide_progress_aware", False)
        or getattr(parsed, "safety_slide_progress_aware", False)
    )
    parsed.safety_slide_progress_fraction = float(
        config.get(
            "safety_slide_progress_fraction",
            getattr(parsed, "safety_slide_progress_fraction", 0.60),
        )
    )
    parsed.safety_slide_progress_cone = float(
        config.get(
            "safety_slide_progress_cone",
            getattr(parsed, "safety_slide_progress_cone", 0.60),
        )
    )
    parsed.safety_barrier_actions = bool(
        config.get("safety_barrier_actions", False)
        or getattr(parsed, "safety_barrier_actions", False)
    )
    parsed.safety_barrier_near_wall = float(
        config.get(
            "safety_barrier_near_wall",
            getattr(parsed, "safety_barrier_near_wall", 0.36),
        )
    )
    parsed.safety_min_ray_speed_cap = bool(
        config.get("safety_min_ray_speed_cap", False)
        or getattr(parsed, "safety_min_ray_speed_cap", False)
    )
    parsed.safety_min_ray_speed_stop = float(
        config.get(
            "safety_min_ray_speed_stop",
            getattr(parsed, "safety_min_ray_speed_stop", 0.18),
        )
    )
    parsed.safety_min_ray_speed_slow = float(
        config.get(
            "safety_min_ray_speed_slow",
            getattr(parsed, "safety_min_ray_speed_slow", 0.45),
        )
    )
    parsed.safety_min_ray_speed_min_scale = float(
        config.get(
            "safety_min_ray_speed_min_scale",
            getattr(parsed, "safety_min_ray_speed_min_scale", 0.0),
        )
    )
    parsed.safety_passage_speed_cap = bool(
        config.get("safety_passage_speed_cap", False)
        or getattr(parsed, "safety_passage_speed_cap", False)
    )
    parsed.safety_passage_speed_near = float(
        config.get(
            "safety_passage_speed_near",
            getattr(parsed, "safety_passage_speed_near", 0.45),
        )
    )
    parsed.safety_passage_speed_front = float(
        config.get(
            "safety_passage_speed_front",
            getattr(parsed, "safety_passage_speed_front", 0.80),
        )
    )
    parsed.safety_passage_speed_side_gap = float(
        config.get(
            "safety_passage_speed_side_gap",
            getattr(parsed, "safety_passage_speed_side_gap", 0.20),
        )
    )
    parsed.safety_passage_speed_stop = float(
        config.get(
            "safety_passage_speed_stop",
            getattr(parsed, "safety_passage_speed_stop", 0.20),
        )
    )
    parsed.safety_passage_speed_slow = float(
        config.get(
            "safety_passage_speed_slow",
            getattr(parsed, "safety_passage_speed_slow", 0.60),
        )
    )
    parsed.safety_passage_speed_min_scale = float(
        config.get(
            "safety_passage_speed_min_scale",
            getattr(parsed, "safety_passage_speed_min_scale", 0.0),
        )
    )
    parsed.safety_passage_center_actions = bool(
        config.get("safety_passage_center_actions", False)
        or getattr(parsed, "safety_passage_center_actions", False)
    )
    parsed.safety_passage_center_near = float(
        config.get(
            "safety_passage_center_near",
            getattr(parsed, "safety_passage_center_near", 0.45),
        )
    )
    parsed.safety_passage_center_front = float(
        config.get(
            "safety_passage_center_front",
            getattr(parsed, "safety_passage_center_front", 0.60),
        )
    )
    parsed.safety_passage_center_gain = float(
        config.get(
            "safety_passage_center_gain",
            getattr(parsed, "safety_passage_center_gain", 0.80),
        )
    )
    parsed.safety_passage_center_max_turn_deg = float(
        config.get(
            "safety_passage_center_max_turn_deg",
            getattr(parsed, "safety_passage_center_max_turn_deg", 35.0),
        )
    )
    parsed.safety_passage_center_min_imbalance = float(
        config.get(
            "safety_passage_center_min_imbalance",
            getattr(parsed, "safety_passage_center_min_imbalance", 0.05),
        )
    )
    parsed.safety_passage_center_actions = bool(
        config.get("safety_passage_center_actions", False)
        or getattr(parsed, "safety_passage_center_actions", False)
    )
    parsed.safety_passage_center_near = float(
        config.get(
            "safety_passage_center_near",
            getattr(parsed, "safety_passage_center_near", 0.45),
        )
    )
    parsed.safety_passage_center_front = float(
        config.get(
            "safety_passage_center_front",
            getattr(parsed, "safety_passage_center_front", 0.60),
        )
    )
    parsed.safety_passage_center_gain = float(
        config.get(
            "safety_passage_center_gain",
            getattr(parsed, "safety_passage_center_gain", 0.80),
        )
    )
    parsed.safety_passage_center_max_turn_deg = float(
        config.get(
            "safety_passage_center_max_turn_deg",
            getattr(parsed, "safety_passage_center_max_turn_deg", 35.0),
        )
    )
    parsed.safety_passage_center_min_imbalance = float(
        config.get(
            "safety_passage_center_min_imbalance",
            getattr(parsed, "safety_passage_center_min_imbalance", 0.05),
        )
    )
    parsed.safety_corridor_width_actions = bool(
        config.get("safety_corridor_width_actions", False)
        or getattr(parsed, "safety_corridor_width_actions", False)
    )
    parsed.safety_corridor_width_near = float(
        config.get(
            "safety_corridor_width_near",
            getattr(parsed, "safety_corridor_width_near", 0.55),
        )
    )
    parsed.safety_corridor_width_front = float(
        config.get(
            "safety_corridor_width_front",
            getattr(parsed, "safety_corridor_width_front", 0.70),
        )
    )
    parsed.safety_corridor_width_side_stop = float(
        config.get(
            "safety_corridor_width_side_stop",
            getattr(parsed, "safety_corridor_width_side_stop", 0.24),
        )
    )
    parsed.safety_corridor_width_side_slow = float(
        config.get(
            "safety_corridor_width_side_slow",
            getattr(parsed, "safety_corridor_width_side_slow", 0.65),
        )
    )
    parsed.safety_corridor_width_min_scale = float(
        config.get(
            "safety_corridor_width_min_scale",
            getattr(parsed, "safety_corridor_width_min_scale", 0.35),
        )
    )
    parsed.safety_corridor_width_max_turn_deg = float(
        config.get(
            "safety_corridor_width_max_turn_deg",
            getattr(parsed, "safety_corridor_width_max_turn_deg", 55.0),
        )
    )
    parsed.safety_corridor_width_side_deg = float(
        config.get(
            "safety_corridor_width_side_deg",
            getattr(parsed, "safety_corridor_width_side_deg", 80.0),
        )
    )
    parsed.safety_corridor_width_progress_floor = float(
        config.get(
            "safety_corridor_width_progress_floor",
            getattr(parsed, "safety_corridor_width_progress_floor", 0.10),
        )
    )
    parsed.safety_trajectory_actions = bool(
        config.get("safety_trajectory_actions", False)
        or getattr(parsed, "safety_trajectory_actions", False)
    )
    parsed.safety_trajectory_near_wall = float(
        config.get(
            "safety_trajectory_near_wall",
            getattr(parsed, "safety_trajectory_near_wall", 0.42),
        )
    )
    parsed.safety_trajectory_steps = int(
        config.get(
            "safety_trajectory_steps",
            getattr(parsed, "safety_trajectory_steps", 8),
        )
    )
    parsed.safety_trajectory_dt = float(
        config.get(
            "safety_trajectory_dt",
            getattr(parsed, "safety_trajectory_dt", 0.08),
        )
    )
    parsed.safety_trajectory_min_fit = float(
        config.get(
            "safety_trajectory_min_fit",
            getattr(parsed, "safety_trajectory_min_fit", -0.25),
        )
    )
    parsed.safety_trajectory_trigger_scale = float(
        config.get(
            "safety_trajectory_trigger_scale",
            getattr(parsed, "safety_trajectory_trigger_scale", 0.45),
        )
    )
    parsed.safety_trajectory_direct_min_fit = float(
        config.get(
            "safety_trajectory_direct_min_fit",
            getattr(parsed, "safety_trajectory_direct_min_fit", 0.20),
        )
    )
    parsed.map_sidecar_encoder = bool(
        config.get("map_sidecar_encoder", getattr(parsed, "map_sidecar_encoder", False))
    )
    parsed.far_map_start = int(config.get("far_map_start", getattr(parsed, "far_map_start", FAR_MAP_START)))
    parsed.far_map_side = int(config.get("far_map_side", getattr(parsed, "far_map_side", FAR_MAP_SIDE)))
    parsed.deploy_patch_obs = bool(
        config.get("deploy_patch_obs", getattr(parsed, "deploy_patch_obs", False))
    )
    parsed.deploy_graph_candidate_obs = bool(
        config.get(
            "deploy_graph_candidate_obs",
            getattr(parsed, "deploy_graph_candidate_obs", False),
        )
        or getattr(parsed, "deploy_graph_candidate_obs", False)
        or parsed.predict_graph_candidate
    )
    parsed.deploy_coverage_obs = bool(
        config.get("deploy_coverage_obs", getattr(parsed, "deploy_coverage_obs", False))
    )
    parsed.deploy_onboard_target_obs = bool(
        config.get(
            "deploy_onboard_target_obs",
            getattr(parsed, "deploy_onboard_target_obs", False),
        )
    )
    parsed.explore_frontier_target_mode = config.get(
        "explore_frontier_target_mode",
        getattr(parsed, "explore_frontier_target_mode", "nearest"),
    )
    parsed.cell_size = float(
        config.get("cell_size", getattr(parsed, "cell_size", 0.25))
    )
    parsed.explore_reachable_coverage = bool(
        config.get(
            "explore_reachable_coverage",
            getattr(parsed, "explore_reachable_coverage", False),
        )
    )
    parsed.explore_onboard_path_lookahead = int(
        config.get(
            "explore_onboard_path_lookahead",
            getattr(parsed, "explore_onboard_path_lookahead", 1),
        )
    )
    parsed.explore_onboard_adaptive_lookahead = bool(
        config.get("explore_onboard_adaptive_lookahead", False)
        or getattr(parsed, "explore_onboard_adaptive_lookahead", False)
    )
    parsed.explore_onboard_adaptive_lookahead_min = int(
        config.get(
            "explore_onboard_adaptive_lookahead_min",
            getattr(parsed, "explore_onboard_adaptive_lookahead_min", 2),
        )
    )
    parsed.explore_onboard_adaptive_lookahead_max = int(
        config.get(
            "explore_onboard_adaptive_lookahead_max",
            getattr(parsed, "explore_onboard_adaptive_lookahead_max", 4),
        )
    )
    parsed.explore_onboard_adaptive_clearance = float(
        config.get(
            "explore_onboard_adaptive_clearance",
            getattr(parsed, "explore_onboard_adaptive_clearance", 0.55),
        )
    )
    parsed.explore_onboard_adaptive_turn_dot = float(
        config.get(
            "explore_onboard_adaptive_turn_dot",
            getattr(parsed, "explore_onboard_adaptive_turn_dot", 0.55),
        )
    )
    parsed.explore_onboard_emit_objective_target = bool(
        config.get(
            "explore_onboard_emit_objective_target",
            getattr(parsed, "explore_onboard_emit_objective_target", False),
        )
        or getattr(parsed, "explore_onboard_emit_objective_target", False)
    )
    parsed.explore_onboard_objective_obs = bool(
        config.get(
            "explore_onboard_objective_obs",
            getattr(parsed, "explore_onboard_objective_obs", False),
        )
        or getattr(parsed, "explore_onboard_objective_obs", False)
    )
    parsed.explore_onboard_region_commit = bool(
        config.get("explore_onboard_region_commit", False)
        or getattr(parsed, "explore_onboard_region_commit", False)
    )
    parsed.explore_onboard_region_commit_stall_ticks = int(
        config.get(
            "explore_onboard_region_commit_stall_ticks",
            getattr(parsed, "explore_onboard_region_commit_stall_ticks", 0),
        )
    )
    parsed.explore_onboard_persistent_regions = bool(
        config.get("explore_onboard_persistent_regions", False)
        or getattr(parsed, "explore_onboard_persistent_regions", False)
    )
    parsed.explore_onboard_region_tracker = bool(
        config.get("explore_onboard_region_tracker", False)
        or getattr(parsed, "explore_onboard_region_tracker", False)
    )
    parsed.explore_onboard_clearance_passable = bool(
        config.get("explore_onboard_clearance_passable", False)
        or getattr(parsed, "explore_onboard_clearance_passable", False)
    )
    parsed.explore_onboard_passable_radius = float(
        config.get(
            "explore_onboard_passable_radius",
            getattr(parsed, "explore_onboard_passable_radius", 0.0),
        )
    )
    parsed.explore_onboard_passage_quality = bool(
        config.get("explore_onboard_passage_quality", False)
        or getattr(parsed, "explore_onboard_passage_quality", False)
    )
    parsed.explore_onboard_passage_weight = float(
        config.get(
            "explore_onboard_passage_weight",
            getattr(parsed, "explore_onboard_passage_weight", 0.35),
        )
    )
    parsed.explore_onboard_clearance_cone_deg = float(
        config.get(
            "explore_onboard_clearance_cone_deg",
            getattr(parsed, "explore_onboard_clearance_cone_deg", 30.0),
        )
    )
    parsed.explore_onboard_budget_frontier = bool(
        config.get("explore_onboard_budget_frontier", False)
        or getattr(parsed, "explore_onboard_budget_frontier", False)
    )
    parsed.explore_onboard_budget_gain_weight = float(
        config.get(
            "explore_onboard_budget_gain_weight",
            getattr(parsed, "explore_onboard_budget_gain_weight", 1.0),
        )
    )
    parsed.explore_onboard_budget_cluster_weight = float(
        config.get(
            "explore_onboard_budget_cluster_weight",
            getattr(parsed, "explore_onboard_budget_cluster_weight", 0.35),
        )
    )
    parsed.explore_onboard_budget_cell_weight = float(
        config.get(
            "explore_onboard_budget_cell_weight",
            getattr(parsed, "explore_onboard_budget_cell_weight", 0.25),
        )
    )
    parsed.explore_onboard_budget_depth_bias = float(
        config.get(
            "explore_onboard_budget_depth_bias",
            getattr(parsed, "explore_onboard_budget_depth_bias", 6.0),
        )
    )
    parsed.explore_onboard_budget_depth_weight = float(
        config.get(
            "explore_onboard_budget_depth_weight",
            getattr(parsed, "explore_onboard_budget_depth_weight", 1.0),
        )
    )
    parsed.explore_onboard_optimistic_unknown = bool(
        config.get("explore_onboard_optimistic_unknown", False)
        or getattr(parsed, "explore_onboard_optimistic_unknown", False)
    )
    parsed.explore_onboard_frontier_graph = bool(
        config.get("explore_onboard_frontier_graph", False)
        or getattr(parsed, "explore_onboard_frontier_graph", False)
    )
    parsed.explore_onboard_graph_candidate_obs = bool(
        config.get("explore_onboard_graph_candidate_obs", False)
        or getattr(parsed, "explore_onboard_graph_candidate_obs", False)
        or getattr(parsed, "deploy_graph_candidate_obs", False)
        or getattr(parsed, "predict_graph_candidate", False)
    )
    parsed.explore_onboard_graph_gain_weight = float(
        config.get(
            "explore_onboard_graph_gain_weight",
            getattr(parsed, "explore_onboard_graph_gain_weight", 0.32),
        )
    )
    parsed.explore_onboard_graph_cluster_weight = float(
        config.get(
            "explore_onboard_graph_cluster_weight",
            getattr(parsed, "explore_onboard_graph_cluster_weight", 0.12),
        )
    )
    parsed.explore_onboard_graph_depth_weight = float(
        config.get(
            "explore_onboard_graph_depth_weight",
            getattr(parsed, "explore_onboard_graph_depth_weight", 0.045),
        )
    )
    parsed.explore_onboard_graph_active_bonus = float(
        config.get(
            "explore_onboard_graph_active_bonus",
            getattr(parsed, "explore_onboard_graph_active_bonus", 0.75),
        )
    )
    parsed.explore_onboard_graph_stale_penalty = float(
        config.get(
            "explore_onboard_graph_stale_penalty",
            getattr(parsed, "explore_onboard_graph_stale_penalty", 0.020),
        )
    )
    parsed.explore_onboard_graph_failure_penalty = float(
        config.get(
            "explore_onboard_graph_failure_penalty",
            getattr(parsed, "explore_onboard_graph_failure_penalty", 0.70),
        )
    )
    parsed.explore_onboard_graph_attempt_penalty = float(
        config.get(
            "explore_onboard_graph_attempt_penalty",
            getattr(parsed, "explore_onboard_graph_attempt_penalty", 0.12),
        )
    )
    parsed.explore_onboard_graph_gain_cap = int(
        config.get(
            "explore_onboard_graph_gain_cap",
            getattr(parsed, "explore_onboard_graph_gain_cap", 192),
        )
    )
    parsed.explore_onboard_approach_viewpoints = bool(
        config.get("explore_onboard_approach_viewpoints", False)
        or getattr(parsed, "explore_onboard_approach_viewpoints", False)
    )
    parsed.explore_onboard_approach_max_depth = int(
        config.get(
            "explore_onboard_approach_max_depth",
            getattr(parsed, "explore_onboard_approach_max_depth", 1),
        )
    )
    parsed.drive_objective_when_close = bool(
        config.get("drive_objective_when_close", False)
        or getattr(parsed, "drive_objective_when_close", False)
    )
    parsed.drive_objective_close_dist = float(
        config.get(
            "drive_objective_close_dist",
            getattr(parsed, "drive_objective_close_dist", 0.04),
        )
    )
    parsed.drive_objective_min_dist = float(
        config.get(
            "drive_objective_min_dist",
            getattr(parsed, "drive_objective_min_dist", 0.08),
        )
    )
    parsed.unclamped_frontier_vector_dist_loss = bool(
        config.get(
            "unclamped_frontier_vector_dist_loss",
            getattr(parsed, "unclamped_frontier_vector_dist_loss", False),
        )
    )
    parsed.frontier_vector_dist_loss_weight = float(
        config.get(
            "frontier_vector_dist_loss_weight",
            getattr(parsed, "frontier_vector_dist_loss_weight", 0.5),
        )
    )
    parsed.freeze_except_frontier_vector_dist = bool(
        config.get(
            "freeze_except_frontier_vector_dist",
            getattr(parsed, "freeze_except_frontier_vector_dist", False),
        )
    )
    parsed.use_pred_vector_fallback = bool(
        config.get("use_pred_vector_fallback", False)
        or getattr(parsed, "use_pred_vector_fallback", False)
    )
    parsed.pred_vector_fallback_primary_threshold = float(
        config.get(
            "pred_vector_fallback_primary_threshold",
            getattr(parsed, "pred_vector_fallback_primary_threshold", 0.2),
        )
    )
    parsed.pred_vector_fallback_min_coverage = float(
        config.get(
            "pred_vector_fallback_min_coverage",
            getattr(parsed, "pred_vector_fallback_min_coverage", 0.0),
        )
    )
    parsed.pred_vector_fallback_min_dist = float(
        config.get(
            "pred_vector_fallback_min_dist",
            getattr(parsed, "pred_vector_fallback_min_dist", 0.02),
        )
    )
    parsed.use_pred_recovery_gate = bool(
        config.get("use_pred_recovery_gate", False)
        or getattr(parsed, "use_pred_recovery_gate", False)
    )
    parsed.pred_recovery_gate_threshold = float(
        config.get(
            "pred_recovery_gate_threshold",
            getattr(parsed, "pred_recovery_gate_threshold", 0.5),
        )
    )
    parsed.pred_recovery_gate_min_coverage = float(
        config.get(
            "pred_recovery_gate_min_coverage",
            getattr(parsed, "pred_recovery_gate_min_coverage", 0.25),
        )
    )
    parsed.pred_recovery_gate_min_dist = float(
        config.get(
            "pred_recovery_gate_min_dist",
            getattr(parsed, "pred_recovery_gate_min_dist", 0.02),
        )
    )
    policy = create_policy(vec.obs_size, data.get("act_sizes", [1, 1, 1, 1]), parsed, device)
    if int(data["input_size"]) == vec.obs_size:
        policy.load_state_dict(data["state_dict"])
    else:
        copied, resized = load_resized_output_state(
            policy,
            data["state_dict"],
            resize_input=True,
            resize_output=False,
        )
        print(
            f"loaded checkpoint {path} with resized input "
            f"input_size={data['input_size']} env_obs_size={vec.obs_size} "
            f"copied={len(copied)} resized={','.join(resized)}",
            flush=True,
        )
    return policy


def apply_eval_env_config_from_checkpoint(path, parsed):
    data = torch.load(path, map_location="cpu", weights_only=False)
    config = data.get("config", {})
    parsed.explore_frontier_target_mode = config.get(
        "explore_frontier_target_mode",
        getattr(parsed, "explore_frontier_target_mode", "nearest"),
    )
    parsed.cell_size = float(
        config.get("cell_size", getattr(parsed, "cell_size", 0.25))
    )
    parsed.explore_reachable_coverage = bool(
        config.get(
            "explore_reachable_coverage",
            getattr(parsed, "explore_reachable_coverage", False),
        )
    )
    parsed.explore_onboard_path_lookahead = int(
        config.get(
            "explore_onboard_path_lookahead",
            getattr(parsed, "explore_onboard_path_lookahead", 1),
        )
    )
    parsed.explore_onboard_adaptive_lookahead = bool(
        config.get("explore_onboard_adaptive_lookahead", False)
        or getattr(parsed, "explore_onboard_adaptive_lookahead", False)
    )
    parsed.explore_onboard_adaptive_lookahead_min = int(
        config.get(
            "explore_onboard_adaptive_lookahead_min",
            getattr(parsed, "explore_onboard_adaptive_lookahead_min", 2),
        )
    )
    parsed.explore_onboard_adaptive_lookahead_max = int(
        config.get(
            "explore_onboard_adaptive_lookahead_max",
            getattr(parsed, "explore_onboard_adaptive_lookahead_max", 4),
        )
    )
    parsed.explore_onboard_adaptive_clearance = float(
        config.get(
            "explore_onboard_adaptive_clearance",
            getattr(parsed, "explore_onboard_adaptive_clearance", 0.55),
        )
    )
    parsed.explore_onboard_adaptive_turn_dot = float(
        config.get(
            "explore_onboard_adaptive_turn_dot",
            getattr(parsed, "explore_onboard_adaptive_turn_dot", 0.55),
        )
    )
    parsed.explore_onboard_emit_objective_target = bool(
        config.get(
            "explore_onboard_emit_objective_target",
            getattr(parsed, "explore_onboard_emit_objective_target", False),
        )
        or getattr(parsed, "explore_onboard_emit_objective_target", False)
    )
    parsed.explore_onboard_objective_obs = bool(
        config.get(
            "explore_onboard_objective_obs",
            getattr(parsed, "explore_onboard_objective_obs", False),
        )
        or getattr(parsed, "explore_onboard_objective_obs", False)
    )
    parsed.explore_onboard_region_commit = bool(
        config.get("explore_onboard_region_commit", False)
        or getattr(parsed, "explore_onboard_region_commit", False)
    )
    parsed.explore_onboard_region_commit_stall_ticks = int(
        config.get(
            "explore_onboard_region_commit_stall_ticks",
            getattr(parsed, "explore_onboard_region_commit_stall_ticks", 0),
        )
    )
    parsed.explore_onboard_persistent_regions = bool(
        config.get("explore_onboard_persistent_regions", False)
        or getattr(parsed, "explore_onboard_persistent_regions", False)
    )
    parsed.explore_onboard_region_tracker = bool(
        config.get("explore_onboard_region_tracker", False)
        or getattr(parsed, "explore_onboard_region_tracker", False)
    )
    parsed.explore_onboard_clearance_passable = bool(
        config.get("explore_onboard_clearance_passable", False)
        or getattr(parsed, "explore_onboard_clearance_passable", False)
    )
    parsed.explore_onboard_passable_radius = float(
        config.get(
            "explore_onboard_passable_radius",
            getattr(parsed, "explore_onboard_passable_radius", 0.0),
        )
    )
    parsed.explore_onboard_passage_quality = bool(
        config.get("explore_onboard_passage_quality", False)
        or getattr(parsed, "explore_onboard_passage_quality", False)
    )
    parsed.explore_onboard_passage_weight = float(
        config.get(
            "explore_onboard_passage_weight",
            getattr(parsed, "explore_onboard_passage_weight", 0.35),
        )
    )
    parsed.explore_onboard_budget_frontier = bool(
        config.get("explore_onboard_budget_frontier", False)
        or getattr(parsed, "explore_onboard_budget_frontier", False)
    )
    parsed.explore_onboard_budget_gain_weight = float(
        config.get(
            "explore_onboard_budget_gain_weight",
            getattr(parsed, "explore_onboard_budget_gain_weight", 1.0),
        )
    )
    parsed.explore_onboard_budget_cluster_weight = float(
        config.get(
            "explore_onboard_budget_cluster_weight",
            getattr(parsed, "explore_onboard_budget_cluster_weight", 0.35),
        )
    )
    parsed.explore_onboard_budget_cell_weight = float(
        config.get(
            "explore_onboard_budget_cell_weight",
            getattr(parsed, "explore_onboard_budget_cell_weight", 0.25),
        )
    )
    parsed.explore_onboard_budget_depth_bias = float(
        config.get(
            "explore_onboard_budget_depth_bias",
            getattr(parsed, "explore_onboard_budget_depth_bias", 6.0),
        )
    )
    parsed.explore_onboard_budget_depth_weight = float(
        config.get(
            "explore_onboard_budget_depth_weight",
            getattr(parsed, "explore_onboard_budget_depth_weight", 1.0),
        )
    )
    parsed.explore_onboard_optimistic_unknown = bool(
        config.get("explore_onboard_optimistic_unknown", False)
        or getattr(parsed, "explore_onboard_optimistic_unknown", False)
    )
    parsed.explore_onboard_frontier_graph = bool(
        config.get("explore_onboard_frontier_graph", False)
        or getattr(parsed, "explore_onboard_frontier_graph", False)
    )
    parsed.explore_onboard_graph_candidate_obs = bool(
        config.get("explore_onboard_graph_candidate_obs", False)
        or getattr(parsed, "explore_onboard_graph_candidate_obs", False)
        or getattr(parsed, "deploy_graph_candidate_obs", False)
        or getattr(parsed, "predict_graph_candidate", False)
    )
    parsed.explore_onboard_graph_gain_weight = float(
        config.get(
            "explore_onboard_graph_gain_weight",
            getattr(parsed, "explore_onboard_graph_gain_weight", 0.32),
        )
    )
    parsed.explore_onboard_graph_cluster_weight = float(
        config.get(
            "explore_onboard_graph_cluster_weight",
            getattr(parsed, "explore_onboard_graph_cluster_weight", 0.12),
        )
    )
    parsed.explore_onboard_graph_depth_weight = float(
        config.get(
            "explore_onboard_graph_depth_weight",
            getattr(parsed, "explore_onboard_graph_depth_weight", 0.045),
        )
    )
    parsed.explore_onboard_graph_active_bonus = float(
        config.get(
            "explore_onboard_graph_active_bonus",
            getattr(parsed, "explore_onboard_graph_active_bonus", 0.75),
        )
    )
    parsed.explore_onboard_graph_stale_penalty = float(
        config.get(
            "explore_onboard_graph_stale_penalty",
            getattr(parsed, "explore_onboard_graph_stale_penalty", 0.020),
        )
    )
    parsed.explore_onboard_graph_failure_penalty = float(
        config.get(
            "explore_onboard_graph_failure_penalty",
            getattr(parsed, "explore_onboard_graph_failure_penalty", 0.70),
        )
    )
    parsed.explore_onboard_graph_attempt_penalty = float(
        config.get(
            "explore_onboard_graph_attempt_penalty",
            getattr(parsed, "explore_onboard_graph_attempt_penalty", 0.12),
        )
    )
    parsed.explore_onboard_graph_gain_cap = int(
        config.get(
            "explore_onboard_graph_gain_cap",
            getattr(parsed, "explore_onboard_graph_gain_cap", 192),
        )
    )
    parsed.explore_onboard_approach_viewpoints = bool(
        config.get("explore_onboard_approach_viewpoints", False)
        or getattr(parsed, "explore_onboard_approach_viewpoints", False)
    )
    parsed.explore_onboard_approach_max_depth = int(
        config.get(
            "explore_onboard_approach_max_depth",
            getattr(parsed, "explore_onboard_approach_max_depth", 1),
        )
    )
    parsed.drive_objective_when_close = bool(
        config.get("drive_objective_when_close", False)
        or getattr(parsed, "drive_objective_when_close", False)
    )
    parsed.drive_objective_close_dist = float(
        config.get(
            "drive_objective_close_dist",
            getattr(parsed, "drive_objective_close_dist", 0.04),
        )
    )
    parsed.drive_objective_min_dist = float(
        config.get(
            "drive_objective_min_dist",
            getattr(parsed, "drive_objective_min_dist", 0.08),
        )
    )
    parsed.predict_graph_candidate = bool(
        config.get(
            "predict_graph_candidate",
            getattr(parsed, "predict_graph_candidate", False),
        )
    )
    parsed.graph_candidate_label_source = config.get(
        "graph_candidate_label_source",
        getattr(parsed, "graph_candidate_label_source", "privileged"),
    )
    parsed.deploy_patch_obs = bool(
        config.get("deploy_patch_obs", getattr(parsed, "deploy_patch_obs", False))
    )
    parsed.deploy_graph_candidate_obs = bool(
        config.get(
            "deploy_graph_candidate_obs",
            getattr(parsed, "deploy_graph_candidate_obs", False),
        )
        or getattr(parsed, "deploy_graph_candidate_obs", False)
        or getattr(parsed, "predict_graph_candidate", False)
    )
    parsed.deploy_coverage_obs = bool(
        config.get("deploy_coverage_obs", getattr(parsed, "deploy_coverage_obs", False))
    )
    parsed.deploy_onboard_target_obs = bool(
        config.get(
            "deploy_onboard_target_obs",
            getattr(parsed, "deploy_onboard_target_obs", False),
        )
    )
    if (
        parsed.deploy_onboard_target_obs
        and parsed.explore_frontier_target_mode != "onboard"
    ):
        raise ValueError(
            "checkpoint deploy_onboard_target_obs requires "
            "explore_frontier_target_mode=onboard"
        )
    parsed.safety_min_ray_speed_cap = bool(
        config.get("safety_min_ray_speed_cap", False)
        or getattr(parsed, "safety_min_ray_speed_cap", False)
    )
    parsed.safety_min_ray_speed_stop = float(
        config.get(
            "safety_min_ray_speed_stop",
            getattr(parsed, "safety_min_ray_speed_stop", 0.0),
        )
    )
    parsed.safety_min_ray_speed_slow = float(
        config.get(
            "safety_min_ray_speed_slow",
            getattr(parsed, "safety_min_ray_speed_slow", 0.0),
        )
    )
    parsed.safety_min_ray_speed_min_scale = float(
        config.get(
            "safety_min_ray_speed_min_scale",
            getattr(parsed, "safety_min_ray_speed_min_scale", 0.0),
        )
    )


def print_metrics(label, metrics):
    keys = (
        "n",
        "episode_length",
        "success",
        "score",
        "crashed",
        "obstacle_hit",
        "wall_hit",
        "ground_hit",
        "timeout",
        "coverage",
        "explored_cells",
        "new_cell_rate",
        "frontier_count",
        "stale_frontier_count",
        "stale_marks",
        "frontier_cluster_count",
        "onboard_target_exists",
        "onboard_target_age",
        "onboard_target_cluster_cells",
        "blocked_cells",
        "min_clearance",
        "path_length",
        "action_delta",
        "region_tracker_regions",
        "region_tracker_active_age",
        "region_tracker_since_progress",
        "region_tracker_switch_rate",
        "region_tracker_stale_rate",
        "frontier_graph_nodes",
        "frontier_graph_unknown_cells",
        "frontier_graph_best_unknown_cells",
        "frontier_graph_best_depth",
        "frontier_graph_best_score",
        "frontier_graph_active_rate",
        "adaptive_lookahead",
        "adaptive_lookahead_long_rate",
        "adaptive_lookahead_clearance_reject_rate",
        "adaptive_lookahead_turn_reject_rate",
    )
    print(label)
    for key in keys:
        if key in metrics:
            print(f"{key} {metrics[key]:.6g}")
    for pattern in range(4):
        n_key = f"procedural_{pattern}_n"
        if n_key not in metrics or metrics[n_key] <= 0:
            continue
        share = metrics[n_key]
        count = share * metrics.get("n", 0.0)
        success = metrics.get(f"procedural_{pattern}_success", 0.0) / share
        timeout = metrics.get(f"procedural_{pattern}_timeout", 0.0) / share
        obstacle = metrics.get(f"procedural_{pattern}_obstacle_hit", 0.0) / share
        coverage = metrics.get(f"procedural_{pattern}_coverage", 0.0) / share
        print(
            f"procedural_{pattern} n={count:.0f} share={share:.6g} success={success:.6g} "
            f"timeout={timeout:.6g} obstacle_hit={obstacle:.6g} "
            f"coverage={coverage:.6g}"
        )


def run_eval(parsed, policy=None):
    from pufferlib import _C

    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    if policy is None and not parsed.ignore_checkpoint_env_config:
        apply_eval_env_config_from_checkpoint(parsed.checkpoint, parsed)
    args = make_vec_config(parsed)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, _, terminals = tensor_views(vec, device)
    vec.reset()

    if policy is None:
        policy = load_policy(parsed.checkpoint, vec, parsed, device)
    policy.eval()
    state = policy.initial_state(vec.total_agents, device)
    action_controller = make_recovery_controller(parsed, device)

    accum = None
    min_eval_steps = parsed.max_steps if parsed.drain_initial_episodes else 0
    try:
        for step in range(1, parsed.max_eval_steps + 1):
            with torch.no_grad():
                x = deployable_input(obs, parsed)
                logits, _, state = policy.forward_eval(x, state)
                if getattr(parsed, "action_policy", False):
                    actions = action_policy_to_actions(logits.mean, parsed, obs)
                else:
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

            step_vec(vec, actions, device)
            done = terminals > 0.5
            state = reset_done_state(state, done)
            state = detach_state(state)
            if action_controller is not None:
                action_controller.reset_done(done)

            if step % parsed.log_interval == 0:
                accum = accumulate_logs(accum, vec.log())
                if (
                    step >= min_eval_steps
                    and finalize_logs(accum).get("n", 0.0) >= parsed.episodes
                ):
                    break
        accum = accumulate_logs(accum, vec.log())
    finally:
        vec.close()

    metrics = finalize_logs(accum)
    label = f"target_distill_eval {getattr(parsed, 'checkpoint', 'in_memory')}"
    print_metrics(label, metrics)
    return metrics


def run_oracle_eval(parsed):
    from pufferlib import _C

    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    args = make_vec_config(parsed)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, _, terminals = tensor_views(vec, device)
    vec.reset()
    action_controller = make_recovery_controller(parsed, device)

    accum = None
    min_eval_steps = parsed.max_steps if parsed.drain_initial_episodes else 0
    try:
        for step in range(1, parsed.max_eval_steps + 1):
            labels = target_labels(obs, parsed)
            if getattr(parsed, "predict_graph_candidate", False):
                cand_index, cand_valid = graph_candidate_labels(obs, labels, parsed)
                labels = graph_candidate_target_from_index(obs, cand_index, cand_valid)
            actions = (
                action_controller.actions(labels, obs)
                if action_controller is not None
                else target_to_actions(labels, parsed, obs)
            )
            step_vec(vec, actions, device)
            if action_controller is not None:
                action_controller.reset_done(terminals > 0.5)
            if step % parsed.log_interval == 0:
                accum = accumulate_logs(accum, vec.log())
                if (
                    step >= min_eval_steps
                    and finalize_logs(accum).get("n", 0.0) >= parsed.episodes
                ):
                    break
        accum = accumulate_logs(accum, vec.log())
    finally:
        vec.close()

    metrics = finalize_logs(accum)
    print_metrics("target_distill_oracle_eval", metrics)
    return metrics


def _heading_ray(obs, direction, parsed):
    return _direction_ray(obs, direction, parsed)


def diagnostic_features(obs, pred_target, oracle_target, action, oracle_action, parsed):
    rays = obs[:, 26 : 26 + parsed.num_rays].float() * parsed.ray_length
    min_ray = rays.min(dim=1).values if parsed.num_rays > 0 else torch.zeros(
        (obs.shape[0],), dtype=torch.float32, device=obs.device
    )

    pred_dir = pred_target[:, 0:2].clamp(-1.0, 1.0)
    oracle_dir = oracle_target[:, 0:2].clamp(-1.0, 1.0)
    pred_unit = pred_dir / torch.linalg.vector_norm(pred_dir, dim=1, keepdim=True).clamp_min(1e-6)
    oracle_unit = oracle_dir / torch.linalg.vector_norm(oracle_dir, dim=1, keepdim=True).clamp_min(1e-6)
    cos = (pred_unit * oracle_unit).sum(dim=1).clamp(-1.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(cos))

    action_dir = action[:, 0:2]
    action_unit = action_dir / torch.linalg.vector_norm(action_dir, dim=1, keepdim=True).clamp_min(1e-6)

    features = {
        "coverage": obs[:, 17].float(),
        "min_ray_m": min_ray,
        "front_ray_m": rays[:, 0] if parsed.num_rays > 0 else min_ray,
        "pred_oracle_angle_deg": angle_deg,
        "pred_dist_err": (pred_target[:, 2] - oracle_target[:, 2]).abs(),
        "pred_exists": (pred_target[:, 3] > parsed.exists_threshold).float(),
        "oracle_exists": (oracle_target[:, 3] > parsed.exists_threshold).float(),
        "action_l2": torch.linalg.vector_norm(action[:, 0:2] - oracle_action[:, 0:2], dim=1),
        "action_speed": torch.linalg.vector_norm(action[:, 0:2], dim=1),
        "pred_target_ray_m": _heading_ray(obs, pred_unit, parsed),
        "oracle_target_ray_m": _heading_ray(obs, oracle_unit, parsed),
        "action_ray_m": _heading_ray(obs, action_unit, parsed),
    }
    return {key: value.detach().clone() for key, value in features.items()}


class DiagnosticStats:
    def __init__(self, parsed):
        self.parsed = parsed
        self.values = {"success": {}, "crash": {}, "timeout": {}}
        self.counts = {"success": 0, "crash": 0, "timeout": 0}

    def add(self, group, features, mask):
        count = int(mask.sum().item())
        if count <= 0:
            return
        self.counts[group] += count
        for key, value in features.items():
            bucket = self.values[group].setdefault(key, [])
            bucket.extend(value[mask].detach().float().cpu().tolist())

    @staticmethod
    def _mean(values):
        if not values:
            return float("nan")
        return sum(values) / len(values)

    @staticmethod
    def _frac(values, pred):
        if not values:
            return float("nan")
        return sum(1.0 for value in values if pred(value)) / len(values)

    @staticmethod
    def _pct(values, q):
        if not values:
            return float("nan")
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[idx]

    def report_group(self, controller, group):
        n = self.counts[group]
        if n <= 0:
            print(f"{controller}.{group}.n 0")
            return
        vals = self.values[group]
        print(f"{controller}.{group}.n {n}")
        for key in (
            "episode_steps",
            "start_coverage",
            "start_min_ray_m",
            "start_pred_exists",
            "start_oracle_exists",
            "episode_min_ray_m",
            "episode_action_speed_mean",
            "episode_action_speed_max",
            "coverage",
            "min_ray_m",
            "pred_oracle_angle_deg",
            "pred_dist_err",
            "action_l2",
            "pred_target_ray_m",
            "oracle_target_ray_m",
            "action_ray_m",
            "pred_exists",
            "oracle_exists",
            "action_speed",
        ):
            print(f"{controller}.{group}.{key}_mean {self._mean(vals.get(key, [])):.6g}")
        print(f"{controller}.{group}.episode_steps_p10 {self._pct(vals.get('episode_steps', []), 0.10):.6g}")
        print(f"{controller}.{group}.episode_steps_p50 {self._pct(vals.get('episode_steps', []), 0.50):.6g}")
        print(f"{controller}.{group}.min_ray_p10 {self._pct(vals.get('min_ray_m', []), 0.10):.6g}")
        print(f"{controller}.{group}.min_ray_p50 {self._pct(vals.get('min_ray_m', []), 0.50):.6g}")
        print(
            f"{controller}.{group}.immediate_frac "
            f"{self._frac(vals.get('episode_steps', []), lambda x: x <= 1.0):.6g}"
        )
        print(
            f"{controller}.{group}.short_frac "
            f"{self._frac(vals.get('episode_steps', []), lambda x: x <= 8.0):.6g}"
        )
        print(
            f"{controller}.{group}.start_min_ray_lt_danger_frac "
            f"{self._frac(vals.get('start_min_ray_m', []), lambda x: x < self.parsed.danger_clearance):.6g}"
        )
        print(
            f"{controller}.{group}.angle_gt45_frac "
            f"{self._frac(vals.get('pred_oracle_angle_deg', []), lambda x: x > 45.0):.6g}"
        )
        print(
            f"{controller}.{group}.pred_target_ray_lt_danger_frac "
            f"{self._frac(vals.get('pred_target_ray_m', []), lambda x: x < self.parsed.danger_clearance):.6g}"
        )
        print(
            f"{controller}.{group}.oracle_target_ray_lt_danger_frac "
            f"{self._frac(vals.get('oracle_target_ray_m', []), lambda x: x < self.parsed.danger_clearance):.6g}"
        )
        pred = vals.get("pred_target_ray_m", [])
        oracle = vals.get("oracle_target_ray_m", [])
        if pred and oracle:
            worse = sum(
                1.0
                for pred_ray, oracle_ray in zip(pred, oracle)
                if oracle_ray - pred_ray > self.parsed.clearance_gap
            ) / len(pred)
            print(f"{controller}.{group}.pred_ray_worse_by_gap_frac {worse:.6g}")

    def report(self, controller):
        total = sum(self.counts.values())
        print(f"target_distill_diagnostic {controller}")
        print(f"{controller}.episodes {total}")
        for group in ("success", "crash", "timeout"):
            print(f"{controller}.{group}_rate {self.counts[group] / max(total, 1):.6g}")
        for group in ("success", "crash", "timeout"):
            self.report_group(controller, group)


def run_diagnostic_controller(parsed, policy, controller):
    from pufferlib import _C

    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    args = make_vec_config(parsed)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, rewards, terminals = tensor_views(vec, device)
    vec.reset()

    policy.eval()
    state = policy.initial_state(vec.total_agents, device)
    stats = DiagnosticStats(parsed)
    episode_steps = torch.zeros((vec.total_agents,), dtype=torch.float32, device=device)
    start_coverage = torch.zeros_like(episode_steps)
    start_min_ray = torch.zeros_like(episode_steps)
    start_pred_exists = torch.zeros_like(episode_steps)
    start_oracle_exists = torch.zeros_like(episode_steps)
    episode_min_ray = torch.full_like(episode_steps, parsed.ray_length)
    episode_action_speed_sum = torch.zeros_like(episode_steps)
    episode_action_speed_max = torch.zeros_like(episode_steps)
    episode_action_count = torch.zeros_like(episode_steps)

    try:
        for step in range(1, parsed.max_eval_steps + 1):
            with torch.no_grad():
                x = deployable_input(obs, parsed)
                logits, _, state = policy.forward_eval(x, state)
                pred_target = (
                    graph_candidate_target_from_logits(logits.mean, obs, parsed)
                    if getattr(parsed, "predict_graph_candidate", False)
                    else logits.mean
                )
                oracle_target = target_labels(obs, parsed)
                student_action = target_to_actions(pred_target, parsed, obs)
                oracle_action = target_to_actions(oracle_target, parsed, obs)
                action = oracle_action if controller == "oracle" else student_action
                features = diagnostic_features(
                    obs, pred_target, oracle_target, action, oracle_action, parsed
                )
                fresh = episode_steps <= 0.0
                start_coverage = torch.where(fresh, features["coverage"], start_coverage)
                start_min_ray = torch.where(fresh, features["min_ray_m"], start_min_ray)
                start_pred_exists = torch.where(fresh, features["pred_exists"], start_pred_exists)
                start_oracle_exists = torch.where(fresh, features["oracle_exists"], start_oracle_exists)
                episode_min_ray = torch.minimum(episode_min_ray, features["min_ray_m"])
                episode_action_speed_sum = episode_action_speed_sum + features["action_speed"]
                episode_action_speed_max = torch.maximum(
                    episode_action_speed_max, features["action_speed"]
                )
                episode_action_count = episode_action_count + 1.0

            step_vec(vec, action, device)
            episode_steps = episode_steps + 1.0
            done = terminals > 0.5
            if done.any():
                episode_features = dict(features)
                episode_features.update({
                    "episode_steps": episode_steps.detach().clone(),
                    "start_coverage": start_coverage.detach().clone(),
                    "start_min_ray_m": start_min_ray.detach().clone(),
                    "start_pred_exists": start_pred_exists.detach().clone(),
                    "start_oracle_exists": start_oracle_exists.detach().clone(),
                    "episode_min_ray_m": episode_min_ray.detach().clone(),
                    "episode_action_speed_mean": (
                        episode_action_speed_sum / episode_action_count.clamp_min(1.0)
                    ).detach().clone(),
                    "episode_action_speed_max": episode_action_speed_max.detach().clone(),
                })
                crash = done & (rewards < parsed.crash_reward_threshold)
                success = done & (rewards > parsed.success_reward_threshold)
                timeout = done & ~(crash | success)
                stats.add("crash", episode_features, crash)
                stats.add("success", episode_features, success)
                stats.add("timeout", episode_features, timeout)
                state = reset_done_state(state, done)
                episode_steps = torch.where(done, torch.zeros_like(episode_steps), episode_steps)
                start_coverage = torch.where(done, torch.zeros_like(start_coverage), start_coverage)
                start_min_ray = torch.where(done, torch.zeros_like(start_min_ray), start_min_ray)
                start_pred_exists = torch.where(done, torch.zeros_like(start_pred_exists), start_pred_exists)
                start_oracle_exists = torch.where(done, torch.zeros_like(start_oracle_exists), start_oracle_exists)
                episode_min_ray = torch.where(
                    done, torch.full_like(episode_min_ray, parsed.ray_length), episode_min_ray
                )
                episode_action_speed_sum = torch.where(
                    done, torch.zeros_like(episode_action_speed_sum), episode_action_speed_sum
                )
                episode_action_speed_max = torch.where(
                    done, torch.zeros_like(episode_action_speed_max), episode_action_speed_max
                )
                episode_action_count = torch.where(
                    done, torch.zeros_like(episode_action_count), episode_action_count
                )
            state = detach_state(state)
            if step % parsed.log_interval == 0 and sum(stats.counts.values()) >= parsed.episodes:
                break
    finally:
        vec.close()

    stats.report(controller)
    return stats


def run_diagnostic(parsed):
    from pufferlib import _C

    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    if not parsed.ignore_checkpoint_env_config:
        apply_eval_env_config_from_checkpoint(parsed.checkpoint, parsed)
    args = make_vec_config(parsed)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    try:
        policy = load_policy(parsed.checkpoint, vec, parsed, device)
    finally:
        vec.close()

    controllers = ("student", "oracle") if parsed.controllers == "both" else (parsed.controllers,)
    out = {}
    for controller in controllers:
        out[controller] = run_diagnostic_controller(parsed, policy, controller)
    return out


def create_train_runner(parsed, device, mode):
    from pufferlib import _C

    run_parsed = copy.copy(parsed)
    run_parsed.mode = mode
    run_parsed.obstacle_mode = OBSTACLE_MODES[mode]
    args = make_vec_config(run_parsed)
    vec = _C.create_vec(args, 0 if device.type == "cpu" else 1)
    obs, _, terminals = tensor_views(vec, device)
    vec.reset()
    return {
        "mode": mode,
        "vec": vec,
        "obs": obs,
        "terminals": terminals,
        "rollout_state": None,
        "anchor_rollout_state": None,
    }


def run_train(parsed):
    from pufferlib import _C

    device = torch.device("cuda" if _C.gpu and not parsed.cpu else "cpu")
    init_data = None
    if parsed.init_checkpoint:
        init_data = torch.load(parsed.init_checkpoint, map_location=device, weights_only=False)
        meta = init_data.get("policy", {})
        parsed.network = meta.get("network", parsed.network)
        parsed.hidden_size = int(meta.get("hidden_size", parsed.hidden_size))
        parsed.num_layers = int(meta.get("num_layers", parsed.num_layers))
        parsed.action_std = float(meta.get("action_std", parsed.action_std))
        config = init_data.get("config", {})
        if not parsed.ignore_checkpoint_env_config:
            parsed.cell_size = float(config.get("cell_size", parsed.cell_size))
            parsed.explore_onboard_clearance_passable = bool(
                config.get("explore_onboard_clearance_passable", False)
                or parsed.explore_onboard_clearance_passable
            )
            parsed.explore_onboard_passable_radius = float(
                config.get(
                    "explore_onboard_passable_radius",
                    parsed.explore_onboard_passable_radius,
                )
            )
            parsed.explore_onboard_passage_quality = bool(
                config.get("explore_onboard_passage_quality", False)
                or parsed.explore_onboard_passage_quality
            )
            parsed.explore_onboard_passage_weight = float(
                config.get(
                    "explore_onboard_passage_weight",
                    parsed.explore_onboard_passage_weight,
                )
            )
            parsed.explore_onboard_clearance_cone_deg = float(
                config.get(
                    "explore_onboard_clearance_cone_deg",
                    parsed.explore_onboard_clearance_cone_deg,
                )
            )
        parsed.predict_speed_scale = bool(
            config.get("predict_speed_scale", parsed.predict_speed_scale)
        )
        parsed.predict_frontier_sectors = bool(
            config.get("predict_frontier_sectors", parsed.predict_frontier_sectors)
            or parsed.predict_frontier_sectors
        )
        parsed.predict_frontier_vector = bool(
            config.get("predict_frontier_vector", parsed.predict_frontier_vector)
            or parsed.predict_frontier_vector
        )
        parsed.predict_recovery_gate = bool(
            config.get("predict_recovery_gate", parsed.predict_recovery_gate)
            or parsed.predict_recovery_gate
        )
        parsed.predict_frontier_patch = bool(
            config.get("predict_frontier_patch", parsed.predict_frontier_patch)
            or parsed.predict_frontier_patch
        )
        parsed.predict_graph_candidate = bool(
            config.get("predict_graph_candidate", parsed.predict_graph_candidate)
            or parsed.predict_graph_candidate
        )
        parsed.graph_candidate_label_source = config.get(
            "graph_candidate_label_source",
            parsed.graph_candidate_label_source,
        )
        parsed.action_policy = bool(
            config.get("action_policy", parsed.action_policy)
            or parsed.action_policy
        )
        parsed.action_policy_safety = bool(
            config.get("action_policy_safety", parsed.action_policy_safety)
            or parsed.action_policy_safety
        )
        parsed.map_sidecar_encoder = bool(
            config.get("map_sidecar_encoder", parsed.map_sidecar_encoder)
            or parsed.map_sidecar_encoder
        )
        parsed.far_map_start = int(config.get("far_map_start", parsed.far_map_start))
        parsed.far_map_side = int(config.get("far_map_side", parsed.far_map_side))
        parsed.use_pred_speed_scale = bool(
            config.get("predict_speed_scale", parsed.use_pred_speed_scale)
        )
        parsed.deploy_patch_obs = bool(
            config.get("deploy_patch_obs", parsed.deploy_patch_obs)
            or parsed.deploy_patch_obs
        )
        parsed.deploy_graph_candidate_obs = bool(
            config.get("deploy_graph_candidate_obs", parsed.deploy_graph_candidate_obs)
            or parsed.deploy_graph_candidate_obs
            or parsed.predict_graph_candidate
        )
        parsed.deploy_coverage_obs = bool(
            config.get("deploy_coverage_obs", parsed.deploy_coverage_obs)
            or parsed.deploy_coverage_obs
        )
        parsed.deploy_onboard_target_obs = bool(
            config.get("deploy_onboard_target_obs", parsed.deploy_onboard_target_obs)
            or parsed.deploy_onboard_target_obs
        )
        parsed.unclamped_frontier_vector_dist_loss = bool(
            config.get(
                "unclamped_frontier_vector_dist_loss",
                parsed.unclamped_frontier_vector_dist_loss,
            )
            or parsed.unclamped_frontier_vector_dist_loss
        )
        parsed.frontier_vector_dist_loss_weight = float(
            config.get(
                "frontier_vector_dist_loss_weight",
                parsed.frontier_vector_dist_loss_weight,
            )
        )
        parsed.freeze_except_frontier_vector_dist = bool(
            parsed.freeze_except_frontier_vector_dist
        )
        parsed.freeze_except_recovery_gate = bool(parsed.freeze_except_recovery_gate)
        parsed.use_pred_vector_fallback = bool(
            config.get("use_pred_vector_fallback", parsed.use_pred_vector_fallback)
            or parsed.use_pred_vector_fallback
        )
        parsed.pred_vector_fallback_primary_threshold = float(
            config.get(
                "pred_vector_fallback_primary_threshold",
                parsed.pred_vector_fallback_primary_threshold,
            )
        )
        parsed.pred_vector_fallback_min_coverage = float(
            config.get(
                "pred_vector_fallback_min_coverage",
                parsed.pred_vector_fallback_min_coverage,
            )
        )
        parsed.pred_vector_fallback_min_dist = float(
            config.get(
                "pred_vector_fallback_min_dist",
                parsed.pred_vector_fallback_min_dist,
            )
        )
        parsed.use_pred_recovery_gate = bool(
            config.get("use_pred_recovery_gate", parsed.use_pred_recovery_gate)
            or parsed.use_pred_recovery_gate
        )
        parsed.pred_recovery_gate_threshold = float(
            config.get(
                "pred_recovery_gate_threshold",
                parsed.pred_recovery_gate_threshold,
            )
        )
        parsed.pred_recovery_gate_min_coverage = float(
            config.get(
                "pred_recovery_gate_min_coverage",
                parsed.pred_recovery_gate_min_coverage,
            )
        )
        parsed.pred_recovery_gate_min_dist = float(
            config.get(
                "pred_recovery_gate_min_dist",
                parsed.pred_recovery_gate_min_dist,
            )
        )
    if parsed.predict_speed_scale and parsed.predict_frontier_sectors:
        raise ValueError("predict_speed_scale and predict_frontier_sectors cannot be combined")
    if parsed.predict_frontier_vector and not parsed.predict_frontier_sectors:
        raise ValueError("predict_frontier_vector requires predict_frontier_sectors")
    if parsed.predict_recovery_gate and not parsed.predict_frontier_vector:
        raise ValueError("predict_recovery_gate requires predict_frontier_vector")
    if parsed.predict_recovery_gate and parsed.predict_frontier_patch:
        raise ValueError("predict_recovery_gate cannot be combined with predict_frontier_patch")
    if parsed.predict_frontier_patch and not parsed.predict_frontier_vector:
        raise ValueError("predict_frontier_patch requires predict_frontier_vector")
    if parsed.predict_graph_candidate:
        if (
            parsed.action_policy
            or parsed.predict_speed_scale
            or parsed.predict_frontier_sectors
            or parsed.predict_frontier_vector
            or parsed.predict_recovery_gate
            or parsed.predict_frontier_patch
        ):
            raise ValueError("predict_graph_candidate cannot be combined with other output heads")
        parsed.deploy_graph_candidate_obs = True
        parsed.explore_onboard_graph_candidate_obs = True

    train_modes = parsed.train_modes or [parsed.mode]
    runners = [create_train_runner(parsed, device, mode) for mode in train_modes]
    vec0 = runners[0]["vec"]
    if parsed.freeze_except_new_inputs:
        if parsed.new_input_start is None:
            if init_data is None:
                raise ValueError("--freeze-except-new-inputs requires --init-checkpoint or --new-input-start")
            parsed.new_input_start = int(init_data["input_size"])
        if parsed.new_input_start >= vec0.obs_size:
            raise ValueError(
                f"--new-input-start={parsed.new_input_start} leaves no new input columns "
                f"for env obs_size={vec0.obs_size}"
            )

    separate_patch_head = (
        parsed.predict_frontier_patch
        and getattr(parsed, "separate_frontier_patch_head", False)
    )

    if parsed.predict_graph_candidate:
        act_sizes = [1] * GRAPH_CANDIDATE_COUNT
    elif parsed.action_policy:
        act_sizes = [1, 1, 1, 1]
    elif parsed.predict_frontier_patch and not separate_patch_head:
        act_sizes = [1] * 27
    elif parsed.predict_recovery_gate:
        act_sizes = [1] * 12
    elif parsed.predict_frontier_vector:
        act_sizes = [1] * 11
    elif parsed.predict_frontier_sectors:
        act_sizes = [1] * 8
    elif parsed.predict_speed_scale:
        act_sizes = [1, 1, 1, 1, 1]
    else:
        act_sizes = [1, 1, 1, 1]
    policy = create_policy(vec0.obs_size, act_sizes, parsed, device)
    patch_head = None
    if separate_patch_head:
        patch_head = torch.nn.Linear(parsed.hidden_size, FRONTIER_PATCH_SIZE).to(device)
    if init_data is not None:
        init_input_matches = int(init_data["input_size"]) == vec0.obs_size
        if not init_input_matches and not parsed.resize_input_head:
            raise ValueError(
                f"init checkpoint input_size={init_data['input_size']} does not match env obs_size={vec0.obs_size}"
            )
        init_act_sizes = list(init_data.get("act_sizes", []))
        if init_act_sizes != act_sizes and not parsed.resize_output_head:
            raise ValueError(
                f"init checkpoint act_sizes={init_data.get('act_sizes')} does not match {act_sizes}"
            )
        if init_input_matches and init_act_sizes == act_sizes:
            policy.load_state_dict(init_data["state_dict"])
            print(f"loaded init_checkpoint {parsed.init_checkpoint}", flush=True)
        else:
            copied, resized = load_resized_output_state(
                policy,
                init_data["state_dict"],
                resize_input=not init_input_matches,
                resize_output=init_act_sizes != act_sizes,
            )
            print(
                f"loaded init_checkpoint {parsed.init_checkpoint} with resized state "
                f"copied={len(copied)} resized={','.join(resized)}",
                flush=True,
            )
    if parsed.zero_deploy_coverage_input_weight:
        changed = zero_encoder_input_columns(policy, [17])
        print(
            "zero_deploy_coverage_input_weight="
            f"{','.join(str(col) for col in changed) if changed else 'none'}",
            flush=True,
        )

    anchor_policy = None
    anchor_modes = set(parsed.anchor_modes or [])
    hard_modes = set(parsed.hard_modes or [])
    false_negative_modes = set(parsed.false_negative_modes or [])
    if parsed.anchor_checkpoint:
        anchor_data = torch.load(parsed.anchor_checkpoint, map_location=device, weights_only=False)
        anchor_act_sizes = list(anchor_data.get("act_sizes", []))
        anchor_policy = create_policy(vec0.obs_size, anchor_act_sizes, parsed, device)
        if int(anchor_data["input_size"]) == vec0.obs_size:
            anchor_policy.load_state_dict(anchor_data["state_dict"])
            anchor_resize_msg = ""
        else:
            copied, resized = load_resized_output_state(
                anchor_policy,
                anchor_data["state_dict"],
                resize_input=True,
                resize_output=False,
            )
            anchor_resize_msg = (
                f" resized_input={anchor_data['input_size']}->{vec0.obs_size} "
                f"copied={len(copied)} resized={','.join(resized)}"
            )
        anchor_policy.eval()
        for param in anchor_policy.parameters():
            param.requires_grad_(False)
        if not anchor_modes:
            anchor_modes = set(train_modes)
        print(
            f"loaded anchor_checkpoint {parsed.anchor_checkpoint} "
            f"modes={','.join(sorted(anchor_modes))} weight={parsed.anchor_loss_weight}"
            f"{anchor_resize_msg}",
            flush=True,
        )
    if parsed.hard_example_weight > 0.0 and hard_modes:
        print(
            f"hard_state_weighting modes={','.join(sorted(hard_modes))} "
            f"weight={parsed.hard_example_weight} angle={parsed.hard_angle_deg} "
            f"clearance_gap={parsed.hard_clearance_gap} low_speed={parsed.hard_low_speed} "
            f"min_coverage={parsed.hard_min_coverage}",
            flush=True,
        )
    if parsed.false_negative_weight > 0.0 and false_negative_modes:
        print(
            f"false_negative_exists modes={','.join(sorted(false_negative_modes))} "
            f"weight={parsed.false_negative_weight} margin={parsed.false_negative_margin} "
            f"min_coverage={parsed.false_negative_min_coverage}",
            flush=True,
        )

    policy.train()
    if parsed.freeze_except_frontier_vector_dist:
        freeze_except_frontier_vector_dist(policy)
        print("freeze_except_frontier_vector_dist=1 trainable=decoder_mean_row10", flush=True)
    if parsed.freeze_except_recovery_gate:
        freeze_except_recovery_gate(policy)
        print("freeze_except_recovery_gate=1 trainable=decoder_mean_row11", flush=True)
    if parsed.freeze_except_new_inputs:
        freeze_except_new_input_columns(policy)
        print(
            "freeze_except_new_inputs=1 "
            f"trainable=encoder_columns_{parsed.new_input_start}:{vec0.obs_size}",
            flush=True,
        )
    if parsed.freeze_except_map_sidecar:
        if not parsed.map_sidecar_encoder:
            raise ValueError("--freeze-except-map-sidecar requires --map-sidecar-encoder")
        freeze_except_map_sidecar(policy)
        print("freeze_except_map_sidecar=1 trainable=encoder.map_net", flush=True)
    optimizer_params = [param for param in policy.parameters() if param.requires_grad]
    if patch_head is not None:
        optimizer_params.extend(patch_head.parameters())
        patch_head.train()
    optimizer = torch.optim.Adam(optimizer_params, lr=parsed.learning_rate)
    for runner in runners:
        runner["rollout_state"] = policy.initial_state(runner["vec"].total_agents, device)
        if anchor_policy is not None:
            runner["anchor_rollout_state"] = anchor_policy.initial_state(
                runner["vec"].total_agents, device
            )

    last_loss = 0.0
    last_anchor_loss = 0.0
    last_false_negative_loss = 0.0
    last_frontier_sector_loss = 0.0
    last_frontier_vector_loss = 0.0
    last_frontier_patch_loss = 0.0
    last_graph_candidate_loss = 0.0
    last_graph_candidate_valid = 0.0
    last_recovery_gate_loss = 0.0
    last_recovery_gate_frac = 0.0
    last_sector_fallback_frac = 0.0
    last_vector_fallback_frac = 0.0
    last_hard_frac = 0.0
    last_hard_weight = 1.0
    last_student_prob = 0.0
    output_path = Path(parsed.output)
    checkpoint_dir = (
        Path(parsed.checkpoint_dir)
        if parsed.checkpoint_dir
        else output_path.parent / "checkpoints"
    )

    def save_interval_checkpoint(update):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{output_path.stem}_u{update:06d}{output_path.suffix}"
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "aux_state_dict": {
                    "frontier_patch_head": patch_head.state_dict() if patch_head is not None else None,
                },
                "input_size": vec0.obs_size,
                "num_atns": len(act_sizes),
                "act_sizes": act_sizes,
                "config": {
                    "task_mode": 3,
                    "obstacle_mode": parsed.obstacle_mode,
                    "train_modes": train_modes,
                    "train_update": update,
                    "action_policy": parsed.action_policy,
                    "action_policy_safety": parsed.action_policy_safety,
                    "action_policy_correction_weight": parsed.action_policy_correction_weight,
                    "action_policy_clearance_weight": parsed.action_policy_clearance_weight,
                    "explore_memory_obs": parsed.explore_memory_obs,
                    "cell_size": parsed.cell_size,
                    "explore_frontier_target_mode": parsed.explore_frontier_target_mode,
                    "explore_reachable_coverage": parsed.explore_reachable_coverage,
                    "explore_onboard_path_lookahead": parsed.explore_onboard_path_lookahead,
                    "explore_onboard_adaptive_lookahead": parsed.explore_onboard_adaptive_lookahead,
                    "explore_onboard_adaptive_lookahead_min": parsed.explore_onboard_adaptive_lookahead_min,
                    "explore_onboard_adaptive_lookahead_max": parsed.explore_onboard_adaptive_lookahead_max,
                    "explore_onboard_adaptive_clearance": parsed.explore_onboard_adaptive_clearance,
                    "explore_onboard_adaptive_turn_dot": parsed.explore_onboard_adaptive_turn_dot,
                    "explore_onboard_stall_viewpoint_ticks": parsed.explore_onboard_stall_viewpoint_ticks,
                    "explore_onboard_stall_viewpoint_max_depth": parsed.explore_onboard_stall_viewpoint_max_depth,
                    "explore_onboard_approach_viewpoints": parsed.explore_onboard_approach_viewpoints,
                    "explore_onboard_approach_max_depth": parsed.explore_onboard_approach_max_depth,
                    "explore_onboard_emit_objective_target": parsed.explore_onboard_emit_objective_target,
                    "explore_onboard_objective_obs": parsed.explore_onboard_objective_obs,
                    "explore_onboard_region_commit": parsed.explore_onboard_region_commit,
                    "explore_onboard_region_commit_stall_ticks": parsed.explore_onboard_region_commit_stall_ticks,
                    "explore_onboard_persistent_regions": parsed.explore_onboard_persistent_regions,
                    "explore_onboard_region_tracker": parsed.explore_onboard_region_tracker,
                    "explore_onboard_clearance_passable": parsed.explore_onboard_clearance_passable,
                    "explore_onboard_passable_radius": parsed.explore_onboard_passable_radius,
                    "explore_onboard_passage_quality": parsed.explore_onboard_passage_quality,
                    "explore_onboard_passage_weight": parsed.explore_onboard_passage_weight,
                    "explore_onboard_clearance_cone_deg": parsed.explore_onboard_clearance_cone_deg,
                    "explore_onboard_budget_frontier": parsed.explore_onboard_budget_frontier,
                    "explore_onboard_budget_gain_weight": parsed.explore_onboard_budget_gain_weight,
                    "explore_onboard_budget_cluster_weight": parsed.explore_onboard_budget_cluster_weight,
                    "explore_onboard_budget_cell_weight": parsed.explore_onboard_budget_cell_weight,
                    "explore_onboard_budget_depth_bias": parsed.explore_onboard_budget_depth_bias,
                    "explore_onboard_budget_depth_weight": parsed.explore_onboard_budget_depth_weight,
                    "explore_onboard_optimistic_unknown": parsed.explore_onboard_optimistic_unknown,
                    "explore_onboard_frontier_graph": parsed.explore_onboard_frontier_graph,
                    "explore_onboard_graph_candidate_obs": parsed.explore_onboard_graph_candidate_obs,
                    "explore_onboard_graph_gain_weight": parsed.explore_onboard_graph_gain_weight,
                    "explore_onboard_graph_cluster_weight": parsed.explore_onboard_graph_cluster_weight,
                    "explore_onboard_graph_depth_weight": parsed.explore_onboard_graph_depth_weight,
                    "explore_onboard_graph_active_bonus": parsed.explore_onboard_graph_active_bonus,
                    "explore_onboard_graph_stale_penalty": parsed.explore_onboard_graph_stale_penalty,
                    "explore_onboard_graph_failure_penalty": parsed.explore_onboard_graph_failure_penalty,
                    "explore_onboard_graph_attempt_penalty": parsed.explore_onboard_graph_attempt_penalty,
                    "explore_onboard_graph_gain_cap": parsed.explore_onboard_graph_gain_cap,
                    "drive_objective_when_close": parsed.drive_objective_when_close,
                    "drive_objective_close_dist": parsed.drive_objective_close_dist,
                    "drive_objective_min_dist": parsed.drive_objective_min_dist,
                    "deploy_patch_obs": parsed.deploy_patch_obs,
                    "deploy_graph_candidate_obs": parsed.deploy_graph_candidate_obs,
                    "deploy_camera_depth_obs": parsed.deploy_camera_depth_obs,
                    "camera_depth_obs": parsed.camera_depth_obs,
                    "camera_depth_width": parsed.camera_depth_width,
                    "camera_depth_height": parsed.camera_depth_height,
                    "camera_depth_fov_x_deg": parsed.camera_depth_fov_x_deg,
                    "camera_depth_fov_y_deg": parsed.camera_depth_fov_y_deg,
                    "deploy_coverage_obs": parsed.deploy_coverage_obs,
                    "deploy_onboard_target_obs": parsed.deploy_onboard_target_obs,
                    "train_mode": "frontier_target_distill",
                    "init_checkpoint": parsed.init_checkpoint,
                    "predict_speed_scale": parsed.predict_speed_scale,
                    "predict_frontier_sectors": parsed.predict_frontier_sectors,
                    "predict_frontier_vector": parsed.predict_frontier_vector,
                    "predict_recovery_gate": parsed.predict_recovery_gate,
                    "predict_frontier_patch": parsed.predict_frontier_patch,
                    "predict_graph_candidate": parsed.predict_graph_candidate,
                    "graph_candidate_label_source": parsed.graph_candidate_label_source,
                    "graph_candidate_loss_weight": parsed.graph_candidate_loss_weight,
                    "graph_candidate_label_align_weight": parsed.graph_candidate_label_align_weight,
                    "graph_candidate_label_clearance_weight": parsed.graph_candidate_label_clearance_weight,
                    "graph_candidate_label_gain_weight": parsed.graph_candidate_label_gain_weight,
                    "graph_candidate_label_cluster_weight": parsed.graph_candidate_label_cluster_weight,
                    "graph_candidate_label_passage_weight": parsed.graph_candidate_label_passage_weight,
                    "graph_candidate_label_selected_weight": parsed.graph_candidate_label_selected_weight,
                    "graph_candidate_label_depth_weight": parsed.graph_candidate_label_depth_weight,
                    "graph_candidate_label_min_align": parsed.graph_candidate_label_min_align,
                    "map_sidecar_encoder": parsed.map_sidecar_encoder,
                    "freeze_except_map_sidecar": parsed.freeze_except_map_sidecar,
                    "far_map_start": parsed.far_map_start,
                    "far_map_side": parsed.far_map_side,
                    "separate_frontier_patch_head": separate_patch_head,
                    "frontier_vector_dist_loss_weight": parsed.frontier_vector_dist_loss_weight,
                    "unclamped_frontier_vector_dist_loss": parsed.unclamped_frontier_vector_dist_loss,
                    "freeze_except_new_inputs": parsed.freeze_except_new_inputs,
                    "new_input_start": parsed.new_input_start,
                    "use_pred_vector_fallback": parsed.use_pred_vector_fallback,
                    "pred_vector_fallback_primary_threshold": parsed.pred_vector_fallback_primary_threshold,
                    "pred_vector_fallback_min_coverage": parsed.pred_vector_fallback_min_coverage,
                    "pred_vector_fallback_min_dist": parsed.pred_vector_fallback_min_dist,
                    "use_pred_recovery_gate": parsed.use_pred_recovery_gate,
                    "pred_recovery_gate_threshold": parsed.pred_recovery_gate_threshold,
                    "pred_recovery_gate_min_coverage": parsed.pred_recovery_gate_min_coverage,
                    "pred_recovery_gate_min_dist": parsed.pred_recovery_gate_min_dist,
                    "speed_fraction": parsed.speed_fraction,
                    "yaw_fraction": parsed.yaw_fraction,
                    "safety_cone_ray_stop": parsed.safety_cone_ray_stop,
                    "safety_cone_deg": parsed.safety_cone_deg,
                    "safety_slow_ray": parsed.safety_slow_ray,
                    "safety_min_ray_speed_cap": parsed.safety_min_ray_speed_cap,
                    "safety_min_ray_speed_stop": parsed.safety_min_ray_speed_stop,
                    "safety_min_ray_speed_slow": parsed.safety_min_ray_speed_slow,
                    "safety_min_ray_speed_min_scale": parsed.safety_min_ray_speed_min_scale,
                    "safety_passage_speed_cap": parsed.safety_passage_speed_cap,
                    "safety_passage_speed_near": parsed.safety_passage_speed_near,
                    "safety_passage_speed_front": parsed.safety_passage_speed_front,
                    "safety_passage_speed_side_gap": parsed.safety_passage_speed_side_gap,
                    "safety_passage_speed_stop": parsed.safety_passage_speed_stop,
                    "safety_passage_speed_slow": parsed.safety_passage_speed_slow,
                    "safety_passage_speed_min_scale": parsed.safety_passage_speed_min_scale,
                    "safety_passage_center_actions": parsed.safety_passage_center_actions,
                    "safety_passage_center_near": parsed.safety_passage_center_near,
                    "safety_passage_center_front": parsed.safety_passage_center_front,
                    "safety_passage_center_gain": parsed.safety_passage_center_gain,
                    "safety_passage_center_max_turn_deg": parsed.safety_passage_center_max_turn_deg,
                    "safety_passage_center_min_imbalance": parsed.safety_passage_center_min_imbalance,
                    "safety_corridor_width_actions": parsed.safety_corridor_width_actions,
                    "safety_corridor_width_near": parsed.safety_corridor_width_near,
                    "safety_corridor_width_front": parsed.safety_corridor_width_front,
                    "safety_corridor_width_side_stop": parsed.safety_corridor_width_side_stop,
                    "safety_corridor_width_side_slow": parsed.safety_corridor_width_side_slow,
                    "safety_corridor_width_min_scale": parsed.safety_corridor_width_min_scale,
                    "safety_corridor_width_max_turn_deg": parsed.safety_corridor_width_max_turn_deg,
                    "safety_corridor_width_side_deg": parsed.safety_corridor_width_side_deg,
                    "safety_corridor_width_progress_floor": parsed.safety_corridor_width_progress_floor,
                    "safety_slide": parsed.safety_slide,
                    "safety_slide_progress_aware": parsed.safety_slide_progress_aware,
                    "safety_slide_progress_fraction": parsed.safety_slide_progress_fraction,
                    "safety_slide_progress_cone": parsed.safety_slide_progress_cone,
                    "safety_barrier_actions": parsed.safety_barrier_actions,
                    "safety_barrier_near_wall": parsed.safety_barrier_near_wall,
                    "safety_trajectory_actions": parsed.safety_trajectory_actions,
                    "safety_trajectory_near_wall": parsed.safety_trajectory_near_wall,
                    "safety_trajectory_steps": parsed.safety_trajectory_steps,
                    "safety_trajectory_dt": parsed.safety_trajectory_dt,
                    "safety_trajectory_min_fit": parsed.safety_trajectory_min_fit,
                    "safety_trajectory_trigger_scale": parsed.safety_trajectory_trigger_scale,
                    "safety_trajectory_direct_min_fit": parsed.safety_trajectory_direct_min_fit,
                },
                "policy": {
                    "network": parsed.network,
                    "hidden_size": parsed.hidden_size,
                    "num_layers": parsed.num_layers,
                    "action_std": parsed.action_std,
                },
                "final_loss": last_loss,
            },
            path,
        )
        print(f"wrote_checkpoint update={update} path={path} loss={last_loss:.6g}", flush=True)

    try:
        for update in range(1, parsed.seq_updates + 1):
            runner = runners[(update - 1) % len(runners)]
            vec = runner["vec"]
            obs = runner["obs"]
            terminals = runner["terminals"]
            rollout_state = runner["rollout_state"]
            anchor_rollout_state = runner["anchor_rollout_state"]
            start_state = clone_detached_state(rollout_state)
            inputs = []
            targets = []
            anchor_targets = []
            sample_weights = []
            coverages = []
            sector_fallback_fracs = []
            vector_fallback_fracs = []
            dones = []
            use_anchor = (
                anchor_policy is not None
                and parsed.anchor_loss_weight > 0.0
                and runner["mode"] in anchor_modes
            )
            use_hard_weights = parsed.hard_example_weight > 0.0 and runner["mode"] in hard_modes
            use_action_weights = (
                parsed.action_policy
                and (
                    parsed.action_policy_correction_weight > 0.0
                    or parsed.action_policy_clearance_weight > 0.0
                )
            )
            use_false_negative = (
                parsed.false_negative_weight > 0.0
                and runner["mode"] in false_negative_modes
            )

            if update <= parsed.teacher_rollout_updates:
                student_prob = 0.0
            else:
                denom = max(1, parsed.seq_updates - parsed.teacher_rollout_updates)
                frac = (update - parsed.teacher_rollout_updates) / denom
                student_prob = min(parsed.max_student_rollout_prob, frac * parsed.max_student_rollout_prob)
            last_student_prob = student_prob

            for _ in range(parsed.seq_len):
                x_t = deployable_input(obs, parsed).detach()
                label_t = target_labels(obs, parsed).detach()
                teacher_drive_target = None
                if parsed.predict_graph_candidate:
                    y_t = graph_candidate_label_targets(obs, label_t, parsed).detach()
                    teacher_drive_target = graph_candidate_target_from_index(
                        obs,
                        y_t[:, 0].long(),
                        y_t[:, 1],
                    ).detach()
                elif parsed.action_policy:
                    y_t = target_to_actions(label_t, parsed, obs).detach()
                    if use_action_weights:
                        raw_y_t = target_to_actions(
                            label_t,
                            parsed,
                            obs,
                            use_safety=False,
                        ).detach()
                        sample_weights.append(
                            action_policy_sample_weights(obs, raw_y_t, y_t, parsed).detach()
                        )
                else:
                    y_t = label_t
                inputs.append(x_t)
                targets.append(y_t)
                if parsed.sector_fallback_target:
                    sector_fallback_fracs.append(
                        sector_fallback_mask(obs, parsed).float().mean().detach()
                    )
                if parsed.vector_fallback_target:
                    vector_fallback_fracs.append(
                        vector_fallback_mask(obs, parsed).float().mean().detach()
                    )
                if use_false_negative:
                    coverages.append(obs[:, 17].float().detach())

                with torch.no_grad():
                    if use_anchor:
                        anchor_logits, _, anchor_rollout_state = anchor_policy.forward_eval(
                            x_t, anchor_rollout_state
                        )
                        anchor_targets.append(anchor_logits.mean.detach())
                    logits, _, rollout_state = policy.forward_eval(x_t, rollout_state)
                    student_pred = logits.mean.detach()
                    if parsed.predict_graph_candidate:
                        student_target = graph_candidate_target_from_logits(
                            student_pred, obs, parsed
                        )
                        y_drive = teacher_drive_target
                        if student_prob <= 0.0:
                            drive_target = y_drive
                        else:
                            use_student = (
                                torch.rand((vec.total_agents, 1), device=device) < student_prob
                            )
                            drive_target = torch.where(use_student, student_target, y_drive)
                    else:
                        student_target = student_pred
                        y_drive = y_t[:, : student_target.shape[1]]
                        if student_prob <= 0.0:
                            drive_target = y_drive
                        else:
                            use_student = (
                                torch.rand((vec.total_agents, 1), device=device) < student_prob
                            )
                            drive_target = torch.where(use_student, student_target, y_drive)
                    if use_hard_weights:
                        student_action = (
                            action_policy_to_actions(student_target, parsed, obs)
                            if parsed.action_policy
                            else target_to_actions(student_target, parsed, obs)
                        )
                        sample_weights.append(
                            hard_state_weights(obs, student_target, y_drive, student_action, parsed).detach()
                        )
                    actions = (
                        action_policy_to_actions(drive_target, parsed, obs)
                        if parsed.action_policy
                        else target_to_actions(drive_target, parsed, obs)
                    )

                step_vec(vec, actions, device)
                done = terminals > 0.5
                dones.append(done.clone())
                if done.any():
                    rollout_state = reset_done_state(rollout_state, done)
                    if use_anchor:
                        anchor_rollout_state = reset_done_state(anchor_rollout_state, done)
                rollout_state = detach_state(rollout_state)
                if use_anchor:
                    anchor_rollout_state = detach_state(anchor_rollout_state)
            runner["rollout_state"] = rollout_state
            if use_anchor:
                runner["anchor_rollout_state"] = anchor_rollout_state

            x = torch.stack(inputs, dim=1)
            target = torch.stack(targets, dim=1)
            done_seq = torch.stack(dones, dim=1)
            weights = torch.stack(sample_weights, dim=1) if (use_hard_weights or use_action_weights) else None
            coverage = torch.stack(coverages, dim=1) if use_false_negative else None
            if patch_head is not None:
                pred, hidden = predict_sequence_with_hidden(policy, x, start_state, done_seq)
            else:
                pred = predict_sequence(policy, x, start_state, done_seq)
                hidden = None
            target_main = target if parsed.predict_graph_candidate else target[..., : pred.shape[-1]]
            graph_loss = torch.zeros((), dtype=torch.float32, device=device)
            if parsed.predict_graph_candidate:
                graph_loss = graph_candidate_loss(pred, target_main, parsed, weights=weights)
                loss = parsed.graph_candidate_loss_weight * graph_loss
            elif parsed.action_policy:
                loss = action_policy_loss(pred, target_main, weights=weights)
            else:
                loss = target_loss(pred, target_main, parsed, weights=weights)
            false_negative_loss = torch.zeros((), dtype=torch.float32, device=device)
            if use_false_negative and not parsed.action_policy and not parsed.predict_graph_candidate:
                false_negative_loss = false_negative_exists_loss(
                    pred, target_main, parsed, coverage=coverage, weights=weights
                )
                loss = loss + parsed.false_negative_weight * false_negative_loss
            sector_loss = torch.zeros((), dtype=torch.float32, device=device)
            if (
                parsed.predict_frontier_sectors
                and parsed.frontier_sector_loss_weight > 0.0
                and not parsed.action_policy
            ):
                sector_loss = frontier_sector_loss(pred, target_main, weights=weights)
                loss = loss + parsed.frontier_sector_loss_weight * sector_loss
            vector_loss = torch.zeros((), dtype=torch.float32, device=device)
            if (
                parsed.predict_frontier_vector
                and parsed.frontier_vector_loss_weight > 0.0
                and not parsed.action_policy
            ):
                vector_loss = frontier_vector_loss(pred, target_main, parsed, weights=weights)
                loss = loss + parsed.frontier_vector_loss_weight * vector_loss
            patch_loss = torch.zeros((), dtype=torch.float32, device=device)
            if (
                parsed.predict_frontier_patch
                and parsed.frontier_patch_loss_weight > 0.0
                and not parsed.action_policy
            ):
                if patch_head is not None:
                    patch_pred = patch_head(hidden.reshape(-1, hidden.shape[-1])).reshape(
                        hidden.shape[0],
                        hidden.shape[1],
                        FRONTIER_PATCH_SIZE,
                    )
                    patch_loss = frontier_patch_aux_loss(patch_pred, target, weights=weights)
                else:
                    patch_loss = frontier_patch_loss(pred, target, weights=weights)
                loss = loss + parsed.frontier_patch_loss_weight * patch_loss
            gate_loss = torch.zeros((), dtype=torch.float32, device=device)
            gate_frac = torch.zeros((), dtype=torch.float32, device=device)
            if (
                parsed.predict_recovery_gate
                and parsed.recovery_gate_loss_weight > 0.0
                and not parsed.action_policy
            ):
                gate_loss, gate_frac = recovery_gate_loss(pred, target_main, x, parsed, weights=weights)
                loss = loss + parsed.recovery_gate_loss_weight * gate_loss
            anchor_loss = torch.zeros((), dtype=torch.float32, device=device)
            if use_anchor:
                anchor_target = torch.stack(anchor_targets, dim=1)
                anchor_loss = F.mse_loss(pred[..., : anchor_target.shape[-1]], anchor_target)
                loss = loss + parsed.anchor_loss_weight * anchor_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if parsed.freeze_except_frontier_vector_dist:
                mask_frontier_vector_dist_grads(policy)
            if parsed.freeze_except_recovery_gate:
                mask_recovery_gate_grads(policy)
            if parsed.freeze_except_new_inputs:
                mask_new_input_grads(policy, parsed.new_input_start)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), parsed.max_grad_norm)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            last_anchor_loss = float(anchor_loss.detach().cpu())
            last_false_negative_loss = float(false_negative_loss.detach().cpu())
            last_frontier_sector_loss = float(sector_loss.detach().cpu())
            last_frontier_vector_loss = float(vector_loss.detach().cpu())
            last_frontier_patch_loss = float(patch_loss.detach().cpu())
            last_graph_candidate_loss = float(graph_loss.detach().cpu())
            if parsed.predict_graph_candidate:
                last_graph_candidate_valid = float(target_main[..., 1].float().mean().detach().cpu())
            else:
                last_graph_candidate_valid = 0.0
            last_recovery_gate_loss = float(gate_loss.detach().cpu())
            last_recovery_gate_frac = float(gate_frac.detach().cpu())
            if sector_fallback_fracs:
                last_sector_fallback_frac = float(
                    torch.stack(sector_fallback_fracs).mean().detach().cpu()
                )
            else:
                last_sector_fallback_frac = 0.0
            if vector_fallback_fracs:
                last_vector_fallback_frac = float(
                    torch.stack(vector_fallback_fracs).mean().detach().cpu()
                )
            else:
                last_vector_fallback_frac = 0.0
            if weights is not None:
                last_hard_frac = float((weights > 1.0).float().mean().detach().cpu())
                last_hard_weight = float(weights.mean().detach().cpu())
            else:
                last_hard_frac = 0.0
                last_hard_weight = 1.0

            if update % parsed.print_interval == 0 or update == parsed.seq_updates:
                logs = vec.log()
                print(
                    f"target_update={update} mode={runner['mode']} loss={last_loss:.6g} "
                    f"anchor_loss={last_anchor_loss:.6g} "
                    f"fn_loss={last_false_negative_loss:.6g} "
                    f"sector_loss={last_frontier_sector_loss:.6g} "
                    f"vector_loss={last_frontier_vector_loss:.6g} "
                    f"patch_loss={last_frontier_patch_loss:.6g} "
                    f"graph_loss={last_graph_candidate_loss:.6g} "
                    f"graph_valid={last_graph_candidate_valid:.4f} "
                    f"gate_loss={last_recovery_gate_loss:.6g} "
                    f"gate_frac={last_recovery_gate_frac:.4f} "
                    f"sector_fallback={last_sector_fallback_frac:.4f} "
                    f"vector_fallback={last_vector_fallback_frac:.4f} "
                    f"hard_frac={last_hard_frac:.4f} hard_weight={last_hard_weight:.3f} "
                    f"student_prob={last_student_prob:.3f} "
                    f"success={float(logs.get('success', 0.0)):.4f} "
                    f"coverage={float(logs.get('coverage', 0.0)):.4f} "
                    f"obstacle={float(logs.get('obstacle_hit', 0.0)):.4f} "
                    f"timeout={float(logs.get('timeout', 0.0)):.4f}",
                    flush=True,
                )
            if parsed.checkpoint_interval > 0 and update % parsed.checkpoint_interval == 0:
                save_interval_checkpoint(update)

        output = output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "aux_state_dict": {
                    "frontier_patch_head": patch_head.state_dict() if patch_head is not None else None,
                },
                "input_size": vec0.obs_size,
                "num_atns": len(act_sizes),
                "act_sizes": act_sizes,
                "config": {
                    "task_mode": 3,
                    "obstacle_mode": parsed.obstacle_mode,
                    "train_modes": train_modes,
                    "action_policy": parsed.action_policy,
                    "action_policy_safety": parsed.action_policy_safety,
                    "action_policy_correction_weight": parsed.action_policy_correction_weight,
                    "action_policy_clearance_weight": parsed.action_policy_clearance_weight,
                    "explore_memory_obs": parsed.explore_memory_obs,
                    "cell_size": parsed.cell_size,
                    "explore_frontier_target_mode": parsed.explore_frontier_target_mode,
                    "explore_reachable_coverage": parsed.explore_reachable_coverage,
                    "deploy_patch_obs": parsed.deploy_patch_obs,
                    "deploy_graph_candidate_obs": parsed.deploy_graph_candidate_obs,
                    "deploy_camera_depth_obs": parsed.deploy_camera_depth_obs,
                    "camera_depth_obs": parsed.camera_depth_obs,
                    "camera_depth_width": parsed.camera_depth_width,
                    "camera_depth_height": parsed.camera_depth_height,
                    "camera_depth_fov_x_deg": parsed.camera_depth_fov_x_deg,
                    "camera_depth_fov_y_deg": parsed.camera_depth_fov_y_deg,
                    "deploy_coverage_obs": parsed.deploy_coverage_obs,
                    "deploy_onboard_target_obs": parsed.deploy_onboard_target_obs,
                    "train_mode": "frontier_target_distill",
                    "init_checkpoint": parsed.init_checkpoint,
                    "resize_input_head": parsed.resize_input_head,
                    "resize_output_head": parsed.resize_output_head,
                    "anchor_checkpoint": parsed.anchor_checkpoint,
                    "anchor_modes": sorted(anchor_modes),
                    "anchor_loss_weight": parsed.anchor_loss_weight,
                    "false_negative_modes": sorted(false_negative_modes),
                    "false_negative_weight": parsed.false_negative_weight,
                    "false_negative_margin": parsed.false_negative_margin,
                    "false_negative_min_coverage": parsed.false_negative_min_coverage,
                    "hard_modes": sorted(hard_modes),
                    "hard_example_weight": parsed.hard_example_weight,
                    "hard_angle_deg": parsed.hard_angle_deg,
                    "hard_clearance_gap": parsed.hard_clearance_gap,
                    "hard_low_speed": parsed.hard_low_speed,
                    "hard_min_coverage": parsed.hard_min_coverage,
                    "explore_onboard_path_lookahead": parsed.explore_onboard_path_lookahead,
                    "explore_onboard_adaptive_lookahead": parsed.explore_onboard_adaptive_lookahead,
                    "explore_onboard_adaptive_lookahead_min": parsed.explore_onboard_adaptive_lookahead_min,
                    "explore_onboard_adaptive_lookahead_max": parsed.explore_onboard_adaptive_lookahead_max,
                    "explore_onboard_adaptive_clearance": parsed.explore_onboard_adaptive_clearance,
                    "explore_onboard_adaptive_turn_dot": parsed.explore_onboard_adaptive_turn_dot,
                    "explore_onboard_stall_viewpoint_ticks": parsed.explore_onboard_stall_viewpoint_ticks,
                    "explore_onboard_stall_viewpoint_max_depth": parsed.explore_onboard_stall_viewpoint_max_depth,
                    "explore_onboard_approach_viewpoints": parsed.explore_onboard_approach_viewpoints,
                    "explore_onboard_approach_max_depth": parsed.explore_onboard_approach_max_depth,
                    "explore_onboard_emit_objective_target": parsed.explore_onboard_emit_objective_target,
                    "explore_onboard_objective_obs": parsed.explore_onboard_objective_obs,
                    "explore_onboard_region_commit": parsed.explore_onboard_region_commit,
                    "explore_onboard_region_commit_stall_ticks": parsed.explore_onboard_region_commit_stall_ticks,
                    "explore_onboard_persistent_regions": parsed.explore_onboard_persistent_regions,
                    "explore_onboard_region_tracker": parsed.explore_onboard_region_tracker,
                    "explore_onboard_clearance_passable": parsed.explore_onboard_clearance_passable,
                    "explore_onboard_passable_radius": parsed.explore_onboard_passable_radius,
                    "explore_onboard_passage_quality": parsed.explore_onboard_passage_quality,
                    "explore_onboard_passage_weight": parsed.explore_onboard_passage_weight,
                    "explore_onboard_clearance_cone_deg": parsed.explore_onboard_clearance_cone_deg,
                    "explore_onboard_budget_frontier": parsed.explore_onboard_budget_frontier,
                    "explore_onboard_budget_gain_weight": parsed.explore_onboard_budget_gain_weight,
                    "explore_onboard_budget_cluster_weight": parsed.explore_onboard_budget_cluster_weight,
                    "explore_onboard_budget_cell_weight": parsed.explore_onboard_budget_cell_weight,
                    "explore_onboard_budget_depth_bias": parsed.explore_onboard_budget_depth_bias,
                    "explore_onboard_budget_depth_weight": parsed.explore_onboard_budget_depth_weight,
                    "explore_onboard_optimistic_unknown": parsed.explore_onboard_optimistic_unknown,
                    "explore_onboard_frontier_graph": parsed.explore_onboard_frontier_graph,
                    "explore_onboard_graph_candidate_obs": parsed.explore_onboard_graph_candidate_obs,
                    "explore_onboard_graph_gain_weight": parsed.explore_onboard_graph_gain_weight,
                    "explore_onboard_graph_cluster_weight": parsed.explore_onboard_graph_cluster_weight,
                    "explore_onboard_graph_depth_weight": parsed.explore_onboard_graph_depth_weight,
                    "explore_onboard_graph_active_bonus": parsed.explore_onboard_graph_active_bonus,
                    "explore_onboard_graph_stale_penalty": parsed.explore_onboard_graph_stale_penalty,
                    "explore_onboard_graph_failure_penalty": parsed.explore_onboard_graph_failure_penalty,
                    "explore_onboard_graph_attempt_penalty": parsed.explore_onboard_graph_attempt_penalty,
                    "explore_onboard_graph_gain_cap": parsed.explore_onboard_graph_gain_cap,
                    "drive_objective_when_close": parsed.drive_objective_when_close,
                    "drive_objective_close_dist": parsed.drive_objective_close_dist,
                    "drive_objective_min_dist": parsed.drive_objective_min_dist,
                    "target_slots_zeroed": True,
                    "seq_len": parsed.seq_len,
                    "teacher_rollout_updates": parsed.teacher_rollout_updates,
                    "max_student_rollout_prob": parsed.max_student_rollout_prob,
                    "exists_loss_weight": parsed.exists_loss_weight,
                    "negative_exists_weight": parsed.negative_exists_weight,
                    "false_positive_weight": parsed.false_positive_weight,
                    "false_positive_margin": parsed.false_positive_margin,
                    "absent_direction_weight": parsed.absent_direction_weight,
                    "absent_fallback": parsed.absent_fallback,
                    "absent_fallback_min_ray": parsed.absent_fallback_min_ray,
                    "absent_fallback_angle_penalty": parsed.absent_fallback_angle_penalty,
                    "predict_speed_scale": parsed.predict_speed_scale,
                    "predict_frontier_sectors": parsed.predict_frontier_sectors,
                    "predict_frontier_vector": parsed.predict_frontier_vector,
                    "predict_recovery_gate": parsed.predict_recovery_gate,
                    "predict_frontier_patch": parsed.predict_frontier_patch,
                    "predict_graph_candidate": parsed.predict_graph_candidate,
                    "graph_candidate_label_source": parsed.graph_candidate_label_source,
                    "graph_candidate_loss_weight": parsed.graph_candidate_loss_weight,
                    "graph_candidate_label_align_weight": parsed.graph_candidate_label_align_weight,
                    "graph_candidate_label_clearance_weight": parsed.graph_candidate_label_clearance_weight,
                    "graph_candidate_label_gain_weight": parsed.graph_candidate_label_gain_weight,
                    "graph_candidate_label_cluster_weight": parsed.graph_candidate_label_cluster_weight,
                    "graph_candidate_label_passage_weight": parsed.graph_candidate_label_passage_weight,
                    "graph_candidate_label_selected_weight": parsed.graph_candidate_label_selected_weight,
                    "graph_candidate_label_depth_weight": parsed.graph_candidate_label_depth_weight,
                    "graph_candidate_label_min_align": parsed.graph_candidate_label_min_align,
                    "map_sidecar_encoder": parsed.map_sidecar_encoder,
                    "freeze_except_map_sidecar": parsed.freeze_except_map_sidecar,
                    "far_map_start": parsed.far_map_start,
                    "far_map_side": parsed.far_map_side,
                    "separate_frontier_patch_head": separate_patch_head,
                    "frontier_sector_loss_weight": parsed.frontier_sector_loss_weight,
                    "frontier_vector_loss_weight": parsed.frontier_vector_loss_weight,
                    "frontier_vector_dist_loss_weight": parsed.frontier_vector_dist_loss_weight,
                    "unclamped_frontier_vector_dist_loss": parsed.unclamped_frontier_vector_dist_loss,
                    "freeze_except_frontier_vector_dist": parsed.freeze_except_frontier_vector_dist,
                    "freeze_except_recovery_gate": parsed.freeze_except_recovery_gate,
                    "freeze_except_new_inputs": parsed.freeze_except_new_inputs,
                    "new_input_start": parsed.new_input_start,
                    "frontier_patch_loss_weight": parsed.frontier_patch_loss_weight,
                    "sector_fallback_target": parsed.sector_fallback_target,
                    "sector_fallback_threshold": parsed.sector_fallback_threshold,
                    "sector_fallback_min_coverage": parsed.sector_fallback_min_coverage,
                    "sector_fallback_distance": parsed.sector_fallback_distance,
                    "vector_fallback_target": parsed.vector_fallback_target,
                    "vector_fallback_min_coverage": parsed.vector_fallback_min_coverage,
                    "vector_fallback_min_dist": parsed.vector_fallback_min_dist,
                    "use_pred_vector_fallback": parsed.use_pred_vector_fallback,
                    "pred_vector_fallback_primary_threshold": parsed.pred_vector_fallback_primary_threshold,
                    "pred_vector_fallback_min_coverage": parsed.pred_vector_fallback_min_coverage,
                    "pred_vector_fallback_min_dist": parsed.pred_vector_fallback_min_dist,
                    "use_pred_recovery_gate": parsed.use_pred_recovery_gate,
                    "pred_recovery_gate_threshold": parsed.pred_recovery_gate_threshold,
                    "pred_recovery_gate_min_coverage": parsed.pred_recovery_gate_min_coverage,
                    "pred_recovery_gate_min_dist": parsed.pred_recovery_gate_min_dist,
                    "recovery_gate_loss_weight": parsed.recovery_gate_loss_weight,
                    "recovery_gate_bad_angle_deg": parsed.recovery_gate_bad_angle_deg,
                    "recovery_gate_margin_deg": parsed.recovery_gate_margin_deg,
                    "recovery_gate_min_coverage": parsed.recovery_gate_min_coverage,
                    "recovery_gate_min_dist": parsed.recovery_gate_min_dist,
                    "speed_loss_weight": parsed.speed_loss_weight,
                    "speed_fraction": parsed.speed_fraction,
                    "yaw_fraction": parsed.yaw_fraction,
                    "safety_cone_ray_stop": parsed.safety_cone_ray_stop,
                    "safety_cone_deg": parsed.safety_cone_deg,
                    "safety_slow_ray": parsed.safety_slow_ray,
                    "safety_min_ray_speed_cap": parsed.safety_min_ray_speed_cap,
                    "safety_min_ray_speed_stop": parsed.safety_min_ray_speed_stop,
                    "safety_min_ray_speed_slow": parsed.safety_min_ray_speed_slow,
                    "safety_min_ray_speed_min_scale": parsed.safety_min_ray_speed_min_scale,
                    "safety_passage_speed_cap": parsed.safety_passage_speed_cap,
                    "safety_passage_speed_near": parsed.safety_passage_speed_near,
                    "safety_passage_speed_front": parsed.safety_passage_speed_front,
                    "safety_passage_speed_side_gap": parsed.safety_passage_speed_side_gap,
                    "safety_passage_speed_stop": parsed.safety_passage_speed_stop,
                    "safety_passage_speed_slow": parsed.safety_passage_speed_slow,
                    "safety_passage_speed_min_scale": parsed.safety_passage_speed_min_scale,
                    "safety_passage_center_actions": parsed.safety_passage_center_actions,
                    "safety_passage_center_near": parsed.safety_passage_center_near,
                    "safety_passage_center_front": parsed.safety_passage_center_front,
                    "safety_passage_center_gain": parsed.safety_passage_center_gain,
                    "safety_passage_center_max_turn_deg": parsed.safety_passage_center_max_turn_deg,
                    "safety_passage_center_min_imbalance": parsed.safety_passage_center_min_imbalance,
                    "safety_corridor_width_actions": parsed.safety_corridor_width_actions,
                    "safety_corridor_width_near": parsed.safety_corridor_width_near,
                    "safety_corridor_width_front": parsed.safety_corridor_width_front,
                    "safety_corridor_width_side_stop": parsed.safety_corridor_width_side_stop,
                    "safety_corridor_width_side_slow": parsed.safety_corridor_width_side_slow,
                    "safety_corridor_width_min_scale": parsed.safety_corridor_width_min_scale,
                    "safety_corridor_width_max_turn_deg": parsed.safety_corridor_width_max_turn_deg,
                    "safety_corridor_width_side_deg": parsed.safety_corridor_width_side_deg,
                    "safety_corridor_width_progress_floor": parsed.safety_corridor_width_progress_floor,
                    "safety_slide": parsed.safety_slide,
                    "safety_slide_progress_aware": parsed.safety_slide_progress_aware,
                    "safety_slide_progress_fraction": parsed.safety_slide_progress_fraction,
                    "safety_slide_progress_cone": parsed.safety_slide_progress_cone,
                    "safety_barrier_actions": parsed.safety_barrier_actions,
                    "safety_barrier_near_wall": parsed.safety_barrier_near_wall,
                    "safety_trajectory_actions": parsed.safety_trajectory_actions,
                    "safety_trajectory_near_wall": parsed.safety_trajectory_near_wall,
                    "safety_trajectory_steps": parsed.safety_trajectory_steps,
                    "safety_trajectory_dt": parsed.safety_trajectory_dt,
                    "safety_trajectory_min_fit": parsed.safety_trajectory_min_fit,
                    "safety_trajectory_trigger_scale": parsed.safety_trajectory_trigger_scale,
                    "safety_trajectory_direct_min_fit": parsed.safety_trajectory_direct_min_fit,
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
    finally:
        for runner in runners:
            runner["vec"].close()

    if parsed.eval_after:
        return run_eval(parsed, policy=policy)
    return {}


def add_common(parser):
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--total-agents", type=int, default=1024)
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--max-eval-steps", type=int, default=50000)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--max-vel", type=float, default=0.7)
    parser.add_argument("--max-z-vel", type=float, default=0.25)
    parser.add_argument("--vel-ctrl-kp-vel", type=float, default=0.0)
    parser.add_argument("--vel-ctrl-ki-z", type=float, default=4.0)
    parser.add_argument("--spawn-radius", type=float, default=2.5)
    parser.add_argument("--spawn-radius-min", type=float, default=0.0)
    parser.add_argument("--spawn-height", type=float, default=1.5)
    parser.add_argument("--room-size", type=float, default=3.0)
    parser.add_argument("--ceiling-height", type=float, default=3.0)
    parser.add_argument("--num-rays", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=4.0)
    parser.add_argument("--camera-depth-obs", action="store_true")
    parser.add_argument("--camera-depth-width", type=int, default=DEPTH_CAMERA_DEFAULT_WIDTH)
    parser.add_argument("--camera-depth-height", type=int, default=DEPTH_CAMERA_DEFAULT_HEIGHT)
    parser.add_argument("--camera-depth-fov-x-deg", type=float, default=90.0)
    parser.add_argument("--camera-depth-fov-y-deg", type=float, default=60.0)
    parser.add_argument("--mode", choices=tuple(OBSTACLE_MODES), default="maze")
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--explore-goal-fraction", type=float, default=0.85)
    parser.add_argument("--explore-memory-obs", type=int, default=3)
    parser.add_argument(
        "--explore-frontier-target-mode",
        choices=tuple(FRONTIER_TARGET_MODES),
        default="nearest",
    )
    parser.add_argument("--explore-reachable-coverage", action="store_true")
    parser.add_argument("--explore-onboard-commit-steps", type=int, default=0)
    parser.add_argument("--explore-onboard-path-lookahead", type=int, default=1)
    parser.add_argument("--explore-onboard-adaptive-lookahead", action="store_true")
    parser.add_argument("--explore-onboard-adaptive-lookahead-min", type=int, default=2)
    parser.add_argument("--explore-onboard-adaptive-lookahead-max", type=int, default=4)
    parser.add_argument("--explore-onboard-adaptive-clearance", type=float, default=0.55)
    parser.add_argument("--explore-onboard-adaptive-turn-dot", type=float, default=0.55)
    parser.add_argument("--explore-onboard-commit-min-clearance", type=float, default=0.35)
    parser.add_argument("--explore-onboard-stall-viewpoint-ticks", type=int, default=0)
    parser.add_argument("--explore-onboard-stall-viewpoint-max-depth", type=int, default=1)
    parser.add_argument("--explore-onboard-approach-viewpoints", action="store_true")
    parser.add_argument("--explore-onboard-approach-max-depth", type=int, default=1)
    parser.add_argument("--explore-onboard-emit-objective-target", action="store_true")
    parser.add_argument("--explore-onboard-objective-obs", action="store_true")
    parser.add_argument("--explore-onboard-region-commit", action="store_true")
    parser.add_argument("--explore-onboard-region-commit-stall-ticks", type=int, default=0)
    parser.add_argument("--explore-onboard-persistent-regions", action="store_true")
    parser.add_argument("--explore-onboard-region-tracker", action="store_true")
    parser.add_argument("--explore-onboard-clearance-passable", action="store_true")
    parser.add_argument("--explore-onboard-passable-radius", type=float, default=0.0)
    parser.add_argument("--explore-onboard-passage-quality", action="store_true")
    parser.add_argument("--explore-onboard-passage-weight", type=float, default=0.35)
    parser.add_argument("--explore-onboard-clearance-cone-deg", type=float, default=30.0)
    parser.add_argument("--explore-onboard-budget-frontier", action="store_true")
    parser.add_argument("--explore-onboard-budget-gain-weight", type=float, default=1.0)
    parser.add_argument("--explore-onboard-budget-cluster-weight", type=float, default=0.35)
    parser.add_argument("--explore-onboard-budget-cell-weight", type=float, default=0.25)
    parser.add_argument("--explore-onboard-budget-depth-bias", type=float, default=6.0)
    parser.add_argument("--explore-onboard-budget-depth-weight", type=float, default=1.0)
    parser.add_argument("--explore-onboard-optimistic-unknown", action="store_true")
    parser.add_argument("--explore-onboard-frontier-graph", action="store_true")
    parser.add_argument("--explore-onboard-graph-candidate-obs", action="store_true")
    parser.add_argument("--explore-onboard-graph-gain-weight", type=float, default=0.32)
    parser.add_argument("--explore-onboard-graph-cluster-weight", type=float, default=0.12)
    parser.add_argument("--explore-onboard-graph-depth-weight", type=float, default=0.045)
    parser.add_argument("--explore-onboard-graph-active-bonus", type=float, default=0.75)
    parser.add_argument("--explore-onboard-graph-stale-penalty", type=float, default=0.020)
    parser.add_argument("--explore-onboard-graph-failure-penalty", type=float, default=0.70)
    parser.add_argument("--explore-onboard-graph-attempt-penalty", type=float, default=0.12)
    parser.add_argument("--explore-onboard-graph-gain-cap", type=int, default=192)
    parser.add_argument("--predict-graph-candidate", action="store_true")
    parser.add_argument("--graph-candidate-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--graph-candidate-label-source",
        choices=("privileged", "selected"),
        default="privileged",
    )
    parser.add_argument("--graph-candidate-label-align-weight", type=float, default=2.0)
    parser.add_argument("--graph-candidate-label-clearance-weight", type=float, default=0.40)
    parser.add_argument("--graph-candidate-label-gain-weight", type=float, default=0.30)
    parser.add_argument("--graph-candidate-label-cluster-weight", type=float, default=0.10)
    parser.add_argument("--graph-candidate-label-passage-weight", type=float, default=0.20)
    parser.add_argument("--graph-candidate-label-selected-weight", type=float, default=0.0)
    parser.add_argument("--graph-candidate-label-depth-weight", type=float, default=0.10)
    parser.add_argument("--graph-candidate-label-min-align", type=float, default=-1.0)
    parser.add_argument("--drive-objective-when-close", action="store_true")
    parser.add_argument("--drive-objective-close-dist", type=float, default=0.04)
    parser.add_argument("--drive-objective-min-dist", type=float, default=0.08)
    parser.add_argument("--deploy-patch-obs", action="store_true")
    parser.add_argument("--deploy-graph-candidate-obs", action="store_true")
    parser.add_argument("--deploy-camera-depth-obs", action="store_true")
    parser.add_argument("--deploy-coverage-obs", action="store_true")
    parser.add_argument("--deploy-onboard-target-obs", action="store_true")
    parser.add_argument("--speed-fraction", type=float, default=0.55)
    parser.add_argument("--yaw-fraction", type=float, default=0.35)
    parser.add_argument("--exists-threshold", type=float, default=0.2)
    parser.add_argument("--absent-fallback", choices=("none", "clear_ray"), default="none")
    parser.add_argument("--absent-fallback-min-ray", type=float, default=0.60)
    parser.add_argument("--absent-fallback-angle-penalty", type=float, default=0.15)
    parser.add_argument("--safety-min-ray-stop", type=float, default=0.0)
    parser.add_argument("--safety-action-ray-stop", type=float, default=0.0)
    parser.add_argument("--safety-cone-ray-stop", type=float, default=0.0)
    parser.add_argument("--safety-cone-deg", type=float, default=45.0)
    parser.add_argument("--safety-slow-ray", type=float, default=0.0)
    parser.add_argument("--safety-min-ray-speed-cap", action="store_true")
    parser.add_argument("--safety-min-ray-speed-stop", type=float, default=0.18)
    parser.add_argument("--safety-min-ray-speed-slow", type=float, default=0.45)
    parser.add_argument("--safety-min-ray-speed-min-scale", type=float, default=0.0)
    parser.add_argument("--safety-passage-speed-cap", action="store_true")
    parser.add_argument("--safety-passage-speed-near", type=float, default=0.45)
    parser.add_argument("--safety-passage-speed-front", type=float, default=0.80)
    parser.add_argument("--safety-passage-speed-side-gap", type=float, default=0.20)
    parser.add_argument("--safety-passage-speed-stop", type=float, default=0.20)
    parser.add_argument("--safety-passage-speed-slow", type=float, default=0.60)
    parser.add_argument("--safety-passage-speed-min-scale", type=float, default=0.0)
    parser.add_argument("--safety-momentum-speed-cap", action="store_true")
    parser.add_argument("--safety-momentum-speed-near", type=float, default=0.45)
    parser.add_argument("--safety-momentum-speed-stop", type=float, default=0.20)
    parser.add_argument("--safety-momentum-speed-slow", type=float, default=0.60)
    parser.add_argument("--safety-momentum-speed-min-vel", type=float, default=0.08)
    parser.add_argument("--safety-momentum-speed-toward", type=float, default=0.02)
    parser.add_argument("--safety-momentum-speed-min-scale", type=float, default=0.0)
    parser.add_argument("--safety-passage-center-actions", action="store_true")
    parser.add_argument("--safety-passage-center-near", type=float, default=0.45)
    parser.add_argument("--safety-passage-center-front", type=float, default=0.60)
    parser.add_argument("--safety-passage-center-gain", type=float, default=0.80)
    parser.add_argument("--safety-passage-center-max-turn-deg", type=float, default=35.0)
    parser.add_argument("--safety-passage-center-min-imbalance", type=float, default=0.05)
    parser.add_argument("--safety-corridor-width-actions", action="store_true")
    parser.add_argument("--safety-corridor-width-near", type=float, default=0.55)
    parser.add_argument("--safety-corridor-width-front", type=float, default=0.70)
    parser.add_argument("--safety-corridor-width-side-stop", type=float, default=0.24)
    parser.add_argument("--safety-corridor-width-side-slow", type=float, default=0.65)
    parser.add_argument("--safety-corridor-width-min-scale", type=float, default=0.35)
    parser.add_argument("--safety-corridor-width-max-turn-deg", type=float, default=55.0)
    parser.add_argument("--safety-corridor-width-side-deg", type=float, default=80.0)
    parser.add_argument("--safety-corridor-width-progress-floor", type=float, default=0.10)
    parser.add_argument("--safety-slide", action="store_true")
    parser.add_argument("--safety-slide-progress-aware", action="store_true")
    parser.add_argument("--safety-slide-progress-fraction", type=float, default=0.60)
    parser.add_argument("--safety-slide-progress-cone", type=float, default=0.60)
    parser.add_argument("--safety-barrier-actions", action="store_true")
    parser.add_argument("--safety-barrier-near-wall", type=float, default=0.36)
    parser.add_argument("--safety-trajectory-actions", action="store_true")
    parser.add_argument("--safety-trajectory-near-wall", type=float, default=0.42)
    parser.add_argument("--safety-trajectory-steps", type=int, default=8)
    parser.add_argument("--safety-trajectory-dt", type=float, default=0.08)
    parser.add_argument("--safety-trajectory-min-fit", type=float, default=-0.25)
    parser.add_argument("--safety-trajectory-trigger-scale", type=float, default=0.45)
    parser.add_argument("--safety-trajectory-direct-min-fit", type=float, default=0.20)
    parser.add_argument("--safety-sample-actions", action="store_true")
    parser.add_argument("--safety-sample-trigger-scale", type=float, default=0.35)
    parser.add_argument("--safety-stateful-recovery", action="store_true")
    parser.add_argument("--recovery-policy", default="")
    parser.add_argument("--safety-progress-floor-scale", type=float, default=0.0)
    parser.add_argument("--safety-progress-floor-ray", type=float, default=1.00)
    parser.add_argument("--safety-progress-floor-cone", type=float, default=0.35)
    parser.add_argument("--recovery-stall-steps", type=int, default=360)
    parser.add_argument("--recovery-duration-steps", type=int, default=72)
    parser.add_argument("--recovery-near-wall", type=float, default=0.36)
    parser.add_argument("--recovery-progress-delta", type=float, default=0.002)
    parser.add_argument("--recovery-rollout-steps", type=int, default=8)
    parser.add_argument("--recovery-rollout-dt", type=float, default=0.08)
    parser.add_argument("--use-pred-speed-scale", action="store_true")
    parser.add_argument("--external-safety-with-pred-speed", action="store_true")
    parser.add_argument("--use-pred-vector-fallback", action="store_true")
    parser.add_argument("--pred-vector-fallback-primary-threshold", type=float, default=0.2)
    parser.add_argument("--pred-vector-fallback-min-coverage", type=float, default=0.0)
    parser.add_argument("--pred-vector-fallback-min-dist", type=float, default=0.02)
    parser.add_argument("--use-pred-recovery-gate", action="store_true")
    parser.add_argument("--pred-recovery-gate-threshold", type=float, default=0.5)
    parser.add_argument("--pred-recovery-gate-min-coverage", type=float, default=0.25)
    parser.add_argument("--pred-recovery-gate-min-dist", type=float, default=0.02)
    parser.add_argument("--action-policy-safety", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--network", choices=("MLP", "MinGRU", "GRU", "LSTM"), default="MinGRU")
    parser.add_argument("--action-std", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--log-interval", type=int, default=64)
    parser.add_argument("--drain-initial-episodes", action="store_true")
    parser.add_argument("--ignore-checkpoint-env-config", action="store_true")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill reachable frontier-path targets from deployable observation history."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--output", required=True)
    train.add_argument("--init-checkpoint")
    train.add_argument("--resize-input-head", action="store_true")
    train.add_argument("--resize-output-head", action="store_true")
    train.add_argument("--zero-deploy-coverage-input-weight", action="store_true")
    train.add_argument("--action-policy", action="store_true")
    train.add_argument("--action-policy-correction-weight", type=float, default=0.0)
    train.add_argument("--action-policy-clearance-weight", type=float, default=0.0)
    train.add_argument("--freeze-except-new-inputs", action="store_true")
    train.add_argument("--new-input-start", type=int)
    train.add_argument("--map-sidecar-encoder", action="store_true")
    train.add_argument("--freeze-except-map-sidecar", action="store_true")
    train.add_argument("--far-map-start", type=int, default=FAR_MAP_START)
    train.add_argument("--far-map-side", type=int, default=FAR_MAP_SIDE)
    train.add_argument("--train-modes", nargs="+", choices=tuple(OBSTACLE_MODES))
    train.add_argument("--anchor-checkpoint")
    train.add_argument("--anchor-modes", nargs="+", choices=tuple(OBSTACLE_MODES))
    train.add_argument("--anchor-loss-weight", type=float, default=0.0)
    train.add_argument("--false-negative-modes", nargs="+", choices=tuple(OBSTACLE_MODES))
    train.add_argument("--false-negative-weight", type=float, default=0.0)
    train.add_argument("--false-negative-margin", type=float, default=0.80)
    train.add_argument("--false-negative-min-coverage", type=float, default=0.0)
    train.add_argument("--hard-modes", nargs="+", choices=tuple(OBSTACLE_MODES))
    train.add_argument("--hard-example-weight", type=float, default=0.0)
    train.add_argument("--hard-angle-deg", type=float, default=60.0)
    train.add_argument("--hard-clearance-gap", type=float, default=0.20)
    train.add_argument("--hard-low-speed", type=float, default=0.0)
    train.add_argument("--hard-min-coverage", type=float, default=0.0)
    train.add_argument("--seq-updates", type=int, default=800)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--teacher-rollout-updates", type=int, default=80)
    train.add_argument("--max-student-rollout-prob", type=float, default=0.5)
    train.add_argument("--print-interval", type=int, default=20)
    train.add_argument("--checkpoint-interval", type=int, default=0)
    train.add_argument("--checkpoint-dir")
    train.add_argument("--eval-after", action="store_true")
    train.add_argument("--dist-loss-weight", type=float, default=0.5)
    train.add_argument("--exists-loss-weight", type=float, default=0.5)
    train.add_argument("--negative-exists-weight", type=float, default=1.0)
    train.add_argument("--false-positive-weight", type=float, default=0.0)
    train.add_argument("--false-positive-margin", type=float, default=0.1)
    train.add_argument("--absent-direction-weight", type=float, default=0.0)
    train.add_argument("--predict-speed-scale", action="store_true")
    train.add_argument("--speed-loss-weight", type=float, default=1.0)
    train.add_argument("--predict-frontier-sectors", action="store_true")
    train.add_argument("--frontier-sector-loss-weight", type=float, default=1.0)
    train.add_argument("--predict-frontier-vector", action="store_true")
    train.add_argument("--frontier-vector-loss-weight", type=float, default=1.0)
    train.add_argument("--frontier-vector-dist-loss-weight", type=float, default=0.5)
    train.add_argument("--unclamped-frontier-vector-dist-loss", action="store_true")
    train.add_argument("--freeze-except-frontier-vector-dist", action="store_true")
    train.add_argument("--predict-recovery-gate", action="store_true")
    train.add_argument("--freeze-except-recovery-gate", action="store_true")
    train.add_argument("--recovery-gate-loss-weight", type=float, default=1.0)
    train.add_argument("--recovery-gate-bad-angle-deg", type=float, default=60.0)
    train.add_argument("--recovery-gate-margin-deg", type=float, default=10.0)
    train.add_argument("--recovery-gate-min-coverage", type=float, default=0.25)
    train.add_argument("--recovery-gate-min-dist", type=float, default=0.02)
    train.add_argument("--predict-frontier-patch", action="store_true")
    train.add_argument("--frontier-patch-loss-weight", type=float, default=1.0)
    train.add_argument("--separate-frontier-patch-head", action="store_true")
    train.add_argument("--sector-fallback-target", action="store_true")
    train.add_argument("--sector-fallback-threshold", type=float, default=0.15)
    train.add_argument("--sector-fallback-min-coverage", type=float, default=0.25)
    train.add_argument("--sector-fallback-distance", type=float, default=0.5)
    train.add_argument("--vector-fallback-target", action="store_true")
    train.add_argument("--vector-fallback-min-coverage", type=float, default=0.25)
    train.add_argument("--vector-fallback-min-dist", type=float, default=0.02)

    eval_p = sub.add_parser("eval")
    add_common(eval_p)
    eval_p.add_argument("checkpoint")

    oracle = sub.add_parser("oracle-eval")
    add_common(oracle)

    diagnose = sub.add_parser("diagnose")
    add_common(diagnose)
    diagnose.add_argument("checkpoint")
    diagnose.add_argument("--controllers", choices=("student", "oracle", "both"), default="both")
    diagnose.add_argument("--crash-reward-threshold", type=float, default=-5.0)
    diagnose.add_argument("--success-reward-threshold", type=float, default=5.0)
    diagnose.add_argument("--danger-clearance", type=float, default=0.45)
    diagnose.add_argument("--clearance-gap", type=float, default=0.20)

    parsed = parser.parse_args()
    parsed.obstacle_mode = OBSTACLE_MODES[parsed.mode]
    if getattr(parsed, "predict_frontier_sectors", False) and getattr(
        parsed, "predict_speed_scale", False
    ):
        parser.error("--predict-frontier-sectors cannot be combined with --predict-speed-scale")
    if getattr(parsed, "predict_frontier_vector", False) and getattr(
        parsed, "predict_speed_scale", False
    ):
        parser.error("--predict-frontier-vector cannot be combined with --predict-speed-scale")
    if getattr(parsed, "predict_frontier_patch", False) and getattr(
        parsed, "predict_speed_scale", False
    ):
        parser.error("--predict-frontier-patch cannot be combined with --predict-speed-scale")
    if getattr(parsed, "predict_recovery_gate", False) and getattr(
        parsed, "predict_speed_scale", False
    ):
        parser.error("--predict-recovery-gate cannot be combined with --predict-speed-scale")
    if getattr(parsed, "predict_frontier_vector", False) and not getattr(
        parsed, "predict_frontier_sectors", False
    ):
        parser.error("--predict-frontier-vector requires --predict-frontier-sectors")
    if getattr(parsed, "predict_recovery_gate", False) and not getattr(
        parsed, "predict_frontier_vector", False
    ):
        parser.error("--predict-recovery-gate requires --predict-frontier-vector")
    if getattr(parsed, "predict_recovery_gate", False) and getattr(
        parsed, "predict_frontier_patch", False
    ):
        parser.error("--predict-recovery-gate cannot be combined with --predict-frontier-patch")
    if getattr(parsed, "predict_frontier_patch", False) and not getattr(
        parsed, "predict_frontier_vector", False
    ):
        parser.error("--predict-frontier-patch requires --predict-frontier-vector")
    if getattr(parsed, "predict_graph_candidate", False):
        incompatible = (
            "action_policy",
            "predict_speed_scale",
            "predict_frontier_sectors",
            "predict_frontier_vector",
            "predict_recovery_gate",
            "predict_frontier_patch",
            "sector_fallback_target",
            "vector_fallback_target",
        )
        active = [name for name in incompatible if getattr(parsed, name, False)]
        if active:
            parser.error("--predict-graph-candidate cannot be combined with " + ", ".join(active))
        parsed.deploy_graph_candidate_obs = True
        parsed.explore_onboard_graph_candidate_obs = True
    if getattr(parsed, "sector_fallback_target", False) and not getattr(
        parsed, "predict_frontier_sectors", False
    ):
        parser.error("--sector-fallback-target requires --predict-frontier-sectors")
    if getattr(parsed, "vector_fallback_target", False) and not getattr(
        parsed, "predict_frontier_vector", False
    ):
        parser.error("--vector-fallback-target requires --predict-frontier-vector")
    if getattr(parsed, "deploy_onboard_target_obs", False) and getattr(
        parsed, "explore_frontier_target_mode", None
    ) != "onboard":
        parser.error("--deploy-onboard-target-obs requires --explore-frontier-target-mode onboard")
    if getattr(parsed, "deploy_graph_candidate_obs", False):
        parsed.explore_onboard_graph_candidate_obs = True
    if getattr(parsed, "predict_speed_scale", False):
        parsed.use_pred_speed_scale = True
    return parsed


def main():
    parsed = parse_args()
    if parsed.cmd == "train":
        run_train(parsed)
    elif parsed.cmd == "eval":
        parsed.checkpoint = parsed.checkpoint
        run_eval(parsed)
    elif parsed.cmd == "oracle-eval":
        run_oracle_eval(parsed)
    elif parsed.cmd == "diagnose":
        run_diagnostic(parsed)
    else:
        raise SystemExit(f"unknown command {parsed.cmd}")


if __name__ == "__main__":
    main()
