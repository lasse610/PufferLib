#!/usr/bin/env python3
import argparse
from collections import deque

import numpy as np


OBSTACLES = {
    "none": (),
    "center_block": ((-0.35, 0.35, -0.35, 0.35),),
    "doorway": ((-3.0, -0.45, -0.08, 0.08), (0.45, 3.0, -0.08, 0.08)),
    "corridor": ((-3.0, 3.0, 0.45, 3.0), (-3.0, 3.0, -3.0, -0.45)),
    "maze": ((-0.08, 0.08, -3.0, 0.0), (-0.55, 3.0, 0.80, 0.95), (-3.0, -1.35, -0.35, 3.0)),
}

MODE_ALIASES = {
    "open": "none",
    "block": "center_block",
    "random_blocks": "maze",
    "layout_mixed": "maze",
}


def obstacle_boxes(mode, room_size):
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in OBSTACLES:
        raise ValueError(f"unsupported static obstacle mode {mode!r}")
    scale = room_size / 3.0
    return tuple(tuple(value * scale for value in box) for box in OBSTACLES[mode])


def in_obstacle(x, y, boxes, margin):
    for xmin, xmax, ymin, ymax in boxes:
        if xmin - margin <= x <= xmax + margin and ymin - margin <= y <= ymax + margin:
            return True
    return False


def valid_xy(x, y, room_size, boxes, margin):
    if x < -room_size + margin or x > room_size - margin:
        return False
    if y < -room_size + margin or y > room_size - margin:
        return False
    return not in_obstacle(x, y, boxes, margin)


def cell_centers(room_size, cell_size):
    dim = int(np.ceil((2.0 * room_size) / cell_size))
    coords = -room_size + (np.arange(dim, dtype=np.float32) + 0.5) * cell_size
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return coords, xx, yy


def ray_box_distance(ox, oy, dx, dy, box):
    xmin, xmax, ymin, ymax = box
    tmin = 0.0
    tmax = 1.0e9
    if abs(dx) < 1e-8:
        if ox < xmin or ox > xmax:
            return None
    else:
        t1 = (xmin - ox) / dx
        t2 = (xmax - ox) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)

    if abs(dy) < 1e-8:
        if oy < ymin or oy > ymax:
            return None
    else:
        t1 = (ymin - oy) / dy
        t2 = (ymax - oy) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)

    if tmax < tmin or tmax < 0.0:
        return None
    return max(tmin, 0.0)


def ray_distance(x, y, dx, dy, room_size, boxes, ray_length):
    best = ray_length
    if dx > 1e-8:
        best = min(best, (room_size - x) / dx)
    elif dx < -1e-8:
        best = min(best, (-room_size - x) / dx)
    if dy > 1e-8:
        best = min(best, (room_size - y) / dy)
    elif dy < -1e-8:
        best = min(best, (-room_size - y) / dy)

    for box in boxes:
        dist = ray_box_distance(x, y, dx, dy, box)
        if dist is not None and dist < best:
            best = dist
    return max(0.0, min(best, ray_length))


def cell_index(x, y, room_size, cell_size, dim, valid_mask):
    ix = int(np.floor((x + room_size) / cell_size))
    iy = int(np.floor((y + room_size) / cell_size))
    if ix < 0 or ix >= dim or iy < 0 or iy >= dim:
        return None
    if not valid_mask[iy, ix]:
        return None
    return iy, ix


def visible_mask_from_pose(x, y, args, boxes, valid_cells):
    dim = valid_cells.shape[0]
    seen = np.zeros_like(valid_cells, dtype=bool)
    own = cell_index(x, y, args.room_size, args.cell_size, dim, valid_cells)
    if own is not None:
        seen[own] = True

    step = max(args.cell_size * 0.75, 0.10)
    for ray in range(args.num_rays):
        angle = 2.0 * np.pi * ray / args.num_rays
        dx = float(np.cos(angle))
        dy = float(np.sin(angle))
        dist = ray_distance(x, y, dx, dy, args.room_size, boxes, args.ray_length)
        t = step
        while t < dist:
            idx = cell_index(x + dx * t, y + dy * t, args.room_size, args.cell_size, dim, valid_cells)
            if idx is not None:
                seen[idx] = True
            t += step
        idx = cell_index(x + dx * dist, y + dy * dist, args.room_size, args.cell_size, dim, valid_cells)
        if idx is not None:
            seen[idx] = True
    return seen


def connected_components(mask):
    labels = -np.ones(mask.shape, dtype=np.int32)
    sizes = []
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x] or labels[y, x] >= 0:
                continue
            label = len(sizes)
            labels[y, x] = label
            size = 0
            queue = deque([(y, x)])
            while queue:
                cy, cx = queue.popleft()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or nx < 0 or ny >= mask.shape[0] or nx >= mask.shape[1]:
                        continue
                    if mask[ny, nx] and labels[ny, nx] < 0:
                        labels[ny, nx] = label
                        queue.append((ny, nx))
            sizes.append(size)
    return labels, sizes


def shortest_paths(start, traversable):
    dist = -np.ones(traversable.shape, dtype=np.int32)
    if start is None:
        return dist
    dist[start] = 0
    queue = deque([start])
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if ny < 0 or nx < 0 or ny >= traversable.shape[0] or nx >= traversable.shape[1]:
                continue
            if traversable[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[cy, cx] + 1
                queue.append((ny, nx))
    return dist


def build_world(args):
    boxes = obstacle_boxes(args.mode, args.room_size)
    coords, xx, yy = cell_centers(args.room_size, args.cell_size)
    valid_cells = np.zeros_like(xx, dtype=bool)
    drone_cells = np.zeros_like(xx, dtype=bool)
    for iy in range(xx.shape[0]):
        for ix in range(xx.shape[1]):
            x = float(xx[iy, ix])
            y = float(yy[iy, ix])
            valid_cells[iy, ix] = valid_xy(x, y, args.room_size, boxes, args.coverage_margin)
            drone_cells[iy, ix] = valid_xy(x, y, args.room_size, boxes, args.drone_radius)
    return boxes, coords, xx, yy, valid_cells, drone_cells


def precompute_visibility(args, boxes, xx, yy, valid_cells, drone_cells):
    poses = []
    masks = []
    for iy in range(drone_cells.shape[0]):
        for ix in range(drone_cells.shape[1]):
            if not drone_cells[iy, ix]:
                continue
            poses.append((iy, ix))
            masks.append(visible_mask_from_pose(float(xx[iy, ix]), float(yy[iy, ix]), args, boxes, valid_cells))
    return poses, masks


def pose_mask_for_component(poses, masks, labels, label):
    visible = np.zeros_like(labels, dtype=bool)
    count = 0
    for (iy, ix), mask in zip(poses, masks):
        if labels[iy, ix] == label:
            visible |= mask
            count += 1
    return count, visible


def greedy_episode(start, args, drone_cells, poses, masks, pose_to_index):
    seen = masks[pose_to_index[start]].copy()
    current = start
    path_cells = 0
    rows = []
    for step in range(args.greedy_steps + 1):
        rows.append((step, current, int(seen.sum()), path_cells))
        if step == args.greedy_steps:
            break
        dist = shortest_paths(current, drone_cells)
        best_score = None
        best_pose = None
        best_new = 0
        for pose, mask in zip(poses, masks):
            d = int(dist[pose])
            if d < 0:
                continue
            new = int((mask & ~seen).sum())
            if new <= 0:
                continue
            score = new / max(d, 1)
            if best_score is None or score > best_score:
                best_score = score
                best_pose = pose
                best_new = new
        if best_pose is None:
            break
        path_cells += max(int(dist[best_pose]), 0)
        current = best_pose
        seen |= masks[pose_to_index[current]]
    return rows, seen


def sample_starts(args, drone_cells, rng):
    candidates = np.argwhere(drone_cells)
    starts = []
    while len(starts) < args.episodes and len(candidates):
        iy, ix = candidates[rng.integers(0, len(candidates))]
        x = -args.room_size + (ix + 0.5) * args.cell_size
        y = -args.room_size + (iy + 0.5) * args.cell_size
        if max(abs(x), abs(y)) <= args.spawn_radius:
            starts.append((int(iy), int(ix)))
    return starts


def fmt(value):
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6g}"


def print_table(rows, columns):
    print(",".join(columns))
    for row in rows:
        print(",".join(fmt(row.get(column, "")) for column in columns))


def main():
    parser = argparse.ArgumentParser(description="Geometry/frontier probe for quadrotor exploration layouts")
    parser.add_argument("--mode", default="maze", choices=sorted(set(OBSTACLES) | set(MODE_ALIASES)))
    parser.add_argument("--room-size", type=float, default=3.0)
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--coverage-margin", type=float, default=0.30)
    parser.add_argument("--drone-radius", type=float, default=0.12)
    parser.add_argument("--num-rays", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=4.0)
    parser.add_argument("--spawn-radius", type=float, default=2.5)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--greedy-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=88)
    args = parser.parse_args()

    boxes, coords, xx, yy, valid_cells, drone_cells = build_world(args)
    poses, masks = precompute_visibility(args, boxes, xx, yy, valid_cells, drone_cells)
    pose_to_index = {pose: idx for idx, pose in enumerate(poses)}
    total_valid = int(valid_cells.sum())
    total_drone = int(drone_cells.sum())

    labels, comp_sizes = connected_components(drone_cells)
    component_rows = []
    for label, size in enumerate(comp_sizes):
        pose_count, visible = pose_mask_for_component(poses, masks, labels, label)
        component_rows.append({
            "component": label,
            "drone_cells": size,
            "poses": pose_count,
            "union_visible_cells": int(visible.sum()),
            "union_coverage": float(visible.sum() / max(total_valid, 1)),
        })

    all_visible = np.zeros_like(valid_cells, dtype=bool)
    for mask in masks:
        all_visible |= mask

    rng = np.random.default_rng(args.seed)
    starts = sample_starts(args, drone_cells, rng)
    episode_rows = []
    for start in starts:
        rows, seen = greedy_episode(start, args, drone_cells, poses, masks, pose_to_index)
        final_step, final_pose, _, path_cells = rows[-1]
        episode_rows.append({
            "start_y": start[0],
            "start_x": start[1],
            "steps": final_step,
            "path_m": path_cells * args.cell_size,
            "coverage": float(seen.sum() / max(total_valid, 1)),
            "visible_cells": int(seen.sum()),
            "component": int(labels[start]),
        })

    coverages = np.array([row["coverage"] for row in episode_rows], dtype=np.float32)
    paths = np.array([row["path_m"] for row in episode_rows], dtype=np.float32)
    summary = {
        "mode": args.mode,
        "valid_cells": total_valid,
        "drone_cells": total_drone,
        "components": len(comp_sizes),
        "all_pose_union_coverage": float(all_visible.sum() / max(total_valid, 1)),
        "spawned_episodes": len(episode_rows),
        "greedy_success_rate_85": float(np.mean(coverages >= 0.85)) if len(coverages) else 0.0,
        "greedy_mean_coverage": float(np.mean(coverages)) if len(coverages) else 0.0,
        "greedy_p10_coverage": float(np.quantile(coverages, 0.10)) if len(coverages) else 0.0,
        "greedy_mean_path_m": float(np.mean(paths)) if len(paths) else 0.0,
    }
    print_table([summary], (
        "mode",
        "valid_cells",
        "drone_cells",
        "components",
        "all_pose_union_coverage",
        "spawned_episodes",
        "greedy_success_rate_85",
        "greedy_mean_coverage",
        "greedy_p10_coverage",
        "greedy_mean_path_m",
    ))
    print()
    print_table(component_rows, (
        "component",
        "drone_cells",
        "poses",
        "union_visible_cells",
        "union_coverage",
    ))
    print()
    print_table(sorted(episode_rows, key=lambda row: row["coverage"])[:10], (
        "component",
        "start_y",
        "start_x",
        "steps",
        "path_m",
        "coverage",
        "visible_cells",
    ))


if __name__ == "__main__":
    main()
