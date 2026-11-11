#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib.pufferl import _resolve_backend
from pufferlib.torch_pufferl import _actions_for_vec_step

from quadrotor4_bottleneck_probe import (
    MODE_NAMES,
    build_cases,
    configure_case_args,
    parse_float_list,
    parse_mode_list,
    resolve_named_checkpoint,
)
from quadrotor4_eval_headless import (
    detach_state,
    resolve_checkpoint,
    reset_done_state,
)


QR_OBS_SIZE = 26
QR_ROUTE_WALL_OBS_SIZE = 6
QR_ROUTE_EXTRA_OBS_SIZE = 8
QR_ROUTE_PERCEPTION_OBS_SIZE = 12
QR_ROUTE_COMPACT_PERCEPTION_OBS_SIZE = 8
QR_ROUTE_AFFORDANCE_OBS_SIZE = 12
QR_ACTION_HISTORY_OBS_SIZE = 4

DEFAULT_CASES = (
    "tiny:lr:straight:y+0.45->+0.45",
    "small:lr:straight:y+0.45->+0.45",
    "small:rl:straight:y+0.65->+0.65",
    "large:lr:straight:y+0.45->+0.45",
)


class ActivationStats:
    def __init__(self):
        self.count = 0
        self.numel = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.abs_max = 0.0
        self.abs_gt_1 = 0
        self.abs_gt_3 = 0
        self.abs_gt_5 = 0
        self.abs_gt_10 = 0
        self.near_zero = 0
        self.nonfinite = 0

    def update(self, tensor):
        x = tensor.detach()
        if x.numel() == 0:
            return
        x = x.float().reshape(-1)
        finite = torch.isfinite(x)
        self.count += 1
        self.numel += int(x.numel())
        self.nonfinite += int((~finite).sum().item())
        if not finite.any():
            return

        xf = x[finite]
        abs_x = xf.abs()
        self.sum += float(xf.sum().item())
        self.sumsq += float((xf * xf).sum().item())
        self.min = min(self.min, float(xf.min().item()))
        self.max = max(self.max, float(xf.max().item()))
        self.abs_max = max(self.abs_max, float(abs_x.max().item()))
        self.abs_gt_1 += int((abs_x > 1.0).sum().item())
        self.abs_gt_3 += int((abs_x > 3.0).sum().item())
        self.abs_gt_5 += int((abs_x > 5.0).sum().item())
        self.abs_gt_10 += int((abs_x > 10.0).sum().item())
        self.near_zero += int((abs_x < 1e-6).sum().item())

    def finalize(self, name):
        finite_numel = max(0, self.numel - self.nonfinite)
        if finite_numel == 0:
            return {
                "module": name,
                "calls": self.count,
                "numel": self.numel,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "abs_max": float("nan"),
                "frac_abs_gt_1": float("nan"),
                "frac_abs_gt_3": float("nan"),
                "frac_abs_gt_5": float("nan"),
                "frac_abs_gt_10": float("nan"),
                "frac_near_zero": float("nan"),
                "frac_nonfinite": 1.0 if self.numel else 0.0,
            }

        mean = self.sum / finite_numel
        var = max(0.0, self.sumsq / finite_numel - mean * mean)
        return {
            "module": name,
            "calls": self.count,
            "numel": self.numel,
            "mean": mean,
            "std": math.sqrt(var),
            "min": self.min,
            "max": self.max,
            "abs_max": self.abs_max,
            "frac_abs_gt_1": self.abs_gt_1 / finite_numel,
            "frac_abs_gt_3": self.abs_gt_3 / finite_numel,
            "frac_abs_gt_5": self.abs_gt_5 / finite_numel,
            "frac_abs_gt_10": self.abs_gt_10 / finite_numel,
            "frac_near_zero": self.near_zero / finite_numel,
            "frac_nonfinite": self.nonfinite / self.numel if self.numel else 0.0,
        }


class ActivationRecorder:
    def __init__(self, policy):
        self.stats = {}
        self.last = {}
        self.handles = []
        for name, module in policy.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.GELU)):
                self.stats[name] = ActivationStats()
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return
            self.stats[name].update(output)
            x = output.detach().float().reshape(-1)
            finite = torch.isfinite(x)
            if finite.any():
                xf = x[finite]
                self.last[name] = {
                    "mean": float(xf.mean().item()),
                    "std": float(xf.std(unbiased=False).item()),
                    "abs_max": float(xf.abs().max().item()),
                    "frac_abs_gt_3": float((xf.abs() > 3.0).float().mean().item()),
                    "frac_nonfinite": float((~finite).float().mean().item()),
                }
            else:
                self.last[name] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "abs_max": float("nan"),
                    "frac_abs_gt_3": float("nan"),
                    "frac_nonfinite": 1.0,
                }
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def summary_rows(self):
        return [self.stats[name].finalize(name) for name in sorted(self.stats)]

    def row_features(self):
        features = {}
        for name in (
            "encoder.encoder",
            "network.net.0",
            "network.net.2",
            "network.net.4",
            "network.net.6",
            "decoder.decoder_mean",
            "decoder.value_function",
        ):
            stats = self.last.get(name)
            if not stats:
                continue
            safe = slugify(name)
            features[f"act_{safe}_abs_max"] = stats["abs_max"]
            features[f"act_{safe}_std"] = stats["std"]
            features[f"act_{safe}_frac_abs_gt_3"] = stats["frac_abs_gt_3"]
        return features


def slugify(value):
    value = value.replace("+", "p").replace("-", "m")
    value = value.replace(">", "to")
    return re.sub(r"[^A-Za-z0-9_.]+", "_", value).strip("_")


def parse_case_labels(value):
    labels = []
    for item in value.split(","):
        item = item.strip()
        if item:
            labels.append(item)
    return labels


def finite_float(value):
    value = float(value)
    if not math.isfinite(value):
        return value
    return value


def obs_slices(env):
    idx = QR_OBS_SIZE
    out = {"base": (0, QR_OBS_SIZE), "target": (14, QR_OBS_SIZE)}

    num_rays = int(env.get("num_rays", 0))
    out["rays"] = (idx, idx + num_rays)
    idx += num_rays

    if int(env.get("route_wall_obs", 0)):
        out["wall"] = (idx, idx + QR_ROUTE_WALL_OBS_SIZE)
        idx += QR_ROUTE_WALL_OBS_SIZE
    else:
        out["wall"] = (idx, idx)

    if int(env.get("route_extra_obs", 0)):
        out["extra"] = (idx, idx + QR_ROUTE_EXTRA_OBS_SIZE)
        idx += QR_ROUTE_EXTRA_OBS_SIZE
    else:
        out["extra"] = (idx, idx)

    perception = int(env.get("route_perception_obs", 0))
    if perception == 1:
        size = QR_ROUTE_PERCEPTION_OBS_SIZE
    elif perception == 2:
        size = QR_ROUTE_COMPACT_PERCEPTION_OBS_SIZE
    elif perception == 3:
        size = QR_ROUTE_AFFORDANCE_OBS_SIZE
    else:
        size = 0
    out["perception"] = (idx, idx + size)
    idx += size

    if int(env.get("action_history_obs", 0)):
        out["action_history"] = (idx, idx + QR_ACTION_HISTORY_OBS_SIZE)
        idx += QR_ACTION_HISTORY_OBS_SIZE
    else:
        out["action_history"] = (idx, idx)

    out["deployed_prefix"] = (0, idx)
    out["patch_tail"] = (idx, None)
    return out


def array_stats(prefix, values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            f"{prefix}_abs_max": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_frac_abs_gt_095": float("nan"),
            f"{prefix}_frac_nonfinite": float("nan"),
        }
    finite = np.isfinite(values)
    if not finite.any():
        return {
            f"{prefix}_abs_max": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_frac_abs_gt_095": float("nan"),
            f"{prefix}_frac_nonfinite": 1.0,
        }
    xf = values[finite]
    return {
        f"{prefix}_abs_max": float(np.max(np.abs(xf))),
        f"{prefix}_mean": float(np.mean(xf)),
        f"{prefix}_std": float(np.std(xf)),
        f"{prefix}_frac_abs_gt_095": float(np.mean(np.abs(xf) > 0.95)),
        f"{prefix}_frac_nonfinite": float(1.0 - finite.mean()),
    }


def slice_values(obs, slices, name):
    start, end = slices[name]
    if end is None:
        return obs[start:]
    return obs[start:end]


def trace_case(checkpoint, parsed, case):
    args = configure_case_args(parsed, case)
    checkpoint = resolve_checkpoint(args, checkpoint)
    if parsed.quiet_dashboard:
        import pufferlib.pufferl as pufferl_mod
        pufferl_mod.print_dashboard = lambda *args, **kwargs: None

    backend = _resolve_backend(args)
    pufferl = backend.create_pufferl(args)
    recorder = None
    rows = []
    summary = {
        "case": case["label"],
        "checkpoint": checkpoint,
        "mode": MODE_NAMES.get(case["mode"], str(case["mode"])),
        "seed": args["env"]["seed"],
        "start_x": case["start_x"],
        "start_y": case["start_y"],
        "target_x": case["target_x"],
        "target_y": case["target_y"],
        "yaw_deg": case["yaw_deg"],
        "max_steps": parsed.max_steps,
    }

    try:
        backend.load_weights(pufferl, checkpoint)
        policy = pufferl.policy
        policy.eval()
        recorder = ActivationRecorder(policy)
        state = policy.initial_state(pufferl.total_agents, pufferl.device)
        slices = obs_slices(args["env"])
        ray_length = float(args["env"].get("ray_length", 1.0))
        prev_action = None
        reward_sum = 0.0
        done_step = None

        for step in range(1, parsed.max_steps + 1):
            obs = pufferl.vec_obs.detach().cpu().numpy()[0].astype(np.float32, copy=True)
            target = obs[14:26]
            rays = slice_values(obs, slices, "rays")
            deployed = slice_values(obs, slices, "deployed_prefix")
            patch_tail = slice_values(obs, slices, "patch_tail")
            action_hist = slice_values(obs, slices, "action_history")

            with torch.no_grad():
                logits, values, state = policy.forward_eval(pufferl.vec_obs, state)
                if isinstance(logits, torch.distributions.Normal):
                    action_tensor = logits.mean if parsed.deterministic else logits.sample()
                    action_std = logits.stddev.detach().cpu().numpy()[0]
                    action_mean = logits.mean.detach().cpu().numpy()[0]
                else:
                    action_tensor = logits
                    action_std = np.full((pufferl._vec.num_atns,), np.nan, dtype=np.float32)
                    action_mean = logits.detach().cpu().numpy()[0]

            action = action_tensor.detach().cpu().numpy()[0].astype(np.float32, copy=True)
            action_delta = (
                float(np.max(np.abs(action - prev_action)))
                if prev_action is not None else 0.0
            )
            prev_action = action.copy()

            row = {
                "step": step,
                "value": float(values.detach().cpu().reshape(-1)[0].item()),
                "target_0": float(target[0]),
                "target_1": float(target[1]),
                "target_2": float(target[2]),
                "target_3": float(target[3]),
                "target_4": float(target[4]),
                "target_5": float(target[5]),
                "vel_body_x": float(obs[0]),
                "vel_body_y": float(obs[1]),
                "vel_body_z": float(obs[2]),
                "omega_x": float(obs[3]),
                "omega_y": float(obs[4]),
                "omega_z": float(obs[5]),
                "motor_0": float(obs[10]),
                "motor_1": float(obs[11]),
                "motor_2": float(obs[12]),
                "motor_3": float(obs[13]),
                "min_ray": float(np.min(rays) * ray_length) if rays.size else float("nan"),
                "mean_ray": float(np.mean(rays) * ray_length) if rays.size else float("nan"),
                "front_ray": float(rays[0] * ray_length) if rays.size else float("nan"),
                "left_ray": float(rays[len(rays) // 4] * ray_length) if rays.size else float("nan"),
                "back_ray": float(rays[len(rays) // 2] * ray_length) if rays.size else float("nan"),
                "right_ray": float(rays[(3 * len(rays)) // 4] * ray_length) if rays.size else float("nan"),
                "action_0": float(action[0]),
                "action_1": float(action[1]),
                "action_2": float(action[2]),
                "action_3": float(action[3]),
                "action_mean_0": float(action_mean[0]),
                "action_mean_1": float(action_mean[1]),
                "action_mean_2": float(action_mean[2]),
                "action_mean_3": float(action_mean[3]),
                "action_std_0": float(action_std[0]),
                "action_std_1": float(action_std[1]),
                "action_std_2": float(action_std[2]),
                "action_std_3": float(action_std[3]),
                "action_abs_max": float(np.max(np.abs(action))),
                "action_delta": action_delta,
                "collective": float(np.mean(action)),
                "roll_proxy": float((action[0] + action[1]) - (action[2] + action[3])),
                "pitch_proxy": float((action[0] + action[3]) - (action[1] + action[2])),
                "yaw_proxy": float((-action[0] + action[1] - action[2] + action[3])),
                "action_hist_abs_max": float(np.max(np.abs(action_hist))) if action_hist.size else float("nan"),
                **array_stats("deployed_obs", deployed),
                **array_stats("target_obs", target),
                **array_stats("ray_obs", rays),
                **array_stats("patch_tail", patch_tail),
                **recorder.row_features(),
            }

            actions_flat = _actions_for_vec_step(action_tensor)
            if pufferl.gpu:
                actions_flat = actions_flat.cuda()
                pufferl._vec.gpu_step(actions_flat.data_ptr())
                torch.cuda.synchronize()
            else:
                pufferl._vec.cpu_step(actions_flat.data_ptr())

            reward = float(pufferl.vec_rewards.detach().cpu().reshape(-1)[0].item())
            terminal = bool((pufferl.vec_terminals.detach().cpu().reshape(-1)[0] > 0.5).item())
            reward_sum += reward
            row["reward"] = reward
            row["reward_sum"] = reward_sum
            row["terminal"] = int(terminal)
            rows.append(row)

            done = pufferl.vec_terminals > 0.5
            state = reset_done_state(state, done)
            state = detach_state(state)
            if terminal:
                done_step = step
                break

        env_logs = pufferl._vec.log()
        summary.update({
            f"env_{key}": float(value)
            for key, value in env_logs.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        })
        summary["done_step"] = done_step
        summary["reward_sum"] = reward_sum
        summary["rows"] = len(rows)
        if rows:
            summary["trace_min_ray"] = min(float(row["min_ray"]) for row in rows)
            summary["trace_max_action_abs"] = max(float(row["action_abs_max"]) for row in rows)
            summary["trace_max_action_delta"] = max(float(row["action_delta"]) for row in rows)
            summary["trace_value_min"] = min(float(row["value"]) for row in rows)
            summary["trace_value_max"] = max(float(row["value"]) for row in rows)
            summary["trace_deployed_obs_abs_max"] = max(float(row["deployed_obs_abs_max"]) for row in rows)

        return summary, rows, recorder.summary_rows()
    finally:
        if recorder is not None:
            recorder.close()
        backend.close(pufferl)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summaries):
    headers = (
        "case",
        "success",
        "obstacle",
        "timeout",
        "ground",
        "done",
        "min_ray",
        "max_abs_a",
        "max_da",
        "v_min",
        "v_max",
    )
    rows = []
    widths = {header: len(header) for header in headers}
    for summary in summaries:
        row = {
            "case": summary["case"],
            "success": f"{summary.get('env_success', 0.0):.3f}",
            "obstacle": f"{summary.get('env_obstacle_hit', 0.0):.3f}",
            "timeout": f"{summary.get('env_timeout', 0.0):.3f}",
            "ground": f"{summary.get('env_ground_hit', 0.0):.3f}",
            "done": str(summary.get("done_step")),
            "min_ray": f"{summary.get('trace_min_ray', float('nan')):.3f}",
            "max_abs_a": f"{summary.get('trace_max_action_abs', float('nan')):.3f}",
            "max_da": f"{summary.get('trace_max_action_delta', float('nan')):.3f}",
            "v_min": f"{summary.get('trace_value_min', float('nan')):.2f}",
            "v_max": f"{summary.get('trace_value_max', float('nan')):.2f}",
        }
        for key, value in row.items():
            widths[key] = max(widths[key], len(value))
        rows.append(row)

    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def main():
    parser = argparse.ArgumentParser(
        description="Trace one-agent AFL bottleneck hard-state rollouts with obs/action/value/activation diagnostics."
    )
    parser.add_argument("checkpoint", nargs="?", default="afl038")
    parser.add_argument(
        "--cases",
        type=parse_case_labels,
        default=list(DEFAULT_CASES),
        help="Comma-separated exact case labels. Use --list-cases to inspect labels.",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--obstacle-modes",
        type=parse_mode_list,
        default=parse_mode_list("agile_pillar_l2_marker_tiny,agile_pillar_l2_marker_small,agile_pillar_l2_marker_large"),
    )
    parser.add_argument("--max-steps", type=int, default=768)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--seed-stride", type=int, default=17)
    parser.add_argument("--start-dist", type=float, default=0.8)
    parser.add_argument("--target-dist", type=float, default=0.8)
    parser.add_argument("--laterals", type=parse_float_list, default=parse_float_list("-0.65,-0.45,0.45,0.65"))
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--include-straight", action="store_true")
    parser.add_argument("--z", type=float, default=5.0)
    parser.add_argument("--jitter-xy", type=float, default=0.02)
    parser.add_argument("--jitter-z", type=float, default=0.0)
    parser.add_argument("--route-goal-count", type=int, default=1)
    parser.add_argument("--route-goal-radius", type=float, default=0.75)
    parser.add_argument("--route-reward-mode", type=int, default=4)
    parser.add_argument("--route-target-obs-mode", type=int, default=3)
    parser.add_argument("--route-walls", type=int, default=0)
    parser.add_argument("--route-clearance-radius", type=float, default=0.80)
    parser.add_argument("--route-clearance-scale", type=float, default=0.60)
    parser.add_argument("--route-clearance-speed-scale", type=float, default=0.02)
    parser.add_argument("--route-clearance-speed-min", type=float, default=0.90)
    parser.add_argument("--route-clearance-speed-full", type=float, default=1.60)
    parser.add_argument("--route-goal-speed", type=float, default=3.0)
    parser.add_argument("--num-rays", type=int)
    parser.add_argument("--ray-length", type=float)
    parser.add_argument("--mass-scale-min", type=float, default=0.0)
    parser.add_argument("--mass-scale-max", type=float, default=0.0)
    parser.add_argument("--thrust-scale-min", type=float, default=0.0)
    parser.add_argument("--thrust-scale-max", type=float, default=0.0)
    parser.add_argument("--drag-scale-min", type=float, default=0.0)
    parser.add_argument("--drag-scale-max", type=float, default=0.0)
    parser.add_argument("--motor-tau-scale-min", type=float, default=0.0)
    parser.add_argument("--motor-tau-scale-max", type=float, default=0.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--quiet-dashboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", default="logs/agile/afl051/bottleneck_trace")
    parsed = parser.parse_args()

    parsed.deterministic = not parsed.stochastic
    parsed.total_agents = 1
    parsed.num_buffers = 1
    parsed.num_threads = 1
    parsed.horizon = 1

    checkpoint = resolve_named_checkpoint(parsed.checkpoint)
    if checkpoint != "latest" and not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)

    cases = build_cases(parsed)
    if parsed.list_cases:
        for case in cases:
            print(case["label"])
        return

    case_by_label = {case["label"]: case for case in cases}
    missing = [label for label in parsed.cases if label not in case_by_label]
    if missing:
        available = "\n  ".join(case_by_label)
        raise SystemExit(
            "Unknown case label(s): "
            + ", ".join(missing)
            + "\nAvailable cases:\n  "
            + available
        )

    out_dir = Path(parsed.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    activation_rows = []

    print(f"checkpoint {checkpoint}")
    print(f"out_dir {out_dir}")
    for case_label in parsed.cases:
        case = case_by_label[case_label]
        print(f"trace_start {case_label}", flush=True)
        summary, rows, activations = trace_case(checkpoint, parsed, case)
        slug = slugify(case_label)
        trace_path = out_dir / f"{slug}.csv"
        activation_path = out_dir / f"{slug}.activations.csv"
        summary_path = out_dir / f"{slug}.summary.json"

        write_csv(trace_path, rows)
        write_csv(activation_path, activations)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        summaries.append(summary)
        for row in activations:
            activation_rows.append({"case": case_label, **row})
        print(
            f"trace_done {case_label} rows={len(rows)} "
            f"done={summary.get('done_step')} "
            f"success={summary.get('env_success', 0.0):.3f} "
            f"obstacle={summary.get('env_obstacle_hit', 0.0):.3f} "
            f"timeout={summary.get('env_timeout', 0.0):.3f} "
            f"trace={trace_path}",
            flush=True,
        )

    write_csv(out_dir / "activation_summary.csv", activation_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print_summary(summaries)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
