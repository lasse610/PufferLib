#!/usr/bin/env python3
import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import quadrotor4_asym_ppo as asym


@dataclass
class Capture:
    step: int
    env_id: int
    snapshot: bytes
    obs: torch.Tensor
    state: tuple


def max_abs(a, b):
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


def clone_state(state):
    if not isinstance(state, tuple):
        return state
    return tuple(value.detach().clone() if torch.is_tensor(value) else value for value in state)


def collect_failure_capture(parsed, vec, obs, rewards, terminals, actor, device):
    actor.eval()
    state = actor.initial_state(vec.total_agents, device)
    ring = deque(maxlen=parsed.rewind + 1)

    for step in range(1, parsed.max_collect_steps + 1):
        snapshots = [vec.snapshot(env_id) for env_id in range(vec.total_agents)]
        ring.append(
            {
                "step": step,
                "snapshots": snapshots,
                "obs": obs.detach().cpu().clone(),
                "state": clone_state(state),
            }
        )

        with torch.no_grad():
            dist, _, state = actor.forward_eval(obs.float(), state)
            actions = dist.mean.clamp(-1.0, 1.0)
        asym.step_vec(vec, actions, device)

        done = terminals > 0.5
        failed = torch.logical_and(done, rewards < parsed.failure_reward_threshold)
        if failed.any() and len(ring) > parsed.rewind:
            env_id = int(failed.nonzero(as_tuple=False)[0].item())
            source = list(ring)[-(parsed.rewind + 1)]
            return Capture(
                step=source["step"],
                env_id=env_id,
                snapshot=source["snapshots"][env_id],
                obs=source["obs"][env_id].clone(),
                state=clone_state(source["state"]),
            )

        if done.any():
            state = asym.reset_done_state(state, done)
        state = asym.detach_state(state)

    raise RuntimeError(
        f"No failure found in {parsed.max_collect_steps} steps; increase "
        "--max-collect-steps or --total-agents"
    )


def rollout_from_capture(parsed, vec, obs, rewards, terminals, actor, capture, device):
    vec.restore(capture.env_id, capture.snapshot)
    restored_obs = obs[capture.env_id].detach().cpu().clone()
    restore_obs_delta = max_abs(restored_obs, capture.obs)

    state = clone_state(capture.state)
    records = []
    actor.eval()
    for local_step in range(parsed.replay_horizon):
        with torch.no_grad():
            dist, _, state = actor.forward_eval(obs.float(), state)
            actions = dist.mean.clamp(-1.0, 1.0)

        env_action = actions[capture.env_id].detach().cpu().clone()
        asym.step_vec(vec, actions, device)

        record = {
            "action": env_action,
            "obs": obs[capture.env_id].detach().cpu().clone(),
            "reward": float(rewards[capture.env_id].detach().cpu().item()),
            "terminal": float(terminals[capture.env_id].detach().cpu().item()),
        }
        records.append(record)
        if record["terminal"] > 0.5:
            break

        done = terminals > 0.5
        if done.any():
            state = asym.reset_done_state(state, done)
        state = asym.detach_state(state)

    return restore_obs_delta, records


def compare_records(a_records, b_records):
    count = min(len(a_records), len(b_records))
    action_delta = 0.0
    obs_delta = 0.0
    reward_delta = 0.0
    terminal_delta = 0.0
    for idx in range(count):
        action_delta = max(
            action_delta,
            max_abs(a_records[idx]["action"], b_records[idx]["action"]),
        )
        obs_delta = max(obs_delta, max_abs(a_records[idx]["obs"], b_records[idx]["obs"]))
        reward_delta = max(
            reward_delta,
            abs(a_records[idx]["reward"] - b_records[idx]["reward"]),
        )
        terminal_delta = max(
            terminal_delta,
            abs(a_records[idx]["terminal"] - b_records[idx]["terminal"]),
        )
    return {
        "len_a": len(a_records),
        "len_b": len(b_records),
        "action_delta": action_delta,
        "obs_delta": obs_delta,
        "reward_delta": reward_delta,
        "terminal_delta": terminal_delta,
    }


def run(parsed):
    from pufferlib import _C

    if parsed.cpu_only and _C.gpu:
        print("Using CPU vec even though CUDA backend is available.", flush=True)
    device = torch.device("cpu")

    args = asym.make_vec_config(parsed, privileged=False)
    vec = _C.create_vec(args, 0)
    obs, rewards, terminals = asym.tensor_views(vec, device)
    vec.reset()

    try:
        actor = asym.create_actor(vec.obs_size, vec.act_sizes, parsed, device)
        asym.load_actor(actor, parsed.checkpoint, device, parsed.load_action_std)

        capture = collect_failure_capture(
            parsed, vec, obs, rewards, terminals, actor, device
        )
        delta_a, records_a = rollout_from_capture(
            parsed, vec, obs, rewards, terminals, actor, capture, device
        )
        delta_b, records_b = rollout_from_capture(
            parsed, vec, obs, rewards, terminals, actor, capture, device
        )
        comparison = compare_records(records_a, records_b)

        print(f"checkpoint {parsed.checkpoint}")
        print(f"captured_step {capture.step}")
        print(f"captured_env_id {capture.env_id}")
        print(f"snapshot_bytes {len(capture.snapshot)}")
        print(f"restore_obs_delta_a {delta_a:.9g}")
        print(f"restore_obs_delta_b {delta_b:.9g}")
        for key, value in comparison.items():
            if isinstance(value, float):
                print(f"{key} {value:.9g}")
            else:
                print(f"{key} {value}")

        passed = (
            delta_a <= parsed.tolerance
            and delta_b <= parsed.tolerance
            and comparison["len_a"] == comparison["len_b"]
            and comparison["action_delta"] <= parsed.tolerance
            and comparison["obs_delta"] <= parsed.tolerance
            and comparison["reward_delta"] <= parsed.tolerance
            and comparison["terminal_delta"] <= parsed.tolerance
        )
        print(f"replay_exact {int(passed)}")
        if not passed:
            raise SystemExit(2)
    finally:
        vec.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe exact resettable hard-state replay for the native quadrotor env."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="experiments/qdr145/asym_mixed15_from_qdr144_s88/qdr135_asym_ppo_000005.bin",
    )
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--total-agents", type=int, default=64)
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--max-z-vel", type=float, default=0.25)
    parser.add_argument("--max-rate", type=float, default=1.0)
    parser.add_argument("--vel-ctrl-ki-z", type=float, default=4.0)
    parser.add_argument("--spawn-radius", type=float, default=2.5)
    parser.add_argument("--spawn-radius-min", type=float, default=2.5)
    parser.add_argument("--spawn-height", type=float, default=1.5)
    parser.add_argument("--room-size", type=float, default=3.0)
    parser.add_argument("--ceiling-height", type=float, default=3.0)
    parser.add_argument("--num-rays", type=int, default=24)
    parser.add_argument("--ray-length", type=float, default=4.0)
    parser.add_argument("--obstacle-mode", type=int, default=7)
    parser.add_argument("--route-goal-radius", type=float, default=1.0)
    parser.add_argument("--route-goal-count", type=int, default=3)
    parser.add_argument("--route-goal-count-min", type=int, default=0)
    parser.add_argument("--route-path-margin", type=float, default=0.35)
    parser.add_argument("--route-target-mode", type=int, default=1)
    parser.add_argument("--route-blocked-target-prob", type=float, default=1.0)
    parser.add_argument("--target-z", type=float, default=1.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--network", choices=("MLP", "MinGRU", "GRU", "LSTM"), default="MinGRU")
    parser.add_argument("--action-std", type=float, default=0.05)
    parser.add_argument("--load-action-std", type=float, default=0.05)
    parser.add_argument("--rewind", type=int, default=8)
    parser.add_argument("--replay-horizon", type=int, default=64)
    parser.add_argument("--max-collect-steps", type=int, default=4096)
    parser.add_argument("--failure-reward-threshold", type=float, default=-1.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--cpu-only", action="store_true", default=True)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
