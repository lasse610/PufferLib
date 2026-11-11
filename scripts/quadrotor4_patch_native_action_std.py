#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np


def aligned(offset):
    return (offset + 7) & ~7


def infer_obs_size(raw_size, hidden_size, action_size, num_layers):
    fixed = (
        hidden_size
        + (action_size + 1) * hidden_size
        + (action_size + 1)
        + action_size
        + num_layers * (hidden_size * hidden_size + hidden_size)
    )
    remaining = raw_size - fixed
    if remaining <= 0 or remaining % hidden_size != 0:
        raise ValueError(
            "Could not infer obs size from native checkpoint size. "
            f"raw_size={raw_size} fixed={fixed} hidden_size={hidden_size}"
        )
    return remaining // hidden_size


def logstd_offset(obs_size, hidden_size, action_size):
    offset = 0
    offset = aligned(offset + hidden_size * obs_size)
    offset = aligned(offset + hidden_size)
    offset = aligned(offset + (action_size + 1) * hidden_size)
    offset = aligned(offset + (action_size + 1))
    return offset


def main():
    parser = argparse.ArgumentParser(
        description="Patch decoder log-std in a native raw PufferLib continuous-action checkpoint."
    )
    parser.add_argument("input", help="Input native raw .bin checkpoint")
    parser.add_argument("output", help="Output native raw .bin checkpoint")
    parser.add_argument("--action-std", type=float, required=True)
    parser.add_argument("--obs-size", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--action-size", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    args = parser.parse_args()

    if args.action_std <= 0:
        raise ValueError("--action-std must be positive")

    input_path = Path(args.input)
    output_path = Path(args.output)
    raw = np.fromfile(input_path, dtype=np.float32)
    obs_size = args.obs_size or infer_obs_size(
        raw.size, args.hidden_size, args.action_size, args.num_layers
    )
    offset = logstd_offset(obs_size, args.hidden_size, args.action_size)
    end = offset + args.action_size
    if end > raw.size:
        raise ValueError(
            f"Log-std block [{offset}:{end}] exceeds checkpoint size {raw.size}"
        )

    patched = raw.copy()
    old = patched[offset:end].copy()
    patched[offset:end] = np.float32(math.log(args.action_std))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    patched.tofile(output_path)
    old_str = ",".join(f"{value:.6g}" for value in old)
    print(
        f"wrote {output_path} obs_size={obs_size} offset={offset} "
        f"old_logstd=[{old_str}] new_logstd={math.log(args.action_std):.6g}"
    )


if __name__ == "__main__":
    main()
