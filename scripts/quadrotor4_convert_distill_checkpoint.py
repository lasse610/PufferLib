#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Convert quadrotor privileged-distillation checkpoints to native PufferLib state_dicts."
    )
    parser.add_argument("input", help="Distillation .pt checkpoint")
    parser.add_argument("output", help="Native .bin state_dict output")
    parser.add_argument(
        "--drop-prev-action",
        action="store_true",
        help="Drop appended previous-action encoder columns for native env observations.",
    )
    parser.add_argument(
        "--action-std",
        type=float,
        default=None,
        help="Overwrite continuous decoder logstd in the converted checkpoint.",
    )
    args = parser.parse_args()

    data = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = OrderedDict(
        (key.replace("module.", ""), value.detach().cpu())
        for key, value in data["state_dict"].items()
    )

    if args.drop_prev_action:
        prev_action_size = int(data.get("prev_action_size", data.get("num_atns", 4)))
        key = "encoder.encoder.weight"
        if key not in state_dict:
            raise KeyError(f"Missing {key!r} in checkpoint")
        weight = state_dict[key]
        if weight.ndim != 2 or weight.shape[1] <= prev_action_size:
            raise ValueError(
                f"Cannot drop {prev_action_size} columns from encoder weight with shape {tuple(weight.shape)}"
            )
        state_dict[key] = weight[:, :-prev_action_size].contiguous()

    if args.action_std is not None:
        if args.action_std <= 0:
            raise ValueError("--action-std must be positive")
        key = "decoder.decoder_logstd"
        if key not in state_dict:
            raise KeyError(f"Missing {key!r} in checkpoint")
        state_dict[key] = torch.full_like(state_dict[key], float(torch.log(torch.tensor(args.action_std))))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
