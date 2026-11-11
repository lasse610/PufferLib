# Quadrotor 4.0 Port

This worktree ports the custom quadrotor benchmark onto PufferLib 4.0:

- Branch: `codex/quadrotor-4-port`
- Worktree: `.worktrees/quadrotor-4`
- Env path: `ocean/quadrotor`
- Config: `config/quadrotor.ini`
- Helper: `scripts/quadrotor4.sh`

## What Changed

PufferLib 4.0 uses the static `ocean/<env>` layout and builds one env into
`pufferlib/_C`. The old `pufferlib.ocean.quadrotor` Python `PufferEnv` wrapper
is not the training path here.

The active 4.0 benchmark is intentionally fixed:

- 56 observations: base 26 state features, 24 lidar/ray features, and six
  wall-distance features.
- 4 continuous velocity-command actions.
- Indoor route task with layout-mixed obstacles, blocked one-goal targets, fixed
  `z=1.5m`, and the old promoted `route_path_margin=0.35` waypoint scaffold.
- Default benchmark policy network is a one-layer biased `MLP`. `MinGRU`,
  `MLP`, and `GELU` are selectable through `[torch] network`, but the biased
  feedforward MLP is currently strongest on this waypoint/ray rung.

This keeps the migration scientifically clean: first reproduce the old solved
velocity-control benchmark under the 4.0 trainer, then compare direct motor
control on the same observation/reward surface.

## Local Checks

From this worktree:

```bash
scripts/quadrotor4.sh smoke
```

This builds the CPU backend, verifies the env has `obs_size=56`, and runs a
tiny PyTorch-backend training loop.

Manual equivalent:

```bash
bash build.sh quadrotor --cpu
python -m pufferlib.pufferl train quadrotor --slowly --vec.total-agents 64 --train.horizon 16 --train.minibatch-size 1024 --train.total-timesteps 2048
```

## GPU Commands

On a CUDA machine from this worktree:

```bash
scripts/quadrotor4.sh build-gpu
scripts/quadrotor4.sh train-gpu
scripts/quadrotor4.sh eval-headless latest
scripts/quadrotor4.sh eval-checkpoints latest --episodes 1024
```

For a short GPU run:

```bash
QUAD_TIMESTEPS=10000000 scripts/quadrotor4.sh train-gpu
```

Remote Aalto command used for this port:

```bash
cd ~/PufferLib4
CUDA_HOME=/usr/local/cuda-13.2 NVCC_ARCH=sm_120 CC=gcc CXX=g++ PYTHON=.venv/bin/python scripts/quadrotor4.sh train-gpu
```

Headless checkpoint eval:

```bash
CUDA_HOME=/usr/local/cuda-13.2 NVCC_ARCH=sm_120 CC=gcc CXX=g++ PYTHON=.venv/bin/python scripts/quadrotor4.sh eval-headless latest --episodes 2048
```

Checkpoint sweep eval:

```bash
CUDA_HOME=/usr/local/cuda-13.2 NVCC_ARCH=sm_120 CC=gcc CXX=g++ PYTHON=.venv/bin/python scripts/quadrotor4.sh eval-checkpoints latest --episodes 1024
```

To evaluate an older checkpoint trained with a non-default network, pass the
matching network shape:

```bash
CUDA_HOME=/usr/local/cuda-13.2 NVCC_ARCH=sm_120 CC=gcc CXX=g++ PYTHON=.venv/bin/python scripts/quadrotor4.sh eval-headless latest --episodes 4096 --network MLP --num-layers 1
```

For checkpoints trained with a non-default bias setting, also pass
`--use-bias 0` or `--use-bias 1`.

## GPU Validation

Validated on `whitespace-aalto` routing to the RTX 5070 host:

- Native build succeeds with `CUDA_HOME=/usr/local/cuda-13.2`, `NVCC_ARCH=sm_120`,
  `CC=gcc`, and `CXX=g++`.
- `MinGRU`, `MLP`, and `GELU` are selectable in the native trainer through
  `[torch] network`.
- Native continuous policies now honor `[train] action_std`; this is needed to
  match the old quadrotor recipe's initial `0.20` std instead of upstream 4.0's
  default `1.0`.
- Native training now honors `[train] reward_clip`; this benchmark sets
  `reward_clip=0.0` because 4.0's default reward clipping distorted the old
  quadrotor reward scale.
- Decoder init now matches the old continuous-control recipe more closely:
  actor/action rows use small initial weights while the value row keeps normal
  scale.
- Native encoder, MLP, and decoder layers support optional linear biases through
  `[policy] use_bias`. The quadrotor benchmark now defaults this on because it
  closes most of the 4.0 reproduction gap.
- 5M-step smoke run reached about `2.1M` SPS.
- Raw-motor 50M-step run reached about `2.5M` SPS and produced
  `checkpoints/quadrotor/1779203390924/0000000049807360.bin`.
- Sampled headless eval of that checkpoint over `3107` completed episodes:
  `success=0.6608`, `crashed=0.3392`, `obstacle_hit=0.3289`,
  `wall_hit=0.0097`, `ground_hit=0.0006`, `route_goals=2.17`.
  That checkpoint used the earlier raw-motor 50-observation setup and is not
  compatible with the current 56-observation velocity benchmark.
- Velocity-control reproduction improved as mismatches were removed:
  - MinGRU with clipped rewards: `55.48%` success.
  - MinGRU with unclipped rewards: `61.50%` success.
  - Native MLP before decoder-init fix: `82.24%` success.
  - Native GELU after decoder-init fix: `85.81%` success.
  - Native one-layer MLP after decoder-init fix:
    `checkpoints/quadrotor/1779205541684/0000000049807360.bin`,
    `90.79%` success, `9.21%` obstacle hits, no wall/ground/timeouts.
  - Native one-layer MLP with linear biases:
    `checkpoints/quadrotor/1779206607382/0000000049807360.bin`,
    `97.91%` success, `2.09%` obstacle hits, no wall/ground/timeouts.
  - Three-seed gate for the same biased MLP did not pass. Seeds `88`, `90`,
    and `91` peaked at `93.28%`, `92.88%`, and `94.06%` in full checkpoint
    sweeps; late checkpoints were often worse. The remaining failures are still
    obstacle contacts.
  - PyTorch-style fan-in bias initialization was tested on weak seed `88` and
    rejected. Run `1779207566902` peaked at `90.52%` in the full checkpoint
    sweep, below the zero-bias-init seed-88 baseline. Keep zero bias
    initialization for now.

## Current Limitations

- Visual-eval scripts are not yet fully ported to 4.0. Headless
  single-checkpoint and checkpoint-sweep eval work.
- Observation variants are fixed at build time in 4.0. Changing ray count or
  adding perception features needs a matching `OBS_SIZE` in `ocean/quadrotor/binding.c`.
- Upstream `pufferl eval` is a render loop and requires a display. Use
  `scripts/quadrotor4.sh eval-headless` for remote non-visual checkpoint checks.
