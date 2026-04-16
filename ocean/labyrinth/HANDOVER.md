# Labyrinth env — handover

PufferLib Ocean env: **tilt-a-ball marble maze**. Physics sim on a board that
the agent tilts in 2 axes; roll a ball from start to goal tile while avoiding
holes and walls.

Branch: `lasse/labyrinth`. Main ref branch: `3.0`.

## Clone + setup on a fresh machine

```bash
# 1. Clone my fork + the labyrinth branch
git clone https://github.com/lasse610/PufferLib.git
cd PufferLib
git checkout lasse/labyrinth

# 2. Python env (uv is what I use; falls back to python -m venv fine).
# Requires Python 3.12 — 3.13+ may work but untested.
uv venv --python 3.12 .venv
source .venv/bin/activate

# 3. Install PufferLib deps from pyproject.toml (plus pybind11 separately —
# build.sh needs it at compile time, not runtime).
uv pip install -e .
uv pip install pybind11

# 4. Raylib archive. build.sh will auto-download on first run, but if the
# download fails, grab the right archive manually:
#   macOS (arm64): raylib-5.5_macos
#   Linux (amd64): raylib-5.5_linux_amd64
# ...and place the extracted dir at ./raylib-5.5_<platform>/

# 5. On Linux GPU box: make sure CUDA toolkit + cuDNN are visible.
# build.sh looks for nvcc in PATH; set CUDA_HOME if needed.
# (nvidia-smi working is a good smoke test.)

# 6. Verify: run the tests + smoke test (see "Build + run" below).
#    ./labyrinth should also open a playable window if your Linux box has
#    a display — otherwise skip that step.

# 7. Build the Python binding for training:
./build.sh labyrinth           # GPU / CUDA
# or --cpu for pytorch CPU fallback
```

If `build.sh` complains about `-fopenmp` on a Mac-like clang, apply the
Mac-specific patches noted at the bottom of this file.

## File layout

- `ocean/labyrinth/labyrinth_physics.h` — pure physics (ball dynamics, tilts,
  walls, holes). No raylib. Included by tests.
- `ocean/labyrinth/labyrinth_maze.h` — maze generator (walls + holes +
  start/goal placement). Includes `labyrinth_physics.h`.
- `ocean/labyrinth/labyrinth.h` — **RL env wrapper**. Pulls raylib in. Defines
  `LabyrinthEnv`, `Log`, `Client`, all constants (`LABYRINTH_*`), `c_reset`,
  `c_step`, `c_render`, `c_close`, `init`, `allocate_LabyrinthEnv`,
  `free_allocated`, `add_log`, `compute_observations`, BFS distance field
  builder, render helpers (`draw_board`, `draw_ball`).
- `ocean/labyrinth/labyrinth.c` — standalone playable game (keyboard). Calls
  physics + render helpers directly, not the RL env.
- `ocean/labyrinth/binding.c` — PufferLib binding (thin). Reads kwargs
  `seed`, `max_steps`, `physics_substeps`, wires `my_init`/`my_log`.
- `ocean/labyrinth/test_labyrinth.c` — unit tests (49 passing as of handover).
  Only includes physics + maze, so no raylib needed.
- `ocean/labyrinth/smoke_test.c` — standalone sanity runner: 5000 random-action
  steps, asserts obs in range, reward/terminal finite, reports terminal
  classification.
- `config/labyrinth.ini` — env + train config.

## Observation

Total size `LABYRINTH_OBS_SIZE = 233`:

- **225 grid cells** — egocentric 15×15 rasterized view (`LABYRINTH_VISION = 7`
  half-window) of the maze around the ball. Cell codes 0=empty, 1=wall,
  2=hole, 3=goal, 4=ball-cell. Cells off-grid mapped as wall.
- **8 scalar features**:
  `[vx, vy, vz, tilt_x, tilt_y, goal_dx, goal_dy, bfs_dist_norm]`.
  Velocity and tilt normalized by their physical maxes so |values| ≤ ~1.
  `goal_dx/dy` are offset to goal in board-fraction coords. `bfs_dist_norm`
  is BFS distance from ball's cell to goal / max-reachable-distance in [0,1].

Grid cell size = 6mm, so 15×15 view ≈ 9cm × 9cm — about 1.5× ball diameter of
context in each direction. Tune via `LABYRINTH_VIEW_CELL_M` and
`LABYRINTH_VISION`.

## Action

Discrete(5): `0=no-op, 1=tilt_x+, 2=tilt_x-, 3=tilt_y+, 4=tilt_y-`. Each tilt
action bumps that axis by `LABYRINTH_TILT_STEP = 0.02 rad`, clamped to
`±MAX_TILT_RAD`. Actions arrive as floats from PufferLib, cast to int in
`c_step`.

## Physics rate vs agent rate

Agent decides every `c_step`, which internally runs
`env->physics_substeps` physics ticks (default 4). At 120Hz physics, that's
a 30Hz agent. Early-exit: the substep loop breaks if `reached_goal` or
`fell_in_hole` is set mid-substep so the ball doesn't drift post-terminal.

`max_steps = 2000` c_steps → 8000 physics ticks → ~67 real-world seconds per
episode max. Matches the feel of a physical labyrinth.

## Reward (current state at handover)

Per step:
```
r = progress_shaping - step_penalty
  + (+1 if reached_goal else -1 if fell_in_hole else 0)
```

- **progress_shaping** — watermark-based. Tracks `min_dist_norm_seen`
  (lowest BFS dist to goal achieved this episode). When current dist beats
  the watermark: reward = (watermark - current) × 0.5. Backtracking earns 0.
  Monotone: never negative; sums to ≤ 0.5 per episode. Prevents oscillation
  farming.
- **step_penalty** = `LABYRINTH_STEP_PENALTY = 0.002` flat per c_step.
  Added because a previous 50M-step training run showed the agent learned
  to **stall to timeout** (avoid fall penalty by not moving). Step penalty
  makes stalling strictly worse than a fall: 2000 × 0.002 = -4.0 max.
- Terminal: `+1` goal, `-1` hole, `0` timeout.

## Known training result (CPU, pytorch backend via `--slowly`)

**50M steps, 20 min on M-series Mac, pre-step-penalty:**
- `reached_goal = 0.000` (never solved)
- `fell_in_hole = 0.378` (down from 1.0 random)
- `episode_length = 1257 / 2000` (stalling to timeout)
- `ep_return = -0.204`
- `entropy = 0.87` (converged)

**Diagnosis:** agent learned to exploit free timeouts. With `gamma=0.995`
the +1 goal reward is discounted to ~0.002 over 1200 steps, invisible
against -1 fall. So agent picks up easy early shaping, then sits still.

**Fix applied just before handover:** step penalty (see above). **Not yet
retrained.** First GPU run should confirm the fix works. If the agent
**still** never reaches the goal after e.g. 200M steps:

1. Try **potential-based dense shaping** instead of watermark:
   `r = gamma*phi(s') - phi(s)` where `phi = -k * bfs_dist_norm`. Provably
   policy-invariant, dense, no oscillation farming (telescopes). Replace
   watermark block in `c_step`.
2. Bump **gamma** to 0.999 in config — extends effective horizon to ~1000
   steps so the goal reward is visible.
3. Shorter mazes for curriculum — `labyrinth_load_random_maze` takes a seed;
   a seed-filtered curriculum of "easy" mazes could jumpstart learning.

## Build + run

**Unit tests** (no raylib):
```
clang -O2 -std=c11 -I. -Iocean/labyrinth \
  ocean/labyrinth/test_labyrinth.c -lm -o test_labyrinth && ./test_labyrinth
```

**Standalone playable game** (arrows to tilt, R reset, N next maze):
```
./build.sh labyrinth --fast
./labyrinth
```

**Smoke test** (5000 random steps, sanity check):
```
clang -O2 -std=c11 -I. -Iocean/labyrinth -I./raylib-5.5_macos/include \
  -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
  -L/opt/homebrew/opt/libomp/lib -DPLATFORM_DESKTOP \
  ocean/labyrinth/smoke_test.c ./raylib-5.5_macos/lib/libraylib.a \
  -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL \
  -lm -lpthread -lomp -o smoke_test && ./smoke_test
```
(On Linux GPU box: drop the macOS frameworks and libomp path tweaks; use
`-fopenmp` directly and `raylib-5.5_linux_amd64`.)

**Python binding** (CPU backend, pytorch):
```
source .venv/bin/activate
./build.sh labyrinth --cpu
```

**Python binding** (GPU backend, CUDA):
```
source .venv/bin/activate
./build.sh labyrinth          # no flag = default CUDA build
```

**Train:**
```
# GPU default:
puffer train labyrinth --train.total-timesteps 100000000

# Pytorch CPU fallback:
puffer train labyrinth --slowly --train.total-timesteps 50000000
```

**Eval a checkpoint:**
```
puffer eval labyrinth --load-model-path checkpoints/labyrinth/<run>/<step>.bin
```

## Mac-specific `build.sh` patches (may not be on other machines)

I patched `build.sh` to use `${OMP_CFLAGS[@]}` instead of hardcoded `-fopenmp`
in the `--cpu` static-lib + bindings compile, because Apple clang needs
`-Xpreprocessor -fopenmp`. On Linux this is a no-op (OMP_CFLAGS=`-fopenmp`).
If the GPU box has a fresh checkout without this patch, either rebase or
re-apply — look for `-fopenmp` around lines 228 and 269. Not strictly needed
for Linux CUDA builds.

## User context

- Lasse Tammela (<lasse@komuhomes.com>)
- Goal: study PufferLib deeply to become a maintainer → wants architectural
  understanding, not just surface-level usage
- Prefers concise explanations, trusts good judgment, pushes back when a
  design feels off (e.g. flagged the stall exploit before I did, asked for
  "not too constrained to the thin line" which led to BFS shaping, asked
  "how do we avoid the agent just going back and forth raking up rewards"
  which led to watermark pattern)

## Pending TODOs

- [ ] Run 100M+ training on GPU to confirm step penalty fixes stall exploit
- [ ] If still not learning: implement potential-based dense shaping
- [ ] If still not learning after that: bump gamma to 0.999
- [ ] Eventually: curriculum (easy seeds first), LSTM policy for longer
      memory, consider ray-based obs vs raster (see `compute_observations`)
