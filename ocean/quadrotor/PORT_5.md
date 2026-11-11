# Port To PufferLib 5.0

Ported from `lasse/quadrotor-4` (PufferLib 4.0 @ `9836f0d2`) to `origin/5.0`
@ `ea7f9893` on 2026-11-11. 5.0 is a rewrite: no Python package, a single
native `puffer` binary (`train|eval|match|sweep`), and envs are header-only
files implementing `puf_init/puf_reset/puf_step/puf_render/puf_close/puf_log`.

## What Moved

- `quadrotor.h`, `quadrotor_physics.h`, `quadrotor_control.h`: env logic is
  unchanged. Only the glue changed:
  - `Log` and `QuadrotorEnv` are now `struct Log` / `struct Env` as
    `pufferenv.h` requires; `Env` gains `agents[1]`, `tag`, `boundary_reached`.
  - The old `binding.c` (`my_init`/`my_log`) is folded into the end of
    `quadrotor.h` as `puf_init`/`puf_log`. All `[env]` keys and defaults are
    identical.
  - `qr_bind_agent` copies the trainer's per-agent buffer pointers into the
    legacy `env->observations/actions/rewards/terminals` fields before each
    reset/step/render, so the rest of the env code is untouched.
  - `OBS_SIZE` is still 416 floats; `puf_init` aborts if the config
    produces a different layout.
- `config/quadrotor.ini`: `[env]` copied verbatim; `[train]` reduced to keys 5.0
  understands.
- Research notes (`ROADMAP.md`, `AGILE_FLIGHT.md`, `TESTS.md`, ...) copied as-is.
  Commands in them refer to the 4.0 tooling.

## What Did Not Move

These stay on `lasse/quadrotor-4` and need rewriting for 5.0 if still wanted:

- Python tooling: `quadrotor.py`, `eval_hover.py`, `diagnose_route.py`,
  `diagnose_explore.py`, `validate_explore.py`, `render_hover.py`,
  `stress_eval.py`, `train_nav_clone.py`, `compare_route_checkpoints.py`, and
  `scripts/quadrotor4*.sh|py` (frontier distill, hard-state replay, ...).
- Trainer patches with no 5.0 equivalent yet: `action_std` /
  `train_action_std` (fixed or initial log-std), `reward_clip`,
  `asymmetric_critic` (privileged critic obs), `use_bias`, the hard-state
  `StaticVec` restore helpers. Warm-start loading is built into 5.0
  (`--base.load_model_path` is loaded before training).
- `check_dynamics_parity.c`: upstream `ocean/drone` was rewritten (RK4
  `physics.h`, new `Params`), so the 4.0 parity check no longer compiles.
  The `dynamics_profile=drone_reference` constants match the *old* drone env.
- Checkpoints: 4.0 `.bin` files (120073 floats for AFL-016/038) do not match
  the 5.0 network layout (103048 floats at hidden 128, 1 layer). Retrain.
- Hyperparameters: 5.0 uses Muon (`train.momentum`), not Adam. The AFL recipe
  hyperparameters need a fresh sweep.

## Verified

- `./build.sh quadrotor --cpu` builds on macOS (needed a bash-3.2 fix in
  `build.sh`: `${ENV^^}` -> `tr`).
- `./quadrotor --headless --base.eval_episodes=200` runs 200 untrained
  episodes without errors.
- `quadrotor.h` passes a C++17 syntax check (nvcc compiles env headers as
  C++), and there are no symbol collisions with `src/` beyond the `puf_*` API.

Not yet verified: the CUDA `./build.sh quadrotor` trainer build and an actual
training run (needs the GPU host; build with `NVCC_ARCH=sm_86` for the RTX 3080).
