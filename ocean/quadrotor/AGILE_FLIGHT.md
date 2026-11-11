# Agile Flight Track

This track is separate from the indoor exploration work. It uses the same
quadrotor physics and PufferLib-native training loop, but the benchmark asks a
different question: can the drone fly a known objective fast, safely, and under
heavy randomization?

Exploration work studies mapping, memory, frontier choice, and search. Agile
flight studies control, perception, obstacle avoidance, speed, and sim2real
robustness. Do not mix claims across the two tracks.

## Rules

- Use native PufferLib PPO as the default training harness.
- Keep experiment IDs in the `AFL-###` namespace, not QDR.
- Actor observations must be deployable for the rung being tested.
- Privileged simulator state may be used for critic inputs, labels, diagnostics,
  reset curricula, or teacher policies, but not as deployed actor input.
- First solve with state/rays, then add depth/pixels, then domain randomization.
- Promote checkpoints on success, collision rate, speed, route time,
  clearance, and visual behavior. Do not promote a policy that only wins by
  moving slowly.

## Current Direction

AFL-038 is the current promoted P1 agile-flight candidate. It starts from the
AFL-033 fixed-dynamics policy, fixes native PufferLib train warm-start loading,
resets the loaded action log-std to `0.05`, and conservatively fine-tunes under
light dynamics randomization (`0.95..1.05` for mass, thrust, drag, and motor
time constant). On the hard historical AFL009 eval sampler at 1024 steps it
reaches `93.94%` success, `5.94%` timeout, `0.072%` obstacle hits, `0.048%`
ground hits, `score=42.54`, and `path_efficiency=0.286`. At 1536 steps under
the same randomization it reaches `99.88%` success with zero timeouts.

The important lessons are now twofold. First, the timeout tail was not solved
by global time pressure or pass-through waypoint credit. AFL-033 worked because
speed was rewarded only when the drone had clearance, while controlled arrival
still prevented reckless goal crossing. Second, native training previously
ignored `--load-model-path`; AFL-035/AFL-036 were effectively from-zero runs.
AFL-038 works because train now calls `backend.load_weights(...)` before the
first rollout and uses conservative fine-tune updates.

AFL-039 through AFL-043 then tested the first harder fixed-geometry branch.
The result is a sharper diagnosis, not a new promoted checkpoint. AFL-038
already solves P3 switchback baffles at about `92%` success and solves the
graded open pillar-weave rungs L0/L1/L2-open at about `99.5%+`. AFL-042 and
AFL-043 do not improve old L2/L3. The hard break is the constrained central
passage created by the L2 center marker and the denser L3 weave, where success
stays around `17-26%` with large timeout/contact tails.

Latest native backend status: the temporary zero-action native eval failure was
fixed on June 23, 2026. It came from compiling for the wrong GPU architecture
(`sm_120` on an RTX 3080 host that needs `sm_86`) plus a raw-checkpoint layout
alignment mismatch. The cleaned native build restores AFL-038 L2-open eval to
about `99.6%` success with nonzero velocity and action deltas.

The same day also exposed a second harness issue: the AFL009 training wrapper
was rebuilding the backend without `--float`, which selected bf16 on the remote
host. For direct-motor policies this was catastrophic: loaded float checkpoints
produced saturated actions and immediate crashes. AFL009 now builds float32 by
default (`AFL009_FLOAT_BUILD=1`, override only for explicit bf16 tests). AFL-048
confirmed the collapse is gone, but AFL-049 showed that stable fixed-logstd
continuation from AFL-038 still does not improve the bottleneck curriculum
(`42.17%` best last-5 success, `46.84%` timeout, `10.81%` obstacle). AFL-050
then added a center-local reset probe. On tiny/small/large marker approaches,
AFL-038 averages `36.26%` success overall and `43.51%` after excluding
physically too-tight large-marker rows; AFL-049 is flat/slightly worse. The
useful new signal is that many valid hard rows fail by timeout hesitation
rather than immediate collision.

AFL-051 added the current debug harness:

```bash
PYTHON=../../.venv/bin/python scripts/quadrotor4.sh reference-waypoint-bottleneck-trace afl038 --max-steps 768 --out-dir logs/agile/afl051/afl038_trace_default
```

It writes per-step obs/action/value/reward traces plus Linear/GELU activation
summaries. The first finding is that the current AFL-038 MLP is behaving
numerically normally in the traced hard states: no nonfinite activations, no
obvious hidden-layer saturation, and decoder means around the same magnitude as
normal direct-motor policies. The concrete issue is the observation/control
contract. Direct motor physics clamps actions to `[-1, 1]`, but action-history
obs stores raw unclipped policy outputs, so the actor can observe previous
commands that were never executed. The harness also confirms that some
center-local reset rows are invalid or right at the physical passability edge.

Next raise the bar in one direction at a time. For geometry, do not train more
open-weave variants and do not repeat conservative AFL-038 continuation on the
same bottleneck mix. Build the next method around valid AFL-050 passage states,
but first clean the AFL-051 contract issues: passability-audit hard resets and
record clipped executed action history or add a compatibility flag. Then test
filtered reset curriculum/hard-state replay, deployable passage-affordance
supervision, or a local waypoint/action interface that commits through tight
gaps. For sim2real, increase dynamics randomization from AFL-038 as a separate
branch.

Current checkpoint:

```text
experiments/agile/afl038/reference_waypoint_p1_speed3_clearance_speed_light_dynamics_ft_conservative/quadrotor/1782216829045/0000000006029312.bin
```

Visual command:

```bash
PYTHON=../../.venv/bin/python scripts/quadrotor4.sh reference-waypoint-visual experiments/agile/afl038/reference_waypoint_p1_speed3_clearance_speed_light_dynamics_ft_conservative/quadrotor/1782216829045/0000000006029312.bin --max-steps 1024 --route-goal-speed 3.0 --route-clearance-speed-scale 0.02 --route-clearance-speed-min 0.90 --route-clearance-speed-full 1.60 --mass-scale-min 0.95 --mass-scale-max 1.05 --thrust-scale-min 0.95 --thrust-scale-max 1.05 --drag-scale-min 0.95 --drag-scale-max 1.05 --motor-tau-scale-min 0.95 --motor-tau-scale-max 1.05
```

## Rungs

0. `drone_hover_motor_reset`: clean PufferLib-drone-style control baseline in
   the custom env. Hover-centered direct motor actions, open 3D target ball,
   reference drone reward coefficients, no obstacle/navigation reward stack.
1. `reference_waypoint_open3`: AFL-008 physics/action profile, but now an
   open 3-waypoint chain with direct body-frame target observation. This tests
   maneuvering before obstacles, pixels, or a velocity controller.
2. `agile_v0_state_rays`: known current waypoint, body-frame target
   observation, lidar rays, action history, velocity-action interface.
3. `agile_v0_depth`: same known route task, but with the egocentric depth
   camera enabled. This is a perception comparison, not exploration.
4. `agile_v0_randomized`: add initial velocity, z targets, light dynamics
   randomization, and harder obstacle layouts.
5. `agile_v1_rate`: move from velocity-action to body-rate/thrust while keeping
   the route task simple enough to diagnose.
6. `agile_v2_pixels`: camera-first actor with heavy visual/domain
   randomization.

## Current Start Point

Reset direction as of AFL-007: before pushing pixels or indoor agility, prove
that the custom quadrotor env can learn the same basic control problem as
PufferLib's built-in `ocean/drone` env. The previous AFL-001 through AFL-006
route line used a velocity controller and indoor route reward. That is useful
for navigation experiments, but it is not a clean comparison to the reference
drone env.

AFL-007 adds `task_mode=drone_hover` (`5`) and `action_mode=motor_hover` (`3`):

- Zero action is hover-centered motor speed.
- Target is sampled in a 5m ball around the drone, like the reference hover
  task.
- Reward uses the reference coefficients from `config/drone.ini`:
  distance progress, hover potential, potential shaping, and omega penalty.
- Extra rays/wall/history observations are kept only to satisfy the current
  compiled custom observation size; in the open baseline they are inert.
- This is not yet physics-identical to `ocean/drone`: the custom env still has
  its own mass/inertia constants and `QR_DT=0.05`. If AFL-007 lags the built-in
  drone env, the next build is exact Crazyflie constants plus reference
  substepping.

Default clean reset command:

```bash
PYTHON=.venv/bin/python scripts/quadrotor4.sh drone-hover-train
```

Small CPU smoke:

```bash
AFL007_CPU=1 AFL007_TIMESTEPS=2048 AFL007_TOTAL_AGENTS=64 AFL007_HORIZON=16 AFL007_MINIBATCH_SIZE=1024 AFL007_NUM_THREADS=1 AFL007_NUM_BUFFERS=1 AFL007_SKIP_EVAL=1 PYTHON=../../.venv/bin/python scripts/quadrotor4.sh drone-hover-train
```

Evaluate a run:

```bash
scripts/quadrotor4.sh drone-hover-eval-checkpoints experiments/agile/afl007/drone_hover/quadrotor/<run-id> --episodes 4096 --fresh-process --last 5
```

AFL-007 result: the first GPU run trained at about `1.7M` SPS and produced a
strong but not perfect custom-env control baseline:

```text
experiments/agile/afl007/drone_hover/quadrotor/1781013559875/0000000042663936.bin
```

Fresh-process 4104-episode eval: `success=95.54%`, `ground_hit=4.46%`,
`drone_oob=0%`, `timeout=0%`, `final_dist=0.059m`, `score=57.21`. This proves
the custom env can learn direct motor hover/reach from zero. The next low-level
gap is ground-contact robustness and physics parity with `ocean/drone`, not
basic PPO/control wiring.

AFL-008 is the first physics-parity follow-up. It keeps the AFL-007 clean
hover/reach task, but changes the lower-level simulator profile:

- `dynamics_profile=drone_reference`: Crazyflie-like mass, inertia, arm length,
  thrust/drag constants, max RPM, motor time constant, and velocity clamps.
- `physics_dt=0.002`, `physics_substeps=5`: match `ocean/drone`'s 500 Hz
  integration and 100 Hz policy action cadence.
- `action_mode=motor_ref_hover`: action `0` is hover and action `-1` maps to
  the reference centered minimum RPM instead of motor-off.

Command:

```bash
PYTHON=.venv/bin/python scripts/quadrotor4.sh drone-reference-train
```

AFL-008 result: the first GPU run trained at about `1.5M` SPS and is now the
promoted clean-control baseline:

```text
experiments/agile/afl008/drone_hover/quadrotor/1781019298456/0000000042663936.bin
```

Fresh-process 4096-episode last-5 eval: `success=99.85%`,
`ground_hit=0.15%`, `drone_oob=0%`, `timeout=0%`, `avg_vel=0.454m/s`,
`path_length=4.59m`, `action_saturation=0.027`, `score=53.56`. This confirms
the missing piece in AFL-007 was physics/action parity with the reference drone
env, not PPO or basic direct motor control. Carry AFL-008's dynamics profile,
substeps, and hover-centered reference motor action mapping forward.

Evaluate on the native GPU/Linux backend:

```bash
scripts/quadrotor4.sh drone-reference-eval-checkpoints experiments/agile/afl008/drone_hover/quadrotor/1781019298456 --episodes 4096 --fresh-process --last 5
```

Current macOS caveat: `build.sh quadrotor --cpu` exposes the vec-only CPU
backend (`create_vec`) but not native binary policy loading (`create_pufferl`),
so local `.bin` eval/visual needs either GPU/Linux native eval or a checkpoint
export/loader pass before it is one-command ready on the Mac.

AFL-009 is the first post-hover maneuvering rung. It keeps the AFL-008
reference physics and direct motor action mapping, but switches from one static
hover target to an open 3-waypoint route. There are no obstacles and no xy wall
termination; the point is to prove moving-target chaining before adding gates,
depth, pixels, or clutter.

```bash
PYTHON=.venv/bin/python scripts/quadrotor4.sh reference-waypoint-train
```

AFL-011 through AFL-016 corrected the first route-flight attempt. The failed
piece was not the dynamics or direct motor interface; it was that the route rung
kept conservative PPO/reward settings from earlier obstacle experiments while
trying to mimic native `ocean/drone`. The recovered custom-env recipe is:

```bash
AFL_EXPERIMENT=afl016 \
AFL009_SCENARIO=reference_waypoint_open3_controlled_arrival \
AFL009_ROUTE_REWARD_MODE=4 \
AFL009_ROUTE_TARGET_OBS_MODE=3 \
AFL009_ROUTE_STEP_COST=0.0 \
AFL009_ROUTE_GOAL_COUNT=3 \
AFL009_ROUTE_GOAL_COUNT_MIN=0 \
AFL009_ROUTE_GOAL_SPEED=2.0 \
AFL009_ROUTE_ARRIVAL_RADIUS=1.5 \
AFL009_ROUTE_ARRIVAL_VEL_SCALE=0.08 \
AFL009_LEARNING_RATE=0.00975033 \
AFL009_GAMMA=0.981094 \
AFL009_GAE_LAMBDA=0.883828 \
AFL009_REPLAY_RATIO=2.25498 \
AFL009_CLIP_COEF=0.067834 \
AFL009_VF_COEF=4.42222 \
AFL009_VF_CLIP_COEF=3.42278 \
AFL009_MAX_GRAD_NORM=0.692201 \
AFL009_ENT_COEF=9.62597e-05 \
AFL009_ACTION_STD=1.0 \
AFL009_TIMESTEPS=40000000 \
PYTHON=.venv/bin/python scripts/quadrotor4.sh reference-waypoint-train
```

This gives `100%` fresh-eval success on controlled-arrival open 3-waypoint 3D
flight
(`experiments/agile/afl016/reference_waypoint_open3_controlled_arrival/quadrotor/1781032088671/0000000040042496.bin`).
Final waypoint speed is `1.85m/s` versus `6.15m/s` in AFL-015, with `0%`
contacts. The current limit is geometry: simple gates/obstacles, path
efficiency, and action saturation on top of AFL-016.

AFL-017 through AFL-019 add the first creative obstacle rung. The new
`gate_slalom` obstacle family has four variants: offset gate walls, pillar
slalom, cross gates, and chicanes. Mixed gates are not solved yet: AFL-017 got
only `35.0%` success with `53.7%` obstacle hits. The useful result is AFL-019,
which trains only the easiest pillar-slalom variant with stronger clearance
shaping:

```bash
AFL_EXPERIMENT=afl019 \
AFL009_SCENARIO=reference_waypoint_pillar_slalom_clearance \
AFL009_OBSTACLE_MODE=35 \
AFL009_ROUTE_REWARD_MODE=4 \
AFL009_ROUTE_TARGET_OBS_MODE=3 \
AFL009_ROUTE_STEP_COST=0.0 \
AFL009_ROUTE_GOAL_COUNT=3 \
AFL009_ROUTE_GOAL_COUNT_MIN=0 \
AFL009_ROUTE_GOAL_SPEED=2.0 \
AFL009_ROUTE_ARRIVAL_RADIUS=1.5 \
AFL009_ROUTE_ARRIVAL_VEL_SCALE=0.08 \
AFL009_ROUTE_CLEARANCE_RADIUS=0.80 \
AFL009_ROUTE_CLEARANCE_SCALE=0.60 \
AFL009_LEARNING_RATE=0.00975033 \
AFL009_GAMMA=0.981094 \
AFL009_GAE_LAMBDA=0.883828 \
AFL009_REPLAY_RATIO=2.25498 \
AFL009_CLIP_COEF=0.067834 \
AFL009_VF_COEF=4.42222 \
AFL009_VF_CLIP_COEF=3.42278 \
AFL009_MAX_GRAD_NORM=0.692201 \
AFL009_ENT_COEF=9.62597e-05 \
AFL009_ACTION_STD=1.0 \
AFL009_TIMESTEPS=40000000 \
PYTHON=.venv/bin/python scripts/quadrotor4.sh reference-waypoint-train
```

Current safe obstacle checkpoint:

```text
experiments/agile/afl019/reference_waypoint_pillar_slalom_clearance/quadrotor/1781033504492/0000000041353216.bin
```

At the original 1024-step horizon this checkpoint gets `76.6%` success with
`0%` obstacle hits and `23.1%` timeouts. At 1536 steps it gets `99.7%` success,
`0%` obstacle hits, and no timeouts. This is the current safe obstacle baseline,
but it is cautious rather than agile. The next target is speed/efficiency under
clearance pressure, then a curriculum back to true gate passages.

AFL-020 through AFL-022 tried the direct speed-pressure variants and did not
beat AFL-019:

- `afl020`: 1536-step training plus `route_step_cost=0.001` reached `68.46%`
  success, `30.77%` timeout.
- `afl021`: 1536-step training with no step cost reached `74.57%` success,
  `24.45%` timeout.
- `afl022`: original 1024-step training plus `route_step_cost=0.001` reached
  `66.28%` success, `33.54%` timeout.

The lesson is that global time pressure makes the policy safe but hesitant. For
the next agile obstacle push, keep AFL-019 as the baseline and prefer a
structured curriculum or clearance-conditioned speed objective over step-cost
tuning.

AFL-023 tested a clearance-conditioned speed reward on top of AFL-019. It did
not promote: the best 1024-step checkpoint reached `75.5%` success with small
obstacle/ground contacts, and the 1536-step regression reached only `90.4%`
success versus AFL-019's `99.7%`. The result is useful: small speed-shaping
terms are not the missing agility lever on the P1 pillar slalom. Keep AFL-019
as the safe obstacle baseline and move to a curriculum/geometry/control change.

AFL-024 starts that structural path with `gate_slalom_easy_mix`: 70% P1 pillar
slalom, 20% P0 gate walls, and 10% P3 chicane walls. P2 cross-gates stay out of
the distribution for now because AFL-017 showed the full mixed family is too
large a jump. This tests whether the missing piece is intermediate geometry
coverage rather than another reward scalar.

AFL-024 and AFL-025 answered that first curriculum question. The 70/20/10
P1/P0/P3 mix was too hard and poisoned even the P1 skill. AFL-025 narrowed the
distribution to 90% P1 and 10% P0. That recovered near-AFL-019 P1 performance
(`78.7%` forced P1 success, almost no contacts) and reached `75.8%` mixed
success, but it did not teach wall-gate passages: forced P0 was only `22.2%`
success with `74.1%` timeouts, and P2/P3 remained poor. The next geometry move
should not be another small random mixture. Use staged per-pattern training,
explicit scheduling, or a richer reset/curriculum that proves P0 passage before
mixing it back into P1.

AFL-039/AFL-040 added a separate `agile_geometry` family with more varied fixed
obstacles. This was a useful stress test but not a promoted skill. The intro
mix scored only `28.3%` success and matched a direct AFL-038 control, so
fine-tuning did not teach the new distribution. Forced evals showed why: the
new dense pillar weave is much harder than the old P1 and poisons the mix,
while the P3 switchback course is already mostly solvable (`~92%` success) but
has a `~7%` contact tail. Stronger clearance fine-tuning on P3 only reduced
contacts from `7.6%` to `6.9%`, not enough to count as a new capability.

Next agile geometry work should build a graded curriculum, not another broad
mix: interpolate pillar density/gap width from old P1 to the failed dense weave,
diagnose near-corner clips, then add local passage affordance or recovery
supervision if scalar reward still leaves the contact tail.

AFL-001 is the first scaffold. It reuses `task_mode=indoor_route` and trains
known-objective multi-goal flight in `layout_mixed` or `maze`.

Initial result: AFL-001 `rays` on `layout_mixed`, 3 goals, `max_vel=1.4`
trained at about `1.8M-2.0M` SPS on `beeblebrox`. The best last-3 deterministic
checkpoint reached `73.97%` success with `26.03%` obstacle contacts. That is
not promotable, but it proves this is the right fast iteration harness.

AFL-002 tested whether raw budget fixes the first scaffold by training the same
setup for `100M` steps. It did not: success fell to `68.72%` and contacts rose
to `31.09%`. Longer training alone is not enough for the original reward/speed
setup.

AFL-003 and AFL-004 changed the task economics instead: `max_vel=1.0`, stronger
clearance pressure, lower velocity reward pressure, and fixed or 1..3-goal
routes. They improved the old 2048-episode fixed 3-goal eval, but that protocol
was too optimistic. Use 4096+ episodes for agile comparisons.

AFL-006 trained the safer fixed 3-goal setup for `200M` steps. This answered
the budget question: more training can find a slightly better early checkpoint,
but taking the final model is actively bad. The last checkpoints drift into a
faster, collision-heavy policy. Under 4096-episode headless eval, the current
best checkpoint is:

```text
experiments/agile/afl006/layout_mixed_rays/quadrotor/1781008995185/0000000033816576.bin
```

On seeds 88/90/91 it gets about `77-78%` success with `22%` obstacle contacts.
That is slightly better than AFL-004 under the same protocol, but not close to
solved. The blocker is now the benchmark/objective/control structure, not raw
SPS or insufficient wall-clock training. The next engineering move should be a
cleaner visible-gate/agile-track benchmark or a better local
avoidance/controller rung before depth/pixels.

Default command:

```bash
scripts/quadrotor4.sh agile-v0-train rays
```

Small CPU smoke:

```bash
AFL001_CPU=1 AFL001_TIMESTEPS=2048 AFL001_TOTAL_AGENTS=64 AFL001_HORIZON=16 AFL001_MINIBATCH_SIZE=1024 AFL001_NUM_THREADS=2 AFL001_NUM_BUFFERS=1 AFL001_SKIP_EVAL=1 PYTHON=../../.venv/bin/python scripts/quadrotor4.sh agile-v0-train rays
```

Remote GPU command:

```bash
PYTHON=.venv/bin/python scripts/quadrotor4.sh agile-v0-train rays
```

Evaluate a run:

```bash
scripts/quadrotor4.sh agile-v0-eval-checkpoints experiments/agile/afl001/layout_mixed_rays/quadrotor/<run-id> --episodes 4096 --fresh-process --last 3
```
