# Codex Iteration Charter: Indoor Quadrotor Research Track

This document is the working contract for future Codex sessions on the custom
PufferLib quadrotor environment. The goal is not to tune a toy benchmark until
it looks good. The goal is to build, diagnose, and harden a simulator workflow
that can plausibly move toward research-grade indoor flight and later sim2real.

## North Star

Build a PufferLib-native indoor flight stack where:

- training uses privileged simulation information when useful;
- deployment observations remain real-drone feasible;
- evaluation keeps exposing the next failure mode instead of hiding it;
- the simulator progressively absorbs real drone sensing, latency, dynamics,
  and actuator constraints;
- every claimed improvement is tied to a checkpoint, command, deterministic
  eval, and written lesson.

The closest research template is the Swift/agile-flight family: privileged
training, deployable onboard observations, strong simulation throughput, hard
failure-state replay, and a deliberate bridge from clean sim to measured real
hardware.

PufferLib remains the training engine. We do not switch stacks just because a
paper used a different simulator; we borrow the method and keep the iteration
speed.

Current agile-flight build: AFL-038 is still the promoted light-randomization
P1 waypoint/slalom candidate. AFL-029/AFL-030 found that hidden light dynamics
randomization had weakened the clean parity run. AFL-032 showed that the
`2.0m/s` arrival-speed gate was part of the 1024-step timeout tail. AFL-033
added clearance-conditioned target-directed speed reward on top of the `3.0m/s`
controlled-arrival gate. AFL-035/AFL-036 then exposed that native training was
not actually loading `--load-model-path`; `pufferlib/pufferl.py` now calls
`backend.load_weights(...)` before the first rollout. AFL-038 uses a low-std
AFL-033 bootstrap plus conservative fine-tuning under 0.95..1.05 dynamics
randomization and reaches `93.94%` success at 1024 steps with near-zero
contacts. At 1536 steps under the same randomization it reaches `99.88%`
success with zero timeouts.

AFL-039 through AFL-043 turned the vague "harder creative obstacles" direction
into a graded geometry diagnosis. P3 switchback baffles are mostly solved by
AFL-038 (`~92%` success), and open pillar-weave rungs L0/L1/L2-open are also
already solved (`~99.5%+`). The first real break is not pillar count; it is the
constrained central passage introduced by the L2 center marker and amplified in
L3. Fine-tuning on old L2 (AFL-042) and L2-open (AFL-043) did not improve that
break: old L2 remains around `23-26%` success with large timeout/contact tails.

June 23, 2026 backend note: the latest native zero-action eval failure was a
tooling/port issue, not a policy regression. The current remote GPU target is
an RTX 3080 and must be built with `NVCC_ARCH=sm_86`; an `sm_120` build makes
custom kernels such as GELU/sampling fail with `no kernel image is available`.
The native allocator also now uses 32-byte alignment so raw `.bin` checkpoints
match the Torch-compatible layout. A cleaned `sm_86` rebuild restored native
AFL-038 L2-open eval to `99.62%` success over a short 512-episode sanity run.

Second June 23, 2026 backend note: direct-motor agile training must also pin
native precision to float32. AFL-046/AFL-047 looked like PPO/action-std collapse
because the AFL009 wrapper rebuilt the backend without `--float`, producing
bf16 inference (`precision_bytes=2`) from float checkpoints. That made actions
saturate (`action_delta~=1.17`) and crash immediately. The wrapper now builds
`build.sh quadrotor --float` by default for GPU AFL009 runs. AFL-048 confirmed
that float32 fixed-std training no longer collapses. AFL-049 then ran the fair
20M-step continuation and failed to improve the full bottleneck task: best
last-5 eval was `42.17%` success, `46.84%` timeout, and `10.81%` obstacle hits.
So the harness bug is fixed, but constrained-passage learning is still the
blocker. AFL-050 added a controlled center-local bottleneck reset probe so this
is no longer just a vague full-episode failure: AFL-038 averages `36.26%`
success over the tiny/small/large center-local grid and `43.51%` after removing
physically too-tight large-marker rows; AFL-049 is flat/slightly worse at
`36.17%` all-case and `41.34%` valid-ish. The valid hard rows are often
timeout-heavy, not contact-heavy, which means the policy frequently hesitates
instead of committing through the passage.

AFL-051 then added a per-step bottleneck trace harness. The key result is
negative but useful: the AFL-038 MLP is not obviously saturating or producing
nonfinite activations in the failed hard states. Encoder activations stay small,
GELU layers mostly stay below abs `3`, and the decoder mean is large but not
exploding. The trace instead points at a control-contract and local-passage
problem. Physics clamps motor actions to `[-1, 1]`, but action-history
observations record the raw unclipped policy action, so the actor can observe
commands that the plant never executed. Some hard-state rows are also invalid
or at the physical passability edge. Before another PPO continuation, fix the
diagnostic/task contract: passability-audit reset states, record clipped
executed action history or gate it behind a compatibility flag, and then train
only on valid constrained-passage states with a stronger passage affordance or
reset replay signal.

Current agile checkpoint:

```text
experiments/agile/afl038/reference_waypoint_p1_speed3_clearance_speed_light_dynamics_ft_conservative/quadrotor/1782216829045/0000000006029312.bin
```

Next agile step: raise only one stressor at a time, but avoid more open-weave
training and avoid more conservative AFL-038 continuation on `obstacle_mode=57`.
The next fixed-geometry branch should first clean the measurement/control
contract exposed by AFL-051, then train directly on valid AFL-050 center-local
states. Filter out physically invalid starts, record executed clipped action
history, then test one of: reset curriculum/hard-state replay over valid passage
states, deployable passage-affordance/recovery labels, or a local subgoal/action
interface that explicitly commits through tight gaps. Stronger dynamics
randomization remains a separate branch from AFL-038; do not mix dynamics
randomization and passage geometry until one branch is proven.

## Benchmark Split

Do not collapse fast indoor flight and exploration into one pass/fail task yet.
They test different skills and need different rewards, horizons, and metrics.
The combined target is fast indoor exploration, but that should be a later
integration benchmark after both halves work separately.

Fast indoor flight means known-objective navigation: waypoint, goal, gate, or
route information may be present depending on the rung. Timeouts are failures,
speed matters, and the main metrics are success, obstacle contact, route time,
clearance, and control aggressiveness. This track is where we harden dynamics,
body-rate/thrust control, high-speed turns, and sim2real stress.

Exploration means unknown-objective information gathering. The actor should not
receive a planner waypoint or final-target breadcrumb. Reward should come from
new free-space coverage, frontier/information gain, target detection when
visible, and crash avoidance. Timeouts are an episode budget, not automatically
evidence of failure. The main metrics are coverage over time, crash rate,
revisit/stagnation, target discovery rate, and time-to-detect.

Do not use a fast route policy as evidence that exploration works, and do not
use a slow cautious explorer as evidence that fast flight works. Until the two
tracks are individually strong, mixed rewards will hide the real blocker.

## Current Swift-Lite Exploration Status

As of QDR-352, the active research track is still the p3/procedural-house
exploration gap exposed by QDR-340 through QDR-350. The strict `0.99` p3 task
is not currently solved by the deployable onboard stack. Privileged
nearest-unexplored planning is still the ceiling, but direct distillation to a
target vector and the first graph-candidate-index abstraction both failed.

QDR-351/QDR-352 added and validated a deployable egocentric depth-camera
observation, but this is only sensor infrastructure. The work now sits on
latest upstream PufferLib `origin/4.0` at `9836f0d2` in the
`codex/quadrotor-4-latest` worktree. The camera defaults to `16x12`, can be
raised to `20x15`, and writes into the existing exploration patch tail when
`camera_depth_obs` is enabled, so old checkpoints remain compatible when the
flag is off. Local probes showed bounded, nonconstant frames on doorway, maze,
and p3 house layouts, and center depth matched the old forward lidar ray. QDR-354
then tested the `20x15` camera in native PufferLib PPO on `maze`: it did not
improve from-zero learning over the non-camera baseline and slowed rollout from
about `450K` SPS to about `190K` SPS. Treat depth as a valid sensor, not a
solution. Camera work needs a sharper diagnostic or architecture that forces
geometry use.

The important QDR-350 lesson is architectural, not a PPO result. A graph
candidate must carry two different meanings: a local path-step direction for
safe action execution and a frontier/topological identity for supervision and
memory. Reusing one direction vector for both creates bad labels. Splitting the
candidate feature into path-step direction plus final-frontier bearing fixed
the representation mismatch, but the top-4 candidate-index oracle still reached
only `16.58%` p3 coverage at horizon `1200` under the strict `0.99` gate, far
below the onboard graph path oracle's `71.92%` coverage on the same local
diagnostic. Do not launch long candidate-index training until the planner
interface exposes richer path/topology state or a better label target.

Current next high-potential build:

- keep PufferLib and the native C env;
- use native PufferLib PPO as the default learning harness for iteration speed;
- stop treating "choose one of four frontier candidates" as a sufficient
  learned planner;
- expose a deployable topological/path trace that includes current path step,
  frontier region identity, branch/doorway commitment, progress/staleness, and
  candidate value;
- use privileged information for labels/critic/reset diagnostics only;
- gate every build against a no-training oracle/probe before GPU training.

## Current Doorway-Stall Status

As of QDR-286/QDR-287, the remaining QDR-285 hard-p1 visual stalls are not
primarily a perception failure. The actor target and onboard planner target
agree closely in the failing traces, rays see the doorway, and the safety layer
often allows full-speed motion. The bad state is local: while near a doorway
frame, the planner/controller chases very shallow current/adjacent frontier
targets and progress stalls until stale-frontier suppression eventually moves
on.

Current best local profile:

```bash
scripts/quadrotor4.sh explore-v0-target-visual experiments/qdr285/qdr285_fast_safe_rollout_qdr282_s88.pt --mode procedural_indoor_hard_p1 --seed 88 --max-steps 2048 --room-size 6.0 --explore-goal-fraction 0.95 --spawn-radius-min 0.0 --speed-fraction 0.50 --explore-onboard-path-lookahead 2 --safety-slide --safety-stateful-recovery --recovery-stall-steps 96 --recovery-duration-steps 64 --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80 --restart-on-stall-steps 3000
```

This profile reduces the visible stall trace and preserves zero contacts, but
it is not a final solution. Drained hard-p1 eval improves only marginally when
stateful recovery is enabled (`96.31%` to `96.40%` on seed `88` in the local
QDR-287 check). Treat it as a useful deployable recovery scaffold and a teacher
candidate, not as proof that the exploration problem is solved.

QDR-288 built the first learned recovery sidecar around that scaffold. The
mechanism works: a small deployable MLP can be trained from hard doorway states
and invoked through `--recovery-policy`, and eval/visual paths can load it. The
lesson is negative but useful. Raw learned recovery actions need the same
deployable slide/cone safety wrapper, and once wrapped they are safe but not
meaningfully better than the hand-coded recovery teacher. The best side-aware
clone reached `96.58%` hard-p1 seed-`88` success with `0%` contacts and still
logged doorway stalls in no-render visual eval. Do not spend GPU time on more
capacity or longer imitation of this teacher. The next recovery build needs
stronger labels or an objective that directly rewards escaping doorway stalls:
short-horizon hard-state search, privileged local recovery labels, or
closed-loop hard-state replay with safety constraints.

QDR-289 tested the first stronger-label rung and ruled out the simplest form.
Short-horizon constant-action search labels can be generated from restored
doorway states, but an unfiltered 256-sample sidecar failed the 512-episode
regression (`95.84%` success with a small ground-crash tail), and a stricter
coverage-gain/clearance filter accepted almost no labels. The important lesson:
doorway recovery is not a one-step label problem. It often needs several setup
actions before coverage changes. The next serious method should train on
resettable hard-state replay or trajectory-level recovery labels, not single
MPC first actions.

QDR-290 moved trajectory search into the existing native vectorized env path.
The first Python implementation was still too slow, so the labeler now restores
one hard state into many `StaticVec` slots and rolls candidate actions together
with native C stepping. That is the right implementation shape for diagnostics:
the GPU-box smoke collected `8` accepted trajectory labels in `18s`, and the
medium run collected `64` labels in a few minutes. The policy result was still
bad. The medium sidecar fell to `87.60%` hard-p1 success; constraining learned
recovery to XY-only removed crashes but left a `12.40%` timeout rate. Keep the
vectorized search machinery and the XY-only recovery safety fix, but do not
continue sparse trajectory-label imitation as the main method. The next step
should be a resettable hard-state RL/replay benchmark or stronger trajectory
optimizer that directly trains closed-loop recovery, not a small BC sidecar
over biased accepted labels.

QDR-291 built that resettable recovery replay benchmark and changed the
interpretation again. The tooling works: `explore-v0-recovery-replay` captures
exact recovery-trigger native snapshots, restores them in batches on the GPU,
and compares base, stateful, learned, and PG-trained recovery controllers on
identical states. The result is negative for the recovery-action sidecar. On
early-trigger states (`recovery_stall_steps=96`), base and stateful are tied
over `128` states (`mean_gain=0.220587` vs `0.220744`). On stricter late-stall
states (`recovery_stall_steps=240`, mean `since_progress=302`), base beats
stateful on replay gain (`0.106726` vs `0.088603`) with zero contacts. A random
PG sidecar is catastrophic in full eval (`17.95%` success), and even a
base-action warm start followed by PG loses the replay metric
(`mean_gain=0.047990`) while merely matching baseline eval within noise
(`94.90%` vs base `95.05%`, zero contacts). Do not spend more time on this
specific action-sidecar/PG loop unless the replay set is first filtered to
states where the base controller actually fails. The next higher-leverage build
is local target/viewpoint/planner quality: the slow doorway behavior appears to
come from the emitted frontier objective, not from needing a slightly different
velocity wrapper.

QDR-292 through QDR-298 tested that hypothesis without finding a promotion.
The late-stall viewpoint trigger was a clean negative: default QDR-285 and
`--explore-onboard-stall-viewpoint-ticks 120` were bit-for-bit identical on
hard-p1 seed `88` (`96.7773%` success, `0%` contacts), and the no-render visual
traces were identical too. Directly swapping the emitted target from waypoint
to deeper objective (`--explore-onboard-emit-objective-target`) was invalid
under the old checkpoint and collapsed saved eval to `6.25%` success. The
safer auxiliary-objective observation (`--explore-onboard-objective-obs`) was
not destructive but still did not beat baseline: old QDR-285 with the aux
slots reached `95.43%`, and a 150-update QDR-294 fine-tune reached `95.85%`
full saved eval; the best intermediate quick check was only `96.21%` on 512
episodes. QDR-295 then rechecked the deployable onboard planner oracle under
the same fast-safe profile and found only `95.92%` success. That is the key
read: the current student is not clearly below the planner/controller ceiling.
QDR-296 then added an opt-in local barrier steering controller over deployable
rays. It safely improved the onboard oracle to `96.61%`, and barrier + slide +
stateful recovery reached `96.80%` on the full 1024 hard-p1 check with `0%`
contacts. That was a real signal but not a breakthrough. The same wrapper hurt
QDR-285 policy eval (`95.35%` versus the rebuilt `96.78%` baseline), which meant
the policy was not distribution-matched to the new controller. QDR-298's
close-waypoint objective-drive gate also failed to promote: objective + barrier
was only `96.56%` on a 512-ish check, and objective without barrier fell to
`93.92%`.

QDR-299/QDR-300 are the current best incremental improvement. The native planner
now has opt-in region-goal hysteresis:
`--explore-onboard-region-commit --explore-onboard-region-commit-stall-ticks 90`.
Hard-locking a first region goal was wrong (`67.69%` local success), but using
the remembered region goal only after a stall and only when its score is close
to the current best gives a small oracle gain. The best stack, native region
commit + barrier + slide + stateful recovery, reached `96.91%` oracle success
on full hard-p1 seed `88`, `0%` contacts. Fine-tuning QDR-285 under the native
region-commit/barrier rollout produced a small promotion at update 100:
`experiments/qdr300/checkpoints/qdr300_region_commit_barrier_ft_qdr285_s88_u000100.pt`.
Full hard-p1 eval with the full stack: seed `88` `96.9844%`, seed `90`
`96.1427%`, seed `91` `95.6268%`, all `0%` contacts. Later updates regressed,
so do not continue this fine-tune recipe blindly. Visual smoke still logs
recoverable near-wall stalls, so this is a stronger candidate, not a solved
planner.

QDR-301 found a more important mechanical cause of those stalls: the hard-p1
exploration grid was too fine. At room size `6.0`, the old `0.25m` cell size
creates a 48x48 grid and often emits one-cell doorway objectives. Going finer
to `0.20m` makes the problem much worse: GPU oracle seed `88` fell to `52.83%`
success with `47.17%` timeout. Coarsening to `0.30m` preserves the task while
reducing frontier churn. The onboard oracle reached `100%` success / `0%`
contacts on seed `88`, and QDR-300 u100 policy eval improved from `95.15%` at
`0.25m` to `99.26%` at `0.30m`. Held-out `0.30m` policy checks were seed `90`
`98.61%` and seed `91` `99.06%`, all `0%` contacts. No-render visual smoke
over three episodes logged `2` near-wall stalls at `0.25m` and `0` at `0.30m`.
Treat `--cell-size 0.30` as the current hard-p1 runtime setting. The next
scientifically honest step is not another eval-only wrapper; train or fine-tune
with `0.30m` in the rollout loop and then run retention checks.

QDR-302 did that rollout-matched fine-tune and failed to promote. Before the
run, target-distill checkpoints were patched to save/load `cell_size` so future
checkpoints cannot silently fall back to the old grid. Training from QDR-300
u100 for 150 updates was safe, but it slightly underperformed the no-finetune
QDR-301 profile. The best saved checkpoint, u100, reached hard-p1 seed `88`
`99.17%`, seed `90` `98.81%`, seed `91` `98.62%`, all `0%` contacts. QDR-300
u100 with explicit `--cell-size 0.30` remains better overall:
`99.26%`/`98.61%`/`99.06%`. Do not promote QDR-302 or spend more time lightly
distilling the same target policy against the coarser grid. The useful lesson
is that the planner/grid runtime change matters more than another supervised
fine-tune of the already competent follower.

Do not repeat these rejected levers without new evidence: longer viewpoint
commitment, velocity-controller gain increases, clearance-centering, target
distance speed scaling, action hysteresis, sampled safety, current-cell
recentring, shallow-depth stall retargeting, sparse trajectory-label BC, or
unfiltered recovery-action sidecar PG, late-stall viewpoint timers, eval-time
objective-target swapping, aux-objective target fine-tuning, objective-drive
without barrier, eval-time barrier wrapping of QDR-285, hard region-goal locks,
long QDR-300 fine-tunes past u100, or finer hard-p1 grids below `0.25m` under
the current planner semantics, or QDR-302-style light target-distill
continuation from QDR-300 u100. The next serious build should improve the
deployable local target/viewpoint planner with deeper topological progress
state or a changed training objective, because the current actor already
follows the coarser onboard target well.

QDR-155 starts the corrected exploration track. The env now credits cells seen
by onboard rays rather than only cells physically occupied by the drone. This
makes open rooms intentionally easy for a 360-degree range sensor and shifts
the useful benchmark toward occluded layouts where the policy must move to
reveal hidden free space.

## Current Procedural Baseline

As of QDR-251, the current procedural-indoor exploration baseline is:

`experiments/qdr251/qdr251_qdr224_spawn_p1p3_anchor_s88.pt`

This result came from a real diagnosis, not a PPO sweep. QDR-244 split the
procedural generator into four pattern families and showed the aggregate
plateau was mostly pattern `p1`, with `p3` secondary. QDR-249 then showed the
privileged oracle's p1 timeouts all started below the safety danger clearance.
QDR-250 fixed that environment-validity issue by requiring exploration spawns
to satisfy `QR_EXPLORE_CLEARANCE_RADIUS`, matching the reward/safety scale
instead of accepting drone-radius-only starts.

With that benchmark reset, QDR-224 rose to `89.26%` procedural success but
still only `52.85%` on p1. QDR-251 trained from QDR-224 on the valid p1/p3
families with p0/p2/fixed anchors. It reaches `100%` procedural success on
seed `88`, `99.88%` on seed `90`, and `99.93%` on seed `91`, with no timeouts
and very low obstacle contact. A local no-render visual smoke passed `3/3`
episodes.

Do not treat this as complete indoor exploration. QDR-252 isolated the
fixed-layout regression to `corridor`: QDR-251 is `100%` on center block,
doorway, maze, and random blocks, but corridor crashed about `99%` under the
strict spawn gate. QDR-253 fixed the invalid-start part by relaxing spawn
clearance down to physical drone radius when the preferred exploration margin
has no valid sample. Procedural performance remained `100%`, and corridor
changed from contact failure to timeout failure (`11.79%` success, `88.21%`
timeout, `0%` obstacle). The next engineering step is therefore narrow-corridor
progress under deployable cone safety, not more p1/procedural training.

QDR-254 found the missing local-control piece: the cone safety gate can freeze
the drone near walls because a nearby side ray inside the commanded cone scales
the whole action to zero. The new `--safety-slide` mode uses the same deployable
rays but projects commanded velocity away from close forward rays instead of
stopping. As a scaffold, QDR-251 plus `--safety-slide` solves the fixed-layout
quick gauntlet almost perfectly: fixed4 `100%`, corridor seeds `88/90/91`
`100%`, layout_mixed `100%`, with one random-block contact in the 18-run check.
This should be treated as a real-drone-feasible low-level safety layer, not a
privileged planner.

QDR-255 through QDR-257 tried to distill that slide reflex into a raw learned
action policy. This did not succeed yet. Plain action imitation from slide
labels (QDR-255) made corridor move but still hit obstacles (`87.56%` best
corridor success, `12.44%` contacts). Correction/clearance weighting (QDR-256)
and teacher-only refinement (QDR-257) did not improve that. The current honest
read: use `--safety-slide` as the deployable safety scaffold for now; teaching
it into the actor likely needs a dedicated residual/barrier module or explicit
no-contact constrained training, not plain action MSE.

QDR-258 creates the next benchmark instead of continuing that safety
distillation loop. The new `procedural_indoor_hard` modes scale the procedural
layouts into larger multi-wall rooms with longer routes and more occlusion. The
useful stress setting is `--room-size 6.0 --explore-goal-fraction 0.98
--max-steps 4096`; a smaller `4.5m`/`95%` smoke was still too easy. Under this
hard setting, QDR-251 plus `--safety-slide` falls to about `54%` aggregate hard
success across seeds `88/90/91`. Pattern p1 is the cleanest new blocker:
roughly `45%` success, `55%` timeout, and `0%` obstacle contacts. That points
back to memory/frontier target selection and exploration pace, not low-level
safety.

QDR-259 and QDR-260 tested two direct responses to the hard-p1 patrol loop.
QDR-259 trained the full QDR-251 target policy on hard p1/p2 labels with
hard-state weighting and old-mode anchors. It failed badly: deterministic eval
collapsed hard-p1 to `2-3%` success and damaged other modes, even while the
training HUD reported successful rollouts. QDR-260 expanded the deployable far
explored/frontier map from `11x11` to `21x21` and trained only the new input
columns from QDR-251. That protected the old weights but still failed hard-p1:
u200 was `1.66%` success and u400 `4.68%`; p2 improved but with a contact tail.
Also, forcing the existing frontier-vector fallback on QDR-251 did not change
hard-p1 (`43.18%`, matching baseline). Current interpretation: the p1 failure
is not fixed by more labels, wider passive map input, or a looser vector
fallback. The next build should make exploration state/action more structured:
explicit topological frontier memory, a deployable local planner target, or
resettable hard-state training that directly rewards leaving known-space loops.

QDR-261 added a guardrail for that next build. The repo now has a deployable
`onboard` frontier-target mode and a verifier command:

```bash
scripts/quadrotor4.sh explore-v0-onboard-frontier-verify --cpu
```

This verifier runs the native environment with the target oracle/controller
before any policy training. It is intentionally allowed to fail. Current result:
blocked occupancy from ray-hit endpoints improves hard-p1 from roughly `3%` to
about `17-20%`, with `0%` obstacle contact. The verifier now defaults to
reachable-component coverage; that fixed a p2 false failure where even
privileged target modes stalled around `40%` full-layout coverage. With the
correct denominator, privileged `nearest` solves hard-p1/p2 and onboard p2 is
partly viable (`~56%` success), but onboard p1 still fails (`~17%` success,
mostly timeouts). Tried variants did not fix p1: unknown-cluster gain scoring,
nearest-frontier selection, path lookahead, longer timeouts, speed/safety
tweaks, and a clear-ray fallback for blocked targets.

Do not train against onboard frontier targets until this verifier passes;
otherwise we are just distilling a broken teacher. The next useful engineering
response is stronger onboard spatial/topological memory and planner-state
diagnostics, not more PPO tuning.

QDR-262 implements the first version of that response. It adds a minimal
stale-frontier manager to the onboard target: persist the selected unknown
frontier cell, track coverage progress, and mark that cell stale after `96`
env steps without enough new-cell progress. Stale cells are then excluded from
future onboard target selection. This is the same shape as classic frontier
exploration's "visited/inaccessible frontier" bookkeeping, but applied only to
deployable onboard grid state.

It also adds a cheap text probe:

```bash
scripts/quadrotor4.sh explore-v0-planner-probe --mode procedural_indoor_hard_p1 --cpu
```

The probe prints coverage, target existence/distance, target ray clearance,
min-ray clearance, action speed, and a no-progress age, and can write CSV with
`--csv <path>`. QDR-262 improves p2 strongly: hard-p2 is `100%` on seeds
`88/90/91` with no contacts. Hard-p1 improves but is still not solved:
`29.00%`, `33.98%`, and `24.51%` across seeds `88/90/91`. The p1 probe shows
coverage rising quickly, then stalling around `0.626` while the target remains
near a local cell. Next step: region-level frontier clusters with stale status
and planner-state diagnostics for actual selected grid/frontier IDs, not
policy training yet.

QDR-263 adds the missing feedback loop for that next step. The native env now
logs planner-state metrics at episode end:
`frontier_count`, `stale_frontier_count`, `stale_marks`,
`onboard_target_exists`, `onboard_target_age`, and `blocked_cells`. Visual eval
also overlays the exploration grid: blue explored cells, yellow frontier cells,
purple stale cells, red blocked cells, green current cell, and a red marker for
the current onboard target.

Use this validator before training against onboard planner labels:

```bash
scripts/quadrotor4.sh explore-v0-planner-state-verify
```

Local result: hard-p1 seed `88` still times out often (`27.69%` success,
`70.02%` coverage, `0%` obstacle), while hard-p2 is solved (`100%` success,
`98.17%` coverage, `0%` obstacle). The stall probes produce stale marks, and
the sampled run had `absent_under_goal_rate=0` and `blocked_target_rate=0`.
That means the current sampled failure is not "no target" or "target directly
through wall"; it is still weak regional memory/progress accounting. The next
engineering build should move from per-cell stale suppression to frontier
clusters or a small topological manager with region-level stale/visited status.

QDR-264 attacked that exact near-wall local frontier flip. Three bounded
variants were tested before any GPU training: cluster-scored frontier targeting,
cluster/local stale marking while keeping per-cell target choice, and a shorter
`64`-tick stale timeout. None should be promoted. Cluster-scored targeting was
destructive: hard-p1 collapsed to `0%` and hard-p2 regressed badly. Cluster/local
stale at `96` ticks preserved hard-p2 but did not improve hard-p1 across seeds
(`26.47%`, `21.88%`, `36.84%`). Shorter `64`-tick stale was worse
(`9.38%`, `9.38%`, `23.53%`). Active planner behavior was reverted to the
QDR-263/QDR-262 per-cell stale default, while keeping only the new diagnostics:
frontier cluster count, target-frontier marker, and target cluster size. The
next build should use persistent frontier-region objects or a graph/search
layer with explicit target commitment and stale/visited/inaccessible state, not
one-shot wider stale marking.

QDR-265 implements that persistent-region version and changes the picture. The
planner now stores a committed frontier-region anchor, reselects targets only
inside that connected frontier region while it remains viable, does not reset
the stale timer when tiny targets churn inside the same region, and marks the
committed frontier region stale only after the no-progress window. This is still
a deployable onboard-grid planner, not privileged map leakage.

Planner-level results are strong. The `128`-episode p1/p2 verifier over seeds
`88/90/91` reached hard-p1 `100%`, `99.22%`, `99.22%`; hard-p2 stayed `100%`
on all three seeds; obstacle contacts stayed `0%`. A broader p0/p1/p2/p3 hard
family gate with `64` episodes per seed passed at `100%` success and `0%`
contacts for every family/seed. The one-command no-render visual smoke passed
`3/3` after increasing the visual restart budget to `3000` steps, because the
new planner can pause near walls for long periods before recovering. The next
phase can now train/distill against onboard planner labels; keep
`explore-v0-planner-state-verify` and `explore-v0-onboard-visual` as regression
gates before spending GPU budget.

QDR-266 through QDR-269 tightened the visual target behavior after QDR-265. The
useful active correction is QDR-268: penalize selecting the current cell as the
frontier target when other cells in the committed region are available. This
reduces repeated tiny adjacent-target behavior without hurting the hard-family
verifier. QDR-266's shorter `64`-tick stale window regressed hard-p1 and was
reverted. QDR-267's free clear-ray fallback caused a worse timeout/restart and
was reverted. QDR-269's slide-safety cone alignment did not materially help and
was reverted. Active behavior is QDR-265 plus the conservative QDR-268
current-cell penalty. The remaining visual issue is local viewpoint/control
quality near walls, not topological region choice alone.

QDR-270 adds the first local viewpoint hygiene layer on top of QDR-268. When a
committed frontier region degenerates to the current cell plus an adjacent
unknown sliver, the planner can now choose a nearby reachable explored viewpoint
that still has a known-clear sightline to the same unknown target. Candidate
first steps are scored by the current lidar cone, so this stays deployable and
does not reintroduce the failed QDR-267 free clear-ray escape. The hard-family
planner gate still passes at `100%` success and `0%` contacts, and the visual
smoke no longer shows the `0.021m` adjacent-target symptom. It does not fully
remove near-wall pauses; some hard-p1 states still have long progress stalls,
which means the next lever is a stronger local motion/viewpoint planner or
policy transfer against the now-valid planner rather than another one-cell
frontier heuristic.

QDR-271 tested the bigger FUEL-style direction directly: score connected
frontier regions by reachable viewpoint quality, expected visible unknown gain,
current lidar-cone clearance, and path cost. It did not promote. The first
version preserved visual success but made the seed-88 smoke slower and stickier.
Adding a weighted explored-cell path graph could remove stall logs in one visual
smoke, but it regressed hard-p1 completion at `128` episodes to roughly
`94-98%` with timeouts, versus QDR-270's near-saturated gate. The weaker
path-cost variant still missed the strong p1 gate. Active code was cleaned back
to QDR-270 and reconfirmed. The useful conclusion is that the frontier planner
is no longer the main missing layer. The next high-potential build should be a
local motion/control layer: short-horizon trajectory or barrier-style velocity
selection around the QDR-270 planner target, then later distill that behavior
into a deployable policy.

QDR-272 moved to that local-control layer with an optional
`--safety-sample-actions` controller: sample candidate body-frame velocity
directions around the planner target, score progress and lidar-cone safety, and
use it only as a low-safety-scale fallback. It is not promoted. The best visual
smoke improved seed-88 new-cell rate to `0.9102`, but hard-p1 progress
statistics worsened at `64` episodes and no variant beat QDR-270 cleanly. Keep
the flag as a diagnostic, not a default. The next serious controller should
optimize a short trajectory or stateful recovery objective with explicit
progress accounting, because single-step ray-safe direction choice can look
better visually while still slowing systematic p1 exploration.

QDR-273 and QDR-274 tried two more local-controller variants and did not
promote. QDR-273 added per-agent progress memory and a short recovery mode, but
the conservative trigger barely changed hard p1 and the wider trigger caused a
visual stall restart. QDR-274 tested a progress speed floor for states where the
direct target ray is clear but the wider cone throttle is slow; it improved some
visual traces but worsened hard-p1 max-stall statistics. Both are left as
optional diagnostics, off by default. The active baseline remains QDR-270. The
lesson is sharper now: local control is the right layer, but single-snapshot
ray heuristics are not enough. The next useful build should collect the hard
stall states and train or optimize a recovery behavior against explicit
coverage-progress labels.

QDR-275 starts that recovery build by adding an exact hard-state probe for the
active onboard planner. It captures real near-wall/no-progress snapshots, then
replays the same snapshots under the baseline target controller and a
short-horizon local trajectory search. The first CPU probes are mixed but useful:
on four real 90-step stall states, a short `128`-step replay improved mean
coverage gain from `0.0498` to `0.0868` without crashes, but a longer `384`-step
replay made the trajectory search more conservative than baseline (`0.1261`
coverage gain vs `0.1606`) while preserving more clearance. This means the
local-control path still has hope, but not as a naive per-step MPC heuristic.
The next iteration should use the captured hard-state loop to test
progress/commitment-aware trajectory scoring or train a recovery residual with
an explicit progress-vs-clearance gate.

Remote QDR-275/QDR-276 sharpened that conclusion. After rebuilding the remote
quadrotor backend with CUDA 13.2, QDR-275 captured 16 real hard-p1 wall-stall
states and found that feasible per-step MPC barely moved the replay metric:
baseline coverage gain was `0.06994`, MPC was `0.07161`, with no successes or
crashes. The full rollout-12 MPC setting was too slow to be an iteration tool.
QDR-276 then tried the obvious "be bolder" fix: commit the chosen action for
8 steps and weight coverage progress harder. That failed badly: `4/16` crashes
and mean coverage gain dropped to `-0.04574`. The honest read is that local
control remains the right failure layer, but neither single-step safe search nor
aggressive macro-actions provide the missing geometry understanding. The next
build should inspect the hard-state geometry more directly, or move to a learned
recovery/viewpoint module with richer local map features, not continue scalar
MPC score tuning.

QDR-277 is the research reassessment. Public systems such as FUEL, TARE, Active
Neural SLAM, FAR Planner, FASTER, Agile Autonomy, and Swift make the same point
from different angles: exploration is not just local obstacle avoidance, and
agile flight is not full exploration. Strong exploration stacks maintain
persistent map/frontier/graph state, choose viewpoints or long-horizon goals,
and use a local planner/controller underneath. Strong agile-flight stacks use
privileged training and deployable onboard observations for short-horizon
trajectory control. The next mainline build should therefore stop optimizing
scalar local MPC weights and instead add a PufferLib-native geometry layer:
persistent deployable occupancy/frontier state, frontier/viewpoint candidate
objects or a small topological graph, diagnostics that prove those candidates
are reachable/useful, and then a learned high-level selector or privileged-label
student. Keep QDR-270/QDR-275 as hard-state diagnostics, not as the architecture.

QDR-278 adds the first candidate-level diagnostic for that geometry layer. The
new probe logs frontier/viewpoint candidate count, viable count, selection/miss
rates, candidate clearance/depth, selected waypoint depth, and draws both the
emitted waypoint and best candidate in visual eval. The result is important:
hard-p1 is not failing because no candidate signal exists. With the default
`2048`-step horizon it still has safe timeouts, but it averages about `37-40`
candidates and `14-15` viable candidates per selector call, with low miss rate
and no obstacle contacts. Giving the same controller `4096` steps solves the
hard-p1 seeds cleanly. This moves the current blocker from "build candidate
existence" to "convert viable viewpoints into faster committed coverage
progress." Next work should improve target commitment/sequencing or train a
high-level selector against these candidate diagnostics.

QDR-279 tested the most direct commitment hypothesis. The code now logs target
switches, objective switches, unproductive objective switches, committed
viewpoint usage, drops, and committed age. It also has opt-in committed
viewpoints via `--explore-onboard-commit-steps`. This made the failure sharper,
but did not solve it. Baseline hard-p1 has moderate unproductive objective
switching, yet long commitment worsens success; the best short setting
(`24` steps, `0.25m` clearance) is only a mixed tiny gain and regresses one
seed. Hard-p2 stays clean. Keep the diagnostics, but do not promote commitment
as the architecture. The next likely issue is frontier/region choice quality:
which candidate or region to pursue, not simply whether to stick with the last
viewpoint longer.

QDR-280 builds the first stronger privileged region teacher. The first version
failed honestly: maximizing far region size was too slow and timed out hard-p1
around `92%` coverage. The fixed version scores connected unknown regions by
near-term unknown-density gain, clearance, and path-cost-normalized progress.
It is privileged because it uses full map validity, so it is a teacher/critic/
label generator only. It solves all hard procedural patterns p0-p3 on seeds
88/90/91 at the strict `2048`-step horizon with no contacts, and it improves
new-cell rate over the old privileged `unexplored` teacher on every case/seed
pair. This is now the best teacher candidate for the next learning pass.

QDR-156 is the first from-zero training proof for that corrected track. It
solves the random-block coverage gate (`99.75%` sampled native success,
`85.23%` coverage, `0.25%` crashes), but fails the first fixed-layout stress
(`55.72%` success, `44.28%` crashes on `layout_mixed`). Treat this as a useful
baseline, not a promoted explorer.

QDR-157 adds the first proper explore-v0 gauntlet and trains directly on the
fixed-layout mixture. It improves the full 1024-step gauntlet mean from
`62.48%` to `68.38%` and cuts mean crash rate from `26.09%` to `7.70%`, but it
does this partly by timing out more. With a 2048-step eval budget the mixed
layout rises to `89.61%`, so distribution mismatch and horizon both matter.

QDR-158 shows the harder truth: standalone maze training from zero still fails
(`2.05%` best success at 1024 steps, `2.58%` at 2048), with coverage stuck near
`51%`. The current exploration blocker is not generic PPO budget. It is the
maze/local-frontier/perception problem: the actor can move and stay mostly
safe, but it does not discover the hidden half of the map.

QDR-159 proves the maze itself is not broken. A static frontier oracle sees
`100%` of valid cells from reachable poses, the maze has one connected drone
component, and a privileged frontier teacher running in the actual native env
solves `98.44%` of maze episodes with the same velocity-control scaffold. This
separates the blocker cleanly: the missing ingredient is learned
frontier-seeking/map memory, not dynamics, impossible geometry, or a bad
coverage threshold.

QDR-160 tested the obvious transfer branch from that diagnosis: imitate the
privileged frontier teacher with a deployable MinGRU actor. This did not solve
the maze. The fair `action_repeat=1` teacher is strong but imperfect
(`91.36%` success); the best short mixed-rollout student reached `49.76%`
success but crashed in `50.24%` of episodes. Pure BC fit the teacher very well
but was worse at native speed (`36.17%` success / `63.81%` crashes), and slower
eval only moved it to about `62%` with too many crashes. Longer DAgger degraded
to `10.84%` success. Lesson: privileged frontier action labels contain signal,
but one-step action imitation is too brittle for maze exploration. Distill
higher-level frontier/waypoint targets or use the teacher for auxiliary losses
and resettable recovery, not raw action MSE.

QDR-161 then tested an onboard-map observation branch. The env now has optional
`explore_memory_obs`, which reuses the exploration task's otherwise-unused
target slots without changing the native 60-float ABI. It exposes egocentric
frontier sector scores, best frontier direction, coverage, and frontier count
derived from the drone's own explored grid. From-zero PPO on standalone maze
still failed: best deterministic 1024-step checkpoint was only `1.31%` success
with `50.34%` coverage and `91.93%` timeouts; even `4096` steps reached only
`3.33%` success and `52.03%` coverage. Lesson: a hand-built frontier-memory
summary makes the policy safer but does not create frontier-seeking under the
current coverage reward. The next branch needs a stronger training signal:
frontier-progress reward, planner-supervised frontier/waypoint target, map
auxiliary loss, or asymmetric critic over map/frontier state.

QDR-162 through QDR-166 narrowed that blocker. Reward-only frontier progress
failed (`0.215%` best success). Direct nearest-frontier actor observation also
failed (`0.386%`), and QDR-164 explained why: a controller following that
target crashed almost every episode because the nearest frontier can sit behind
or against a wall. A reachable explored-grid path target was a better signal:
the hand controller reached `46.60%` success with only `8.72%` obstacle
contacts. But PPO still did not learn it from scalar coverage/progress reward:
QDR-165 reached only `0.292%`, and the path-consistent progress fix in QDR-166
only reached `0.364%`. The current blocker is therefore no longer just
"perception is poor." The environment needs a higher-level exploration module,
supervised/auxiliary target learning, resettable RL from hard exploration
states, or an asymmetric critic/objective that directly trains frontier
selection. Do not spend more GPU budget on scalar frontier-progress variants.

QDR-167 proves the high-level module direction is real. A MinGRU target
predictor trained from privileged reachable-frontier path labels, with those
label slots zeroed from the deployed actor input, reached about `83.2-83.6%`
maze success on the stricter single-buffer eval profile and
`85.65%`/`85.00%`/`85.20%` on the matched `2048 agents / 2 buffers` profile at
`speed_fraction=0.35`. Coverage was about `78%`, obstacle contacts about
`14-17%`, and timeouts were gone. This is a large break from the half-coverage
PPO plateau and a useful Swift-lite-style decomposition: privileged labels for
training, deployable observations at runtime, fixed low-level velocity control
underneath.

QDR-168 tested whether matching the training rollout speed to the best eval
speed closes the safety gap. It is a modest real improvement, not a leap:
matched-profile seeds `88/90/91` reached `87.35%`/`85.91%`/`86.76%` success
versus QDR-167's `85.65%`/`85.00%`/`85.20%`; stricter single-buffer eval is
lower at `83.36%`/`83.43%`/`83.64%`. Do not spend the next session on more
speed/controller knob sweeps unless they are tied to a specific diagnosed
failure.

QDR-169 gives that diagnosis. In QDR-168 crash states, coverage is only about
`0.26`, min ray clearance is about `0.09m`, and the privileged target often has
`exists=0.0` while the student still predicts `exists~0.5` and moves at
`0.18-0.19` normalized speed. The predicted target/action ray is also much
less safe than the oracle target ray on roughly one third of crash states. The
next blocker is therefore target-existence/safety on induced hard states, not
basic flight or lack of scalar reward. QDR-170 should add explicit
target-existence/stop supervision, hard-state negative target replay, or
resettable recovery for the target module before more end-to-end PPO.

QDR-170 tried the simplest version of that: heavier negative-existence loss,
false-positive margin loss, and absent-target direction-to-zero loss. It failed
badly. Matched-profile seeds `88/90/91` dropped to about `59.7%` success, with
`24.5-25.3%` obstacle contacts and `15%` timeouts. Diagnostics showed the model
still produced false-positive existence in crash states, while also stopping in
timeout states and distorting target direction in successes. Do not repeat this
weighted-loss recipe. The next smaller diagnostic is QDR-171: calibrate the
existing QDR-168 target-existence threshold. If calibration fails, use
hard-state negative replay rather than more global loss weighting.

QDR-171 failed too. Raising QDR-168's action gate threshold from `0.20` to
`0.30` was slightly worse, `0.50` was clearly worse, and `0.70+` mostly turned
movement into timeouts. The target-existence output contains useful signal, but
not enough clean separation to fix crashes with a scalar threshold. QDR-172 is
a final cheap diagnostic before hard-state replay: test a deployable ray-based
safety gate to see whether local braking can close the contact gap or whether
the high-level module itself needs targeted hard-negative training.

QDR-172 was a strong positive diagnostic. A simple deployable min-ray slowdown
on QDR-168, `--safety-min-ray-stop 0.20 --safety-slow-ray 0.60`, improved
matched-profile seeds `88/90/91` to `93.51%`/`92.64%`/`92.86%` with no
timeouts, and stricter single-buffer eval to `92.39%`/`91.84%`/`92.08%`. The
remaining failures are no longer normal flight-into-wall events: seed-88
diagnostic shows crash coverage `0.004`, min ray `0.001m`, predicted target
existence `0.007`, and action speed `0`. This exposes a likely env issue:
exploration spawns are accepted with only physical drone-radius clearance
(`0.12m`) while exploration valid cells/frontiers require `0.30m` clearance.
QDR-173 should align spawn validity with the exploration cell margin before
more policy work.

QDR-173 falsified that simple spawn-margin fix. Temporarily requiring
`QR_EXPLORE_CELL_MARGIN` at spawn dropped the QDR-172 policy to
`83.35%`/`81.99%`/`82.01%` success and increased obstacle contacts to about
`16-18%`, so the change was reverted. The stricter spawn distribution is not a
free hygiene fix for this trained policy. Treat QDR-172 as the current strongest
exploration scaffold, and diagnose remaining failures with trajectory/start
classification rather than changing the default spawn sampler blindly.

QDR-174 performed that trajectory/start classification. The remaining QDR-172
crashes are almost all one-step episodes: immediate fraction `99.27%`/`99.69%`/
`100%` on seeds `88/90/91`, start min ray around `0.0-0.002m`, start coverage
near zero, and commanded speed effectively zero. This points at the
`spawn_radius_min=0.0` eval setup rather than policy behavior. In the maze, very
small spawn bounds around the origin can be fully covered by the center wall's
drone-radius-expanded invalid region; after 64 failed attempts the fallback
also lands invalid. QDR-175 should test a non-invasive minimum spawn radius.

QDR-175 confirms that diagnosis. Adding `--spawn-radius-min 0.25` to QDR-168
with QDR-172's deployable ray safety gate gives matched-profile
`100%`/`100%`/`100%` on seeds `88/90/91` and stricter single-buffer
`99.984%`/`100%`/`100%`. The control without the safety gate is still only
`92.88%`/`92.56%`/`92.37%`, so the solved standalone-maze diagnostic needs two
pieces: avoid invalid tiny-bound starts, and use local ray slowdown for
mid-episode safety. This is the current best standalone maze command:
`scripts/quadrotor4.sh explore-v0-target-distill-eval experiments/qdr168/qdr168_frontier_target_slowrollout_s88.pt --seed 88 --episodes 4096 --total-agents 2048 --num-buffers 2 --num-threads 16 --speed-fraction 0.35 --safety-min-ray-stop 0.20 --safety-slow-ray 0.60 --spawn-radius-min 0.25`.

Do not mark the full research goal solved from QDR-175 alone. It uses
privileged target labels during training and an external deployable safety
scaffold at inference. The next harder limit is mixed/held-out exploration
layouts, and then folding this safety behavior into the learned policy or a
real low-level safety layer.

QDR-176 evaluated that next limit. The stack transfers well to doorway, maze,
random blocks, and the mixed layout, but not uniformly. `layout_mixed` is
`96-97%` across seeds, maze is solved, and random blocks are essentially solved.
The failures are concentrated in two simpler fixed layouts: corridor reaches
only about `69-70%` and all remaining failures are timeouts, while center block
is about `87%` with obstacle contacts. The corridor failure is likely caused by
nearest-ray safety braking on the side walls; QDR-177 should compare
action-direction ray braking and weaker min-ray braking before retraining.

QDR-177 confirmed that diagnosis but did not produce a promotable global fix.
The weak min-ray gate `--safety-min-ray-stop 0.15 --safety-slow-ray 0.45`
solves corridor on seeds `88/90/91`, all `100%`, but it regresses the layouts
that need obstacle discipline: maze is about `95.6-96.2%`, layout mixed about
`93.7-94.3%`, and center block about `81.5-81.8%`. The current best global
recipe remains QDR-175/QDR-176's stricter min-ray safety, while QDR-177 exposes
the next blocker: safety needs to distinguish forward progress through narrow
clearance from actual collision risk. A single nearest-ray scalar gate is too
blunt.

QDR-178 implements that distinction as a deployable commanded-motion cone gate.
The best deterministic recipe so far is QDR-168 plus `--spawn-radius-min 0.25
--speed-fraction 0.35 --safety-cone-ray-stop 0.20 --safety-cone-deg 60
--safety-slow-ray 0.60`. It solves corridor on seeds `88/90/91` (`100%` each)
while keeping maze about `99.2%`, layout mixed about `96.2-96.9%`, doorway
`100%`, and random blocks about `99.8%`. Center block remains the weak point at
`86.5-86.7%` with obstacle contacts. This is real progress over QDR-176 because
the corridor timeout mode is gone without the QDR-177 weak-min regression, but
it is not the full research goal: the stack still relies on a hand-built
deployable safety layer and center-block contacts remain unsolved.

Current deterministic modular eval command:
`scripts/quadrotor4.sh explore-v0-target-distill-eval experiments/qdr168/qdr168_frontier_target_slowrollout_s88.pt --mode <mode> --seed <seed> --episodes 4096 --total-agents 2048 --num-buffers 2 --num-threads 16 --speed-fraction 0.35 --spawn-radius-min 0.25 --safety-cone-ray-stop 0.20 --safety-cone-deg 60 --safety-slow-ray 0.60`.

QDR-179 shows that the remaining center-block weakness is another reset
validity issue, not a learned navigation failure. With QDR-178 cone safety, the
privileged oracle is also only about `85.7%` on center block and its failures
are `100%` immediate zero-speed crashes with start min ray `0`. Raising
`--spawn-radius-min` to `0.50` removes the invalid tiny-bound starts and gives a
much stronger deterministic fixed-layout gauntlet: center block
`99.80%`/`99.63%`/`99.54%`, corridor `100%`, maze about `99.4-99.6%`, layout
mixed about `99.8%`, doorway `100%`, and random blocks about `99.7-99.8%`.

The actual environment bug is in the room-spawn fallback: if the sampled spawn
bound is too small and fully covered by the obstacle-expanded invalid region,
all 64 attempts fail, then the fallback `(-0.5*bound, -0.5*bound)` can also be
inside the obstacle. QDR-180 should fix that fallback to always find a valid
room spawn, then rerun the same gauntlet with the normal spawn distribution
before claiming the benchmark is clean.

QDR-180 fixed the room-spawn fallback. After 64 failed local-bound attempts, the
env now searches the full room for a valid spawn before using the final
deterministic fallback. Local CPU build and remote GPU rebuild both passed.
With the normal `--spawn-radius-min 0.0` distribution and QDR-178 cone safety,
the fixed-layout gauntlet is now near-solved: center block
`99.72%`/`99.64%`/`99.62%`, corridor `100%`, maze
`98.67%`/`98.53%`/`98.80%`, layout mixed `99.62%`/`99.58%`/`99.77%`, doorway
`100%`/`99.98%`/`100%`, and random blocks `99.88%`/`99.78%`/`99.81%`.

Current cleaned fixed-layout eval command:
`scripts/quadrotor4.sh explore-v0-target-distill-eval experiments/qdr168/qdr168_frontier_target_slowrollout_s88.pt --mode <mode> --seed <seed> --episodes 4096 --total-agents 2048 --num-buffers 2 --num-threads 16 --speed-fraction 0.35 --spawn-radius-min 0.0 --safety-cone-ray-stop 0.20 --safety-cone-deg 60 --safety-slow-ray 0.60`.

Post-fix diagnostic: the old immediate zero-clearance crashes are gone.
Privileged oracle reached `100%` on center block; the student still has a small
real hard-state tail (`0.33%` crash on seed `88`) with non-immediate episodes,
low starting clearance, full-speed commands, and large pred-oracle angle. That
is now a genuine target/safety learning issue, but it is no longer the dominant
fixed-layout limit.

QDR-181 tried to absorb the cone safety layer into the learned target model by
adding a fifth supervised output for a deployable local-ray speed scale. The
implementation works mechanically, but the method failed: layout-mixed training
finished with safe but timeout-heavy sampled rollouts, and deterministic eval on
the same training distribution reached only `1.10%` success with `98.90%`
timeouts, `43.55%` coverage, and almost no action variation. This is a clear
failure mode: naive continuous speed-scale distillation learns excessive
caution. Do not repeat this recipe as the path to removing the low-level safety
gate.

Current interpretation: the QDR-178 cone gate should be treated as a
real-drone-feasible low-level safety layer for now. The next research step
should push beyond the cleaned fixed-layout suite or use a better safety
learning formulation, such as hard-state recovery or classifier-style
intervention labels, rather than a continuous speed head.

QDR-182 pushed beyond the cleaned fixed-layout suite by raising the exploration
success threshold from `85%` to `95%` coverage without changing the policy. The
result is a useful harder limit, not a regression to the old broken setup.
Maze remains about `98.0%` and random blocks about `99.5%`, both with no
timeouts. The difficult modes are now `layout_mixed` and `center_block`, both
around `81-83%` success, mostly because `17-19%` of episodes hit the 1024-step
budget. Contacts are low: roughly `1-2%` for mixed and below `0.5%` for center
block. Coverage in the timeout-heavy modes still reaches about `93-94%`.

Current interpretation after QDR-182: the modular explorer is no longer mainly
failing by invalid resets or immediate obstacle contacts. At the higher coverage
threshold it is running out of budget on occluded fixed layouts. The next
diagnostic should change only the eval horizon, for example `--max-steps 2048`
with the same QDR-182 settings, to separate a pure time-budget limit from true
frontier-target exhaustion or late-episode stagnation.

QDR-183 falsified the simple horizon explanation. Doubling eval horizon to
`2048` did not materially improve the `95%` gate: `layout_mixed` stayed around
`81-82%`, and `center_block` around `83%`. Timeouts remained about `16-17%`.
The policy is reaching a late coverage plateau, not merely needing more control
steps.

QDR-184 and QDR-185 split that plateau into separate causes. `center_block` is
a real student gap: the privileged target oracle solves it at `100%`, while the
student times out about `17%` and its timeout states still contain oracle
targets. `doorway` is also a student gap at the stricter `95%` gate: oracle is
`100%`, but the student is only about `60%`. `corridor` is different. Static
geometry says the corridor is fully coverable, but the current privileged
reachable-frontier target oracle only reaches about `63%` and often stalls,
while the student reaches about `68%`. This means the privileged target is not
reliable in thin corridor-like spaces.

Current interpretation after QDR-185: do not train harder against the aggregate
`layout_mixed@95%` metric yet. It mixes at least one real student weakness
(`doorway`/`center_block`) with a teacher/frontier-target failure (`corridor`).
The next aligned engineering work is to fix or replace the high-level frontier
target for thin spaces, then rerun oracle-first before distilling or training a
student on the stricter high-coverage benchmark.

QDR-186 fixed that teacher issue without replacing the whole target. The root
cause was narrower than expected: corridor starts can be physically valid while
outside the stricter exploration-valid cell margin, causing
`qr_explore_cell_index` to fail and the frontier teacher to output no target.
The fix is scoped to active corridor layouts: steer first toward the nearest
exploration-valid cell, then resume the existing reachable-frontier path target.
After this, the privileged oracle solves `corridor`, `doorway`, and aggregate
`layout_mixed` at the `95%` gate across seeds `88/90/91`, with no contacts or
timeouts. This makes `layout_mixed@95%` a valid student benchmark again.

QDR-187 then trained a new deployable target student on `layout_mixed@95%` with
the fixed teacher and QDR-178 cone safety. This is a real improvement:
deterministic held-out seeds reach about `98.6-98.8%` on `layout_mixed`, `100%`
on `corridor`, about `99.7-99.9%` on `doorway`, about `99.6%` on
`center_block`, and about `99.5%` on `random_blocks`, all with no timeouts.
The tradeoff is standalone `maze`: QDR-187 drops to about `62%` there at the
`95%` gate, with one third timeouts. Current best is therefore not one universal
checkpoint. QDR-168 remains stronger for standalone maze; QDR-187 is much
stronger for the high-coverage fixed-layout mixture.

QDR-188 explains the apparent mismatch between QDR-187's `layout_mixed` score
and its standalone `maze` score. The mixed sampler does include maze, but the
aggregate eval is asynchronous and episode-count weighted: fast layouts finish
many episodes while slow maze workers contribute fewer episodes before the eval
episode budget is exhausted. Treat QDR-187's fair balanced fixed-layout score as
roughly `90%`, not `98.7%`. The attempted native per-layout log fields were
reverted after exposing fixed-log memory-safety issues.

QDR-189 turns that into the new promotion gate. The new
`explore-v0-target-balanced-eval` command runs each layout separately and reports
equal-mode means. On QDR-187 at the `95%` coverage gate, seeds `88/90/91`, the
aggregate `layout_mixed` score is still `98.70%`, but the equal-weight fixed4
mean is `90.39%`: center block `99.63%`, doorway `99.79%`, corridor `100%`, and
maze only `62.14%`. The fixed5/all-nonaggregate mean is `92.22%` after adding
random blocks. This is now the authoritative progress signal for the current
exploration track.

QDR-190 tested the smallest direct repair: initialize from QDR-187 and continue
target distillation on standalone `maze@95%`. It improved the balanced fixed4
mean from `90.39%` to `96.67%` and lifted maze to `90.90%`, so the maze/frontier
gap is trainable. But it also raised balanced obstacle contacts to `3.33%` and
dropped doorway from about `99.8%` to `96.1%`. This is progress, not promotion:
single-distribution continuation trades timeout failures for contact failures
and loses some non-maze safety.

QDR-191 implemented true interleaved fixed-layout target-distill updates, but
equal interleaving is not enough. It preserved non-maze safety (`doorway` about
`99.75%`, `corridor` `100%`, `center_block` about `99.31%`) while leaving maze
near the old timeout-heavy regime (`64.24%`). Fixed4 is `90.82%`, barely above
QDR-187 and far below QDR-190. The important lesson is not that interleaving is
bad; it is that the curriculum weight on maze must be much higher, or we need a
hard-state replay/oversampling mechanism for maze late-frontier states.

QDR-192 tried that first explanation with 3x maze-heavy interleaving. It helps
relative to QDR-191, but not enough: fixed4 rises to `91.99%` and maze to
`68.82%`, still far behind QDR-190's `90.90%` maze. Non-maze layouts remain
strong. This weakens the "just balance the curriculum" hypothesis. The next
scientific target is the hard-state distribution inside maze: recover QDR-190's
maze behavior, then reduce its `~9%` maze contacts and doorway regression with
resettable hard-state replay or a targeted target/safety loss.

QDR-193 confirms that a gentle low-LR safety refresh from QDR-190 is too blunt:
it restores doorway/non-maze safety but erases the maze recovery, dropping maze
to `58.47%` and fixed4 to `89.41%`. QDR-194 then diagnoses the real contact
mode. In maze, QDR-190's student is better than the privileged oracle because it
moves through absent-target states where the oracle stops; student crash states
have `oracle_exists=0`, `pred_exists=1`, high target-angle disagreement, and
worse predicted target clearance. Doorway crashes show the same signature, while
the oracle solves doorway perfectly.

QDR-195 tested the obvious deployable fallback: when the oracle target is absent,
label the clearest local ray. That failed hard: maze collapses to `1.73%`,
corridor drops to `83.16%`, and fixed4 falls to `70.93%`. The absence state is
not solved by generic local clearance. The next build should capture/replay the
successful-but-risky QDR-190 absent-target states and supervise a structured
target there, or improve the privileged target so those states no longer produce
no target.

QDR-196 took the better-teacher path. The frontier-path teacher now recovers
from invalid-margin positions in all layouts by steering to the nearest valid
exploration cell, and when no explored-frontier path exists it uses privileged
BFS over valid cells to the nearest unexplored cell and returns the first
waypoint. This converts the absent-target state into a structured privileged
label. The balanced oracle is now strong: fixed4 `99.70%`, maze `99.41%`,
doorway `99.39%`, corridor `100%`, with zero timeouts.

QDR-197 then continued QDR-190 with the QDR-196 teacher on `maze maze doorway`.
This is the current best high-coverage exploration checkpoint:
`experiments/qdr197/qdr197_qdr190_recovery_teacher_s88.pt`. Balanced eval at
the `95%` gate over seeds `88/90/91`: fixed4 `99.26%`, fixed5/all-nonaggregate
`99.35%`, aggregate `layout_mixed` `99.44%`, no timeouts. Per-mode success:
center block `99.61%`, doorway `99.99%`, corridor `100%`, maze `97.45%`, random
blocks `99.72%`. The remaining weakness is maze contacts around `2.5%`, not
exploration failure.

QDR-198 added the target-distill visual loop and diagnosed those remaining maze
contacts. The visual path now runs the same target-distill `.pt` checkpoint and
controller stack through the native vector backend:
`scripts/quadrotor4.sh explore-v0-target-visual
experiments/qdr197/qdr197_qdr190_recovery_teacher_s88.pt --mode maze`.
The contact diagnostic shows the old absent-target hallucination is gone:
student crash states have `pred_exists=1` and `oracle_exists=1`. The excess
student crashes are low-clearance, high-speed corner cases with higher
pred-oracle angle/action mismatch. The oracle has only a small immediate
tight-start crash tail.

QDR-199 then tested the direct deployable lever suggested by QDR-198. Stronger
commanded-cone braking, `--safety-cone-ray-stop 0.25 --safety-slow-ray 0.70`,
reduces balanced fixed-layout contacts to about `0.11%` without adding
timeouts. Balanced eval over seeds `88/90/91`: fixed4 `99.89%`,
fixed5/all-nonaggregate `99.89%`, aggregate `layout_mixed` `99.95%`; per-mode
maze `99.85%`, doorway `99.996%`, corridor `100%`, center block `99.74%`, random
blocks `99.87%`.

Current interpretation after QDR-199: the QDR-190/QDR-197 branch is now a strong
fixed-layout high-coverage explorer under a deployable cone safety scaffold. The
remaining question is no longer whether the drone can explore these fixed
layouts. The next hard limit is whether we can remove/learn the external safety
scaffold and generalize beyond these handcrafted layouts without reintroducing
slow, over-cautious behavior.

QDR-200 added that next limit as `procedural_indoor`: randomized wall/gap,
staggered-corridor, and simple multi-room obstacle layouts. The benchmark is
valid enough because the privileged oracle solves it: `99.90%` success,
`0.10%` obstacle, `0%` timeout at the `95%` coverage gate. QDR-197+QDR-199 does
not generalize zero-shot: `36.03%` success, `63.44%` timeout, only `0.53%`
obstacle, and coverage around `78.8%`. Diagnostics show timeouts are target
mismatch/caution: timeout states have about `126deg` pred-oracle angle,
`48.1%` predicted target rays below danger, and very low action speed.

QDR-201 confirmed this gap is trainable but exposed forgetting. Continuing
QDR-197 on a procedural-heavy mix raised procedural held-out seeds `88/90/91`
to mean `62.48%` success, with only `0.56%` contacts, but standalone maze
collapsed to about `15.96%` success and `83.15%` timeout. Center block, doorway,
corridor, random blocks, and aggregate layout-mixed remained high, so this is
not a total policy collapse; it is a distribution/curriculum conflict between
procedural patterns and the standalone maze skill.

Current interpretation after QDR-201: the missing fixed-layout pieces are in
place, but broad procedural generalization is not solved. The next method should
not be another naive continuation. It needs explicit anti-forgetting or balanced
rehearsal: per-update fixed/procedural batches, KL/anchor loss to QDR-197 on
fixed layouts, hard-state replay/model selection, or a fresh student trained
from the start on balanced procedural plus fixed modes.

QDR-202 tested the KL/anchor-loss idea. The trainer now supports
`--anchor-checkpoint`, `--anchor-modes`, and `--anchor-loss-weight`; on anchor
modes it penalizes drift from the frozen QDR-197 target predictions for the same
deployable observation sequence. This successfully prevents QDR-201-style
forgetting: fixed4/fixed5 stay at about `99.88%`, aggregate layout-mixed is
`99.94%`, and standalone maze recovers to about `99.85%`. But it does not solve
procedural generalization: held-out procedural success is `61.64%`, essentially
tied with QDR-201's `62.48%`, with `37.85%` timeouts.

Current interpretation after QDR-202: anti-forgetting is solved for this
checkpoint family, but the procedural timeout/stall remains. Training logs show
procedural rollout success falls as student rollout probability rises, so the
next blocker is induced-distribution target prediction on procedural layouts,
not fixed-layout retention. The next attempt should focus on procedural hard
states: slower/lower student rollout schedule, explicit DAgger replay of
procedural timeout states, or an auxiliary deployable map/frontier memory head.

QDR-203 tested the smallest hard-state version: keep QDR-202/QDR-197 anchor
rehearsal, but upweight procedural labels when the current student target has a
large pred-oracle angle, points into worse clearance, or becomes too slow after
some coverage. This preserved fixed layouts but did not improve procedural
generalization. Held-out procedural seeds `88/90/91` reached only `59.18%`
mean success, with `40.36%` timeout and `0.47%` obstacle, below QDR-202's
`61.64%`. Fixed5 stayed high at `99.85%` with no timeouts.

Current interpretation after QDR-203: the hard-state problem is not solved by
reweighting the same supervised target loss. We need a stronger change in what
the model learns or where it is reset: actual timeout-window replay/restarts,
auxiliary deployable spatial-memory/frontier prediction, or an asymmetric
training objective that gives useful map/planner structure to the critic/labels
without leaking privileged inputs to the actor.

QDR-204 ran the exact-reset diagnostic before building another replay trainer.
It captures QDR-202 procedural timeout-window states, then restores them under
the student and under the privileged frontier-path oracle. From 64 states at a
256-step rewind, the student rescued `0/64`; the oracle rescued only `23/64`.
A 512-step rewind did not improve the oracle count, and a 1024-step rewind
dropped it to `16/64`. The most important new feature statistic is target
existence: in the 256-step window the oracle still had a valid target in
`68.8%` of states, but the student predicted target existence in only `7.8%`.
Student action speed was exactly `0` on average.

Current interpretation after QDR-204: resettable replay is useful as a probe,
but not enough as the next whole method. The student is learning to stop in
late procedural exploration because its target-existence/memory estimate
collapses, and the current privileged target oracle is also not a strong
recovery expert for many late states. The next serious build should change the
representation/objective: target-existence classification from deployable
history, auxiliary explored/frontier map prediction, or a stronger privileged
procedural oracle before hard-state replay training.

QDR-205 tested the narrow target-existence hypothesis directly. The trainer now
supports a mode-gated false-negative existence margin loss for procedural
positive labels after some coverage. This changed the timeout-window signature:
on replay, the student predicted existence in `100%` of sampled hard states
instead of almost never. But it did not improve behavior. Held-out procedural
seeds `88/90/91` averaged only `61.53%` success, essentially tied with
QDR-202, and replay still rescued `0/64` student states. Fixed-layout retention
remained strong.

Current interpretation after QDR-205: existence collapse is real, but scalar
existence supervision is not the root fix. It can make the model say that a
target exists without giving it a good spatial memory of where useful frontier
directions are. Do not keep tuning existence margins in isolation.

QDR-206 added a stronger spatial-memory label while keeping the actor input
deployable. The C env writes four privileged frontier-sector scores into target
label slots `22:26`; those slots are still zeroed from the deployed actor input.
The target model output head is resized from 4 to 8 outputs, copying QDR-202's
old target head and adding sector supervision. This was the first meaningful
procedural gain in this branch: held-out procedural seeds `88/90/91` reached
`64.78%`/`64.57%`/`65.93%`, mean `65.09%`, with `34.37%` timeout and `0.54%`
obstacle. Fixed retention stayed near `99.9%` with no timeouts.

Current interpretation after QDR-206: privileged frontier-sector auxiliary
supervision is a real lever. It improves procedural generalization without
fixed-layout forgetting, but it mostly helps the model avoid the worst tail.
When QDR-206 still times out, the old hard-state symptom remains:
`pred_exists=10.9%`, `oracle_exists=70.3%`, and student replay from those states
still rescues `0/64`. The next test is QDR-207: continue from QDR-206 and add
QDR-205's existence correction only as a companion signal. If that fails, move
to richer deployable map/frontier prediction or a stronger procedural oracle,
not more scalar target tuning.

QDR-207 is the current strongest procedural checkpoint. Continuing QDR-206 with
the QDR-205 false-negative existence companion loss raises held-out procedural
success to `67.28%` over seeds `88/90/91`, with `32.37%` timeout and only
`0.36%` obstacle. Fixed retention still passes: fixed5/all-nonaggregate is
`99.89%`, aggregate layout-mixed is `99.90%`, and timeouts stay at `0%`.

Current interpretation after QDR-207: the average benchmark moved, but the
late-state mechanism is still not solved. Exact replay of QDR-207 timeout
windows again rescues `0/64` student states. The signature changed from "no
target, stop" to "target exists, but the target/action is not a useful recovery
decision": `pred_exists=100%`, `oracle_exists=59.4%`, and action speed is
`0.173`. The next build, QDR-208, should improve teacher/label quality when the
path target is absent but privileged frontier sectors still identify useful
unexplored structure. If that fails, move to richer map/frontier prediction
rather than another scalar existence variant.

QDR-208 tested that direct sector fallback and failed. Relabeling absent-path
states with the best coarse sector direction collapsed procedural held-out
success to `54.21%`. This is a useful negative result: sector labels should not
be used directly as action targets while the sector head itself is unreliable.

The updated hard-state probe then added sector diagnostics. On QDR-207 timeout
windows, `sector_top_match` is only `21.9%` and predicted sector confidence is
lower than the oracle (`0.30` vs `0.44`). QDR-209 therefore kept sectors as an
auxiliary only, increased sector-loss pressure, and weighted procedural hard
states. This is now the strongest procedural checkpoint: held-out procedural
success is `68.12%`, timeout `31.47%`, obstacle `0.41%`; fixed5 retention is
`99.90%` with no timeouts. It also improves hard-state sector top-match to
`37.5%`. But exact replay still rescues `0/64` student hard states.

Current interpretation after QDR-209: four scalar sector auxiliaries help but
are too weak to solve late exploration. The next engineering build should stop
asking a 4-sector head to carry map memory. Build richer deployable
spatial-memory supervision or a stronger recovery oracle: local occupancy or
frontier map prediction from history, higher-resolution sector labels with a
safe observation-index migration, or a privileged oracle that can rescue more
late states before hard-state replay training.

QDR-210 tested the first richer spatial-memory label without changing deployed
actor inputs or ray indices. The C env now writes a privileged nearest-frontier
vector into slots `19:21`; the old four frontier sectors remain in `22:26`.
The trainer can expand the target model to 11 outputs:
`target4 + sector4 + nearest_frontier_vector3`. Slots `14:26` are still zeroed
from actor input, so the new vector is label-only. This is now the strongest
procedural checkpoint: held-out procedural seeds `88/90/91` reach `70.13%`
mean success, `29.37%` timeout, `0.49%` obstacle, and `90.06%` coverage. Fixed5
retention is `99.92%` with no timeouts.

QDR-210 is progress, but not a solved recovery mechanism. Hard-state replay
still rescues `0/64` student timeout-window states. The new vector head is
useful but imperfect in those states (`vector_angle_mean=40.95deg`,
`vector_dist_error=0.195`), and the target/action head still does not use it
well enough to recover. Oracle replay rises only slightly to `20/64`.

QDR-211 tested the obvious follow-up: use the true nearest-frontier vector as a
direct fallback target when the path oracle is absent. It failed like the
earlier sector-fallback attempt. Held-out procedural success fell to `55.00%`
with `44.51%` timeout. The lesson is now stronger: direct fallback relabeling
is the wrong mechanism even when the label is continuous and privileged. These
labels should supervise memory/representation, not replace the target oracle.

Current interpretation after QDR-211: the project made real progress this turn,
from QDR-209's `68.12%` procedural success to QDR-210's `70.13%`, but the
scientific blocker remains exact late-state recovery. The next build should be
actual local map/frontier patch prediction or a stronger privileged recovery
oracle. Do not keep trying direct fallback target labels.

QDR-212 is reserved for the map-memory auxiliary build. The environment now
appends a 16-float privileged egocentric frontier/explored-free patch after the
existing 60 deployable inputs. The actor input path zeros everything after slot
`60`, so the patch cannot leak into deployment. The target model grows from 11
to 27 outputs: `target4 + sector4 + nearest_frontier_vector3 + patch16`.
Checkpoint loading supports resizing QDR-210's 60-input encoder to the new
76-input environment and resizing the decoder head to 27 outputs. The
hypothesis is that richer patch supervision should train recurrent spatial
memory better than one vector while preserving QDR-210 behavior.

QDR-212 and QDR-213 falsified the passive-map-auxiliary path as a plateau
breaker. QDR-212's expanded 27-output decoder learned the patch label but
regressed procedural success to `69.51%` and raised obstacle contacts to
`0.63%`. QDR-213 moved the patch prediction into a separate auxiliary head so
the deployed policy stayed at QDR-210's 11 outputs. That reduced the contact
regression (`0.46%`) but still missed QDR-210, with `69.66%` procedural
success and `29.88%` timeout. Both variants preserve fixed-layout skill near
`99.9%`, so the failure is not fixed-layout forgetting.

The hard-state signature is the important finding. QDR-212 improves vector
agreement and learns a nontrivial patch (`patch_mse=0.062`), and QDR-213 keeps
sector agreement near QDR-210, but both still rescue `0/64` student replay
states. Target-vs-oracle angle remains huge (`~99-107deg`). The model can learn
passive spatial-memory labels, but the target/action head still does not turn
that memory into recovery behavior. The next build should stop adding passive
auxiliary heads and instead train behavior from resettable timeout-window
states or another objective that directly couples memory to recovery actions.

QDR-214 through QDR-217 then built that resettable hard-state path and falsified
several naive versions of it. The infrastructure works: the trainer can collect
timeout-window snapshots, filter to oracle-recoverable states, restore them in
batch, and train from those exact states. QDR-214 found `55/256`
oracle-recoverable states from QDR-210, but naive replay collapsed broad
procedural success to `57.67%` and preserved `0/64` student replay. QDR-215
added positive-only labels and false-negative existence pressure; it preserved
existence but still collapsed to `54.40%`. QDR-216 recollected hard states
between short training rounds to avoid stale hidden snapshots; it improved only
to `59.36%`. QDR-217 added a frozen QDR-210 behavior anchor on normal
procedural rollouts; that prevented catastrophic forgetting and reached
`68.17%`, but still missed QDR-210's `70.13%` and did not improve hard-state
replay (`0/64`).

QDR-218 and QDR-219 closed that replay branch for now. QDR-218 added burn-in:
restore an earlier snapshot, replay recorded deployable action history through
the current policy, then train at the hard state. It failed hard: saved
procedural seed 88 collapsed to `28.77%` success / `70.40%` timeout, and
student hard-state replay remained `0/64` with action speed `0.048`. QDR-219
added a behavior-coupled action imitation loss against the oracle controller's
actual velocity command. It also collapsed (`28.18%` success / `71.18%`
timeout) and still rescued `0/64` hard states.

Current interpretation after QDR-219: exact resettable replay is useful as a
diagnostic, but the current target-vector head is the wrong near-term recovery
interface. Naive, positive-only, iterative, behavior-anchored, burn-in, and
action-coupled variants all fail to transfer recovery into normal procedural
rollouts. Do not spend more time on small replay loss variants until the
observation or recovery module changes.

QDR-220 made that larger change by treating the existing 16-float egocentric
explored/frontier patch (`obs[60:76]`) as deployable onboard local-map input.
The actor still zeros privileged path/target/planner slots `14:26`; the new
`--deploy-patch-obs` flag only leaves the local patch visible. Training from
QDR-210 with a resized input head improved saved procedural seeds `88/90/91` to
`69.89%`/`70.73%`/`74.02%`, mean `71.55%`, versus QDR-210's `70.13%`. Fixed
retention passed at `99.90%`. The hard-state signature improved
(`81.83deg` target angle, `0.171` action speed), but replay is still `0/64`.

Current interpretation after QDR-220: deployable spatial memory is a real lever.
The current 4x4 patch is not enough to solve late procedural recovery, but it
is the first method after QDR-210 that improves average procedural exploration
without fixed-layout forgetting. The next engineering build should strengthen
the deployable map representation or introduce a planner-style hierarchical
target/recovery module; it should not return to exact target-replay loss
variants as the main path.

QDR-221 strengthened that representation. The native observation ABI grew from
`76` to `174`: QDR-220's first 16 patch slots remain at `60:76`, and the new
slots append a `7x7` egocentric known-free map plus a `7x7` unknown-adjacent-to-
known frontier-boundary map. These are derived from the actor's explored grid,
not from privileged path/target slots. Training from QDR-220 with input-head
resize improved saved procedural seeds `88/90/91` to
`76.77%`/`77.17%`/`78.57%`, mean `77.51%`, with timeout down to `22.18%` and
obstacle contacts down to `0.32%`. Fixed retention passed at `99.91%`.
Hard-state replay still rescues `0/64`, but vector angle improved to `19.59deg`
and action speed to `0.200`.

Current interpretation after QDR-221: deployable local-map memory is not a
side issue; it is the strongest plateau breaker so far. The remaining failure
is late target selection/recovery from states where the current privileged
oracle also only rescues a minority. The next build should use QDR-221 as the
base and add a hierarchical recovery/target module, a stronger oracle, or an
asymmetric critic over map state. Do not keep adding flat local-map slots as
the primary method without first proving what missing map information the hard
states require.

QDR-222 tested the first hierarchical recovery interface on top of QDR-221
without retraining. The policy's auxiliary frontier-vector direction is useful
in late states, but its distance scalar is dead: hard-state diagnostics showed
`pred_vector_dist_mean=0` against `oracle_vector_dist_mean=0.2445`. A fallback
that required predicted vector distance therefore never engaged. A direction-
only fallback works if it is gated by primary target confidence: threshold
`0.6`, min vector distance `0.0`, min coverage `0.25` improves procedural seeds
`88/90/91` to `82.84%`/`83.40%`/`83.90%`, mean `83.38%`, with timeout down to
`16.32%` and obstacle contacts `0.30%`. Fixed-layout retention remains
`99.91%` with no timeouts. This is now the best procedural controller setting,
but hard-state replay is still `0/64`, so it is not a solved recovery module.

Current interpretation after QDR-222: the missing piece was not absence of a
heading signal; it was conflating heading with a weak/dead existence-distance
scalar. The next engineering build should turn this into a learned gate or
recovery head trained from privileged frontier labels, not keep expanding flat
map slots or hand-tune a threshold indefinitely.

QDR-223 and QDR-224 tested that gate cleanup. QDR-223 repaired the vector
distance scalar with an unclamped loss, but updating the full recurrent policy
damaged procedural exploration: saved seeds `88/90/91` fell to mean `64.68%`.
QDR-224 repeated the scalar repair while freezing the policy and training only
decoder mean row 10. That preserves QDR-222 behavior while making the vector
distance scalar nonzero: procedural seeds `88/90/91` are
`82.84%`/`83.40%`/`83.93%`, mean `83.39%`, timeout `16.31%`, obstacle
`0.30%`; fixed retention remains `99.91%`; hard-state replay is still `0/64`.

Current interpretation after QDR-224: promote QDR-224 as the cleaner current
procedural candidate because it no longer relies on ignoring vector distance.
This is not a new capability jump over QDR-222. The next real build should be
an explicit recovery/gating module or stronger late-state oracle that improves
the `0/64` hard-state replay signature.

## Current State

The env is no longer suspected broken in the broad sense. We have found and
fixed several real issues:

- invalid/overhard geometry and overly conservative waypoint settings;
- wrapper/profile defaults overriding scenario intent;
- PufferLib 4.0 trainer parity gaps: reward clipping, network selection,
  decoder init, and missing linear biases;
- invalid checkpoint conversion when previous action was cropped away;
- weak direct navigation from zero without privileged help.

The current research track is split into Swift-lite direct flight and indoor
exploration. For the active exploration branch:

- actor observation: state, previous action, rays, and now optionally the
  deployable local explored/frontier map via QDR-221;
- actor still must not see privileged path/target slots `14:26`;
- privileged side: frontier/path labels, fixed-layout anchors, diagnostics,
  hard-state probes, and future critic/teacher signals;
- current best procedural exploration candidate: QDR-224,
  `experiments/qdr224/qdr224_qdr221_vector_dist_row_only_s88.pt`, using the
  checkpoint's saved fallback flags `--use-pred-vector-fallback`,
  `--pred-vector-fallback-primary-threshold 0.6`,
  `--pred-vector-fallback-min-coverage 0.25`, and
  `--pred-vector-fallback-min-dist 0.02`.

The older Swift-lite direct actor track remains useful:

- actor observation: deployable direct target, local rays/walls, state, previous
  action;
- privileged side: waypoint/planner observations for teacher, critic, labels,
  diagnostics, and hard-state recovery;
- action interface for now: velocity commands through our low-level velocity
  controller, kept as a scaffold before returning to body-rate/thrust actions.

Best current checkpoints:

- QDR-136: safest one-goal deployable direct actor,
  `experiments/qdr136/qdr136_failure_recovery_tail24_ep1_s88.bin`.
- QDR-139: old sequential direct-routing baseline,
  `experiments/qdr139/qdr139_goals3_recovery_lr1e5_s88.bin`.
- QDR-145 update 5: current strongest sequential candidate,
  `experiments/qdr145/asym_mixed15_from_qdr144_s88/qdr135_asym_ppo_000005.bin`.

Important measured gates:

- QDR-136 one-goal direct: `94.91%`.
- QDR-139 one-goal direct: `94.68%`.
- QDR-136 direct three-goal: `48.41%`.
- QDR-139 direct three-goal: `65.05%`.
- QDR-139 direct five-goal: `45.88%`.
- QDR-145 direct three-goal: `65.44%`, with `34.56%` obstacle contacts.
- QDR-145 direct five-goal: `50.75%`.
- QDR-145 direct combo: `54.70%`.

The next failure is not basic flight or light dynamics. It is sequential direct
routing under partial observability: the policy becomes more decisive, reaches
more goals, then fails by obstacle contact.

The latest iteration moved the limit only marginally. QDR-141 clearance-speed
anchoring made the policy cautious and lost the QDR-139 sequential gain.
QDR-142 iterative correction saturated around QDR-139. QDR-143 showed
deployable perception variants need a separate 4.0 rebuild because observation
size is fixed at `60`. QDR-144 mixed-length hard-state collection and QDR-145
early asymmetric PPO produced the current best checkpoint, but only by about
`+0.39` points on `direct_goals3` versus QDR-139. The honest blocker is still
sequential obstacle contact, not lack of a small PPO/recovery tweak.

QDR-146 through QDR-150 tested a more aggressive "solve the general problem"
branch. Keeping the 60-float ABI by replacing 24 raw rays with 12 rays plus
local affordance features was valid infrastructure but a bad observation
contract: QDR-146 reached only `54.87%` one-goal and `26.63%` three-goal
success while the privileged teacher was `95.97%`. Raw 24-ray native-action
history distillation isolated the issue: QDR-147 improved to `67.44%`
one-goal and `40.58%` three-goal, so the affordance/ray-budget trade was
harmful, but one-step sequential teacher cloning still fails badly. Native PPO
from zero (QDR-148) was worse, peaking at `30.94%` stochastic three-goal
success. Full student-rollout DAgger (QDR-149) degraded to `37.21%`, and
asymmetric PPO from the weak QDR-147 actor (QDR-150) collapsed to `14.14%`.
Conclusion: the current direct three-goal formulation is not solved by
hand-built local affordance features, vanilla PPO from zero, stronger one-step
DAgger, or PPO rescue from a weak cloned actor.

QDR-151 through QDR-153 changed the diagnosis. Native resettable hard-state
replay is now implemented and proven: restoring a QDR-145 pre-crash env state
plus MinGRU hidden state reproduced the exact action/observation/reward/
terminal trajectory with zero drift. The blocker is not replay mechanics. The
blocker is the expert/recovery target: on fixed QDR-145 pre-crash states, the
simple privileged waypoint teacher rescued only `14/128` at `rewind=8`,
`55/128` at `rewind=32`, and `32/64` at `rewind=64` with a longer horizon. A
ray-repulsion teacher did not improve this (`29/64`, then `28/64` for slower
variants). These states are not all last-frame collisions; at `rewind=64`, mean
minimum ray clearance was about `0.555m`. This explains the plateau: repeated
BC/DAgger is training toward a teacher that is itself weak on the induced hard
states.

QDR-154 tested a combined resettable + stronger-privilege idea: local
simulator-rollout search over velocity commands from the exact hard state. The
mechanics worked, but the teacher did not. The best bounded search variant
rescued `33/64` hard states versus `32/64` for the waypoint teacher on the same
setup; longer commits and richer candidate sets were worse. Do not convert this
constant-action MPC probe into labels. The stronger privilege has to be a real
trajectory/planning expert or a resettable RL objective.

## Iteration Rules

Every nontrivial experiment gets a QDR ID in `TESTS.md` before running.

Each experiment must answer one question:

- What exact failure are we trying to move?
- What is the falsifiable prediction?
- What checkpoint and command prove the result?
- What should not be repeated?

Promotion requires deterministic held-out eval, not sampled training HUDs.
For direct-actor work, the minimum held-out seeds are `88/90/91`.

A result is not a win if it only moves the easiest metric while making the next
gate worse. One-goal direct success, sequential direct success, obstacle hits,
timeouts, and min clearance must be read together.

When a method helps but changes the failure type, document that as progress and
as the new limit. QDR-139 was the first example: it improved sequential success
but turned timeouts into obstacle contacts. QDR-145 is the current marginal
extension of that line, not a solved benchmark.

## Immediate Ladder

### 1. Safer Sequential Recovery

Question: can we keep QDR-145's sequential progress while reducing obstacle
contacts?

Likely probes:

- collect hard states specifically around post-goal transitions;
- separate timeout/hesitation windows from obstacle-contact windows;
- add clearance or speed anchoring during recovery training;
- replay hard states with resettable env state if we add that capability;
- compare QDR-136, QDR-139, QDR-144, and QDR-145 on identical failing episode
  seeds.

Pass gate:

- preserve one-goal direct success near QDR-136/QDR-139;
- improve `direct_goals3` over QDR-145;
- reduce obstacle-contact rate versus QDR-145, not merely remove timeouts.

### 2. Resettable Hard-State Replay

Question: can we stop learning only from the final few frames before impact and
instead train from the states where the bad decision starts?

Native snapshot/restore support now exists via `VecEnv.snapshot(env_id)` and
`VecEnv.restore(env_id, snapshot)`. It captures the route state, obstacle
layout, dynamics state, previous action, and output buffers; the evaluator must
also save/restore recurrent policy state.

Pass gate:

- deterministic replay of the same hard state gives the same failure;
- a stronger privileged expert or resettable RL objective improves that hard-
  state set;
- the update transfers to normal rollout eval.

The exact replay part is solved. The highest-priority engineering build is now
a stronger recovery expert/objective. QDR-152 and QDR-153 show the current
one-step waypoint/ray teachers are not good enough on the fixed hard-state set,
and QDR-154 shows shallow local rollout search is also not enough. Do not
launch more supervised recovery runs until the teacher or resettable RL
objective can beat the hard-state ceiling.

### 3. Better Deployable Perception

Question: is direct target plus rays enough for hidden sequential routing, or do
we need a more structured onboard observation?

Possible deployable additions:

- short history of local rays and previous actions;
- egocentric local occupancy/range image;
- optical-flow-like or depth-change features;
- goal-bearing memory instead of planner breadcrumbs.

Constraint: no map/planner waypoint in the actor observation unless we mark the
benchmark as scaffolded. Privileged planner data may remain in teacher, critic,
reward, reset curriculum, and diagnostics.

Pass gate:

- improves sequential direct routing from zero or from QDR-139;
- does not secretly reintroduce privileged waypoint information at deployment.

Do not repeat compact/affordance hand-designed features as the next default
perception fix. QDR-146 failed the one-goal gate badly even though the teacher
ceiling was high. The more promising perception build is an explicit egocentric
local memory/range image derived from rays plus odometry, but only after we have
a hard-state replay loop that can prove it changes the failing states.

### 4. Return To Lower-Level Control

Question: can the policy learn a more real drone action interface after the
navigation problem is shaped enough?

Current velocity control is a scaffold, not the final sim2real controller.
The next action ladder should be:

1. hover with body-rate/thrust action;
2. visible target move with body-rate/thrust action;
3. obstacle-free direct routing;
4. one-goal obstacle routing;
5. sequential obstacle routing.

Pass gate:

- body-rate/thrust policy reaches the same simple rung before we combine it
  with the hard sequential navigation problem.

### 5. Sim2Real Modelling

Question: which real-drone effects dominate policy failure?

The simulator should absorb measured or at least explicit models for:

- IMU/proprioceptive state noise;
- range/depth noise and dropout;
- command latency and action delay;
- motor time constant and thrust curve;
- mass, inertia, drag, thrust margin;
- floor/ceiling/wall geometry assumptions;
- onboard compute rate.

Pass gate:

- clean-sim champion survives named stress profiles;
- stress failures are attributed to specific model knobs;
- real drone measurements replace placeholder randomization ranges.

## SOTA Gap

What we do not yet have compared with stronger research systems:

- resettable hard-state replay that transfers into normal rollouts;
- recurrent training that reliably improves induced-distribution behavior;
- a hierarchical recovery/target module on top of QDR-221-style local maps;
- body-rate/thrust control at the same competence as velocity control;
- measured real-drone dynamics and sensing;
- system identification and hardware validation loop;
- large randomized layout distribution and unseen-building evaluation.

The path to close that gap is not a single huge rewrite. It is a sequence of
hard gates:

1. one-goal direct actor works;
2. sequential direct actor works;
3. sequential direct actor survives noise/dynamics;
4. body-rate/thrust actor works on the same ladder;
5. measured drone profile replaces synthetic profile;
6. real indoor hover/move validation;
7. real indoor obstacle and sequential-goal validation.

## Default Next Prompt

Use this when starting a fresh Codex session:

```text
Continue the PufferLib quadrotor Swift-lite research track. Read
ocean/quadrotor/CODEX_ITERATION.md, ROADMAP.md, and TESTS.md first. Keep
PufferLib as the training stack. Work scientifically: reserve a QDR ID, make a
falsifiable hypothesis, run deterministic held-out evals, document the lesson,
and do not tune one-goal success if the active limit is procedural indoor
exploration. Current active limit is exploration efficiency after QDR-283, not
basic perception or collision safety. QDR-280 built a strong privileged
`region` teacher, QDR-281 proved privileged labels alone do not transfer, and
QDR-282 added the missing deployable scaffold: `--deploy-onboard-target-obs`,
which exposes only the onboard frontier-planner target to the actor. Corrected
hard-p1 at `2048` steps rose from QDR-251's `28.16%` seed-88 success to about
`75%`, with held-out seeds `90/91` around `76-78%` and zero contacts. QDR-283
then showed the same checkpoint reaches about `97-99%` success on hard-p1 over
seeds `88/90/91` when the horizon is `4096`, still with zero contacts. QDR-284
promotes the current fast-safe controller profile for `2048`: QDR-282 plus
`--speed-fraction 0.50 --safety-slide --safety-cone-ray-stop 0.30
--safety-slow-ray 0.80` reaches about `88.8%` / `93.5%` / `90.3%` on hard-p1
seeds `88/90/91`, with zero contacts and fixed/procedural regression intact.
The remaining problem is therefore frontier/path efficiency under the shorter
fast-exploration budget, not raw safety or basic target perception. Use
`--drain-initial-episodes` or `explore-v0-target-horizon-probe`, because the
old early-stop eval censored slow timeouts and can falsely report `100%`.
```

## 2026-05-21 QDR-280 to QDR-281 Update

QDR-280 is the first clearly stronger privileged exploration teacher. The first
implementation overvalued huge far regions and timed out on hard-p1 around
`92%` coverage, so it was rejected. The revised region-rate selector solved all
hard procedural families p0/p1/p2/p3 over seeds `88/90/91` with `100%` success
and `0%` contacts at `2048` steps. It also improved `new_cell_rate` over the
old privileged `unexplored` teacher on every tested pair, especially hard-p1
where old `unexplored` was about `1.7` cells/step and region is about
`2.3-2.4`.

The important constraint remains: QDR-280 is privileged. It is allowed for
teacher labels, diagnostics, curriculum, critic targets, and oracle comparison;
it is not allowed as deployed actor input. QDR-281 starts the deployable-student
step by using `--explore-frontier-target-mode region` inside
`explore-v0-target-distill`, while `deployable_input()` still zeroes the
privileged target/path slots and only preserves explicitly deployable map patch
observations.

QDR-281 local smoke passed: it loaded QDR-251 as both init and anchor, trained
two CPU updates with the region teacher, saved
`experiments/qdr281/qdr281_region_teacher_smoke_s88.pt`, and completed tiny
in-memory eval without crashes or contacts. The real evidence now requires a
GPU-sized run. At the time of this note, the SSH master was not reusable from
Codex and remote commands fell back to password authentication through
`kosh.aalto.fi`, so goal work is paused under the GPU-connection rule.

QDR-281 later completed after the GPU connection was restored. It preserved easy
procedural behavior but failed the real hard-p1 transfer test: corrected
hard-p1 stayed around the old baseline. The lesson is important: a privileged
teacher can be perfect, but a deployable recurrent actor still needs enough
deployable spatial/planner state to act on.

QDR-282 adds that missing state in the conservative way: `--deploy-onboard-target-obs`
preserves target/path obs slots only when the target source is `onboard`, which
is built from the actor's deployable exploration grid/rays. It explicitly
rejects privileged target modes. This is closer to the research stack shape:
mapping/frontier state proposes a local objective, while the learned policy
follows and smooths it.

The first QDR-282 checkpoint is the current strongest candidate:

```bash
experiments/qdr282/qdr282_onboard_target_obs_qdr251_s88.pt
```

Corrected drained hard-p1 evals:

- QDR-251 seed `88`: `28.16%` success, `71.84%` timeout, `0%` obstacle.
- QDR-282 seed `88`: `75.51%` success, `24.49%` timeout, `0%` obstacle.
- QDR-282 seeds `90/91`: `76.46%` / `77.82%`, both `0%` obstacle.

This also uncovered a benchmark issue: high-parallel eval can stop after enough
fast successes and censor slow timeout episodes. Use `--drain-initial-episodes`
or small-agent/single-agent confirmation when evaluating hard exploration.
Without that, QDR-282 falsely looked like `100%` on hard-p1. Visual/no-render
still exposes wall-stall timeouts, so the next real limit is recovery/unsticking
around onboard targets, not collision safety.

## 2026-05-22 QDR-283 Horizon/Efficiency Update

QDR-283 reframes the immediate blocker. The new result is not another small
policy gain; it shows that our strict `2048`-step hard-p1 gate was partly
measuring speed rather than exploration validity.

Onboard-oracle stall replay collected `64` near-wall/no-progress states. These
states averaged `0.571` coverage, `0.337m` min-ray clearance, `0.449m` target
distance, and only `0.179` action speed. Replaying the onboard target controller
from those states solved only `2/64` within `512` steps, but gained coverage
without crashing. That points at slow/local objective behavior, not unsafe
targets through walls.

QDR-282 policy timeout hard-state replay reached the same conclusion. Over a
`60000`-step collection budget it found only `19` timeout states. The predicted
target was close to the onboard oracle target: mean angle `7.47deg`, both
target-exists rates `1.0`, and sector top-match `0.947`. Student replay timed
out all `19`; oracle replay rescued `9/19`, with zero crashes. This is not a
strong perception-collapse signature.

The decisive check is horizon:

- QDR-282 hard-p1 seed `88`, `2048` steps: about `67.6%` success, `32.4%`
  timeout, `0%` obstacle in the local drained eval.
- QDR-282 hard-p1 seed `88`, `4096` steps: about `98.1%` success, `1.9%`
  timeout, `0%` obstacle.
- QDR-282 hard-p1 seeds `90/91`, `4096` steps: about `97.2%` / `98.8%`,
  both zero contacts.

Added `episode_length` and `path_length` to target-distill eval output, and
added this reproducible probe:

```bash
scripts/quadrotor4.sh explore-v0-target-horizon-probe experiments/qdr282/qdr282_onboard_target_obs_qdr251_s88.pt --horizons 2048,4096 --seeds 88,90,91 --episodes 256
```

Next work should optimize exploration rate while preserving zero contacts:
better frontier/path ordering, fewer local viewpoint stalls, explicit
time-to-coverage metrics, and eventually a learned high-level selector over
planner candidates. Do not keep treating this as a raw vision failure unless a
new diagnostic proves that.

## 2026-05-22 QDR-284 Fast-Safe Controller Update

QDR-284 tested whether QDR-282/QDR-283 were simply too conservative under the
`2048`-step hard-p1 budget. The answer is partly yes, but only with the
deployable slide safety scaffold. Raw speed is not enough:

- speed `0.45`, no slide: seed `88` hard-p1 `73.05%`, zero contacts.
- speed `0.55`, no slide: seed `88` hard-p1 `73.08%`, but `5.0%` obstacle
  contacts.
- speed `0.45` plus slide: seeds `88/90/91` `87.21%` / `89.23%` / `85.71%`,
  zero contacts.
- speed `0.50` plus slide with earlier slowdown (`0.30/0.80`): seeds
  `88/90/91` `88.76%` / `93.54%` / `90.27%`, zero contacts.

The current promoted controller profile is therefore QDR-282 checkpoint plus:

```bash
--speed-fraction 0.50 --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80
```

Regression smoke with that profile held: procedural aggregate seed `88`
`99.84%`, layout_mixed `99.12%`, maze `100%`, corridor `99.02%`, all zero
contacts. No-render hard-p1 visual smoke passed `3/3`.

This is a controller-profile improvement, not a new trained checkpoint. It
supports the SOTA-style decomposition: learned target/planner following plus a
deployable local safety layer. Next build should train/evaluate with this
fast-safe layer in the loop and then attack the remaining `~6-11%` timeout tail
through frontier/path ordering, not by increasing raw speed again.

## 2026-05-22 QDR-285 Fast-Safe Rollout Training

QDR-285 trained with the QDR-284 fast-safe controller in the rollout loop rather
than only applying it at eval time. This is a real improvement on the strict
hard-p1 `2048` fast-exploration gate, but it is not a full solution.

Checkpoint:

```bash
experiments/qdr285/qdr285_fast_safe_rollout_qdr282_s88.pt
```

Use it with the same deployable controller profile:

```bash
--speed-fraction 0.50 --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80
```

Corrected drained hard-p1 results:

- QDR-284 eval-only profile seeds `88/90/91`: `88.76%` / `93.54%` / `90.27%`.
- QDR-285 final checkpoint seeds `88/90/91`: `94.79%` / `92.70%` / `92.12%`.
- All of the above QDR-285 hard-p1 checks had `0%` obstacle contacts.

This moves the three-seed mean from about `90.86%` to `93.20%`. The result is
not just a lucky final checkpoint: the QDR-285 checkpoint sweep on seed `88`
improved monotonically from u200 through u800/final (`93.73%`, `93.96%`,
`94.56%`, `94.79%`), all with zero contacts.

Regression smoke is acceptable but not perfect:

- procedural aggregate seed `88`: `99.14%`, zero contacts;
- layout_mixed: `98.94%`, zero contacts;
- maze: `97.90%`, zero contacts;
- corridor: `99.24%`, zero contacts.

So QDR-285 is the current hard-p1 fast-exploration checkpoint, but it slightly
trades away easy/fixed retention, especially maze timeout rate. No-render visual
smoke passed `3/3` on hard-p1 with zero contacts/timeouts, while still logging
recoverable near-wall stalls. The next limit is therefore narrower: local
frontier/motion efficiency near walls. Do not push raw speed as the main line;
build a trajectory-level local motion controller, learned recovery residual, or
better frontier/path efficiency metric around this checkpoint.

## 2026-05-21 QDR-225 to QDR-233 Update

The main finding is an environment-validity bug in the procedural exploration
benchmark. Old `proc95` counted all valid cells in the layout denominator, even
when the spawn was in only one connected component. That made many episodes
partly impossible and explained why even privileged oracle variants saturated
around `62-65%` success and about `90%` coverage.

Implemented `--explore-reachable-coverage`. It computes the valid-cell
component reachable from the spawn and uses that as the coverage denominator.
The first implementation used a reachable denominator but still counted
visible cells outside the component, producing coverage above `1.0`; this was
fixed by storing a reachable mask and only counting cells in that component.

Corrected reachable `proc95` results:

- oracle: `80.90%` success, `19.01%` timeout, `0.09%` obstacle;
- QDR-224: `68.71%` success, `30.88%` timeout, `0.41%` obstacle;
- QDR-232 continuation: `54.31%`, failed;
- QDR-233 shorter/lower-student continuation: `57.61%`, failed.

Other tested paths:

- QDR-225 row-only recovery gate: no positive labels in normal rollouts;
  learned gate never fired.
- QDR-226 clear-ray hard-state replay: overfit the replay subset and collapsed
  full proc95 to `30.09%`.
- QDR-227 speed-only eval: faster commands traded timeouts for contacts.
- QDR-228/QDR-229 deployable scalar coverage observation: not enough.
- QDR-230 gain-scored frontier teacher: oracle smoke was not better than the
  student, so full training was skipped.

Current best is still QDR-224, but it must now be evaluated with
`--explore-reachable-coverage`. The next useful work is not another minor
continuation schedule. Either add stable checkpoint selection/early stopping
for target distillation, train a stronger deployable spatial-memory module, or
replace the teacher with a structured frontier planner that beats the student
under corrected reachable proc95.

## 2026-05-23 QDR-303 Shallow Approach Viewpoint Probe

QDR-303 tested the most direct version of "do not drive into the doorway
frontier; first pick a nearby clearance-friendly approach cell." The
implementation is opt-in:

```bash
--explore-onboard-approach-viewpoints --explore-onboard-approach-max-depth <n>
```

Local hard-p1 oracle checks on seed `88`, `256` episodes, `--cell-size 0.30`,
QDR-301/QDR-300 stack:

- baseline: `100%` success, `0%` contacts/timeouts, coverage `0.950574`,
  path `168.063m`, min clearance `0.3836m`;
- approach depth `1`: `97.318%` success, `2.682%` timeout, `0%` contacts,
  coverage `0.947767`, path `189.375m`, min clearance `0.3975m`;
- approach depth `2`: `98.099%` success, `1.901%` timeout, `0%` contacts,
  coverage `0.948679`, path `188.780m`, min clearance `0.4018m`;
- old late-stall viewpoint trigger at `90` ticks was essentially neutral:
  `100%` success, `0%` contacts/timeouts, path `169.659m`.

Conclusion: the existing local viewpoint selector is not strong enough as a
generic approach planner. It increases clearance slightly but adds detours and
timeouts. Do not use the new flag in the current best profile. The next serious
planner build should value candidate approach states by closed-loop progress or
region completion, not by a shallow frontier-depth rule.

## 2026-05-23 QDR-304 Runtime Local Trajectory Scorer

QDR-304 tested the concern that "trajectory-valued local planning" might just
repeat the old hard-state recovery branch. To make it a fair new test, it did
not capture hard states, generate labels, or train a sidecar. It added only an
opt-in deployable runtime action wrapper:

```bash
--safety-trajectory-actions
```

The wrapper samples short constant-velocity headings from current lidar/target
state and scores target progress, clearance, rollout room, wall escape, and
smoothness. It was tested oracle-first on hard-p1 with the QDR-301/QDR-300
stack.

Result: safe but not useful enough. The standalone trajectory controller kept
`100%` local seed-88 success with zero contacts, but path length worsened from
`168.063m` to `178.393m`. Gating it more tightly still stayed around `178m`.
Stacking a strict near-wall-only version on top of the existing barrier/slide
controller was almost neutral: seed `88` slightly improved path (`167.801m`),
but seed `90` slightly regressed (`163.997m` versus baseline `163.744m`).

Conclusion: this is not the missing lever. The current-lidar local candidate
scorer mostly rephrases the existing barrier/slide behavior. Future local
planning work needs topological or region-progress value, not another
constant-heading scorer.

## 2026-05-23 QDR-305 Active Persistent Regions

QDR-305 implemented the first active persistent frontier-region manager behind:

```bash
--explore-onboard-persistent-regions
```

The code gives current frontier clusters a stable memory slot, centroid,
attempt/failure counters, progress tick/cell state, and a score penalty for
failed regions. The hypothesis was that stable region identity would reduce
doorway target churn and improve hard-p1 oracle efficiency.

Result: strong negative. Baseline hard-p1 seed `88`, 256-episode local oracle
with the QDR-301/QDR-300 stack is `100%` success, `0%` contacts/timeouts,
coverage `0.950574`, path `168.063m`. The first active persistent selector
collapsed to `59.93%` success and `40.07%` timeout. Conservative variants were
still bad (`62.16%`, then `50.58%`). After removing a progress-clock bug where
region selection reset progress without new cells, stale-region churn collapsed
coverage to `0.4797` and success to `13.67%`.

This is useful evidence, not a promotion. The concept is plausible, but active
region memory must not own target selection or stale marking before it is
validated. The next version should be passive: compute stable region IDs,
switches, active-region progress, and predicted stale decisions while leaving
the existing planner output unchanged. Once that passive tracker predicts the
visual near-wall failures, only then should it influence target scoring.

## 2026-05-23 QDR-306 Passive Region Tracker

QDR-306 implemented the safe version of the QDR-305 idea. It adds an opt-in
passive tracker:

```bash
--explore-onboard-region-tracker
```

The tracker clusters the current reachable frontier cells, matches them to
stable region slots, tracks the active region selected by the existing planner,
and reports region count, active-region age, no-progress age, switch rate, and
stale prediction rate. It also adds a visual HUD line for these values. It does
not choose targets, mark stale cells, or modify action selection.

The behavior-equivalence gate passed exactly. On hard-p1 seed `88`, `256`
episodes, with the QDR-301/QDR-300 stack and `--cell-size 0.30`, baseline and
tracker both produced `100%` success, `0%` contacts/timeouts, coverage
`0.950574`, path `168.063m`, min clearance `0.383617`, action delta
`0.0640693`, and new-cell rate `0.788219`. The tracker emitted useful
diagnostics on the same run: regions `15.4784`, active age `386.797`,
since-progress `34.1852`, switch rate `0.0428793`, stale rate `0.0236302`.
A no-render visual smoke with the tracker solved `1/1` and logged
`wall_stall_logs=0`. The current QDR-300 policy with the same tracker visual
profile solved `32/32`, coverage `0.9505`, no contacts/timeouts, and
`wall_stall_logs=3`. The visual summary now prints tracker metrics too; that
32-episode run reported regions `14.6912`, active age `361.2634`,
since-progress `33.8738`, switch rate `0.0473`, and stale rate `0.0240`.
The stall-bearing text traces often still had valid far onboard targets and
low safety cone clearance, so the current visible stalls are not automatically
evidence that topological region identity is the sole blocker.

Repro commands:

```bash
PYTHON=../../.venv/bin/python scripts/quadrotor4.sh build-cpu

PYTHON=../../.venv/bin/python scripts/quadrotor4.sh explore-v0-target-oracle-eval --mode procedural_indoor_hard_p1 --seed 88 --episodes 256 --total-agents 256 --num-buffers 1 --num-threads 8 --max-steps 2048 --room-size 6.0 --cell-size 0.30 --explore-goal-fraction 0.95 --speed-fraction 0.50 --explore-frontier-target-mode onboard --explore-reachable-coverage --explore-onboard-path-lookahead 2 --explore-onboard-region-commit --explore-onboard-region-commit-stall-ticks 90 --explore-onboard-region-tracker --safety-barrier-actions --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80

PYTHON=../../.venv/bin/python scripts/quadrotor4.sh explore-v0-target-visual --controller oracle --mode procedural_indoor_hard_p1 --seed 88 --max-steps 2048 --room-size 6.0 --cell-size 0.30 --explore-goal-fraction 0.95 --speed-fraction 0.50 --explore-frontier-target-mode onboard --explore-reachable-coverage --explore-onboard-path-lookahead 2 --explore-onboard-region-commit --explore-onboard-region-commit-stall-ticks 90 --explore-onboard-region-tracker --safety-barrier-actions --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80 --restart-on-stall-steps 3000
```

Conclusion: QDR-306 is promoted only as instrumentation. The next useful loop
is to run visual traces with the tracker enabled and inspect whether the
remaining near-wall stalls are associated with excessive region switches,
long active-region no-progress, or stale predictions. Do not make region memory
active again until this passive signal explains failures.

## 2026-05-23 QDR-307/QDR-308 Local Execution Diagnosis

QDR-307 split the near-wall stall action path into raw desired action,
barrier-only action, slide-after-barrier action, and final action. The same
32-episode hard-p1 no-render visual loop stayed safe:

- QDR-300 policy: `32/32` success, `0%` contacts/timeouts, coverage `0.9505`,
  new-cell rate `0.7815`, `wall_stall_logs=3`.
- One stall was not concerning: target cone scale `0.952`, final progress
  ratio `0.851`.
- Two stalls exposed the actual local-execution artifact: target cone scale was
  moderate (`0.698`/`0.761`), barrier still preserved some target progress
  (`0.181`/`0.197`), but slide cut progress almost to zero
  (`0.006`/`0.042`) while turning the command `89deg`/`85deg`.

This changes the interpretation. The remaining visible stall tail is not mainly
missing targets, planner target absence, or topological region identity. It is
partly a local projection issue: slide is globally useful, but in some doorway
states it over-projects a moderately valid command into near-zero-progress
wall-parallel motion.

QDR-308 tested the smallest existing correction:

```bash
--safety-progress-floor-scale 0.50 --safety-progress-floor-cone 0.60
```

This was not promotable. Oracle seed `88` stayed `100%` and slightly shortened
path (`167.363m` vs `168.063m`), but min clearance dropped (`0.379m` vs
`0.384m`). The QDR-300 policy eval regressed slightly: baseline `512` requested
episodes reached coverage `0.950551`, new-cell rate `0.761294`, path
`171.314m`, min clearance `0.3730m`; progress-floor eval reached coverage
`0.950526`, new-cell rate `0.757832`, path `171.533m`, min clearance
`0.3689m`. The 32-episode visual still logged `3` stalls. A temporary
barrier-progress-floor prototype had no effect and was removed.

Conclusion: do not remove slide globally and do not promote naive forward
forcing. The next local-control build should either make slide projection
progress-aware in a clearance-conditioned way, or replace the one-step
projection with a short closed-loop controller that explicitly trades target
progress against clearance. Keep the QDR-307 diagnostics in the visual harness;
they are the right feedback loop for this class of bug.

## 2026-05-23 QDR-309 Progress-Aware Slide Probe

QDR-309 tried the conservative version of that recommendation. It added an
opt-in progress-aware slide projection:

```bash
--safety-slide-progress-aware --safety-slide-progress-fraction <f> --safety-slide-progress-cone <m>
```

Instead of forcing raw target progress after all safety layers, this only
interpolates from the slide output back toward the pre-slide command, and only
when the target cone has enough clearance. Defaults keep existing behavior.

The mechanism worked, but the method did not promote. With
`f=0.60, cone=0.60`, the two QDR-307 slide-erased stall lines improved from
progress ratios `0.006`/`0.042` to `0.108`/`0.118`, with no contacts/timeouts.
But the 32-episode visual still logged `3` stalls and new-cell rate fell from
`0.7815` to `0.7565`. Stronger restoration (`f=1.00`) was destructive:
`93.75%` success, `6.25%` timeout, and `6` stall logs.

Oracle eval at `f=0.60` was basically neutral: seed `88`, `256` episodes,
`100%` success, `0%` contacts, path `168.006m`, min clearance `0.3820m`
versus baseline `168.063m` and `0.3836m`. QDR-300 policy eval slightly
regressed: baseline seed `88`, `512` requested episodes had coverage
`0.950551`, new-cell rate `0.761294`, path `171.314m`, min clearance
`0.3730m`; progress-aware slide had coverage `0.950560`, new-cell rate
`0.758808`, path `171.731m`, min clearance `0.3721m`.

Conclusion: keep the opt-in flag as a diagnostic, but do not use it in the
current best profile. The scalar "restore some progress" model is too crude.
The next controller should evaluate a short local trajectory or explicitly
model corridor tangent/doorway passage geometry, not just preserve a fixed
fraction of pre-slide progress.

## 2026-05-23 QDR-310 Clearance Passability Probe

QDR-310 tested the hypothesis that the doorway stalls come from the onboard
grid treating too-narrow gaps as traversable. It added a default-off planner
flag:

```bash
--explore-onboard-clearance-passable --explore-onboard-passable-radius <m>
```

This inflates known blocked cells before onboard BFS/path traversal. It is
planner-only and does not change physics, rewards, or actor observations.

The result was negative but useful. On the current `0.30m` hard-p1 grid, radius
`0.30` is behavior-equivalent to the baseline because it does not exceed one
cell: seed `88`, `256` oracle episodes stayed at `100%` success, coverage
`0.950574`, path `168.063m`, min clearance `0.383617`. Raising the radius just
above one cell (`0.31`/`0.35`) collapsed the oracle to `35.13%` success and
`64.87%` timeout. A path-only variant that left frontier unknown cells
unfiltered still collapsed (`30.47%` success). Combining finer `0.20m` cells
with a `0.25m` passability radius was not enough either: `57.81%` success on a
128-episode oracle smoke, far below the current `0.30m` baseline.

Conclusion: binary passability is the wrong abstraction at this grid scale. It
has no useful middle ground: below one cell it does nothing, above one cell it
closes the doorway graph. The user's intuition is still pointing at the right
failure family, but the next build should not decide "gap passable or not."
It should represent passage quality: doorway centerline / medial-axis targets,
clearance-weighted local waypoints, or a small local passage planner that keeps
valid gaps open while avoiding wall-edge targets.

## 2026-05-24 QDR-311 Passage Quality Centerline Bias

QDR-311 implemented the corrected version of the user's doorway intuition. It
does not mark narrow passages unreachable. Instead it keeps the onboard graph
connected and adds a default-off passage-quality scoring layer:

```bash
--explore-onboard-passage-quality --explore-onboard-passage-weight 0.35
```

The score rewards frontier/viewpoint cells farther from known blocked cells and
penalizes wall-edge cells. If the selected waypoint is still a wall-edge cell,
the existing local viewpoint selector gets a chance to replace it with a more
centered reachable viewpoint.

This passed the local gates strongly. Oracle hard-p1 seed `88`, `256` episodes
kept `100%` success and zero contacts/timeouts, while path dropped from
`168.063m` to `150.994m`, new-cell rate rose from `0.788219` to `0.898366`,
and min clearance slightly improved from `0.383617m` to `0.387858m`. Held-out
oracle seeds `90/91` also stayed `100%`, with paths `143.825m` and `145.804m`.

The learned QDR-300 policy benefited without retraining. Hard-p1 seeds
`88/90/91`, `512` requested episodes, all reached `100%` success with zero
contacts/timeouts. Paths were `148.253m` / `145.983m` / `147.259m`; new-cell
rates were `0.907516` / `0.926014` / `0.917715`. The 32-episode no-render
visual loop reached `32/32`, zero contacts/timeouts, new-cell rate `0.8545`,
and `wall_stall_logs=0` versus the prior `3`. A small hard-layout regression
smoke on p0/p2/p3 seed `88` stayed at `100%` success, zero contacts/timeouts.

Follow-up hard-family validation held. The forced-pattern matrix p0/p1/p2/p3
across seeds `88/90/91` passed all `12/12` cases at `100%` success with zero
contacts/timeouts. The random `procedural_indoor_hard` mix also passed seeds
`88/90/91` at `100%`, zero contacts/timeouts, with coverage `0.9536-0.9542`.

Current best hard-p1 runtime profile:

```bash
PYTHON=../../.venv/bin/python scripts/quadrotor4.sh explore-v0-target-visual experiments/qdr300/checkpoints/qdr300_region_commit_barrier_ft_qdr285_s88_u000100.pt --mode procedural_indoor_hard_p1 --seed 88 --max-steps 2048 --room-size 6.0 --cell-size 0.30 --explore-goal-fraction 0.95 --speed-fraction 0.50 --explore-frontier-target-mode onboard --explore-reachable-coverage --explore-onboard-path-lookahead 2 --explore-onboard-region-commit --explore-onboard-region-commit-stall-ticks 90 --explore-onboard-region-tracker --explore-onboard-passage-quality --explore-onboard-passage-weight 0.35 --safety-barrier-actions --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80 --restart-on-stall-steps 3000
```

Next step: distill/train the target model against this passage-quality target
distribution rather than treating it as an eval-only wrapper forever.

## 2026-05-24 QDR-312 Passage Quality Distillation

QDR-312 took the QDR-311 passage-quality target distribution and trained it
into the deployable target policy instead of leaving it only as a runtime
planner wrapper. The run used:

```bash
scripts/quadrotor4_qdr312_passage_quality_distill.sh
```

The GPU run completed on `beeblebrox`. The experiment script now handles the
current remote build reality explicitly: current PufferLib uses `build.sh`, not
`setup.py`; the remote needs `.venv/bin` and `/usr/local/cuda/bin` on `PATH`;
and the available C compiler is `gcc/g++`, not `clang`.

Training wrote checkpoints under `experiments/qdr312/checkpoints/`. The final
u800 checkpoint was safe but showed late timeout drift as student rollout
probability reached `0.5`. Final in-memory eval was `99.51%` success,
`0.49%` timeout, `0%` contacts. Its 32-episode no-render visual smoke had
one timeout, so final is not promotable.

The useful checkpoint is:

```text
experiments/qdr312/checkpoints/qdr312_passage_quality_distill_s88_u000400.pt
```

Fair hard-p1 drained eval against QDR-300 plus QDR-311 runtime passage quality,
seeds `88/90/91`, `1024` requested episodes:

- QDR-300+QDR-311: success `99.42/99.71/99.22%`, paths
  `149.445/147.239/149.940m`, new-cell `0.899/0.917/0.896`.
- QDR-312 u400: success `99.51/99.71/99.52%`, paths
  `148.552/145.010/148.999m`, new-cell `0.907/0.933/0.907`.
- Both had `0%` obstacle contacts.

The no-render visual smoke is mixed but useful. QDR-312 u400 solved `32/32`
hard-p1 seed `88` episodes with `0%` contacts/timeouts and new-cell rate
`0.9113`, but logged one recoverable near-wall stall. QDR-300+QDR-311 solved
the same smoke with `wall_stall_logs=0` and new-cell rate `0.8569`.

Expanded hard-family u400 regression on p0/p2/p3 seeds `88/90/91` stayed
zero-contact and broadly matched the QDR-300+QDR-311 baseline. Under the larger
drained sample, tiny timeout tails appear in both policies, so the earlier
`100%` hard-family result was partly sample-size optimistic.

Conclusion: QDR-312 is modest progress, not a breakthrough. Passage-quality
distillation can improve efficiency, but overtraining with too much student
rollout causes timeout drift. Treat u400 as the strongest trained checkpoint.
Keep QDR-300+QDR-311 as the cleanest runtime visual reference.

Useful local visual command:

```bash
PYTHON=../../.venv/bin/python scripts/quadrotor4.sh explore-v0-target-visual experiments/qdr312/checkpoints/qdr312_passage_quality_distill_s88_u000400.pt --mode procedural_indoor_hard_p1 --seed 88 --max-steps 2048 --room-size 6.0 --cell-size 0.30 --explore-goal-fraction 0.95 --speed-fraction 0.50 --explore-frontier-target-mode onboard --explore-reachable-coverage --explore-onboard-path-lookahead 2 --explore-onboard-region-commit --explore-onboard-region-commit-stall-ticks 90 --explore-onboard-region-tracker --explore-onboard-passage-quality --explore-onboard-passage-weight 0.35 --safety-barrier-actions --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80 --restart-on-stall-steps 3000
```

Next: stop optimizing this same saturated hard-p1 suite. Run QDR-300+QDR-311
and QDR-312 u400 on a harder layout suite with larger rooms, more bottlenecks,
or randomized multi-room topology. The next useful engineering build should be
driven by the failure pattern there, not by squeezing another tiny hard-p1
percentage point.

## 2026-05-24 QDR-313 Tight Procedural Limit

QDR-313 checked whether the next limit is just more room/horizon or whether the
layout topology itself needs to get harder.

The first probe enlarged hard-p1 to an `8.0m` room with `4096` max steps. That
was not a useful next benchmark. At coverage fractions `0.95` and `0.98`,
QDR-300 plus QDR-311 runtime passage quality stayed `100%` on seeds `88/90/91`.
QDR-312 u400 was also effectively solved: `99.93/100/100%` at `0.95` and
`99.85/100/100%` at `0.98`, with zero contacts and only a tiny seed-88 timeout
tail. Bigger rooms alone mostly give the same planner more time to solve the
same topology.

The useful result is a new benchmark family:

```text
procedural_indoor_tight
procedural_indoor_tight_p0
procedural_indoor_tight_p1
procedural_indoor_tight_p2
procedural_indoor_tight_p3
```

Compared with `procedural_indoor_hard`, tight mode uses narrower passable gaps
(`0.34-0.50m` instead of `0.46-0.64m`) and extra clutter. This finally exposes
the next real failure without turning it into a crash task.

Tight-p1 GPU eval, room `6.0m`, max steps `2048`, goal fraction `0.95`, seeds
`88/90/91`:

- Oracle: success `92.62/91.89/92.22%`, all `0%` contacts.
- QDR-300+QDR-311 runtime passage-quality stack: success
  `93.84/93.42/94.89%`, all `0%` contacts.
- QDR-312 u400: success `94.27/94.61/94.19%`, all `0%` contacts.

This changes the interpretation. The next failure is not low-level flight and
not a student checkpoint being wildly below its teacher. Even the oracle leaves
about `7-8%` safe timeouts on tight-p1, while learned policies slightly exceed
that one-step oracle in aggregate. The blocker is late exploration objective
selection in tight topology: finishing the last few visible/nearby frontier
regions through narrow passages, without wall-edge target churn or excessive
caution.

Current references:

- Clean visual/runtime reference: QDR-300 u100 plus QDR-311 flags.
- Strongest trained checkpoint: `experiments/qdr312/checkpoints/qdr312_passage_quality_distill_s88_u000400.pt`.
- New limit benchmark: `procedural_indoor_tight_p1`.

Next engineering target: improve the oracle/planner ceiling on tight-p1 before
training more students. Good candidates are a small passage graph, doorway
centerline search, late-frontier completion scoring, or topological progress
state over persistent frontier regions. If the oracle does not improve, policy
training is mostly optimizing around a planner ceiling.

## 2026-05-24 QDR-314 Tight-p1 Oracle Timeout Diagnosis

QDR-314 reproduced the new tight-p1 failure in a small no-render oracle visual
loop and tested one concrete hypothesis.

Baseline tight-p1 seed `88`, `32` no-restart episodes, room `6.0m`,
`cell_size=0.30`, max steps `2048`, goal fraction `0.95`, QDR-311 passage
quality and current safety stack:

- success `90.62%`, timeout `9.38%`, contacts `0%`;
- coverage `0.9121`;
- timeout episodes were mid-coverage (`0.4821`, `0.4602`, `0.6260`);
- `wall_stall_logs=101`.

A restart-on-stall visual loop counted `100%` success, but only because the
harness restarted coverage-stalled episodes before the real timeout. Treat that
as a stall exposure tool, not as a success metric.

The tested hypothesis was planner/safety cone mismatch. The C planner had been
scoring candidate directions with a fixed 30 degree cone while the Python safety
layer used `--safety-cone-deg 60`. I added an opt-in
`--explore-onboard-clearance-cone-deg` and wired it through the C env, binding,
checkpoint config, and parser. Rebuilding and rerunning with planner cone
`60deg` regressed the same trace:

- success `84.38%`, timeout `15.62%`, contacts `0%`;
- coverage `0.9011`;
- `wall_stall_logs=106`.

Conclusion: cone mismatch was real but not the missing piece. The oracle is not
failing because the target stream is blind to walls; the debug lines show target
and onboard target agree (`angle_err=0`) and the rays/cone metrics see nearby
geometry. Widening the planner cone mostly makes target selection more
conservative and does not fix stale/late frontier completion.

Next probe: QDR-315 should test the representation scale directly. Tight gaps
are `0.34-0.50m`, while the current onboard grid is `0.30m`; nearby cell sizes
may decide whether narrow gaps are represented as passable doorways, unstable
one-cell targets, or effectively unpassable clutter.

## 2026-05-24 QDR-315 Tight-p1 Grid Scale Breakthrough

QDR-315 tested the representation-scale hypothesis from QDR-314. This is the
first clear progress on the tight-topology plateau.

Oracle sweep on `procedural_indoor_tight_p1`, seed `88`, room `6.0m`, max steps
`2048`, goal fraction `0.95`, QDR-311 passage-quality flags and safety stack:

- `cell_size=0.25`: success `73.33%`, timeout `26.67%`, contacts `0%`;
- `cell_size=0.30`: success `88.63%`, timeout `11.37%`, contacts `0%`;
- `cell_size=0.35`: success `96.63%`, timeout `3.37%`, contacts `0%`;
- `cell_size=0.40`: success `86.15%`, timeout `13.85%`, contacts `0%`.

Held-out oracle at `cell_size=0.35` also passed:

- seed `90`: success `96.89%`, timeout `3.11%`, contacts `0%`;
- seed `91`: success `96.84%`, timeout `3.16%`, contacts `0%`.

This is a good falsifiable result. Tight gaps are `0.34-0.50m`; the old
`0.30m` grid was close enough to represent them, but fine enough to create
one-cell/stale frontier churn. `0.25m` made that worse. `0.40m` hid too much
structure and regressed. `0.35m` is the useful middle: it filters unstable
micro-frontiers without closing the task.

I added `--ignore-checkpoint-env-config` to the target eval and visual harnesses
so saved checkpoints can be evaluated under explicit env overrides. Without
this, QDR-312 silently restores its saved `cell_size=0.30`.

Existing QDR-312 u400 with the `0.35m` override is now the best tight-topology
candidate:

```text
experiments/qdr312/checkpoints/qdr312_passage_quality_distill_s88_u000400.pt
```

Tight-p1 seeds `88/90/91`:

- success `97.51/97.41/97.83%`;
- timeout `2.49/2.59/2.17%`;
- contacts `0/0/0%`.

No-render visual smoke for tight-p1 seed `88` solved `32/32` with no contacts
or timeouts. It still logged `88` near-wall stall lines, so this is not a claim
that local control is perfect; it is a claim that the previous timeout plateau
was largely caused by a bad grid scale for tight topology.

Broader tight-family regression with QDR-312 u400 and `cell_size=0.35`:

- p0 seeds `88/90/91`: `99.62/99.57/99.67%`, contacts `0%`;
- p2 seeds `88/90/91`: `99.80/99.78/99.77%`, contacts `0%`;
- p3 seeds `88/90/91`: `99.93/99.90/99.95%`, contacts `0%`.

QDR-312 eval/visual logs are local under `logs/qdr315_remote/`; the oracle
sweep metrics above are recorded from GPU stdout.

Useful local visual command from the worktree:

```bash
PYTHON=../../.venv/bin/python scripts/quadrotor4.sh explore-v0-target-visual experiments/qdr312/checkpoints/qdr312_passage_quality_distill_s88_u000400.pt --ignore-checkpoint-env-config --mode procedural_indoor_tight_p1 --seed 88 --max-steps 2048 --room-size 6.0 --cell-size 0.35 --explore-goal-fraction 0.95 --speed-fraction 0.50 --explore-frontier-target-mode onboard --explore-reachable-coverage --explore-onboard-path-lookahead 2 --explore-onboard-region-commit --explore-onboard-region-commit-stall-ticks 90 --explore-onboard-region-tracker --explore-onboard-passage-quality --explore-onboard-passage-weight 0.35 --deploy-patch-obs --deploy-onboard-target-obs --safety-barrier-actions --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80
```

Next: treat QDR-312 u400 plus `cell_size=0.35` as the tight-topology baseline,
then push the remaining p1 `2-3%` safe-timeout tail. The next useful test is
either a random `procedural_indoor_tight` mix at `0.35m`, or training/distilling
directly with `0.35m` so the policy is no longer evaluated under an env override.

## 2026-05-24 QDR-316 Direct 0.35m Training Baseline

QDR-316 turned the QDR-315 eval-time grid-scale fix into the checkpoint's own
training distribution.

Important diagnostic: the first QDR-316 GPU run was invalid. It appeared to
train with `--cell-size 0.35`, but init-checkpoint loading restored QDR-312's
saved env config and wrote checkpoints with `cell_size=0.30`. Those invalid
checkpoints landed around `91-92%` tight-p1 success and should not be cited as a
real `0.35m` result. I fixed this by making train-time init loading honor
`--ignore-checkpoint-env-config`, while still loading the model architecture and
auxiliary heads from the checkpoint.

Corrected training command on the GPU box:

```bash
cd ~/PufferLib
QDR316_UPDATES=300 PYTHON=.venv/bin/python scripts/quadrotor4_qdr316_cell035_distill.sh
```

The corrected checkpoints live under:

```text
experiments/qdr316/checkpoints_preserve/
```

The promoted candidate is:

```text
experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt
```

I verified the saved checkpoint config directly:

- `cell_size=0.35`;
- `explore_onboard_passage_quality=True`;
- `explore_onboard_passage_weight=0.35`.

Tight-p1 deterministic drained eval, `2048` steps, goal fraction `0.95`,
speed `0.50`, QDR-311/QDR-315 safety and passage-quality stack:

- seed `88`: success `97.96%`, timeout `2.04%`, contacts `0%`;
- seed `90`: success `98.16%`, timeout `1.84%`, contacts `0%`;
- seed `91`: success `97.24%`, timeout `2.76%`, contacts `0%`.

This is a modest average improvement over QDR-312 u400 with an eval override
(`97.51/97.41/97.83%`) rather than a step-function gain. The more convincing
improvement is qualitative/stability in the visual harness: no-render tight-p1
seed `88`, `32` episodes, no early restart within the episode budget, solved
`32/32` with `0%` contacts/timeouts and only `5` near-wall stall logs. The prior
QDR-312+0.35 smoke also solved `32/32`, but logged `88` near-wall stall lines.

Tight-family regression for the promoted QDR-316 u200 checkpoint stayed safe:

- p0 seeds `90/91`: `99.70/99.74%`, contacts `0%`;
- p2 seeds `90/91`: `99.80/99.85%`, contacts `0%`;
- p3 seeds `90/91`: `99.91/99.93%`, contacts `0%`.

Local visual command from this worktree:

```bash
PYTHON=../../.venv/bin/python scripts/quadrotor4.sh explore-v0-target-visual experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt --mode procedural_indoor_tight_p1 --seed 88 --max-steps 2048 --room-size 6.0 --explore-goal-fraction 0.95 --spawn-radius-min 0.0 --speed-fraction 0.50 --explore-frontier-target-mode onboard --explore-reachable-coverage --explore-onboard-path-lookahead 2 --explore-onboard-region-commit --explore-onboard-region-commit-stall-ticks 90 --explore-onboard-region-tracker --explore-onboard-passage-quality --explore-onboard-passage-weight 0.35 --deploy-patch-obs --deploy-onboard-target-obs --safety-barrier-actions --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80
```

Conclusion: promote QDR-316 u200 as the current tight-topology baseline. It is
not research-SOTA exploration yet, but it is a cleaner baseline than QDR-312
with an env override and removes most visible near-wall stall churn on this
suite. The next useful move is to stop squeezing this exact task and expose a
harder failure again: random tight mixtures, larger connected maps, higher
coverage thresholds, or a completion task where the policy must revisit less and
finish late frontiers reliably.

## 2026-05-24 QDR-317 Procedural House v0

QDR-317 added the first PufferLib-native house exploration generator. This is
inspired by HouseExpo/ProcTHOR-style room topology, but deliberately stays in
the existing fast C backend: axis-aligned wall and furniture boxes, existing
raycasts, existing onboard planner, existing renderer, and deployable actor
observations.

Implemented obstacle modes:

- `procedural_house`
- `procedural_house_p0`
- `procedural_house_p1`
- `procedural_house_p2`
- `procedural_house_p3`

The four templates cover:

- a central hallway with side rooms;
- a long side corridor with room branches;
- a loop/courtyard-style plan;
- an irregular open-plan layout with clutter.

I raised `QR_MAX_OBSTACLES` from `16` to `32` so walls plus clutter do not get
silently dropped. Python mode maps and the gauntlet registry now know about the
new modes. CPU backend build and remote GPU rebuild both passed.

GPU results on `beeblebrox`, room `8.0m`, `cell_size=0.35`, `4096` steps,
QDR-311/QDR-316 passage-quality and safety stack:

```text
procedural_house, seed 88, goal fraction 0.98
oracle:   99.82% success, 0.18% timeout, 0% contacts
QDR-316:  99.84% success, 0.16% timeout, 0% contacts

procedural_house, seed 88, goal fraction 0.99
oracle:   99.68% success, 0.32% timeout, 0% contacts
QDR-316:  99.74% success, 0.26% timeout, 0% contacts
```

Pattern 2 is currently the weakest template. At `0.99`, oracle pattern 2 is
`98.87%`, and QDR-316 pattern 2 is `99.05%`. Logs are local under:

```text
logs/qdr317_remote/
```

Visual command from this worktree:

```bash
PYTHON=../../.venv/bin/python scripts/quadrotor4.sh explore-v0-target-visual experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt --mode procedural_house --seed 88 --max-steps 4096 --room-size 8.0 --cell-size 0.35 --explore-goal-fraction 0.99 --spawn-radius-min 0.0 --speed-fraction 0.50 --explore-frontier-target-mode onboard --explore-reachable-coverage --explore-onboard-path-lookahead 2 --explore-onboard-region-commit --explore-onboard-region-commit-stall-ticks 90 --explore-onboard-region-tracker --explore-onboard-passage-quality --explore-onboard-passage-weight 0.35 --deploy-patch-obs --deploy-onboard-target-obs --safety-barrier-actions --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80
```

Conclusion: QDR-317 is good infrastructure, not yet a harder learning limit.
The house generator is reachable, fast, visible in the existing renderer, and
compatible with the deployable stack, but QDR-316 already solves it. Do not
spend GPU training on house v0. The next build should make the house benchmark
qualitatively harder: multi-zone houses, floor-to-floor connectors/stairwell or
shaft volumes, explicit room-discovery metrics, and late-frontier completion
pressure that the current planner cannot solve by the same 2D pattern.

## 2026-05-24 QDR-318 to QDR-322 House Scale and Fast-Exploration Limit

QDR-318 built the next house generator rung: `procedural_house_multizone` plus
forced templates `procedural_house_multizone_p0..p3`. The first draft was
rightfully rejected because the oracle ceiling was too low, especially p0. The
fixed version widened doors, removed the malformed p0 central wall, and reduced
clutter/snake density.

At room `8.0m`, `cell_size=0.35`, `4096` steps, and goal `0.99`, QDR-316 u200
still reaches `99.03%` success with `0%` contacts. A no-render visual smoke
over `32` episodes reached `100%` success and `0%` contacts/timeouts. QDR-318
is useful infrastructure, not a training target.

QDR-319 then scaled the same topology:

```text
10m, cell 0.35, 4096 steps: oracle 100%, QDR-316 100%, 0% contacts
10m, cell 0.35, 3072 steps: oracle 100%, QDR-316 100%, 0% contacts
12m, cell 0.40, 4096 steps: oracle 100%, QDR-316 100%, 0% contacts
```

This saturated pure single-floor 2D coverage again. The policy is not below the
oracle on these cases; the metric is too forgiving when the horizon is loose.

QDR-320 finally exposed a useful limit by keeping the `12m` multizone house but
tightening the horizon to `1200` steps at goal `0.99`:

```text
speed 0.50: oracle 26.53%, QDR-316 26.11%, 0% contacts
speed 0.70: oracle 40.96%, QDR-316 40.24%, ~0.2-0.3% contacts
speed 0.90: oracle 44.42%, QDR-316 41.14%, ~27-34% contacts
```

QDR-321 tested a simple global clearance schedule at speed `0.70`
(`safety_slow_ray=1.20`, cone stops `0.30/0.35`). It removed nearly all
contacts but reduced success to about `36%`, giving back too much speed.

QDR-322 tested the existing deployable short-horizon trajectory wrapper. It did
not solve the tradeoff. At speed `0.70`, QDR-316 fell to `38.96%` success with
`0.416%` contacts, worse than the simpler QDR-320 speed-`0.70` profile. At
speed `0.90`, contacts remained very high (`31.60%`). The wrapper is not the
right adaptive-speed implementation in its current form.

Current interpretation:

- QDR-316 plus onboard planner is now very strong at eventual single-floor
  coverage.
- The next honest limit is fast exploration: reach high coverage quickly while
  preserving zero contacts.
- The bottleneck is still mostly the deployable planner/controller profile,
  because QDR-316 tracks the oracle closely.
- Do not train a student on QDR-318/QDR-319. Use QDR-320 as the next pressure
  benchmark and improve adaptive speed/local control around narrow passages.

Current fast-house command family:

```bash
bash scripts/quadrotor4_qdr320_speed_probe.sh
bash scripts/quadrotor4_qdr321_clearance_schedule_probe.sh
```

Recommended next build: collect/diagnose QDR-320/QDR-322 contact states,
especially p3, and build a passage-aware speed target or learned local
controller from those states. The target is speed-`0.70` coverage with
speed-`0.50` safety, not speed-`0.90` contact-heavy behavior.

## 2026-05-24 QDR-323 to QDR-324 Fast-House Failure Diagnosis

QDR-323 forced the worst current template, `procedural_house_multizone_p3`, and
compared QDR-316 against the privileged oracle controller at speed `0.70` and
`0.90`.

The important crash-state signal was not target hallucination:

- target disagreement stayed small, about `5-9deg`;
- target and action rays were usually long, about `3.1-3.5m`;
- the episode min ray collapsed to about `0.18-0.19m`;
- the oracle also crashed badly at speed `0.90`.

That means the current target stream is mostly sensible. The failure is local
side-clearance and passage dynamics under speed, not the student pointing at
obvious walls.

QDR-324 added a deployable lidar-only min-ray speed cap. It is applied after
target, barrier, and slide action selection. It is intentionally simple:
scale XY speed down when any ray drops between a slow threshold and a stop
threshold. New flags:

```bash
--safety-min-ray-speed-cap
--safety-min-ray-speed-stop 0.18
--safety-min-ray-speed-slow 0.45
```

GPU result, room `12m`, `cell_size=0.40`, goal `0.99`, max steps `1200`:

```text
p3 speed 0.70, cap 0.18/0.45
oracle:   5.37% success, 0% contacts
QDR-316:  4.57% success, 0% contacts

p3 speed 0.90, cap 0.18/0.45
oracle:   7.47% success, 14.36% contacts
QDR-316:  5.57% success, 30.67% contacts

p3 speed 0.90, cap 0.20/0.60
oracle:   5.14% success, 6.28% contacts
QDR-316:  4.79% success, 20.99% contacts

aggregate speed 0.70, cap 0.18/0.45
oracle:   42.53% success, 0% contacts
QDR-316:  41.85% success, 0.021% contacts, 87.99% coverage
```

Compared with QDR-320 aggregate speed `0.70`, the cap is a small net positive:
QDR-316 goes from `40.24%` success / `0.231%` contacts to `41.85%` /
`0.021%`. But p3 remains the blocker. At high speed the cap only trades
contacts for timeouts.

Current recommendation: keep QDR-324's light min-ray cap as the default
speed-`0.70` fast-house safety profile, but do not treat it as the solution.
The next engineering build should be a passage-aware local controller: choose
speed and steering based on corridor/doorway geometry and side-clearance trend,
not just a global min-ray scalar.

## 2026-05-24 QDR-325 Passage-Centering Rejected

QDR-325 tested the obvious next idea after QDR-324: use left/right lidar
imbalance to steer away from the closer side wall while keeping target progress.
This was implemented as an opt-in post-processor:

```bash
--safety-passage-center-actions
--safety-passage-center-near 0.50
--safety-passage-center-front 0.60
--safety-passage-center-gain 0.85
--safety-passage-center-max-turn-deg 35
```

It failed the forced p3 gate:

```text
p3 speed 0.70, centering only
oracle:   4.79% success, 13.17% contacts
QDR-316:  4.73% success, 14.94% contacts

p3 speed 0.70, centering + QDR-324 cap
oracle:   4.74% success, 1.22% contacts
QDR-316:  4.64% success, 1.88% contacts

p3 speed 0.90, centering + QDR-324 cap
oracle:   5.17% success, 40.87% contacts
```

The run was stopped before aggregate evaluation because the p3 gate was already
worse than QDR-324. Logs are local under:

```text
logs/qdr325_remote/
```

Lesson: naive centering is not a passage controller. Local left/right imbalance
is not enough because the controller can oversteer in cluttered or angled
passages. The next serious build needs temporal/local-planning structure:
evaluate a short sequence of candidate velocities against ray geometry, or
train a local hard-state controller from p3 passage states. Do not spend more
time on stateless side-imbalance steering.

## 2026-05-24 QDR-326 Existing Resettable MPC Rejected

Before building new passage-recovery infrastructure, QDR-326 checked the
existing resettable hard-state probe:

```bash
scripts/quadrotor4.sh explore-v0-onboard-hardstate-probe
```

This probe already captures onboard wall-stall snapshots with `vec.snapshot`
and replays them under the baseline target controller versus a constant-action
MPC-style search. I configured it for the current p3 fast-house failure:
`procedural_house_multizone_p3`, room `12m`, `cell_size=0.40`, goal `0.99`,
max steps `1200`, speed `0.70`, and the QDR-324 light min-ray cap.

Local smoke, `8` p3 stall states:

```text
baseline: gain 0.1808, 0 success, 0 crash
MPC:      gain 0.1140, 0 success, 0 crash
```

GPU gate, `16` p3 stall states:

```text
baseline_replay: 0/16 success, 0/16 crash, gain 0.1516, min clearance 0.3574
mpc_replay:      0/16 success, 0/16 crash, gain 0.1401, min clearance 0.3716
```

The existing MPC gets slightly more clearance but less progress and no
recoveries. It is not worth tuning as the main path.

Updated interpretation: "compact local MPC" as previously phrased is already
too close to a saturated family. The unsaturated version would have to be a
new closed-loop resettable passage task, where a learned local skill is trained
against full rollout outcomes from p3 snapshots. That is materially different
from choosing one constant action by scalar-weight search.

## 2026-05-24 QDR-327 to QDR-330: p3 Is Mostly Slow, Not Unrecoverable

QDR-327 built the first closed-loop resettable passage-skill gate for the
current p3 line. The harness is useful, but the existing recovery-PG sidecar is
not:

```text
64 p3 hard states, speed 0.70, QDR-324 cap
base replay:      gain 0.1534, 0 crashes
stateful replay:  gain 0.1346, 0 crashes
trained sidecar:  gain 0.0737, 2/64 crashes
```

Do not repeat that recovery-PG configuration. It learns a worse local policy.

QDR-328 then tested the more important question: are these p3 snapshots truly
stuck? On the same state distribution, base replay kept gaining coverage as the
horizon increased:

```text
base h256:  best gain 0.1534, 0 crashes
base h512:  best gain 0.2946, 0 crashes
base h1024: best gain 0.5176, 0 crashes, 2/64 success, mostly timeouts
```

That changed the diagnosis. p3 is not mainly an unrecoverable escape problem;
it is a slow-passage problem under a tight `1200`-step fast-exploration budget.

QDR-329 tested the first planner-level response: increase onboard path
lookahead from the long-standing value `2`.

Forced p3, speed `0.70`, cap `0.18/0.45`:

```text
lookahead 2, QDR-316:  4.54% success, 0% contacts,     80.20% coverage
lookahead 4, QDR-316: 13.82% success, 0.098% contacts, 86.06% coverage
lookahead 6, QDR-316:  8.74% success, 0.195% contacts, 81.56% coverage
```

Aggregate multizone:

```text
lookahead 4, QDR-316: 42.23% success, 0.138% contacts, 88.00% coverage
lookahead 6, QDR-316: 19.48% success, 0.936% contacts, 75.21% coverage
```

This is the first real progress lever after QDR-324. The missing piece was
partly target horizon: lookahead `2` is too local for p3 passage traversal.
Lookahead `6` is too far and destabilizes routing. Lookahead `4` is the sweet
spot so far.

QDR-330 tried to retune safety around lookahead `4`:

```text
cap 0.20/0.50 p3, QDR-316:       16.55% success, 0.049% contacts
cap 0.20/0.50 aggregate, QDR-316: 40.20% success, 0.139% contacts
cap 0.20/0.60 p3, QDR-316:       14.11% success, 0% contacts
cap 0.20/0.60 aggregate, QDR-316: 37.26% success, 0.185% contacts
```

Do not promote QDR-330 over QDR-329. Stronger caps help p3 but hurt aggregate
throughput and do not cleanly lower aggregate contacts. The current best
progress profile is QDR-329: lookahead `4`, speed `0.70`, cap `0.18/0.45`.
It is not a final promotion because contacts are higher than QDR-324; it is the
right next branch because it attacks the actual p3 bottleneck.

QDR-331 made that lookahead adaptive instead of fixed. The selector uses
lookahead `4` only when the deployable local geometry looks clear and straight,
otherwise it falls back to lookahead `2`.

GPU gate, speed `0.70`, cap `0.18/0.45`, QDR-316:

```text
balanced adaptive p3:        16.85% success, 0% contacts, 86.86% coverage
balanced adaptive aggregate: 45.12% success, 0% contacts, 89.76% coverage
strict adaptive p3:          14.16% success, 0% contacts, 86.15% coverage
strict adaptive aggregate:   43.37% success, 0% contacts, 89.42% coverage
```

This promotes the balanced adaptive profile:

```text
--explore-onboard-adaptive-lookahead
--explore-onboard-adaptive-lookahead-min 2
--explore-onboard-adaptive-lookahead-max 4
--explore-onboard-adaptive-clearance 0.55
--explore-onboard-adaptive-turn-dot 0.55
```

Reproduce the gate:

```bash
QDR331_PROFILES=balanced bash scripts/quadrotor4_qdr331_adaptive_lookahead_probe.sh
```

Visual eval:

```bash
scripts/quadrotor4_qdr331_visual.sh
```

The missing piece was not more global caution. It was choosing a longer target
horizon only when the corridor geometry supports it. This keeps the p3 progress
gain from lookahead `4`, beats the QDR-324/QDR-329 aggregate result, and removes
the QDR-329 contact tail. The next harder limit is now timeout-heavy completion
on p3 and late-house layouts under the `1200`-step budget.

QDR-332 tested whether QDR-331 created speed headroom. It did, but the window
is narrow.

Forced p3, QDR-316:

```text
speed 0.72: 16.36% success, 0% contacts,     87.03% coverage
speed 0.75: 19.38% success, 0.146% contacts, 87.45% coverage
speed 0.80: 21.44% success, 1.172% contacts, 87.41% coverage
speed 0.85: 21.08% success, 7.897% contacts, 85.53% coverage
speed 0.90: 19.32% success, 23.81% contacts, 80.11% coverage
```

Aggregate, QDR-316:

```text
speed 0.72: 47.91% success, 0.046% contacts, 90.38% coverage
speed 0.75: 48.54% success, 0.182% contacts, 90.47% coverage
```

Interpretation: speed `0.72` is the conservative pushed profile. It improves
aggregate success over QDR-331 by about `+2.8` points while keeping contacts to
one event in about `2177` episodes. Speed `0.75` is a useful pressure test but
not clean enough to replace the safety baseline. Do not treat raw speed as the
next solution above `0.75`; the old side-clearance cliff returns fast even with
adaptive lookahead.

Reproduce:

```bash
QDR332_SPEEDS=0.72 QDR332_MODES=p3,aggregate bash scripts/quadrotor4_qdr332_adaptive_speed_headroom.sh
```

Visual eval:

```bash
scripts/quadrotor4_qdr332_visual.sh
```

QDR-333 tested whether the speed `0.75` contact tail could be removed with a
deployable passage-risk speed cap. It helped the oracle but did not transfer to
the fixed QDR-316 student.

Forced p3, speed `0.75`:

```text
balanced oracle: 21.19% success, 0.049% contacts, 88.11% coverage
balanced QDR-316: 16.70% success, 0.146% contacts, 86.49% coverage
wide oracle:     19.97% success, 0% contacts,     88.00% coverage
wide QDR-316:    18.95% success, 0.244% contacts, 86.56% coverage
late oracle:     22.75% success, 0.049% contacts, 88.83% coverage
late QDR-316:    18.60% success, 0.244% contacts, 86.65% coverage
```

Do not promote QDR-333. The oracle gain says the passage-risk idea is not
nonsense, but the student result says post-hoc target/action geometry is not
the missing system-level piece.

QDR-334 tested a velocity-aware momentum cap. It was a no-op: mild, balanced,
and strict profiles reproduced the QDR-332 speed `0.75` numbers exactly.

```text
oracle: 22.95% success, 0% contacts,     88.94% coverage
QDR-316: 19.38% success, 0.146% contacts, 87.45% coverage
```

Do not tune this branch without activation diagnostics. The current predicate
does not activate on the failure states that matter.

QDR-335 tested the next more meaningful explanation: maybe QDR-316 was simply
stale relative to the pushed/adaptive house planner. It fine-tuned QDR-316 for
`200` updates on procedural house p3/aggregate at train speed `0.75`, with
anchor loss on older indoor/layout modes and hard-state weighting for p3/house
bad-angle, bad-clearance, and low-speed samples.

GPU eval:

```text
speed 0.72 p3:        17.48% success, 0% contacts,     86.68% coverage
speed 0.72 aggregate: 45.02% success, 0% contacts,     89.57% coverage
speed 0.75 p3:        17.92% success, 0.049% contacts, 87.18% coverage
speed 0.75 aggregate: 47.97% success, 0.137% contacts, 90.36% coverage
```

Do not promote QDR-335 over QDR-332. It slightly improves conservative p3
success, but loses aggregate success; at speed `0.75` it mostly trades contacts
for throughput and does not beat QDR-332. This falsifies the strongest simple
"stale target student" explanation. The remaining high-probability blocker is
still time-efficient passage traversal in the planner/controller, not just
distillation distribution.

Reproduce:

```bash
QDR335_UPDATES=200 QDR335_TOTAL_AGENTS=2048 QDR335_EVAL_EPISODES=2048 bash scripts/quadrotor4_qdr335_pushed_target_adapt.sh
```

## 2026-05-24 QDR-336 to QDR-338: p3 Is a Budgeted Coverage-Rate Problem

QDR-336 did the duplicate-safe corridor-width-aware local selector. This was
not the same as the earlier naive side-centering branch: it explicitly scored
candidate forward actions by side clearance, local corridor width, and target
progress. It failed as a promotion.

Forced p3, speed `0.75`:

```text
light oracle:     21.88% success, 0.049% contacts, 88.92% coverage
light QDR-316:    18.46% success, 0.098% contacts, 87.16% coverage
balanced oracle:  21.97% success, 0.049% contacts, 88.85% coverage
balanced QDR-316: 18.65% success, 0.244% contacts, 86.78% coverage
strong oracle:    20.75% success, 0% contacts,     88.14% coverage
strong QDR-316:   17.68% success, 0.195% contacts, 86.60% coverage
```

Do not tune QDR-336 profile constants next. It is another local
safety/throughput tradeoff and does not beat the QDR-332 pushed profile.

QDR-337 was the decisive diagnostic. It swept the full p3 eval horizon for the
current QDR-331/QDR-332 adaptive-lookahead profile.

Forced p3, speed `0.72`:

```text
horizon 1200 oracle:  20.07% success, 0.049% contacts, 88.77% coverage
horizon 1200 QDR-316: 16.36% success, 0% contacts,     87.03% coverage
horizon 1600 oracle:  69.30% success, 0.049% contacts, 96.75% coverage
horizon 1600 QDR-316: 58.94% success, 0% contacts,     95.40% coverage
horizon 2048 oracle:  94.36% success, 0.048% contacts, 98.63% coverage
horizon 2048 QDR-316: 90.52% success, 0% contacts,     98.36% coverage
```

This explains the recent wall. The p3 controller/planner can traverse the map
safely when given enough time. The `1200`-step gate is a severe coverage-rate
stress test, not mainly a doorway escape or collision-control failure. Treat
`2048` as the solvability sanity check, `1600` as the near-term benchmark, and
`1200` as the stress gate.

QDR-338 tested the most obvious response: adaptive max-6 lookahead under
stricter clear/straight geometry. It also failed.

Forced p3, speed `0.72`:

```text
strict oracle:      15.43% success, 0.098% contacts, 86.91% coverage
strict QDR-316:     13.43% success, 0% contacts,     85.12% coverage
very_strict oracle: 12.45% success, 0% contacts,     85.90% coverage
very_strict QDR-316:12.65% success, 0.049% contacts, 83.75% coverage
ultra oracle:       11.04% success, 0% contacts,     84.75% coverage
ultra QDR-316:       9.22% success, 0.098% contacts, 82.31% coverage
```

Longer target horizon alone is not the solution. The adaptive max-6 target
rate stayed high enough to reduce target quality and new-cell rate. Keep the
QDR-331 max-4 adaptive lookahead as the default target-horizon profile.

Current next branch: planner-level exploration efficiency. The next build
should improve frontier ordering and target commitment by expected new-cell
gain per path/time, reduce target churn, and prefer high-gain regions under
tight budget. This is different from more local velocity safety. A good QDR-339
candidate is a budget-aware frontier scorer: cluster frontiers, estimate
reachable new-cell gain and travel cost, and choose targets by expected
coverage-rate rather than nearest shallow frontier.

## 2026-05-25 QDR-339 to QDR-342: Privileged Unexplored Is The Ceiling

QDR-339 tested the obvious deployable planner-level response from QDR-338:
choose onboard frontier clusters by estimated unknown gain per path depth.
This failed badly.

Forced p3, speed `0.72`, horizon `1200`:

```text
local oracle:       2.69% success, 77.26% coverage, 0% contacts
local QDR-316:      2.10% success, 75.59% coverage, 0% contacts
balanced oracle:    1.76% success, 75.73% coverage, 0% contacts
balanced QDR-316:   1.61% success, 74.32% coverage, 0% contacts
regional oracle:    1.12% success, 74.17% coverage, 0.049% contacts
regional QDR-316:   0.98% success, 72.79% coverage, 0.049% contacts
```

Do not tune that scorer. It walked about the same path length as baseline but
reduced new-cell rate and target switching. The old local frontier churn was
doing useful work.

QDR-340 then ran the ceiling diagnostic that we should have run before more
scorer tuning: compare current onboard target selection to privileged target
selection on the same p3 budget.

Forced p3, speed `0.72`:

```text
horizon 1200 onboard oracle:      20.07% success, 88.77% coverage, 0.049% contacts
horizon 1200 privileged region:   57.42% success, 97.96% coverage, 0.146% contacts
horizon 1200 privileged unexplored:98.34% success, 98.34% coverage, 0.098% contacts
horizon 1600 onboard oracle:      69.30% success, 96.75% coverage, 0.049% contacts
horizon 1600 privileged region:   97.37% success, 98.38% coverage, 0.146% contacts
horizon 1600 privileged unexplored:99.91% success, 98.96% coverage, 0.095% contacts
```

This is the important result: the p3 stress gate is solvable. The bottleneck is
our deployable onboard target abstraction, not the flight controller or the
episode budget itself. The surprising winner is simple privileged
nearest-unexplored BFS, not the heavier region scorer.

QDR-341 tested the naive deployable approximation: allow onboard BFS to plan
through any unknown cell unless lidar has already marked it blocked/stale. It
failed decisively.

```text
horizon 1200 oracle:  0% success, 26.95% coverage, 0% contacts
horizon 1200 QDR-316: 0% success, 27.31% coverage, 0.049% contacts
horizon 1600 oracle:  0% success, 27.32% coverage, 0% contacts
horizon 1600 QDR-316: 0% success, 27.64% coverage, 0.146% contacts
```

Do not use naive unknown traversal. It selects low-value adjacent unknowns or
impossible unknown corridors and collapses the new-cell rate. The next serious
direction is not another hand-tuned frontier score; it is privileged-label
learning or a constrained topology/visibility planner.

QDR-342 is the current run: train a recurrent target policy against the
privileged unexplored teacher while keeping actor inputs deployable. Privileged
map truth is used only to generate labels and teacher rollout actions. The
deployed policy gets state/history, rays, coverage, and local patch observation,
not privileged target/map slots.

Result: QDR-342 failed as a first draft. It trained from zero for `250`
updates with teacher rollout updates `60` and max student rollout probability
`0.45`. The training HUD looked strong under mixed teacher/student control, but
deterministic all-student eval collapsed:

```text
final p3:       0% success, 28.56% coverage, 0.83% contacts
final aggregate:3.50% success, 26.43% coverage, 0.43% contacts
u50 p3:         0% success, 20.06% coverage, 0.20% contacts
u100 p3:        0% success, 23.94% coverage, 0.049% contacts
u150 p3:        0% success, 16.68% coverage, 0.098% contacts
u200 p3:        0% success, 28.96% coverage, 0.29% contacts
```

QDR-343 tested the immediate hypothesis: maybe QDR-342 simply had too little
student-controlled data. It initialized from QDR-342 u200, shortened teacher
rollout to `20` updates, raised max student rollout probability to `1.0`, and
used seq len `128`.

That also failed:

```text
final p3:       0% success, 30.97% coverage, 3.90% contacts
final aggregate:4.19% success, 36.94% coverage, 2.45% contacts
u50 p3:         0% success, 18.24% coverage, 0.20% contacts
u100 p3:        0% success, 29.06% coverage, 0.39% contacts
u150 p3:        0% success, 31.21% coverage, 1.95% contacts
```

This is a strong negative for the current privileged-label target-vector
policy. The privileged unexplored teacher remains the right ceiling, but a
small recurrent target predictor with local rays/patch/coverage does not learn
closed-loop search. More DAgger schedule tuning is unlikely to be the main
answer. The next build needs either a richer deployable map/memory state, an
explicit topology/search module, or auxiliary training that teaches spatial
state directly rather than only a four-value target vector.

QDR-344 implements the first topology/search probe in the deployable planner:
an opt-in frontier graph selector. It still paths only through known passable
cells, but clusters reachable frontier regions and scores each by persistent
region identity plus the onboard-estimated unknown component behind it. This is
intended to test whether the QDR-340 gap is mainly missing spatial state and
frontier-region memory rather than another local safety or DAgger schedule
problem.

Local validation passed:

```text
scripts/quadrotor4.sh build-cpu
QDR344_BUILD_ARGS=--cpu QDR344_EPISODES=4 QDR344_TOTAL_AGENTS=4 QDR344_THREADS=1 QDR344_BUFFERS=1 QDR344_HORIZONS=64 scripts/quadrotor4_qdr344_frontier_graph_probe.sh
```

The smoke run confirmed the new log metrics are emitted:
`frontier_graph_nodes`, `frontier_graph_unknown_cells`,
`frontier_graph_best_unknown_cells`, `frontier_graph_best_depth`,
`frontier_graph_best_score`, and `frontier_graph_active_rate`.

GPU eval completed on `beeblebrox` with logs pulled to `logs/qdr344_remote/`.

Forced p3, speed `0.72`, 2048 episodes:

```text
horizon 1200 graph oracle:  29.54% success, 91.49% coverage, 0% contacts
horizon 1200 QDR-316 graph: 24.76% success, 88.43% coverage, 0.049% contacts
horizon 1600 graph oracle:  77.20% success, 97.92% coverage, 0% contacts
horizon 1600 QDR-316 graph: 66.13% success, 96.64% coverage, 0.049% contacts
```

This is real progress over the old onboard selector:

```text
horizon 1200 old onboard oracle: 20.07% success, 88.77% coverage
horizon 1200 old QDR-316:        16.36% success, 87.03% coverage
horizon 1600 old onboard oracle: 69.30% success, 96.75% coverage
horizon 1600 old QDR-316:        58.94% success, 95.40% coverage
```

It is not the missing SOTA-style capability. QDR-340 privileged unexplored is
still far ahead (`98.34%` at horizon `1200`, `99.91%` at `1600`). The graph
selector proves persistent spatial state helps, but unknown-component scoring
over reachable frontiers only closes a small fraction of the gap.

QDR-345 ablated the graph selector:

```text
horizon 1200 default oracle:        29.54% success, 91.49% coverage, 0% contacts
horizon 1200 default QDR-316:       24.76% success, 88.43% coverage, 0.049% contacts
horizon 1200 no_gain oracle:         8.01% success, 82.65% coverage, 0% contacts
horizon 1200 no_gain QDR-316:        7.28% success, 80.47% coverage, 0.098% contacts
horizon 1200 no_persistence oracle: 38.67% success, 92.79% coverage, 0% contacts
horizon 1200 no_persistence QDR-316:31.54% success, 89.64% coverage, 0% contacts
horizon 1200 score_only oracle:     10.25% success, 82.42% coverage, 0% contacts
horizon 1200 score_only QDR-316:     8.74% success, 80.98% coverage, 0.049% contacts
horizon 1600 no_persistence oracle: 82.91% success, 98.25% coverage, 0% contacts
horizon 1600 no_persistence QDR-316:75.16% success, 97.29% coverage, 0% contacts
```

This is a clean result. Unknown-component gain is necessary; removing it
collapses below the old onboard baseline. The persistent active/attempt/failure
penalties hurt, probably because frontier IDs are not stable enough in p3 and
the planner over-commits to stale identities. The immediate candidate profile is
`no_persistence`: keep unknown-component gain, remove sticky persistence
penalties.

QDR-346 checked whether `no_persistence` generalizes beyond forced p3 to the
broader `procedural_house_multizone` aggregate.

```text
aggregate h1200 oracle:  56.41% success, 95.19% coverage, 0% contacts
aggregate h1200 QDR-316: 50.82% success, 93.14% coverage, 0.091% contacts
aggregate h1600 oracle:  83.64% success, 98.50% coverage, 0% contacts
aggregate h1600 QDR-316: 78.78% success, 97.73% coverage, 0.083% contacts
```

This is enough to promote `graph_no_persistence` as the current deployable
planner baseline. The contact tail is tiny and isolated mostly to p3 in the
student, while success and coverage improve. The result is still not close to
privileged unexplored, so the next build should move the graph state into the
policy observation/training interface instead of continuing frontier scalar
tuning.

QDR-347 moved the graph state into the actor observation in the smallest ABI-safe
way: the final 32 floats of the existing exploration patch tail can now hold
four graph frontier candidates, each with body direction, normalized path depth,
unknown-component gain, cluster size, passage quality, selected marker, and
exists marker. `--deploy-graph-candidate-obs` preserves that tail while zeroing
the rest of the patch, so the actor input remains deployable and compact.

This did not solve closed-loop learning. GPU from-zero training was stopped at
update `150` after deterministic evals reproduced the old imitation failure:
u100 and u150 both had `0%` p3 success, with u150 at `30.57%` coverage and
`99.80%` timeout. The important diagnostic is sharper than QDR-342: target
existence is no longer the main collapse. On u150 timeout episodes the student
and oracle both report target existence, but the student target is wrong by
about `116deg`; it points into much worse local clearance (`pred_target_ray`
about `0.58m` versus oracle `3.54m`) and drives at only `0.055` action speed.
The privileged oracle under the same profile still gets `98.25%` success.

The lesson is that passive candidate features are not enough when the loss is
still only a four-value target vector. The policy is not being forced to learn
which candidate is useful. The next build should make candidate selection an
explicit supervised or RL decision: a discrete candidate-index/scoring head,
candidate-progress auxiliary labels from the privileged unexplored teacher, or
a candidate-selection policy trained with privileged critic/reward. Also cache
candidate computation before larger training runs; recomputing the graph inside
observation packing made QDR-347 rollout noticeably slower.

Agile-flight reset: AFL-011 through AFL-015 resolved the apparent regression
from native `ocean/drone`. AFL-011 and AFL-012 failed even after matching
route-hover semantics and adding the native coarse/fine target observation:
both had `0%` success because they still used the conservative route PPO setup
and a route step cost inherited from earlier NaN-prone waypoint tests.

AFL-013 changed only the training/reward recipe to the successful AFL-008
native-drone settings: high learning rate, shorter gamma, replay ratio, smaller
clip, larger value weight, low entropy, `action_std=1.0`, `route_step_cost=0`,
and `route_target_obs_mode=3`. That immediately recovered route-mode reach-hover
to `99.41%` success over 4096 fresh eval episodes.

AFL-014 then solved the one-level waypoint rung at `100%` success, and AFL-015
solved the original open 3-waypoint route at `98.37%` success. AFL-016 added a
controlled-arrival mode that only accepts waypoints below `2.0m/s` and penalizes
excess speed in the arrival shell. That run reached `100%` fresh-eval success,
`0%` contacts, `final_vel=1.85m/s`, and `avg_vel=2.80m/s`.

This means the custom env can now reproduce native-style direct motor flight
and chained open-space waypoint pursuit with controlled arrival. The next limit
is no longer "can it learn to fly"; it is safety and route quality under
geometry: gates, simple obstacles, path efficiency, and action saturation. Next
tests should add simple gates/obstacles on top of AFL-016 rather than returning
to conservative PPO tuning.

AFL-017 through AFL-019 added the first creative obstacle rung. Code now has a
native `gate_slalom` obstacle family (`obstacle_mode=33`) with forced variants
`34..37`, plus optional clearance shaping inside the hover-like route reward.
The mixed family was too large a jump: AFL-017 reached only `35.0%` success with
`53.7%` obstacle hits. Forced-pattern eval showed the only promising variant was
P1 pillar slalom (`48.6%` success / `50.6%` obstacle from the mixed checkpoint).

AFL-018 trained P1 only and improved to `53.7%` success, but still had `45.9%`
obstacle hits. AFL-019 then increased clearance shaping to radius `0.80` and
scale `0.60`. That solved contacts on P1: at the original 1024-step horizon it
gets `76.6%` success, `0%` obstacle hits, and `23.1%` timeouts; with a
1536-step horizon the same checkpoint gets `99.7%` success, `0%` obstacle hits,
and no timeouts. Current safe obstacle checkpoint:

```text
experiments/agile/afl019/reference_waypoint_pillar_slalom_clearance/quadrotor/1781033504492/0000000041353216.bin
```

The next limit is not basic obstacle avoidance. It is speed/efficiency under
clearance pressure and transfer from pillar slalom to real gate passages. Do not
go back to mixed gate walls until the safe P1 policy is faster under a
scientifically chosen horizon or curriculum.

Current agile-flight iteration: AFL-023 added a clearance-gated target-speed
bonus on top of AFL-019. It failed to promote. The 1024-step result was
`75.5%` success with small obstacle/ground contacts, and the 1536-step
regression was only `90.4%` success. AFL-019 remains the safe obstacle baseline.
Do not spend more time on small P1 speed-shaping variants unless the task
structure changes.

Remote GPU note: Liftarit hosts (`beeblebrox`, `effrafax`) became unreachable
from `lyta` with `No route to host`. The working fallback is Paniikki
`whitespace` through the existing `whitespace-aalto` alias. It is an RTX 3080
10GB host, slower/weaker than the Liftarit RTX 5070 machines, but it has CUDA
and PyTorch working and successfully evaluated AFL-023.

Next experiment reserved: AFL-024 `reference_waypoint_easy_gate_curriculum`.
It adds `gate_slalom_easy_mix` (`obstacle_mode=38`), a narrow from-zero
geometry curriculum that samples P1/P0/P3 as 70/20/10. It deliberately excludes
P2 cross-gates until the policy can preserve AFL-019-like safety on easier gate
geometry.

AFL-024 early diagnosis: the 70/20/10 mix was still too hard. Aggregate
success was about `49%` with mostly timeouts, and forced eval showed P1
regressed to about `69%`, P0 to about `20%`, P3 to about `2%`, and P2 to about
`23%`. AFL-025 then narrowed this to `gate_slalom_p1_p0_mix`
(`obstacle_mode=39`), 90% P1 and 10% P0 only. It is partial progress, not a
promotion. Paniikki `whitespace` run
`experiments/agile/afl025/reference_waypoint_p1_gate_intro/quadrotor/1781105729229`
reached `75.81%` mixed success at 1024 steps with `23.85%` timeout and only
tiny contacts, best checkpoint `0000000041353216.bin`. Forced-pattern eval:
P1 `78.72%` success, P0 `22.16%`, P3 `10.77%`, P2 `27.30%`. This proves the
strict first rung can recover P1 from zero, but 10% P0 exposure does not teach
wall-gate passage.

AFL-020 through AFL-022 tested the obvious speed-pressure levers and rejected
them. AFL-020 trained the safe P1 pillar slalom at `1536` steps with
`route_step_cost=0.001`; it reached only `68.46%` success, `30.77%` timeout,
and `0.49%` obstacle hits. AFL-021 removed the step cost but kept the `1536`
training horizon; it improved only to `74.57%` success with `24.45%` timeout.
AFL-022 put the tiny step cost back onto the original `1024` training horizon;
it fell to `66.28%` success with `33.54%` timeout.

This is a useful negative result. AFL-019 remains the best obstacle-flight
checkpoint. The winning pattern is to train the safe P1 rung at `1024` steps
with no global step cost, then use `1536` as a solvability/visual-eval horizon.
Global time pressure is too blunt: it preserves clearance but increases
hesitation. The next promising lever should be a curriculum or task-structure
change, not step-cost tuning: for example clearance-scale scheduling from
faster/looser P1 to safe P1, staged P1-to-P0 geometry with explicit
per-pattern scheduling, or a reset/curriculum that over-samples true P0 passage
states until forced P0 itself is solved. Do not repeat small random P1/P0/P3
mixtures as the main method.

AFL-039/AFL-040 added a separate `agile_geometry` obstacle family to stress
the promoted AFL-038 policy with more varied fixed geometry: staggered wall
apertures, dense pillar weave, island/pinball gates, and switchback baffles.
Local CPU build and mode-instantiation checks passed for modes `40..45`, and
visual eval now accepts `--obstacle-mode`.

The result is diagnostic, not a promotion. The AFL-039 intro mix
(`obstacle_mode=45`, 60% dense P1 / 30% P0 / 10% P2) scored only `28.3%`
success with `55.0%` timeouts and `16.3%` obstacle hits. Direct AFL-038 eval on
the same mix was essentially identical, so 20M conservative fine-tuning did not
teach the distribution. Forced-pattern diagnosis showed the real shape of the
problem: P3 switchbacks are already mostly solved (`~92%` success, `~7%`
obstacle hits), P0/P2 are middling (`~40-45%` success), and the new dense P1
weave is poison (`~18-20%` success, `~15-20%` obstacle hits).

AFL-040 isolated P3 and increased clearance shaping. It preserved the high P3
success rate but only moved obstacle hits from `7.6%` to `6.9%`. That is not a
meaningful new skill. Do not continue scalar clearance fine-tuning on P3. The
next useful agile branch is to redesign the geometry curriculum itself:
graded pillar-density variants, explicit near-corner/recovery supervision, or
policy-facing local passage affordance features before remixing P0/P1/P2.
