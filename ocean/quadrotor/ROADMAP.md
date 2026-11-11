# Quadrotor Indoor Navigation Roadmap

Goal: get useful indoor navigation working in simulation quickly, then make the simulator progressively resemble the real indoor drone.

Experiment tracking has two layers: `TESTS.md` is the quick duplicate-check
registry, and `EXPERIMENTS.md` is the detailed lab notebook. Every
evidence-producing train, eval, or diagnostic run should get a `QDR-###` ID
before launch, with the hypothesis, exact command, artifact, result, and "do
not repeat" lesson.

For current research references and open-source quadrotor stacks to borrow from,
see `RESEARCH_SOTA.md`.

For the current Codex operating plan, iteration rules, checkpoint roles, and
milestone ladder toward research-grade sim2real, see `CODEX_ITERATION.md`.

## Current Scope Split: Flight Versus Exploration

As of May 20, 2026, keep two separate benchmark tracks:

- Fast indoor flight: known-objective navigation with route/waypoint/gate/goal
  information allowed according to the rung. This track should optimize stable
  aggressive flight, obstacle clearance, body-rate/thrust control, dynamics
  stress, and eventually sim2real transfer.
- Indoor exploration: unknown-objective coverage and search. The actor should
  not receive planner waypoints or hidden target breadcrumbs. Reward should be
  based on newly observed free space, frontier/information gain, target
  detection when visible, and crash avoidance. Episode timeout is a fixed budget
  for measuring coverage and discovery, not a hard failure by itself.

The combined problem, fast indoor exploration, is the eventual research goal.
Do not use it as the next training task until the two simpler tracks have their
own reliable baselines and diagnostics. Mixing them too early makes failures
ambiguous: a policy can fail because it cannot fly fast, because it cannot
avoid obstacles, because it lacks memory/perception, or because the reward asks
it to know a target it has not observed.

## Current Agile-Flight Direction

The agile-flight track is in `AFL-###` IDs in `AGILE_TESTS.md`. AFL-038 is the
promoted light-randomization P1 waypoint/slalom candidate:

```text
experiments/agile/afl038/reference_waypoint_p1_speed3_clearance_speed_light_dynamics_ft_conservative/quadrotor/1782216829045/0000000006029312.bin
```

It uses direct motor control, deployable state/rays/action-history
observations, P1 geometry, a `3.0m/s` controlled-arrival speed gate,
clearance-conditioned target-speed reward, and light per-episode dynamics
randomization (`0.95..1.05`). Hard historical AFL009 eval at 1024 steps gives
`93.94%` success, `5.94%` timeout, `0.072%` obstacle hits, and `0.048%`
ground hits. At 1536 steps under the same randomization it reaches `99.88%`
success with zero timeouts. The train warm-start bug is fixed: native train now
loads `--load-model-path` before rollouts.

Latest geometry diagnosis: AFL-039 through AFL-050 show that the fixed-geometry
branch has a specific bottleneck, not a generic "more pillars" problem.
Switchback P3 is mostly solved by AFL-038, and the graded open pillar rungs
L0/L1/L2-open are already solved at about `99.5%+` success. The first hard
break is old L2/L3, where a center obstruction creates a constrained central
passage. AFL-042 fine-tuning on old L2 and AFL-043 fine-tuning on L2-open did
not improve that hard break; old L2 remains around `23-26%` success with large
timeout/contact tails. AFL-050 now gives a faster passage-state probe:
center-local tiny/small/large marker resets average only `36.26%` success for
AFL-038, and many valid hard cases fail by timeout hesitation rather than
immediate contact.

AFL-051 added per-step tracing for those hard states. It did not find a simple
network saturation explanation: AFL-038 hidden activations look numerically
normal in representative failures. It did expose two engineering issues to fix
before deeper training: hard-state rows need a passability audit, and
action-history observations currently store raw unclipped motor commands while
the plant executes clipped `[-1, 1]` commands.

GPU operational note: when running on the current `whitespace-aalto` target via
`kosh-aalto`, build the CUDA backend for the actual RTX 3080 target:
`NVCC_ARCH=sm_86`. A stale `sm_120` build caused native zero-action evals
because custom kernels could not execute on that GPU. The latest backend also
depends on 32-byte allocator alignment for raw `.bin` checkpoint compatibility.

Next roadmap item: choose one stressor and keep the rest controlled. The
highest-value fixed-geometry branch is now passage-specific training on valid
AFL-050 center-local states, after fixing the AFL-051 measurement/control
contract. First filter invalid or physically too-tight starts and record clipped
executed action history, then test reset curriculum/hard-state replay,
passage-affordance labels, or a local subgoal/action interface that commits
through the gap. Increase the dynamics-randomization ladder from AFL-038 only
as a separate branch; mixing dynamics randomization with passage geometry would
make the next failure ambiguous.

## Current Research Direction After QDR-352

The p3 procedural-house exploration limit is now the main research blocker. The
graph-frontier planner is useful but far below the privileged nearest-unexplored
ceiling. QDR-347 through QDR-350 ruled out three insufficient shortcuts:

- passive graph-candidate observations added to a target-vector student;
- a discrete top-4 candidate head trained from projected privileged labels;
- selected-candidate cloning without stable planner candidate state.

QDR-350 fixed part of the engineering interface by moving graph candidates into
native planner/cache state and splitting candidate features into local
path-step direction versus final-frontier bearing. That was necessary, but not
sufficient: the candidate-projected oracle is still safe but nearly stationary
on strict p3 (`0%` success, `16.58%` coverage, `100%` timeout at horizon
`1200`). The roadmap should not spend another round on candidate-head PPO
unless a no-training oracle/probe first shows that the candidate abstraction can
actually explore.

QDR-351/QDR-352 added a validated virtual depth camera as sensor
infrastructure on latest upstream PufferLib `origin/4.0` (`9836f0d2`):
`camera_depth_obs` emits a forward egocentric depth frame into the existing
exploration patch tail. It defaults to `16x12` and can be raised to `20x15`
without changing the `416`-float quadrotor ABI. QDR-354 then tested `20x15`
camera observations in native PufferLib PPO on `maze`. It was fast enough to
iterate, but not a learning improvement: baseline got `0%` success and
`45.56%` coverage, while `cam20` got `0%` success and `44.96%` coverage. This
rules out "append depth pixels to the current flat PPO task" as the immediate
fix. It does not rule out vision; it says the next vision rung must include a
diagnostic/auxiliary objective or architecture that makes the policy actually
use local geometry.

Next roadmap item: build a richer deployable topological planner interface
instead of a one-index candidate selector. It should expose current path step,
frontier/region identity, branch or doorway commitment, progress/staleness, and
candidate value. Privileged map information can supervise or critique those
signals, but the deployed actor must still see only feasible onboard state.

## Current Hard-P1 Limit

The current hard-p1 checkpoint is QDR-285:

```bash
experiments/qdr285/qdr285_fast_safe_rollout_qdr282_s88.pt
```

It flies safely and is much stronger than the earlier procedural policies, but
visual eval still exposes recoverable near-wall stalls in doorways. QDR-286
and QDR-287 show the problem is not missing wall perception: the actor target
matches the onboard planner target, rays see the doorway, and contacts remain
zero. The weak point is local doorway efficiency around tiny current/adjacent
frontier targets.

Use this visual command when checking the current best local recovery behavior:

```bash
scripts/quadrotor4.sh explore-v0-target-visual experiments/qdr285/qdr285_fast_safe_rollout_qdr282_s88.pt --mode procedural_indoor_hard_p1 --seed 88 --max-steps 2048 --room-size 6.0 --explore-goal-fraction 0.95 --spawn-radius-min 0.0 --speed-fraction 0.50 --explore-onboard-path-lookahead 2 --safety-slide --safety-stateful-recovery --recovery-stall-steps 96 --recovery-duration-steps 64 --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80 --restart-on-stall-steps 3000
```

Next roadmap item: turn this recovery scaffold into a principled deployable
local planner/action interface that can be trained or distilled. Do not keep
adding one-off doorway heuristics; QDR-287 rejected sampled safety,
shallow-depth stall retargeting, clearance centering, action hysteresis, target
distance speed scaling, and current-cell recentering. The useful next build is
a learned/residual recovery policy or local planner module trained on these
hard states, with privileged planner/map information allowed only for labels,
critic, reset curriculum, or diagnostics.

QDR-288 implemented the first version of that learned recovery interface. It
adds `explore-v0-recovery-policy-train` plus `--recovery-policy` eval/visual
support for a small deployable MLP sidecar. The result is a good engineering
checkpoint but not a promoted navigation improvement: the sidecar can clone the
stateful recovery safely when wrapped with slide/cone safety, but it does not
remove the doorway-stall tail. Best local hard-p1 seed-`88` eval was `96.58%`
success, `3.42%` timeout, and `0%` contacts; no-render visual still showed
doorway stalls. The next roadmap item is therefore not more sidecar capacity.
Generate better recovery supervision from short-horizon hard-state search or
privileged local recovery labels, then train/evaluate the sidecar only if the
label source itself beats the current stateful recovery on replay.

QDR-289 tried that supervision idea with the simplest short-horizon search
label: restore a doorway stall state, evaluate constant candidate actions, and
train the sidecar on the chosen first action. This did not hold up. The small
probe briefly looked better, but the 512-episode regression fell below QDR-288
and introduced a small ground-crash tail. When search labels were filtered to
require immediate coverage gain and clearance, almost all triggered states were
rejected. The roadmap moves one level up: build a resettable hard-state
recovery benchmark and train/evaluate closed-loop recovery trajectories from
captured doorway states, instead of treating recovery as a one-step action
classification problem.

QDR-290 confirmed that the labeler implementation needed to move closer to the
native env. The new trajectory search path batches candidate rollouts by
restoring a hard state into many native `StaticVec` slots and stepping all
candidates together in C. That fixes enough throughput to make this diagnosable,
but the resulting 64-label trajectory sidecar is worse than the baseline:
`87.60%` hard-p1 success after constraining recovery to XY-only. The roadmap
therefore keeps the vectorized native search helper as infrastructure but
drops sparse trajectory-label BC as the next method. Build a resettable
hard-state replay/RL task where the recovery policy is optimized closed-loop
from doorway states under the real eval safety constraints.

QDR-291 built the resettable replay/RL task and ruled out the obvious
action-sidecar version. The benchmark is useful: it replays exact captured
doorway states in batches and reports coverage gain, clearance, terminal
outcome, and full eval. The method result is not promotable. Base and stateful
recovery are tied on early-trigger replay, and base is better on stricter
late-stall replay (`mean_gain=0.106726` vs `0.088603`). A base-warmed PG
sidecar preserves zero contacts but does not improve replay or full eval
(`94.90%` vs base `95.05%` on the same seed/settings). The roadmap should not
continue with "make the recovery velocity smarter" as the main line. The next
high-potential step is to improve the local target/viewpoint planner itself, or
first add a replay filter that only trains on restored states where base has
low coverage gain.

QDR-292 through QDR-298 narrowed that again. A late-stall viewpoint trigger did
nothing in the real QDR-285 rollout. Emitting the deeper planner objective as
the main target destroyed the old checkpoint (`6.25%` hard-p1 success), and
adding it as an auxiliary target plus a 150-update fine-tune did not beat
baseline (`95.85%` saved eval versus QDR-285 rebuilt baseline `96.78%`). The
current onboard oracle under the same fast-safe profile is only `95.92%`, so
the student is no longer the obvious bottleneck. A Python-side local barrier
controller did improve the oracle slightly (`96.61%`, or `96.80%` when combined
with stateful recovery, all `0%` contacts), but it hurt the QDR-285 policy when
applied only at eval time (`95.35%`). A close-waypoint objective-drive gate was
also not enough.

QDR-299/QDR-300 are the current small promotion. The native onboard planner now
has opt-in stall-gated region-goal hysteresis. Hard region locks were bad, but
stall-gated hysteresis plus the barrier/slide/stateful stack improved full
hard-p1 oracle seed `88` to `96.91%`. A short QDR-285 fine-tune under the native
region-commit/barrier rollout produced the current candidate:
`experiments/qdr300/checkpoints/qdr300_region_commit_barrier_ft_qdr285_s88_u000100.pt`.
Full hard-p1 eval with the full stack is seed `88` `96.98%`, seed `90`
`96.14%`, seed `91` `95.63%`, all `0%` contacts. This improves the old held-out
hard-p1 tail but still shows recoverable near-wall stalls in visual smoke. Next
roadmap item: move from local frontier-cell hysteresis to a real topological
planner/progress state that understands doorway/bottleneck regions over time.

QDR-301 moved the hard-p1 runtime setting forward by diagnosing the grid itself.
The old `--cell-size 0.25` was too fine for the current onboard frontier
planner in procedural hard-p1. It produced many one-cell doorway objectives and
near-wall stalls. Finer `0.20m` was much worse (`52.83%` oracle success, `0%`
contacts), while `0.30m` reached `100%` oracle success and lifted QDR-300 u100
policy eval to seed `88` `99.26%`, seed `90` `98.61%`, seed `91` `99.06%`, all
with `0%` contacts. Current hard-p1 visual checks should use `--cell-size 0.30`
with QDR-300 u100:

```bash
PYTHON=../../.venv/bin/python scripts/quadrotor4.sh explore-v0-target-visual experiments/qdr300/checkpoints/qdr300_region_commit_barrier_ft_qdr285_s88_u000100.pt --mode procedural_indoor_hard_p1 --seed 88 --max-steps 2048 --room-size 6.0 --explore-goal-fraction 0.95 --speed-fraction 0.50 --explore-frontier-target-mode onboard --explore-reachable-coverage --explore-onboard-path-lookahead 2 --explore-onboard-region-commit --explore-onboard-region-commit-stall-ticks 90 --safety-barrier-actions --safety-slide --safety-stateful-recovery --recovery-stall-steps 96 --recovery-duration-steps 64 --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80 --cell-size 0.30
```

QDR-302 tried the rollout-matched fine-tune and did not promote. The best saved
checkpoint, `experiments/qdr302/checkpoints/qdr302_cell030_ft_qdr300_s88_u000100.pt`,
is safe but slightly worse than QDR-300 u100 with the runtime `0.30m` grid:
hard-p1 seeds `88/90/91` were `99.17%`/`98.81%`/`98.62%`, all `0%` contacts.
QDR-300 u100 plus explicit `--cell-size 0.30` remains the current best hard-p1
profile. Next roadmap item: improve the onboard planner/topological state or
training objective rather than lightly fine-tuning the same follower. Do not
spend more time on finer hard-p1 grids or QDR-302-style continuation unless the
planner representation changes.

## Indoor Explore V0 Direction

Exploration should be a mapping/search benchmark, not hidden-target route
following. The actor remains deployable: state, local velocity/attitude,
previous action if enabled, and range/depth-like rays. Privileged map/frontier
information may be used for diagnostics, teacher labels, reset curricula, or a
critic, but not as actor input.

The first corrected benchmark is sensor-visible coverage:

- mark cells observed by the ray fan, not only the XY cell under the drone;
- reward first-time observed valid free cells with a small per-cell reward;
- use a high coverage gate (`85%`) only as an eval label;
- treat timeout as a fixed horizon for measuring coverage, not a dense failure
  penalty;
- report coverage curve/AUC, final coverage, crash rate, new-cell rate,
  revisit/stagnation, and clearance.

QDR-155 implements the first native step of this direction. A stationary drone
now maps a large fraction of an open room immediately, which is correct for a
360-degree range sensor and means open-room exploration is not the right hard
case. The next useful explore task should use occluded random-block or corridor
layouts and judge whether the policy deliberately moves to reveal unseen space.

### Current Procedural Baseline After QDR-251

The procedural-indoor exploration plateau has moved. QDR-244 through QDR-251
found that the old plateau was partly invalid: the mixed procedural benchmark
hid one hard slalom-like family (`procedural_indoor_p1`), and many p1 oracle
timeouts began inside the same clearance band where the deployable safety layer
intentionally slows the drone. QDR-250 aligned exploration spawn validity with
`QR_EXPLORE_CLEARANCE_RADIUS`, which made the p1 oracle solve the task at
`100%` instead of about `70%`.

The current procedural checkpoint is:

```bash
experiments/qdr251/qdr251_qdr224_spawn_p1p3_anchor_s88.pt
```

Main eval command:

```bash
scripts/quadrotor4.sh explore-v0-target-distill-eval experiments/qdr251/qdr251_qdr224_spawn_p1p3_anchor_s88.pt --deploy-patch-obs --explore-reachable-coverage --mode procedural_indoor --seed 88 --episodes 4096 --total-agents 1024 --num-buffers 2 --num-threads 16 --max-steps 2048 --speed-fraction 0.35 --spawn-radius-min 0.0 --explore-goal-fraction 0.95 --safety-cone-ray-stop 0.25 --safety-cone-deg 60 --safety-slow-ray 0.70 --use-pred-vector-fallback --pred-vector-fallback-primary-threshold 0.6 --pred-vector-fallback-min-coverage 0.25 --pred-vector-fallback-min-dist 0.02
```

Results: seed `88` `100%`, seed `90` `99.88%`, seed `91` `99.93%`
procedural success, with no timeouts and very low contacts. This is now the
default procedural exploration baseline.

Next limit: narrow-corridor progress under deployable safety. QDR-252 showed
the fixed-layout aggregate was hiding a corridor-specific failure: QDR-251 is
`100%` on center block, doorway, maze, and random blocks, but corridor crashed
about `99%` with the first strict spawn-clearance gate. QDR-253 refined the
spawn rule so it prefers exploration clearance but relaxes down to physical
drone radius in valid narrow passages. This preserves procedural `100%` and
turns corridor from crashes into safe timeouts (`11.79%` success, `0%`
contacts). The next roadmap item is corridor progress/coverage without
weakening obstacle safety globally.

QDR-254 adds the current best deployable safety scaffold: `--safety-slide`.
Instead of stopping whenever any ray in the commanded cone is close, it projects
the commanded velocity away from close forward rays. With QDR-251 and the
original `0.25m/60deg/0.70m` safety thresholds, slide safety reached fixed4
`100%`, corridor seeds `88/90/91` `100%`, and layout_mixed `100%` in the quick
balanced check, with only one random-block contact.

Trying to teach this into a raw action policy is not solved. QDR-255 plain
slide-action imitation, QDR-256 weighted imitation, and QDR-257 teacher-only
refinement all still produced corridor contacts without the external slide
layer. Best learned-action checkpoint so far is QDR-255 u600 at about `87.56%`
corridor success / `12.44%` contacts. The next build should either keep
`--safety-slide` as a real-drone-feasible safety layer while pushing navigation,
or train a more explicit residual/barrier safety module instead of plain action
MSE.

The next benchmark is now QDR-258 `procedural_indoor_hard`. Use it with:

```bash
--room-size 6.0 --explore-goal-fraction 0.98 --max-steps 4096
```

QDR-251 plus `--safety-slide` is no longer enough: aggregate hard success is
only about `54%` across seeds `88/90/91`. Forced p1 is the cleanest target for
the next method: about `45%` success, `55%` timeout, and `0%` obstacle contact.
This should drive the next training work toward better frontier/memory target
selection or hard-procedural curriculum, while keeping slide safety as the
deployable low-level scaffold.

QDR-259 and QDR-260 show what not to do next. Full-policy hard p1/p2
distillation from QDR-251 collapses deterministic behavior, and simply
expanding the passive deployable map to `21x21` then training only new input
columns still leaves hard-p1 near zero success. The next roadmap item should be
a more structured exploration module: topological frontier memory, a
deployable local-planner target/action interface, or resettable hard-state
training focused on escaping already-known loops.

QDR-261 adds the pass/fail gate for that module.
`explore-v0-onboard-frontier-verify` runs the native env with a deployable
onboard frontier target before any policy imitation. Blocked occupancy from
ray-hit endpoints is now modelled, and the verifier uses reachable-component
coverage by default so p2 is not falsely judged against disconnected cells.
The target is still not promotable: hard-p2 is partly viable, but hard-p1 stays
around `17-20%` success while privileged `nearest` solves it. Therefore the
roadmap should not train against the current onboard target. First make the
target/planner itself pass this verifier with real-drone-feasible inputs, then
distill or use it as privileged labels.

QDR-262 makes the first useful topological-memory step: stale frontier
suppression. The onboard planner now remembers the current unknown target cell,
marks it stale after a no-progress window, and skips stale cells. This raises
hard-p2 to `100%` across seeds `88/90/91` and improves hard-p1 into the
`24-34%` range with zero contacts, but p1 still fails the verifier. The next
roadmap item is therefore not GPU training against this target yet.

QDR-263 adds the required planner-debug gate. The native env now logs
planner-state metrics and visual eval draws the explored/frontier/stale/blocked
grid state plus the selected onboard target. The new command is:

```bash
scripts/quadrotor4.sh explore-v0-planner-state-verify
```

This passed locally on hard-p1/p2 seed `88`. Hard-p1 still timed out often
(`27.69%` success / `70.02%` coverage), but the sanity counters showed no
missing target signal and no direct blocked-target commands in that sample. The
roadmap therefore moves to region-level frontier clusters with
stale/visited/inaccessible status and last-progress accounting. Use QDR-263 as
the pre-training gate for that work.

QDR-264 tested the simplest version of that response and ruled it out.
Cluster-scored frontier targeting, one-shot cluster/local stale marking, and a
shorter stale timeout did not solve hard-p1; some variants were destructive.
The active code is back on the QDR-263/QDR-262 per-cell stale planner with only
cluster diagnostics retained. The next roadmap item is a persistent
frontier-region or topological graph/search planner that tracks region
identity, target commitment, stale/visited/inaccessible state, and progress
over time. Do not repeat wider one-shot stale marking as the main fix.

QDR-265 is the first planner-level solution to the hard-procedural onboard
frontier gate. The native planner now commits to a connected frontier region,
continues choosing targets inside that region while it remains viable, and only
marks the committed region stale after no real coverage progress. Intra-region
target churn no longer resets the no-progress timer. This passed the local hard
family gate: p0/p1/p2/p3 all `100%` across seeds `88/90/91` in the 64-episode
verifier, with `0%` obstacle contacts. The visual harness default for onboard
visual eval now uses `--restart-on-stall-steps 3000`, because successful
committed-region episodes can pause near walls for much longer than the old
300-step visual restart budget.

The roadmap now moves from planner validity to policy transfer: train or
distill a deployable student/high-level target model against the QDR-265
planner, then require the same planner-state verifier, no-render visual smoke,
and fixed-layout/procedural regression checks before promotion.

QDR-268 adds the active visual-target hygiene fix: a small penalty for selecting
the current cell as the target when another frontier cell in the committed
region is available. This reduces tiny adjacent wall targets while preserving
hard-family success. QDR-266, QDR-267, and QDR-269 were not promoted: shorter
stale timing regressed p1, a free clear-ray fallback caused worse timeout
behavior, and slide-cone alignment did not address the observed sticky case.
QDR-270 adds a bounded local viewpoint layer for the remaining current-cell
frontier slivers: choose a reachable explored viewpoint with a known-clear
sightline to the same unknown target, scored by current lidar cone clearance.
This preserves the hard-family `100%`/`0%` planner gate and removes the most
obvious tiny adjacent-target visual symptom, but does not fully remove long
near-wall pauses. The next improvement should focus on richer local
viewpoint/motion planning or policy transfer against this planner instead of
changing the topological frontier-region planner again.

QDR-271 tried the richer frontier/viewpoint planner directly and was reverted.
Scoring whole frontier regions by expected visible unknown gain and adding
weighted explored-cell path costs made the behavior look more structured in one
visual smoke, but it reduced hard-p1 robustness and introduced timeouts. This
updates the roadmap: do not keep pushing frontier scoring as the main lever.
Keep QDR-270 as the planner baseline, then build a local trajectory/barrier
controller that chooses safe short-horizon velocity or waypoint commands around
the planner target. That controller can still be trained/distilled later, but
the engineering target is now local motion quality rather than global frontier
selection.

QDR-272 tried the simplest version of that controller: candidate velocity
sampling from current rays. It is useful as a diagnostic flag but not a
promotion. Single-step direction sampling can improve one visual metric while
worsening hard-p1 progress stalls, so the next controller should be explicitly
short-horizon and stateful: evaluate several steps of candidate motion or
waypoints, score expected coverage/progress and clearance over the rollout, and
include an escape/recovery state when progress is flat near walls.

QDR-273/QDR-274 show that even simple stateful recovery and speed-floor
controllers are still too shallow. They can improve selected visual traces, but
they do not reduce the hard-p1 max-stall statistics that matter. The roadmap
should now move from hand-written local ray heuristics to data-driven hard-state
recovery: save the actual long-stall states, label them with a stronger
multi-step oracle or offline trajectory search, and train/evaluate a recovery
head or residual controller on those states before trying to replace the
default slide controller.

QDR-275 adds the first hard-state loop for that roadmap item. It saves exact
active-planner wall-stall snapshots and compares baseline replay against a
short-horizon trajectory-search controller. Early CPU results show real but
incomplete signal: the trajectory search improves short replay coverage gain
without crashes, but becomes too cautious over longer replay. Keep using this
probe as the next local-control pass/fail loop, but do not ship the current
MPC scoring. The next controller should include commitment/progress memory or
be learned as a recovery residual from these snapshots.

QDR-276 tested that commitment/progress-memory idea in its simplest form and
failed: committing progress-heavy local actions caused obstacle contacts on
exactly the hard states QDR-270 otherwise survives. This updates the roadmap
again. The missing module is not "more aggressive local velocity"; it is better
local geometry/viewpoint understanding. Future recovery work should either use
the deployable local grid/ray history to choose a real viewpoint/escape target,
or train a recovery residual from hard-state rollouts with contact-aware labels.
Do not spend more time hand-tuning scalar MPC weights unless a diagnostic first
shows which geometric feature the controller is missing.

QDR-277 adds the public-research reassessment. The next step is larger than a
local recovery tweak: build a proper exploration architecture inside PufferLib.
FUEL, TARE, Active Neural SLAM, FAR Planner, and FASTER all point toward
persistent spatial state plus frontier/viewpoint/graph planning before local
control. Swift and Agile Autonomy remain the right references for the separate
fast-flight track: privileged training, deployable onboard observations, and
trajectory/control labels. The roadmap now has two near-term engineering jobs:

1. Exploration: add a PufferLib-native frontier/viewpoint candidate layer with
   a debug/probe view that proves candidate reachability, expected coverage
   gain, stale/visited status, and why the active target was selected.
2. Learning: train a deployable high-level selector or target student from
   privileged candidate labels only after the candidate layer beats the current
   hard-state planner diagnostics.

Do not merge these back into fast indoor flight yet. First get exploration's
spatial memory/viewpoint layer right, then later combine it with the Swift-lite
control stack.

QDR-278 completes the first half of the exploration item: candidate existence
and visibility are now measurable. The hard-p1 issue is not absence of reachable
viewpoint candidates. The probe sees many viable candidates, low miss rate, and
zero contacts; hard-p1 only fails the strict `2048`-step gate by timing out, and
solves at `4096` steps. The next roadmap item is therefore a candidate
commitment/sequencing build: track selected candidate identity over time,
penalize unproductive switching/revisits in diagnostics, and either hand-test a
small committed waypoint policy or train a high-level selector before spending
more effort on low-level velocity heuristics.

QDR-279 tried that small committed waypoint policy and found it is not enough.
The new churn metrics are useful, but the intervention is not promotable:
longer commitment reduces switching while worsening hard-p1 success, and the
best short setting gives only a small mixed change. This moves the roadmap one
level up. The next engineering build should improve frontier/region selection
itself: rank persistent regions by expected gain, travel cost, stale/revisit
history, and clearance, then expose those candidate labels to a learned
high-level selector. Treat commitment as a diagnostic/control primitive, not as
the main method.

QDR-280 completes the privileged half of that next item. The new `region`
frontier mode is a stronger teacher than `unexplored`: it keeps `100%` hard
pattern success and no contacts while increasing new-cell rate on every tested
hard pattern/seed. This should become the default privileged teacher for the
next learning pass. Roadmap priority is now distillation/asymmetric learning:
actor receives only deployable observations, while `region` supplies target
vectors, action labels, candidate rankings, or critic features. Keep the old
`unexplored` mode as a simple regression baseline.

QDR-156 proves the corrected task is learnable from zero, but also exposes the
next limit. A 25M-step native PPO run on random-block layouts reached `99.75%`
success on the random-block coverage gate with only `0.25%` crashes and
`85.23%` final coverage. The same checkpoint is trivial in open rooms, but
falls to `55.72%` success and `44.28%` crashes on `layout_mixed`. The next
exploration benchmark must therefore train/evaluate on a mixed occluded-layout
gauntlet, not just random blocks.

QDR-157 builds that gauntlet and trains directly on `layout_mixed`. This is a
useful partial step: the best checkpoint improves 1024-step gauntlet mean
success from `62.48%` to `68.38%` and cuts mean crash rate from `26.09%` to
`7.70%`. It preserves open/random-block performance and solves corridor, but
it still fails center-block contact and standalone maze exploration. A longer
2048-step budget lifts `layout_mixed` to `89.61%`, showing that part of the
mixed-layout issue is horizon/caution.

QDR-158 trains on standalone maze and fails: best success is only `2.05%` at
1024 steps and `2.58%` at 2048 steps, with coverage stuck near `51%`. The next
exploration build should diagnose the maze as a frontier/visibility/reward
problem instead of spending more blind PPO budget on the same observation and
reward.

QDR-159 does that diagnosis. The maze has one connected flyable component, the
valid coverage cells are all observable from reachable poses, and a greedy
frontier oracle gets every sampled start above the `85%` gate. A privileged
frontier teacher running through the actual native quadrotor env solves
`98.44%` of maze episodes in about `23` velocity-control steps. This means the
current failure is algorithmic: PPO with only immediate new-cell reward does
not learn the mapping/frontier behavior needed after the easy visible region is
exhausted.

QDR-160 tries to transfer that teacher directly into a deployable MinGRU actor
with sequence BC/DAgger. It improves over blind PPO in the sense that teacher
labels produce moving policies, but it is not safe or reliable: the best short
student reached about `49.8%` success with `50.2%` obstacle contacts, pure BC
fit low MSE but crashed even more at native speed, and longer DAgger collapsed.
The lesson is that raw action imitation of a frontier planner is too brittle;
small action errors cut corners into walls.

QDR-161 tries the other obvious branch: expose a deployable onboard-map
frontier summary in the exploration task's unused target slots while keeping
the 60-float ABI. From-zero PPO still stalls at the half-maze barrier. Best
deterministic maze eval is only `1.31%` success at 1024 steps with `50.34%`
coverage and `91.93%` timeouts; even 4096 steps only reaches `3.33%` success
and `52.03%` coverage. The feature made behavior safer, not smarter.

QDR-162 through QDR-166 show that small frontier shaping is not enough.
Frontier-progress reward and direct nearest-frontier target observations both
failed. The direct target was diagnosed as actively bad: a simple controller
following it crashed `97-99%` of episodes because nearest frontier does not
mean reachable frontier in a maze. A reachable explored-grid path target was
better and gave a hand controller `46.60%` success, but PPO still collapsed to
the same half-coverage timeout policy (`0.292-0.364%` best deterministic
success). This means the next exploration work should become an engineering
build around a high-level exploration module or training objective, not another
scalar reward tweak.

QDR-167 is the first strong evidence for that engineering build. A recurrent
target predictor learned the reachable frontier-path target from deployable
history while privileged target slots were zeroed from actor input. Closed-loop
eval with a fixed velocity target follower reached `85.65%`/`85.00%`/`85.20%`
success on seeds `88/90/91` under the matched training/eval vector profile, and
about `83.2-83.6%` on the stricter single-buffer profile. Coverage is about
`78%`, failures are obstacle contacts, and timeouts are gone. This is not pure
end-to-end exploration, but it is a scientifically honest decomposition that
beats the half-maze plateau.

QDR-168 matched the student rollout speed to the safer eval speed. It improved
matched-profile seeds `88/90/91` to `87.35%`/`85.91%`/`86.76%` success, with
`12.65%`/`14.09%`/`13.24%` obstacle contacts. Treat this as the current
target-distill checkpoint, but not a solved baseline. The stricter
single-buffer profile still sits near `83-84%`, so evaluation profile must be
reported with every result.

QDR-169 diagnosed the remaining obstacle contacts. In student crash states the
privileged reachable-frontier target frequently says there is no current target
(`oracle_exists=0.0`), while the student still predicts a target about half the
time and moves into low clearance. Crash-state min-ray clearance is only about
`0.09m`, the predicted/action target ray is roughly `0.94-0.99m`, and the
oracle target ray is roughly `1.54-1.60m`. About a third of crash states have
the predicted ray worse than oracle by at least `0.20m`. The next exploration
milestone should therefore train the high-level module to know when to stop,
reject unsafe targets, or recover from these target-negative hard states.

QDR-170 tested a blunt weighted-loss version of that idea and failed. Stronger
negative-existence and absent-direction penalties dropped matched-profile
success to about `59.7%`, increased obstacle contacts to about `25%`, and added
about `15%` timeouts. The model did not simply become safer; it both stopped in
some recoverable states and still hallucinated targets in many crash states.
This points away from global loss weighting and toward either calibrated
existence gating or targeted hard-state negative replay.

QDR-171 then tested calibrated existence gating on QDR-168. Threshold `0.30`
was slightly worse, `0.50` was worse, and `0.70+` caused timeout collapse. This
rules out a simple scalar action gate. QDR-172 should test whether a deployable
local ray safety scaffold can stop the specific corner-contact mode. If that
also fails or trades too much success for timeouts, the next engineering build
is hard-state target replay rather than another inference-time gate.

QDR-172 passed as a scaffold/diagnostic. A local ray slowdown using only
deployable range observations, `--safety-min-ray-stop 0.20 --safety-slow-ray
0.60`, raised QDR-168 to `93.51%`/`92.64%`/`92.86%` on matched-profile seeds
`88/90/91`, with no timeouts. Stricter single-buffer eval stayed around
`92%`. Diagnostics show remaining crashes have nearly zero coverage, nearly
zero ray clearance, predicted target absent, and zero commanded speed. This
suggests an initialization-validity issue rather than a remaining learned
navigation behavior. The next roadmap item is to align exploration spawn
sampling with the exploration valid-cell margin and rerun the same policy.

QDR-173 tested that spawn-margin idea and failed. Tightening exploration spawn
validity to the exploration cell margin made the current safety-gated policy
worse (`82-83%` success, `16-18%` obstacle contacts), so the change was
reverted. The lesson is narrower: QDR-172's remaining failures need a more
careful start/failure classifier, not a blind spawn-distribution change.

QDR-174 added that classifier. It showed the remaining QDR-172 failures are
almost all one-step, zero-speed crashes from zero-clearance starts. The likely
source is the `spawn_radius_min=0.0` evaluation distribution: very small spawn
bounds around the maze origin can make all spawn attempts invalid, after which
the fallback is also invalid. The next roadmap item is not policy training; it
is to test a minimum spawn radius or a better spawn fallback.

QDR-175 tested the minimum-spawn-radius version and effectively solves the
standalone maze diagnostic. With QDR-168's learned target module, QDR-172's
deployable ray slowdown, and `--spawn-radius-min 0.25`, matched-profile seeds
`88/90/91` all reached `100%` success with no crashes/timeouts; stricter
single-buffer eval was `99.984%`/`100%`/`100%`. Without the ray safety scaffold,
the same spawn fix is only about `92.4-92.9%`, so the current recipe is a
modular stack: learned high-level frontier target, local ray safety, and sane
valid-start sampling.

Next: move beyond standalone maze. Evaluate this recipe on the full exploration
gauntlet and mixed/held-out layouts, then decide whether to train the safety
gate into the policy, keep it as a real-drone-feasible low-level safety layer,
or replace it with resettable hard-state training.

QDR-176 did the first gauntlet pass. The current modular recipe is strong on
maze, doorway, random blocks, and mixed layouts (`layout_mixed` about
`96-97%`), but weak on corridor (`69-70%`, all timeouts) and center block
(`87%`, obstacle contacts). This gives a sharper next roadmap split: fix
corridor by making safety directional instead of nearest-ray-only, then address
center-block contacts.

QDR-177 confirmed the corridor diagnosis. Weakening the nearest-ray gate to
`--safety-min-ray-stop 0.15 --safety-slow-ray 0.45` solves corridor on seeds
`88/90/91`, but it is not a global improvement: it increases obstacle contacts
and drops maze, mixed layout, and center-block success. The roadmap should not
continue with scalar safety threshold tuning. The next useful change is a
more structured deployable safety/perception interface: separate forward
clearance from lateral wall proximity, or train the target policy with hard
states so it learns the corridor/obstacle distinction instead of relying on an
external one-number brake.

QDR-178 made that structured interface explicit with a commanded-motion cone
safety gate. The current best deterministic modular recipe uses the QDR-168
target model with `--spawn-radius-min 0.25 --safety-cone-ray-stop 0.20
--safety-cone-deg 60 --safety-slow-ray 0.60`. It solves corridor across seeds
`88/90/91`, keeps maze around `99.2%`, mixed around `96.2-96.9%`, doorway at
`100%`, and random blocks around `99.8%`. The remaining fixed-layout weakness
is center block at about `86.5-86.7%` with obstacle contacts.

Roadmap consequence: do not spend more time on global ray-threshold sweeps. The
next blocker is obstacle-contact recovery/avoidance near the center block. Use
resettable hard-state diagnostics, planner-labelled target correction, or train
the safety behavior into the high-level target module. Also keep the cone gate
available as a real-drone-feasible low-level safety layer, but do not mistake it
for a learned from-zero navigation solution.

QDR-179 reclassified the center-block blocker. The privileged oracle fails at
the same rate as the student, and the failures are immediate zero-speed
zero-clearance starts. A `spawn-radius-min 0.50` eval removes almost all of
those failures and lifts the fixed-layout gauntlet to roughly `99.5-100%` on
every tested layout/seed. The next roadmap item is therefore an environment
reset validity fix: room-spawn fallback must never place the drone inside an
obstacle when the sampled bound is too small. After that, rerun the gauntlet
under the normal spawn distribution. If it remains strong, move on from fixed
layouts to harder exploration/randomization; if it regresses, diagnose genuine
post-start obstacle contacts.

QDR-180 completed that reset-validity fix. The fixed-layout exploration
diagnostic is now clean enough to stop treating it as the main blocker:
normal-spawn deterministic eval with QDR-168 target model plus QDR-178 cone
safety reaches `98.5-100%` across center block, corridor, maze, mixed, doorway,
and random-block layouts on seeds `88/90/91`. The remaining student failures are
small real hard states rather than invalid starts; the oracle solves center
block after the reset fix.

Roadmap consequence: the next useful milestone should push beyond this fixed
layout diagnostic. Options that still align with the research goal:

- train the cone/safety behavior into the high-level target module instead of
  keeping it as an external gate;
- add randomized/more complex indoor layouts and evaluate generalization;
- move from velocity scaffold toward body-rate/thrust only after the
  exploration/navigation stack is stable;
- build resettable hard-state training only if the new harder layouts expose a
  nontrivial post-start contact tail.

QDR-181 tested the first option and failed. A naive learned speed-scale head,
supervised from the cone safety labels, becomes too conservative: deterministic
layout-mixed eval drops to `1.10%` success with almost all episodes timing out.
This means the current cone gate should remain an explicit deployable low-level
safety layer while we push the next benchmark. If we revisit learned safety, use
a different formulation: intervention classification, hard-state recovery, or a
separate safety critic/controller, not continuous speed-scale regression on the
target head.

QDR-182 pushed the current modular explorer to a `95%` coverage gate. This
shows progress and a new limit at the same time. Maze and random-block layouts
remain strong (`98-99.5%`, no timeouts), so the policy is not merely solving the
old `85%` gate by luck. But occluded fixed layouts with larger coverage targets
now expose a horizon/efficiency problem: `layout_mixed` and `center_block` are
only about `81-83%` successful under a 1024-step budget, mostly from timeouts,
while obstacle contacts stay low. The next roadmap item is a controlled horizon
diagnostic, not another safety-threshold sweep: rerun the same QDR-182 setup
with a longer eval horizon and decide whether the next build should optimize
coverage rate or handle true late-frontier stagnation.

QDR-183 through QDR-185 completed that diagnosis. A longer horizon alone does
not fix the `95%` gate. More importantly, the aggregate mixed-layout metric is
not a clean single failure. Center block and doorway are student target/model
gaps: the privileged target oracle solves them, but the learned target module
does not. Corridor is a target-oracle gap: static geometry is fully coverable,
but the current reachable-frontier target oracle stalls around `63%`. This is
an important benchmark hygiene result. Training the student harder against the
current `layout_mixed@95%` label would mix real learning signal with a flawed
teacher in thin spaces.

Next roadmap item: improve the privileged frontier target before more student
training. The target should remain training-only privileged information, but it
must solve simple fully-coverable layouts like corridor before being used as a
label/teacher for stricter high-coverage exploration. Candidate fixes are a
coverage-gain viewpoint target, a frontier target that reasons over visibility
gain rather than only adjacent unknown cells, or a fallback that moves toward
unseen valid cells when no adjacent-frontier path exists despite coverage below
goal.

QDR-186 resolved the immediate teacher problem with a smaller fix than a new
viewpoint planner. The corridor failure came from physically valid starts
outside the stricter exploration-grid margin, where the teacher produced no
target. A scoped corridor-only recovery to the nearest exploration-valid cell
restores the privileged oracle: `corridor`, `doorway`, and aggregate
`layout_mixed` all reach `100%` at the `95%` gate across seeds `88/90/91`.

QDR-187 used that fixed teacher to train a new high-coverage layout-mixed
student. This moves the benchmark frontier: `layout_mixed@95%` is now about
`98.7%`, corridor is `100%`, doorway/center/random blocks are about
`99.5-99.9%`, and the old timeout plateau is gone. The new blocker is
standalone maze generalization, where the QDR-187 checkpoint falls to about
`62%`. The roadmap should now treat QDR-187 as the best fixed-layout
high-coverage checkpoint, not as the universal exploration checkpoint.

QDR-188 tightens that interpretation. `layout_mixed` does sample maze, but the
aggregate metric is asynchronous and episode-count weighted. Because successful
layouts finish episodes faster than slow maze timeouts, aggregate
`layout_mixed@95%` can overstate balanced fixed-layout competence. Use
standalone per-mode evals and an equal-mode average for promotion decisions; for
QDR-187 that means roughly `90%` balanced fixed-layout success, not the headline
`98.7%` aggregate.

QDR-189 adds that promotion gate as code. The balanced target eval command shows
QDR-187 at `90.39%` fixed4 success and `92.22%` fixed5/all-nonaggregate success
at the `95%` coverage gate, while aggregate `layout_mixed` remains `98.70%`.
This confirms the current blocker is standalone maze/frontier preservation, not
low-level flight or the non-maze fixed layouts.

QDR-190 confirms a single model can move toward that blocker, but also shows why
single-distribution continuation is not enough. Continuing QDR-187 on
standalone `maze@95%` raises fixed4 success to `96.67%` and maze to `90.90%`,
but balanced obstacle contacts rise to `3.33%` and doorway drops to `96.1%`.
The next method should interleave fixed layouts during target distillation so
maze recovery and doorway/center/corridor safety are optimized in the same
training loop.

First interleaving attempt, QDR-191, shows equal weights are insufficient:
non-maze safety stays strong, but maze only reaches `64.24%`, leaving fixed4 at
`90.82%`. Continue with maze-heavy interleaving or hard-state replay instead of
equal layout sampling.

QDR-192 tests maze-heavy interleaving and still falls short: maze reaches only
`68.82%` and fixed4 `91.99%`. This points away from plain curriculum weighting.
The next roadmapped build should target hard-state replay/reset or a loss change
for late maze states, using QDR-190 as the useful-but-unsafe recovery point.

QDR-193 through QDR-195 sharpen that next build. QDR-193's gentle safety refresh
from QDR-190 restores easy-layout safety but loses maze. QDR-194 shows why:
QDR-190 succeeds in maze by moving through states where the privileged target is
absent; crashes happen when those hallucinated targets point into unsafe
directions. QDR-195 proves a generic "clearest ray" fallback is not enough and
can collapse maze/corridor. The roadmap should now prioritize hard-state replay
or a better privileged absent-target planner, not more simple curriculum or
local-clearance heuristics.

QDR-196 through QDR-199 resolve that branch. The teacher now provides a
privileged recovery target when the explored-frontier path is absent: nearest
valid exploration cell for invalid-margin poses, otherwise a valid-cell BFS to
the nearest unexplored cell. Continuing QDR-190 with this teacher gives QDR-197,
the current best checkpoint:
`experiments/qdr197/qdr197_qdr190_recovery_teacher_s88.pt`. QDR-198 then showed
the remaining contacts were not absent-target hallucination anymore; they were
low-clearance/high-speed corner cases. QDR-199 tightened the deployable
commanded-cone safety from `0.20/0.60` to `0.25/0.70` and raised the balanced
fixed-layout eval to `99.89%` fixed4 and `99.89%` fixed5/all-nonaggregate at the
`95%` coverage gate, with no timeouts and about `0.11%` obstacle contacts.

Current visual/eval command:

```bash
scripts/quadrotor4.sh explore-v0-target-visual experiments/qdr197/qdr197_qdr190_recovery_teacher_s88.pt --mode maze
```

Current balanced eval command:

```bash
scripts/quadrotor4.sh explore-v0-target-balanced-eval experiments/qdr197/qdr197_qdr190_recovery_teacher_s88.pt --seeds 88 90 91 --episodes 4096 --total-agents 2048 --num-buffers 2 --num-threads 16 --max-steps 2048 --max-eval-steps 100000 --speed-fraction 0.35 --spawn-radius-min 0.0 --explore-goal-fraction 0.95 --safety-cone-ray-stop 0.25 --safety-cone-deg 60 --safety-slow-ray 0.70
```

Next roadmap item: stop polishing this fixed-layout benchmark and push the next
limit. Good candidates are broader procedural indoor layouts, learning the
cone-safety behavior instead of hand applying it, or moving the target-distill
student toward a real onboard mapping/perception representation.

QDR-200 implemented the broader-layout rung as `procedural_indoor`. It is a
valid next benchmark: the privileged oracle reaches `99.90%` success with no
timeouts. The current best fixed-layout explorer does not transfer: QDR-197 with
QDR-199 safety reaches only `36.03%` success and `63.44%` timeout, mostly from
student target mismatch rather than crashes. QDR-201 procedural continuation
proves the gap is partly trainable by raising procedural held-out success to
`62.48%`, but it catastrophically forgets standalone maze, which falls to about
`15.96%`.

Next roadmap item: solve procedural generalization without forgetting fixed
maze. Do not repeat naive procedural-heavy continuation. Use balanced replay or
anti-forgetting explicitly: train with separate procedural and fixed batches,
add a QDR-197 anchor/KL loss on fixed-layout states, collect hard timeout states
from procedural layouts, or train a fresh target student with procedural and
standalone maze weighted as first-class distributions from the start.

QDR-202 implements the first anti-forgetting path. It works for retention but
does not solve procedural exploration. Fixed4/fixed5 remain about `99.88%` and
standalone maze returns to about `99.85%`, but procedural held-out success stays
near `61.64%` with `37.85%` timeout. The roadmap should now pivot from
"prevent forgetting" to "fix induced procedural timeout states." Candidate next
tests:

- lower and slow the student-rollout schedule for procedural target distillation;
- collect procedural timeout windows and replay them with teacher targets;
- add an auxiliary deployable map/frontier prediction head so the recurrent
  state is explicitly trained to remember explored/free/frontier structure.

QDR-203 rules out the simplest hard-state loss weighting variant. Upweighting
procedural samples with large student/teacher target disagreement, worse
predicted clearance, or low speed preserved fixed retention (`99.85%` fixed5,
no timeouts) but dropped procedural held-out success to `59.18%`. Do not keep
optimizing weighted target MSE as the main lever. The next roadmap item should
change the training distribution or representation: reset from actual
timeout-window states, train an auxiliary deployable spatial-memory/frontier
head, or use privileged map/planner data as critic/label structure while
keeping the actor observation deployable.

QDR-204 makes this more specific. Exact replay of QDR-202 procedural timeout
windows shows the student rescues `0/64` while the current privileged
frontier-path oracle rescues only `23/64`. The sharp actor failure is
target-existence collapse: the oracle still has a valid frontier-path target in
`68.8%` of replayed states, but the student predicts target existence in only
`7.8%` and stops. The roadmap is now continuous iteration on that failure, not
a narrow replay-only phase:

- build target-existence/spatial-memory supervision from deployable history;
- add auxiliary map/frontier prediction if existence-only supervision is too
  weak;
- improve the privileged procedural oracle before using it as hard-state replay
  labels;
- train/evaluate each build against procedural held-out success and fixed-layout
  retention, then immediately move to the next build if it fails.

QDR-205 shows that direct existence supervision is not sufficient. It forces the
student to predict a target exists in late procedural hard states, but held-out
procedural success stays near QDR-202 and exact replay still rescues `0/64`
student states. Keep this as evidence that the failure is spatial memory and
target quality, not just a missing binary existence bit.

QDR-206 is the strongest current procedural result. Privileged frontier-sector
auxiliary labels, zeroed from deployed actor input but used as supervised
training targets, raise procedural held-out success to `65.09%` while retaining
fixed layouts at about `99.9%`. This should be the new baseline for the
procedural branch. It is not solved: the remaining timeout tail still shows
target-existence collapse and is not recoverable by the current student from
exact reset.

QDR-207 improves that baseline by adding the QDR-205 existence companion signal
to QDR-206. Held-out procedural success rises to `67.28%`, timeout falls to
`32.37%`, and obstacle contact drops to `0.36%`, while fixed layouts remain near
`99.9%`. This is the current procedural checkpoint to beat. The replay probe is
clear, though: the tail is still not recoverable from exact timeout windows.
Existence is now forced (`pred_exists=100%`), but the spatial target/action is
still wrong enough that student replay rescues `0/64`.

Next roadmap item: improve teacher/label quality for the tail. First test a
sector-fallback target label: when the path oracle says no target after enough
coverage, but privileged frontier sectors still show a useful frontier
direction, train the target head on that coarse direction. If that fails, stop
using four scalar sectors as the main representation and build a richer
deployable spatial-memory objective: local occupancy/frontier sector history,
map patch prediction, or a stronger privileged procedural oracle that can
rescue more late states before being used for hard-state replay.

QDR-208 failed the direct sector-fallback target idea: procedural success fell
to `54.21%`. The follow-up sector diagnostic explains the failure: QDR-207's
remaining hard states have weak sector prediction (`sector_top_match=21.9%`).
QDR-209 kept the sector head auxiliary-only, increased sector pressure on hard
procedural states, and became the new strongest procedural checkpoint:
`68.12%` success, `31.47%` timeout, `0.41%` obstacle, with fixed5 at `99.90%`
and no fixed-layout timeouts. It also improved hard-state sector top-match to
`37.5%`, but replay remains `0/64`.

Next roadmap item: stop treating four scalar sectors as enough map memory.
Design the next representation build around richer deployable spatial memory or
stronger recovery labels. The likely options are:

- local occupancy/frontier map-patch prediction from rays and odometry history;
- higher-resolution egocentric frontier sectors, with careful checkpoint input
  migration so ray indices are not silently shifted;
- a stronger privileged late-state recovery oracle, then resettable replay once
  the oracle can rescue more than the current `18/64`-`23/64` tail.

QDR-210 validates the first richer label. A privileged nearest-frontier vector,
written into existing privileged-only slots `19:21` and trained as an auxiliary
head, raises procedural success to `70.13%` while keeping fixed5 at `99.92%`
with no timeouts. This is the current procedural checkpoint. It deliberately
does not change deployed actor inputs: slots `14:26` are still zeroed before
the policy.

QDR-211 then tested whether that vector should become a direct absent-path
target fallback. It should not. Procedural success fell to `55.00%`, repeating
the same lesson as the failed sector fallback with a better label. Direct
fallback relabeling creates behavior that is locally plausible but globally
bad under the controller/recurrent state.

Next roadmap item: build a true map-memory auxiliary, not another direct target
fallback. QDR-212 is reserved to predict a small egocentric local frontier patch
from recurrent state, using privileged explored/unexplored/free grid labels only
as training targets. Keep actor inputs deployable: the appended patch labels are
zeroed before policy inference and are never used as actor observations. Gate it
against QDR-210:

- procedural held-out success must beat `70.13%` or reduce timeout without a
  contact tradeoff;
- fixed5 must remain near `99.9%` with no timeouts;
- hard-state replay should improve beyond `0/64` student recovery, or at least
  expose a sharper blocker than vector/sector mismatch.

QDR-212 and QDR-213 exposed that sharper blocker. The patch label is learnable,
and fixed-layout retention remains near `99.9%`, but procedural exploration
does not beat QDR-210 and hard-state replay remains `0/64`. The issue is no
longer "can the recurrent model predict richer memory?" It can. The issue is
"how does that memory become recovery behavior?" The next roadmap item is
resettable timeout-window training or another behavior-coupled objective:

- collect timeout-window snapshots from the current best procedural policy;
- restore those states during training instead of hoping ordinary rollouts
  revisit them often enough;
- supervise target/actions with the privileged recovery oracle where it succeeds;
- protect fixed-layout behavior with the QDR-197 anchor or interleaved rehearsal;
- promote only if procedural success beats QDR-210 without a contact tradeoff
  and hard-state student replay improves beyond `0/64`.

QDR-214 through QDR-219 implemented and then saturated the resettable replay
path. Exact snapshot replay, oracle-success filtering, positive-only masks,
iterative recollection, a QDR-210 behavior anchor, burn-in/history
reconstruction, and direct action-imitation loss all work as infrastructure,
but none promote a checkpoint. QDR-217 was the least destructive replay variant
at `68.17%` procedural success and `0/64` hard-state replay. QDR-218 and
QDR-219 were worse, collapsing saved procedural seed 88 to about `28%` success
and `71%` timeout while still replaying `0/64`.

The resettable-state lesson is now clear: do not run another small target-replay
loss variant as the main plan. Use replay as a diagnostic, or only return to it
after changing the observation or recovery interface.

QDR-220 changed the observation instead. It made the existing 16-float
egocentric explored/frontier patch actor-visible through `--deploy-patch-obs`
while still zeroing privileged path/target slots. This is deployable if treated
as an onboard local map built from rays and odometry. It improves procedural
held-out success to `71.55%` mean on seeds `88/90/91`, preserves fixed-layout
retention at `99.90%`, and improves hard-state target angle/action speed, but
hard-state replay remains `0/64`.

QDR-221 confirms this direction. Expanding the actor-visible deployable map to
QDR-220's legacy `4x4` patch plus a `7x7` known-free channel and a `7x7`
frontier-boundary channel improves procedural success to `77.51%` mean on
seeds `88/90/91`, lowers timeout to `22.18%`, lowers obstacle contacts to
`0.32%`, and keeps fixed retention at `99.91%`. The new current checkpoint is
`experiments/qdr221/qdr221_qdr220_wide_local_map_s88.pt`.

QDR-222 adds the first successful hierarchical recovery controller on that
checkpoint. It does not change actor inputs or use privileged actor data: when
the primary target head is weak, it falls back to the model's own auxiliary
frontier-vector direction. The useful setting is
`--use-pred-vector-fallback`, `--pred-vector-fallback-primary-threshold 0.6`,
`--pred-vector-fallback-min-coverage 0.25`, and
`--pred-vector-fallback-min-dist 0.0`.
This improves procedural success to `83.38%` mean on seeds `88/90/91`, lowers
timeout to `16.32%`, keeps obstacle contacts at `0.30%`, and preserves fixed
retention at `99.91%`. The hard-state replay gate remains unsolved at `0/64`,
so this is a better controller, not a finished recovery model.

QDR-223 showed that repairing the auxiliary vector-distance scalar by updating
the whole recurrent policy is destructive: procedural success fell to `64.68%`
despite the scalar becoming nonzero. QDR-224 isolates that repair by freezing
the policy and training only decoder mean row 10. This keeps the QDR-222
controller behavior while making the distance gate deployable:
`experiments/qdr224/qdr224_qdr221_vector_dist_row_only_s88.pt` reaches
`83.39%` procedural mean, `16.31%` timeout, `0.30%` obstacle, and `99.91%`
fixed retention. Hard-state replay is still `0/64`.

Next resettable/perception milestone:

- keep QDR-224 as the current procedural candidate;
- train a proper deployable recovery head or sidecar gate so fallback decisions
  are learned from privileged late-state labels, not just primary confidence;
- use hard-state replay/probes to verify the new representation changes the
  late timeout states before running long sweeps.

Next exploration milestone:

- keep actor inputs deployable, but train a stronger objective/module than
  local coverage reward: planner-supervised frontier target prediction,
  auxiliary map/frontier prediction, asymmetric critic over map state, or
  resettable RL from exploration hard states;
- keep privileged map/frontier information out of actor observations unless it
  is explicitly derived from the actor's own observed map;
- judge progress by maze and full-gauntlet coverage, crash, and timeout, not
  just random-block success.

## Direction Pivot: Swift-Style Privileged Learning

As of May 19, 2026, the main research direction is no longer to keep tuning the
current waypoint-assisted velocity-control benchmark. That benchmark remains
useful as a regression and throughput harness, but it is not the path toward
research-grade indoor flight.

The new direction is to copy the structure used by the strongest agile-flight
systems while keeping PufferLib as the fast iteration engine:

- train with privileged information, but deploy policies that only consume
  real-drone-feasible observations;
- use a teacher, privileged critic, planner label, or distillation target
  instead of expecting PPO to rediscover navigation from sparse local signals;
- move the deployable action interface toward collective thrust plus body rates,
  with velocity control kept only as a scaffold/regression rung;
- model onboard sensing explicitly: IMU/proprioception, lidar/depth-like local
  geometry, previous actions, latency, noise, and actuator lag;
- evaluate on unseen layouts, long horizons, dynamics shift, sensor shift, and
  recovery, not just the training layouts.

The immediate implementation target is a "Swift-lite" PufferLib track:

1. Add an actor/privileged observation split or a staged teacher-distillation
   path if asymmetric critic support is too invasive for the first step.
2. Build a small state/gate or indoor-route benchmark where the actor sees only
   onboard-realistic state/range/goal features and previous action.
3. Keep the planner/geodesic/map information available only to the teacher,
   critic, reward, diagnostics, or reset curriculum.
4. Train and evaluate with thrust/body-rate actions once the scaffolded version
   has a clean pass/fail loop.

Do not spend more GPU time trying to push the current 4.0 waypoint velocity
benchmark from `93-98%` to `99%` unless the change directly supports this new
track or protects against regression.

First proof target: `QDR-130 swiftlite_privileged_distill_direct_actor_v0`.
This uses a staged teacher-distillation path before implementing asymmetric PPO:
two synchronized PufferLib envs run the same states/actions, the privileged
teacher view exposes waypoint observations, and the deployable student view sees
direct final-target observations plus previous action appended by the trainer.
The proof is only accepted if the student beats the known direct-from-zero cliff
and reaches `>90%` deterministic success on the one-goal mixed-layout eval.

QDR-130 result: passed the first proof with a recurrent student. The analytic
privileged waypoint teacher scored `99.98%`. A feedforward student only reached
`57.86%`, but a MinGRU student reached `90.96%`/`91.19%`/`91.55%` on eval
seeds `88`/`90`/`91`. Remaining failures are obstacle contacts. This validates
the new direction and shows the direct actor needs memory; it does not yet solve
the harder indoor-navigation problem.

QDR-131 through QDR-133 result: imitation-only sequence distillation did not
improve QDR-130. Reset-aligned BPTT, online BPTT, and short-chunk online BPTT
all failed in pure-student eval (`48-56%` success) despite clean mixed-rollout
losses. This is a useful negative result: the student is failing under its own
induced state distribution, so the next rung should follow the research pattern
of distillation followed by RL fine-tuning or an asymmetric critic, not more
teacher-command cloning.

QDR-134 bridge: exact fine-tuning needs the QDR-130 deployable observation
contract, including previous action. Cropping the previous-action columns to fit
the old native env dropped the converted student to `43.09%`, so that bridge was
invalid. Native `action_history_obs=1` now appends the previous command in the C
env, giving a `60`-float observation. With that bridge, the converted QDR-130
student evaluates at `91.01%`, matching the original result. Plain PPO
fine-tuning is not enough yet: the best low-std fine-tune was `90.95%`, and
final checkpoints drifted lower. The next research step is a safer fine-tune:
value warmup, actor regularization, lower policy LR/clip, or an
asymmetric/privileged critic.

QDR-135 result: the first asymmetric actor-critic probe works. The actor kept
deployable direct-target observations plus previous action, while the critic saw
privileged waypoint/planner observations in a synchronized env. With a light
KL anchor to QDR-130, the best checkpoint improved deterministic success to
`91.94%`/`92.24%`/`92.56%` on seeds `88`/`90`/`91`. This is only about a
one-point gain, but it is the first method after QDR-130 that improves held-out
deployable-direct performance. The failure mode remains obstacle contact, and
longer training drifts, so the next work is stabilizing asymmetric PPO rather
than adding more environment complexity.

QDR-136 result: hard-state recovery is useful when it is small. Collecting
near-crash windows from the QDR-135 actor and applying one epoch of privileged
waypoint recovery cloning improved the deployable direct actor to
`94.90%`/`95.21%`/`94.61%` on seeds `88`/`90`/`91`. All remaining failures are
still obstacle contacts. Two epochs were weaker and four epochs erased the gain,
so this should become an online hard-state replay/correction inside the
asymmetric trainer, not another long imitation stage.

QDR-137 result: simple online replay almost works, but does not beat QDR-136.
Adding a near-failure replay buffer to asymmetric PPO produced a best checkpoint
at `94.79%`/`94.76%`/`94.67%`, still slightly below QDR-136's mean. The useful
lesson is that hard-state recovery is real, but sampled PPO rollouts are a
worse source of recovery data than deterministic hard-state collection. The next
version should add deterministic hard-state collection, resettable hard-state
replay, or a DAgger-style loop rather than only changing replay weights.

QDR-138 result: the next harder direct-actor limit is sequential routing, not
small height or dynamics shifts. QDR-136 stays near `93%` under target-height
variation and light native dynamics randomization, but drops to about `48%` on
three sequential goals and `36%` on five goals. QDR-137 is not more robust. The
next rung should train post-goal transitions and late-route hidden-target states
explicitly, while preserving the QDR-136 one-goal gate.

QDR-139 result: sequential hard-state recovery helps, but makes the remaining
failure sharper. Collecting `direct_goals3` failure windows from QDR-136 and
training a smaller `1e-5` recovery pulse keeps one-goal success near `95%` while
raising three-goal success to about `65%` and five-goal success to about `46%`.
The new failures are mostly obstacle contacts, not timeouts, so the next step is
not "more recovery"; it is safer recovery with clearance/speed anchoring or
resettable hard-state replay around post-goal transitions.

QDR-140 result: the new broad goal's baseline is reconfirmed. QDR-136 is the
safe one-goal checkpoint (`94.91%`) but weak on direct sequential routing
(`48.41%` on `direct_goals3`). QDR-139 is the sequential baseline to beat:
`94.68%` one-goal, `65.05%` `direct_goals3`, and `45.88%` `direct_goals5`.
Its `direct_goals3` failures are almost entirely obstacle contacts, so the next
method-family tests must improve success and reduce contact rate.

QDR-141 through QDR-145 result: the sequential direct-routing limit moved, but
only marginally. Clearance-speed anchoring (QDR-141) preserved one-goal success
but lost the sequential gain by reintroducing timeouts. A second induced-state
DAgger-style recovery pulse (QDR-142) saturated around QDR-139. The first
deployable-perception branch (QDR-143) exposed an infrastructure blocker: 4.0
quadrotor observation size is fixed at build time, so `route_perception_obs`
needs a separate rebuild/worktree instead of a runtime flag. Mixed-length
hard-state recovery (QDR-144) became the first marginal pass, and a very short
asymmetric privileged-critic PPO fine-tune improved it slightly at update 5.
Current strongest checkpoint:
`experiments/qdr145/asym_mixed15_from_qdr144_s88/qdr135_asym_ppo_000005.bin`.
Held-out means are `94.87%` one-goal, `65.44%` `direct_goals3` with `34.56%`
obstacle contacts, `50.75%` `direct_goals5`, and `54.70%` `direct_combo`.
This is not solved; the improvement over QDR-139 is about `+0.39` points on
the main three-goal gate and combo is still below QDR-139. The next real step
is resettable hard-state replay or a clean rebuilt deployable perception track,
not longer shallow PPO from the same checkpoint.

QDR-146 through QDR-150 result: broader perception/training probes did not
break the plateau. A fixed-ABI local affordance observation (`12` raw rays plus
target-aligned gap features) was valid but harmful: `54.87%` one-goal and
`26.63%` three-goal, despite a `95.97%` privileged teacher. Raw `24`-ray
native-action-history distillation was better but still weak: `67.44%`
one-goal and `40.58%` three-goal. Native PPO from zero peaked at only `30.94%`
stochastic three-goal success, full student-rollout DAgger regressed to
`37.21%`, and asymmetric PPO from the weak cloned actor collapsed. The useful
conclusion is that the next build should not be another hand-designed
affordance or plain PPO run. We need resettable hard-state replay and, after
that, a deployable spatial memory representation that can be tested on the
same failing states.

QDR-151 through QDR-153 result: resettable replay is now an available diagnostic
primitive, but it exposed a stronger blocker. `VecEnv.snapshot(env_id)` and
`VecEnv.restore(env_id, snapshot)` can reproduce a QDR-145 pre-crash trajectory
with zero action/observation/reward/terminal drift when the evaluator also
restores MinGRU state. On that exact replay seam, however, the privileged
waypoint teacher is not a reliable recovery expert: at `rewind=8` it rescued
only `14/128` hard states, and at `rewind=64` with a longer horizon it rescued
`32/64`. Adding a local ray-repulsion teacher did not improve the ceiling. The
next build should therefore be a better privileged expert or a resettable
hard-state RL objective, not more cloning of the current waypoint labels.

QDR-154 result: combining resettable hard states with shallow privileged
rollout search is not enough. A bounded local MPC-style teacher over constant
velocity actions rescued at best `33/64` hard states versus `32/64` for the
plain waypoint teacher on the same setup. This proves the machinery works but
also rules out cheap local action search as the missing expert. The next
privileged build should use real route/obstacle trajectory optimization or
train a recovery policy directly from resettable hard states.

## May 19, 2026 Next Rungs

PufferLib remains the main training and environment stack. The next rungs are
implemented as named presets so we can push harder ideas without hiding setup in
long environment strings:

- `route_layout_mixed_finaltarget_waypoint_level`: current QDR-113/QDR-114
  recipe as a one-command preset. It keeps the planner waypoint, appends
  final-target context, trains route lengths 1..5 by default, and uses stronger
  clearance shaping.
- `route_layout_mixed_direct_level`: reduced-waypoint-dependence probe. The
  actor sees the direct final target in the primary target slots. The internal
  route/planner machinery still provides rewards and diagnostics, so this is a
  direct-target actor benchmark, not fully planner-free navigation.
- `route_layout_mixed_swiftlite_rate`: first state-racing rung. It switches from
  the velocity-action interface to hover-centered thrust plus body rates, keeps
  state/ray/waypoint observations, appends one previous action, and exposes
  `QUAD_MAX_RATE`.
- `QUAD_DRONE_PROFILE=latency_v0|micro_indoor_v0`: named real-drone modelling
  hooks for action delay, previous-action observation, sensor/action noise, and
  simple per-episode mass/thrust/drag/motor-time-constant randomization. These
  are placeholders for measured drone parameters, not calibrated physics yet.

Example commands:

```bash
scripts/quadrotor_remote_gpu.sh train route_layout_mixed_finaltarget_waypoint_level 75000000 91 none

scripts/quadrotor_remote_gpu.sh train route_layout_mixed_direct_level 50000000 91 none

scripts/quadrotor_remote_gpu.sh train route_layout_mixed_swiftlite_rate 50000000 91 none

QUAD_DRONE_PROFILE=micro_indoor_v0 scripts/quadrotor_remote_gpu.sh train route_layout_mixed_finaltarget_waypoint_level 75000000 91 none
```

First-pass results:

- QDR-114: final-target plus 1..5 route-length curriculum is strong on fixed
  5-goal routes across seeds (`97.71%`/`96.24%`/`95.17%`), but not stress
  robust. Seed 90 falls to `85.19%` mean stress, with `goals_10=52.05%` and
  `combo_long=58.89%`.
- QDR-115: direct final-target actor training fails at `40.65%`; the waypoint
  scaffold is still carrying essential local navigation information.
- QDR-116: the first rate-action Swift-lite rung fails at `43.03%`; lower-level
  control needs its own hover/move curriculum before obstacle routing.
- QDR-117: `micro_indoor_v0` profile plumbing works for train and eval. The
  first profile run reaches `84.67%`, so latency/noise/dynamics tolerance is a
  real next gap.

## May 18, 2026 Current Reduced Benchmark

The fixed-altitude reduced benchmark is much cleaner than where we started, but
it is not yet proven reliable. QDR-055 showed the earlier `wall` failures were
actually ceiling exits. QDR-056/QDR-057 showed that lowering vertical authority
to `QUAD_MAX_Z_VEL=0.25` removes that class and can train a very strong run.

QDR-058 is the current robustness check. Seeds `88` and `90` trained from zero
to `93.18%` and `98.05%` deterministic success with no wall, ground, or timeout
failures. A seed `91` retrain reached only `86.54%`, while the older QDR-057
seed `91` checkpoint still evaluates at `99.02%` under the current code. The
recipe is therefore promising but still training-run fragile.

The next hard limit is hidden/low-clearance first-waypoint obstacle contact.
QDR-058 seed `91` diagnostics found all failures were obstacle hits, with
failed target visibility `0.145` versus `0.410` for reached waypoints and
failed clearance-at-min `0.143m`.

QDR-059 retested compact target-aligned perception under the lower-z recipe on
the fragile seed `91`. It improved final deterministic success from QDR-058's
`86.54%` to `91.99%`, with no wall, ground, or timeout failures. QDR-060 then
ran the compact recipe as a three-seed gate and failed: seed `88` reached only
`85.37%`, seed `90` reached `89.84%`, and seed `91` stayed at `91.99%`. Compact
perception helps one seed but worsens the previously strong seeds, so it should
not be promoted.

QDR-061 tested the simpler waypoint-geometry hypothesis with
`QUAD_ROUTE_PATH_MARGIN=0.50` and `QUAD_ROUTE_PERCEPTION_OBS=0` on seed `91`.
It reached `95.66%` best and `95.12%` final deterministic success, with no wall,
ground, or timeout failures. This is better than both QDR-058's `86.54%` and
QDR-059's compact-observation `91.99%` on the same seed.

QDR-062 passed the small reduced benchmark gate. Final deterministic success:
seed `88` `96.09%`, seed `90` `93.95%`, seed `91` `95.12%`. Failures are only
obstacle contacts; there are no wall, ground, or timeout failures. The current
reduced benchmark is therefore:

```bash
QUAD_MAX_Z_VEL=0.25
QUAD_ROUTE_TARGET_WALL_MARGIN=1.0
QUAD_ROUTE_PATH_MARGIN=0.50
QUAD_ROUTE_WALL_OBS=1
QUAD_ROUTE_ARRIVAL_RADIUS=1.5
QUAD_ROUTE_ARRIVAL_BONUS=0.04
QUAD_ROUTE_ARRIVAL_VEL_SCALE=0.02
QUAD_ROUTE_REACH_BONUS=5.0
QUAD_ACTION_STD=0.15
QUAD_TRAIN_ACTION_STD=true
QUAD_ROUTE_DIRECT_OBS=0
QUAD_ROUTE_PERCEPTION_OBS=0
```

This is still waypoint-assisted one-goal center-block navigation, not general
indoor exploration. The next harder limit is residual first-waypoint obstacle
contact and generalization beyond one block/layout.

## May 17, 2026 Fixed-Altitude Diagnosis

The previous `route_center_block_waypoint_flat` benchmark was not the clean
indoor-hover obstacle task we thought it was. Its route target height was
`z=0.0`, while episodes spawn the drone near `z=1.5`. The best old deterministic
checkpoint learned to descend close to the floor target without quite crashing;
sampled-policy eval often turned that into ground hits. That explains a large
part of the "good checkpoint but stupid visual behavior" loop.

Keep the old flat preset as historical data, but use the new fixed-altitude
benchmark for the next rung:

```bash
scripts/quadrotor_remote_gpu.sh train route_center_block_waypoint_level 30000000 88 none

scripts/quadrotor_remote_gpu.sh eval-checkpoints route_center_block_waypoint_level 4 512 latest

scripts/quadrotor_visual_eval.sh route_center_block_waypoint_level
```

The new preset is the same one-goal center-block waypoint task, but with
`route_target_z_min=1.5`, `route_target_z_max=1.5`, feedforward velocity control,
fixed `action_std=0.15`, and LR annealing disabled by default. This is still a
waypoint-assisted benchmark, not pure vision navigation.

Current selected visual checkpoint:

```text
experiments/puffer_quadrotor_177905227205/model_puffer_quadrotor_000035.pt
```

On a 2048-episode deterministic eval it reached `95.02%` success, `4.24%`
center-block hits, `0.00%` wall hits, `0.00%` ground hits, and `0.73%`
timeouts. The same fixed-altitude task also trained from zero with the normal
annealed schedule on two seeds: seed `88` peaked at `90.93%`, and seed `89`
peaked at `92.44%`. That is a much more honest indoor-flight benchmark than the
floor-target version, but it is not fully solved because later checkpoints still
collapse into wall-hit policies.

Negative probes:

- Per-axis std with small z/yaw noise did not fix training. With the old
  floor-target task it failed immediately; with the fixed-altitude target it
  still stayed near `5%` sampled success. The small vertical exploration made
  it hard to recover from early downward policies.
- Wider planner margin plus stronger clearance shaping reduced some sampled
  obstacle contact but worsened deterministic selected-checkpoint success to
  `87.30%`.
- Sampled-action eval remains a different deployment mode. On the fixed-altitude
  scalar-std run, the best mean checkpoint was `90.93%`, while the best sampled
  checkpoint was later and only `80.71%`.

Current interpretation: the root environment mistake was the floor-level target.
After fixing the target altitude, the remaining hard limit is PPO stability:
the policy reliably discovers a good mean-action route policy, then overtrains
into wall-hit modes. Until that is fixed, always select checkpoints by
deterministic checkpoint eval instead of trusting the final checkpoint or the
sampled training HUD.

## May 17, 2026 Diagnosis Checkpoint

The pre-obstacle route benchmark is no longer the blocker. After fixing the
controller and evaluation mismatches, a feedforward velocity policy trains from
zero on the open one-goal route task and reaches `98.8%` deterministic success
at the selected 10M-step checkpoint:
`experiments/puffer_quadrotor_177904357224/model_puffer_quadrotor_000040.pt`.

The first simple obstacle rung is now a flat center-block route benchmark:
one blocked target, fixed altitude, 24 horizontal rays, no wall/perception
observation blocks, and a planner-waypoint target observation. This is
scientifically useful but not pure vision-only navigation; the preset name keeps
that explicit. The best current from-zero checkpoint is
`experiments/puffer_quadrotor_177904387853/model_puffer_quadrotor_000040.pt`.
On a 2048-episode deterministic check it reached `87.7%` success, `10.4%`
center-block hits, and `2.0%` timeouts. A cautious PD waypoint baseline reaches
about `95.3%` on the same benchmark, so there is still a real policy gap.

Follow-up zero-start sweeps on May 17 clarified that this is still not a robust
solved benchmark. A 20M-step feedforward run at the default `lr=0.003`
(`experiments/puffer_quadrotor_177904540160`) learned the task but overtrained:
deterministic eval peaked at checkpoint `000040` with `81.95%` success, then
collapsed toward ground crashes. Lowering PPO learning rate to `0.001`
(`experiments/puffer_quadrotor_177904552433`) made the sampled training HUD look
stable around `93-95%`, but deterministic checkpoint eval still peaked at only
`69.76%`. Reducing initial action std to `0.03`
(`experiments/puffer_quadrotor_177904569833`) failed to learn reliably, peaking
at `19.77%`. Moving the planner waypoint outward with
`route_path_margin=0.25` (`experiments/puffer_quadrotor_177904577408`) also did
not close the gap, peaking at `72.85%`. Treat deterministic eval, not the PPO
sampled training HUD, as the pass/fail signal.

Concrete fixes made in this pass:

- `indoor_route` crash rewards now use the configured route crash penalty in
  reference reward mode. They previously fell through to a cheap `-1.0` crash.
- Rate, velocity, and hover-centered motor modes now reset with motors already
  at hover speed instead of zero RPM.
- Route target sampling now enforces a minimum start-to-target distance based
  on `route_goal_radius`; one-goal route episodes can no longer be solved at
  step 1 just because the random target spawned inside the success radius.
- The velocity/position controller had three control bugs: roll and pitch mixer
  axes were swapped, the attitude proportional sign was wrong, and yaw mixer
  sign disagreed with the physics yaw torque. These made simple x/y velocity
  commands drift diagonally and eventually crash.
- Velocity control is now recomputed inside the `action_repeat` physics loop.
  Previously the outer loop used stale thrust/rate targets across repeated
  substeps.
- Python eval/visual PD helpers now decode route target observations in the
  right frame and scale. Earlier diagnostics were partly measuring helper bugs,
  not the env.
- CLI `--rnn-name None` now disables the recurrent wrapper correctly. This was
  important because the simple one-goal velocity rung learns much better with a
  feedforward policy than with the default recurrent wrapper.

What the probes showed:

- `action_mode=rate` is not a from-zero benchmark action interface: action zero
  means zero thrust, not hover. It requires discovering a thrust bias before it
  can even be a flight task.
- `action_mode=motor_hover` has a sane zero-action hover baseline, but PPO from
  zero still collapses into high-speed divergence on open one-goal route flight.
- `action_mode=velocity` is now a useful low-level benchmark rung. It is not the
  final sim2real control interface, but it lets us separate navigation/reward
  failures from raw motor stabilization failures.
- The recurrent wrapper is harmful on the simple fully observable route rung:
  a 3M recurrent run stayed below `20%`, while a feedforward run reached about
  `87.5%` by 3M and `98.8%` by the selected 10M checkpoint.
- Direct final-target plus rays is still too hard: the center-block direct
  target run reached only about `8%` success and mostly hit the block.
- Planner-waypoint plus rays is the first meaningful obstacle result:
  checkpoint `000040` from `experiments/puffer_quadrotor_177904387853` reaches
  `87-89%` depending on eval horizon, but later checkpoints overtrain into more
  crashes. Select by deterministic eval, not final checkpoint.
- The sampled training HUD can be misleading for this continuous policy. Recent
  `lr=0.001` runs looked `93-95%` successful while deterministic mean-action
  eval peaked below `70%`, so visual/deployment checks must use deterministic
  checkpoint eval unless intentionally testing stochastic control.
- Masking rays on that checkpoint drops success from `87.7%` to `74.0%`, so the
  policy does use range sensing. Replacing the planner waypoint with the direct
  final-target observation drops it to about `20.4%`, so the waypoint still
  carries most of the obstacle-routing information.

The honest ladder is now:

1. Open one-goal velocity route from rest: solved enough for this rung.
2. Flat single center-block route with waypoint observation and rays: usable
   zero-start benchmark; current selected-checkpoint best is about `96%`.
3. Same center-block route without planner-waypoint observation: current direct
   target policy fails.
4. Random blocks and multi-goal routes.
5. Targetless exploration and richer perception.

The next harder limit is deterministic reliability on the flat center-block
benchmark. The current policy can fly and can follow the waypoint, but PPO still
overtrains into wall-hit modes after the best checkpoint. Compare every
candidate against the `95%` PD ceiling. After that, the real navigation leap is
direct-target plus rays without the planner waypoint.

## May 17, 2026 Zero-Start Obstacle Findings

The flat center-block route task is learnable from zero. The policy must be
judged by deterministic mean-action checkpoint eval, not by the sampled PPO HUD.
Several runs looked healthy in sampled eval while their deterministic policies
drove into walls.

Current default visual checkpoint:

```text
experiments/puffer_quadrotor_177904905573/model_puffer_quadrotor_000050.pt
```

On the 2048-episode checkpoint sweep it reached `96.39%` success, `3.07%`
center-block hits, `0.00%` wall hits, `0.24%` ground hits, and `0.29%`
timeouts. This came from the now-default zero-start command:

```bash
scripts/quadrotor_remote_gpu.sh train route_center_block_waypoint_flat 30000000 88

scripts/quadrotor_remote_gpu.sh eval-checkpoints route_center_block_waypoint_flat 4 512 latest

scripts/quadrotor_visual_eval.sh route_center_block_waypoint_flat
```

For this scenario the train wrapper now defaults to feedforward, zero-start,
fixed `action_std=0.15`, `train_action_std=false`, `ent_coef=0.0002`, and
`lr=0.003`. A source checkpoint can still be passed as the fifth argument.

Recent zero-start sweep:

- Fixed `action_std=0.15`: seed `88` peaked at `96.39%`; seed `89` peaked at
  `83.64%`. This is the current recommended simple benchmark recipe.
- Fixed `action_std=0.10`: seed `88` peaked at `95.17%`; seed `89` peaked at
  `59.52%`.
- Fixed `action_std=0.20`: seed `89` peaked at `92.68%`; seed `88` peaked at
  `31.72%`.
- Trainable std from zero with `action_std=0.20` and seed `89` peaked at only
  `46.58%`.

Other probes:

- `eval_hover.py` now supports `--action-selection mean|sample` and reports
  learned policy std. Mean-action eval remains the benchmark.
- On a good selected checkpoint, sampled actions can be worse than mean actions:
  `93.14%` mean success versus `0.17%` sampled success for
  `experiments/puffer_quadrotor_177904753160/model_puffer_quadrotor_000040.pt`.
- On a collapsed late checkpoint, sampled actions can hide mean-policy failure:
  `0.32%` mean success versus `57.59%` sampled success for
  `experiments/puffer_quadrotor_177904753160/model_puffer_quadrotor_000115.pt`.
- `ent_coef=0.0` with `action_std=0.2` is bad for deterministic control. The
  sampled HUD reached about `82%`, but deterministic eval peaked at only
  `9.37%` and mostly hit walls.
- A stronger route crash penalty (`route_crash_reward=-60`) improved sampled
  training to about `87%`, but deterministic eval peaked at only `38.53%`.
- Tightening PPO clip to `0.06` made the best checkpoint worse (`58.34%`).
- Shortening the run is not equivalent to selecting an early checkpoint because
  LR annealing changes the whole update schedule.

Interpretation: the current benchmark exposes a real PPO stability/mean-policy
problem, not a broken raycast or an impossible task. A fixed mid-sized action
std gives strong zero-start checkpoints across two seeds, but we do not yet have
a robust final-checkpoint result. The next limit is to make deterministic
control robust after convergence without adding a teacher or changing dynamics.

## Current Direction

The main line is now the custom `puffer_quadrotor` indoor env with Puffer
drone-style direct motor dynamics. The built-in `puffer_drone` env remains the
control benchmark: if our custom env regresses below it, the first suspect is
our task/observation/reward layer, not the underlying dynamics.

1. Keep Puffer-style RK4 motor dynamics and raw four-motor actions in our env.
2. Reuse our custom indoor task machinery: rooms, obstacles, ray sensing,
   exploration grids, diagnostics, visual eval, and sim2real knobs.
3. Validate against `puffer_drone_route` before adding complexity. The same
   basic waypoint task should be learnable in both stacks.
4. Add indoor geometry and local sensing only after direct-motor waypoint flight
   is solid in the custom env.
5. Move to exploration reward, recurrent memory, and camera/depth after route
   failures are visible and diagnosed in simpler settings.
6. Treat the older PID/velocity controller path as a teacher/diagnostic option,
   not as the target control stack.

## Current Environment Shape

- Package path: `pufferlib/ocean/quadrotor`.
- Registered PufferLib env: `puffer_quadrotor`.
- Observation: 26 Puffer-drone-style base state values plus optional horizontal
  range rays. The clean route line exposes the true current route target in the
  base observation and disables the optional wall/perception observation blocks.
  The old local-planner target path remains available only as a diagnostic.
- Actions: 4 continuous values.
- `action_mode = motor`: raw direct motor actions using Puffer-style dynamics.
- `action_mode = rate`: legacy thrust/body-rate control for old hover checkpoints.
- `action_mode = velocity`: legacy body-frame `vx`, body-frame `vy`, world `vz`, yaw-rate.
- `task_mode = indoor_nav`: random start and random point goal inside a square room.
- `task_mode = indoor_route`: chain several random point goals inside one episode.
- `task_mode = indoor_explore`: randomized indoor exploration with internal coverage reward.
- Room sensing: `room_size`, `ceiling_height`, `num_rays`, `ray_length`.
- Exploration reward: new grid-cell coverage, altitude/clearance regularization, crash penalty, and an initial `10%` coverage success threshold.
- Stress knobs: `obstacle_mode`, `obs_noise`, `action_noise`, fixed
  `action_delay`, randomized per-episode `action_delay_min/max`, and fixed
  dynamics scales for mass, inertia, thrust, torque, drag, and motor lag.

Default config now targets the direct-motor route scaffold:

```bash
.venv/bin/python -m pufferlib.pufferl train puffer_quadrotor
```

## What We Know

- Legacy hover/recovery is solved well enough for now by
  `experiments/puffer_quadrotor_177797599382/model_puffer_quadrotor_000077.pt`.
- Current default exploration checkpoint by the safety-weighted validation
  score is `experiments/puffer_quadrotor_177887664537/model_puffer_quadrotor_000010.pt`.
  It was fine-tuned from the open-room policy on `random_blocks`. On the robust
  random-block validation suite it kept about `75%` success while cutting crash
  rate from about `17%` to `7%`; the tradeoff is more timeouts. This is progress,
  not a solved explorer.
- A later mixed open/random-block curriculum run added `obstacle_mode=mixed`.
  Its best checkpoint, `experiments/puffer_quadrotor_177887800450/model_puffer_quadrotor_000020.pt`,
  improved the mixed-suite score, but it gave back open-room robustness and added
  wall crashes. It should not replace the current default yet.
- Follow-up low-LR continuations on May 16, 2026 did not beat the incumbent.
  The best open-biased mixed run, `experiments/puffer_quadrotor_177888042110/model_puffer_quadrotor_000002.pt`,
  reached `0.8613` on the mixed suite, but the incumbent reached `0.8795` on the
  same suite. A block-focused continuation, `experiments/puffer_quadrotor_177888063463/model_puffer_quadrotor_000002.pt`,
  reduced crash rate on `blocks_hard` from `7.15%` to `5.80%`, but success fell
  from `74.75%` to `71.47%`, so it is also not promoted.
- Automated rollout diagnostics now live in `diagnose_explore.py`. The current
  policy is not mainly failing by flying straight into a visible front wall.
  It has a stereotyped backward-left sweep with positive yaw. In random-block
  rooms, crashes are usually side/rear contacts while the front ray remains
  clear; timeouts are slower episodes with high blocked fraction and fewer
  unique cells. A simple ray-reactive policy avoids the center-block crashes
  but times out more, which points to policy/reward/curriculum issues rather
  than a broken raycast.
- Safety shaping has started with an obstacle inward-motion penalty for
  `indoor_explore`. It only activates close to block surfaces and penalizes
  actual or commanded velocity into the nearest block. The first GPU
  continuation produced `experiments/puffer_quadrotor_177891825668/model_puffer_quadrotor_000002.pt`.
  It reduced `blocks_hard` crash rate from `7.15%` to `5.36%`, but success fell
  from `74.75%` to `71.79%`. On the full non-obstacle suite it fell from
  `93.78%` to `87.42%` success. It is useful evidence but should not be
  promoted over the incumbent.
- The room/raycast scaffold works and produces `16 + num_rays` observations.
- Fresh PPO with raw thrust/body-rate actions in a room collapsed into wall/ceiling crashes.
- That failure is why the env now has `action_mode = velocity`.
- Plain PPO from scratch with `indoor_nav + velocity` improved distance but did not solve navigation. Best deterministic success from the long scratch run was about `7%`.
- Controller cloning was the best bootstrap for point-goal navigation. `experiments/quadrotor_nav_clone/pd_velocity_clone.pt` reached about `95%` deterministic success, and a cautious PPO fine-tune reached about `96%` at `experiments/puffer_quadrotor_177809531177/model_puffer_quadrotor_000060.pt`.
- Exploration is now a separate task. It does not expose a point-goal target; the policy gets local dynamics state plus range rays, and reward comes from covering new free-space grid cells.
- The new `indoor_route` task is the next curriculum rung. It keeps the same
  observation/action shape as exploration, but exposes a target offset and
  requires reaching `3` waypoints in one episode. A PD velocity baseline solves
  open-room routes at `100%` on a 512-episode local check, while direct PD in
  random-block rooms only reached about `37%` and mostly crashed into obstacles.
  That is the intended pressure: learn obstacle-aware route following before
  returning to targetless exploration.
- Scratch PPO on `route_open` failed: a 50M-step GPU run collapsed into wall
  crashes. A reset-state recurrent controller clone reached about `84%` local
  open-route success, and PPO fine-tuning from it peaked at
  `experiments/puffer_quadrotor_177891984495/model_puffer_quadrotor_000020.pt`
  with about `90%` open-route success. The final checkpoint was worse, so use
  checkpoint selection.
- Random-block route following is now the active bottleneck. The open-route
  checkpoint transfers to about `34%` success on random blocks. A 25M-step
  random-block fine-tune peaked at
  `experiments/puffer_quadrotor_177891996830/model_puffer_quadrotor_000025.pt`
  with about `38%` success; that is only a small improvement. A simple
  target-PD plus ray-repulsion teacher (`--include-raypd`) reaches about
  `58%` success locally, so the next bootstrap should clone this obstacle-aware
  teacher before further PPO.
- Pure behavioral cloning of the `raypd` route teacher was not enough. A
  recurrent reset-state clone reached about `41%` random-block route success,
  and PPO from it peaked at about `43%`. The failure was distribution shift:
  the clone had low supervised loss on teacher trajectories but was brittle in
  closed loop.
- DAgger-style cloning fixed most of that gap. The clone
  `experiments/quadrotor_route_clone/raypd_route_blocks_24ray_recurrent_dagger.pt`
  reached about `56%` success on the 1024-episode local random-block route
  check, close to the `raypd` teacher at about `60%`. Conservative PPO from
  that clone peaked at
  `experiments/puffer_quadrotor_177892235054/model_puffer_quadrotor_000010.pt`
  with about `57.5%` success, `37%` obstacle hits, and `5%` timeouts. This is
  the current route policy to inspect visually. It is a major improvement over
  the old PPO route policy, but it has not beaten the teacher yet.
- A route reward safety-shaping test on May 16, 2026 did not provide a clear
  win. Its best checkpoint was statistically tied with the old PPO result and
  did not reduce obstacle hits. The shaping was reverted; keep the simpler
  route reward until a diagnostic shows exactly what term is missing.
- The custom PID/velocity route stack is no longer the primary path. Its best
  DAgger/PPO policy is close to the simple ray-PD teacher, but both are far
  behind the capability implied by upstream `puffer_drone`. The new benchmark is
  `puffer_drone_route`, which keeps upstream direct motor actions and adds a
  multi-waypoint route objective.
- The custom env now has Puffer-style RK4 motor dynamics and a 26-feature
  Puffer-compatible base observation. With `num_rays=0`, the strong
  `puffer_drone_route` checkpoint loads into `puffer_quadrotor`; appended ray
  sensors should be introduced through checkpoint expansion or fine-tuning.
- May 16, 2026 direct-motor baseline: the first custom-env transfer run
  `experiments/puffer_quadrotor_177892945856` was initially underreported
  because custom route success was hard-coded at `0.30m`, while upstream
  `puffer_drone_route` uses a `1.0m` route goal radius. `route_goal_radius` is
  now a first-class env/script parameter.
- With `route_goal_radius=1.0`, `num_rays=0`, and `route_goal_count=3`,
  `experiments/puffer_quadrotor_177892945856/model_puffer_quadrotor_000130.pt`
  reaches about `86.6%` deterministic open-room 3-goal success. A short 3-goal
  continuation improved this to about `89.6%`:
  `experiments/puffer_quadrotor_177893026034/model_puffer_quadrotor_000020.pt`.
  Later checkpoints degrade, so choose checkpoints by deterministic sweeps, not
  by final training output.
- First 24-ray random-block route attempt:
  `experiments/puffer_quadrotor_177893026034/model_puffer_quadrotor_000020_rays24.pt`
  starts at about `39.1%` success and `53.8%` obstacle hits. A 25M-step PPO
  continuation peaked at only `40.7%` success with `53.3%` obstacle hits
  (`experiments/puffer_quadrotor_177893054959/model_puffer_quadrotor_000010.pt`)
  and later degraded. This is not good enough; the policy is mostly preserving
  open-room flight and not learning obstacle avoidance from rays yet.
- Local planner target observations were the first clean route-obstacle win.
  Keeping the final goal/reward unchanged but exposing the next path corner as
  the target offset raised dense random-block route success for
  `experiments/quadrotor_route_obstacle_curriculum_20260516_153954/mixed_25/model_puffer_quadrotor_000010.pt`
  from about `46.7%` to about `52.5%` on a 5120-episode check, while preserving
  open-route performance at about `94.4%`. A short low-LR continuation from
  that checkpoint did not improve the new benchmark, so do not promote
  `experiments/quadrotor_route_planner_obs_low_lr_20260516_1604/model_puffer_quadrotor_000005.pt`.
- The local-navigation benchmark now lives at
  `scripts/quadrotor_local_nav_benchmark.sh`. The May 16, 2026 full benchmark
  for the planner-observation baseline is
  `experiments/quadrotor_local_nav_benchmark_compare_20260516_162045`: mean
  success `55.6%`, open density `94.5%`, dense random blocks `52.7%`,
  512-step dense blocks `51.7%`, `0.75m` goal radius `34.9%`, and `0.50m`
  goal radius `0.05%`. This says obstacle handling is still weak, but precision
  near the waypoint is the sharpest current failure.
- Precision-reward shaping did not solve waypoint arrival. A May 16, 2026 GPU
  sweep over tighter goal radii, near-goal bonuses, and near-goal velocity
  penalties (`experiments/quadrotor_precision_experiment_20260516_165834`)
  mostly degraded with training. Velocity-penalty-only shaping collapsed, and
  the tiny `0.50m` success bump from the `bonus_slow_050` early checkpoint was
  not useful because it lost too much general route success.
- Early stopping on a plain `0.75m` continuation was a small measurable win but
  not a new capability. In the full local benchmark
  `experiments/quadrotor_precision_best_full_20260516_1730`,
  `experiments/quadrotor_precision_earlystop_20260516_1710/plain_075/model_puffer_quadrotor_000003.pt`
  improved mean success from `55.6%` to `56.7%`, dense random blocks from
  `52.7%` to `53.4%`, `0.75m` precision from `34.9%` to `36.0%`, and mean
  obstacle hits from `28.7%` to `27.5%`. The `0.50m` goal-radius case stayed at
  `0.05%`, so this should be treated as a slightly better checkpoint selection,
  not as a solution to precise waypoint arrival.
- Route-arrival diagnostics now live at
  `pufferlib/ocean/quadrotor/diagnose_route.py` and
  `scripts/quadrotor_diagnose_route.sh`. On the current best route checkpoint,
  the default `1.0m` radius diagnostic
  `experiments/quadrotor_route_diagnostics_20260516_current` found `55.3%`
  episode success and `82.9%` waypoint reach; failed waypoints almost never
  crossed the goal plane and `99.1%` belonged to obstacle/wall crash episodes.
  This says the default-radius bottleneck is still obstacle route following.
  At `0.75m`, success fell to `34.8%`; only `3.6%` of failed waypoints crossed
  the goal plane, but `25.7%` were near misses. At `0.50m`, success was `0%`,
  `60.5%` of failed waypoints were near misses, and `31.2%` crossed the goal
  plane. This is the first clean evidence that sub-meter precision is a
  separate arrival-control problem from the random-block route task.

## Useful Commands

Rebuild the C extension after C changes:

```bash
.venv/bin/python setup.py build_quadrotor --inplace --force
.venv/bin/python setup.py build_drone --inplace --force
```

Train the current open one-goal velocity route rung from zero:

```bash
QUAD_RNN_NAME=None scripts/quadrotor_train_obstacle.sh route_open_velocity_flat 10000000 71 none
```

Evaluate the selected open-route checkpoint:

```bash
scripts/quadrotor_eval_latest.sh route_open_velocity_flat 8 256 \
  experiments/puffer_quadrotor_177904357224/model_puffer_quadrotor_000040.pt
```

Train the current flat center-block waypoint benchmark from zero:

```bash
QUAD_ACTION_MODE=velocity QUAD_RNN_NAME=None \
  scripts/quadrotor_train_obstacle.sh route_center_block_waypoint_flat 20000000 81 none
```

Evaluate all saved checkpoints deterministically and select by success, not by
the final checkpoint or PPO training HUD:

```bash
QUAD_EVAL_RNN_NAME=None \
  scripts/quadrotor_eval_checkpoints.sh route_center_block_waypoint_flat 4 512 \
  experiments/puffer_quadrotor_177904540160
```

Evaluate and visually inspect the selected center-block checkpoint:

```bash
scripts/quadrotor_eval_latest.sh route_center_block_waypoint_flat 8 256 \
  experiments/puffer_quadrotor_177904387853/model_puffer_quadrotor_000040.pt
QUAD_VISUAL_FETCH=0 scripts/quadrotor_visual_eval.sh route_center_block_waypoint_flat
```

Run the matching route-failure diagnostic:

```bash
QUAD_DIAG_BUILD=0 scripts/quadrotor_diagnose_route.sh \
  experiments/puffer_quadrotor_177904387853/model_puffer_quadrotor_000040.pt \
  --episodes 512 --num-envs 64 --max-steps 512 --route-goal-count 1 \
  --route-goal-radius 1.0 --route-target-mode blocked \
  --route-blocked-target-prob 1.0 --route-target-z-min 0.0 \
  --route-target-z-max 0.0 --route-walls 1 --route-direct-obs 0 \
  --route-wall-obs 0 --route-perception-obs 0 --route-target-obs-mode body \
  --obstacle-mode center_block --num-rays 24 --ray-length 4.0 \
  --room-size 3.0 --spawn-radius 2.5 --action-mode velocity \
  --action-repeat 1 --action-std 0.1 --rnn-name None
```

Train the new upstream-drone waypoint benchmark locally:

```bash
.venv/bin/python -m pufferlib.pufferl train puffer_drone_route
```

Train it on the remote GPU helper:

```bash
scripts/quadrotor_remote_gpu.sh train-drone-route 50000000 1
```

Run the maintained visual scenarios:

```bash
.venv/bin/python -m pufferlib.ocean.quadrotor.render_hover --scenario room-nav
.venv/bin/python -m pufferlib.ocean.quadrotor.render_hover --scenario explore-random
.venv/bin/python -m pufferlib.ocean.quadrotor.render_hover --scenario hover
.venv/bin/python -m pufferlib.ocean.quadrotor.render_hover --scenario recovery
```

Run the local route-navigation benchmark:

```bash
scripts/quadrotor_local_nav_benchmark.sh
scripts/quadrotor_local_nav_benchmark.sh --suite quick
scripts/quadrotor_local_nav_benchmark.sh checkpoint_a.pt checkpoint_b.pt
```

Run the reference-parity direct-motor route ladder:

```bash
scripts/quadrotor_route_reference_ladder.sh
```

Run route-arrival diagnostics:

```bash
scripts/quadrotor_diagnose_route.sh
scripts/quadrotor_diagnose_route.sh --route-goal-radius 0.75
scripts/quadrotor_diagnose_route.sh --route-goal-radius 0.50
```

Train the current exploration task:

```bash
scripts/quadrotor_train_obstacle.sh explore 50000000 1 none
```

Train the current route-following curriculum:

```bash
QUAD_ACTION_MODE=motor QUAD_ACTION_REPEAT=1 QUAD_NUM_RAYS=0 \
  QUAD_ROUTE_GOAL_RADIUS=1.0 QUAD_ROUTE_GOAL_COUNT=3 \
  scripts/quadrotor_train_obstacle.sh route_open 50000000 1 none
QUAD_ACTION_MODE=motor QUAD_ACTION_REPEAT=1 QUAD_ROUTE_GOAL_RADIUS=1.0 \
  scripts/quadrotor_train_obstacle.sh route 50000000 2 \
  experiments/puffer_quadrotor_<run_id>/model_puffer_quadrotor_<update>.pt
```

On the remote GPU helper:

```bash
scripts/quadrotor_remote_gpu.sh train explore 50000000 1 none
scripts/quadrotor_remote_gpu.sh eval explore
scripts/quadrotor_remote_gpu.sh eval-checkpoints explore
scripts/quadrotor_remote_gpu.sh validate latest-dir
QUAD_VALIDATE_SUITE=blocks_hard scripts/quadrotor_remote_gpu.sh validate latest-dir
```

Bias the mixed open/random-block curriculum when testing curriculum balance:

```bash
QUAD_OBSTACLE_BLOCK_PROB=0.25 scripts/quadrotor_remote_gpu.sh train explore_mixed 20000000 8 \
  experiments/puffer_quadrotor_177887664537/model_puffer_quadrotor_000010.pt
```

Run the default local visual exploration eval. If the checkpoint is not present
locally, the script fetches it from the configured GPU host:

```bash
scripts/quadrotor_visual_eval.sh best
scripts/quadrotor_visual_eval.sh blocks
scripts/quadrotor_visual_eval.sh mixed
scripts/quadrotor_visual_eval.sh route_open
scripts/quadrotor_visual_eval.sh route
scripts/quadrotor_visual_eval.sh route_combo
```

Run the route stress gauntlet on one checkpoint or every checkpoint in an
experiment directory:

```bash
scripts/quadrotor_remote_gpu.sh route-gauntlet \
  experiments/puffer_quadrotor_177901561690/model_puffer_quadrotor_000075.pt frontier
QUAD_GAUNTLET_CASES=combo scripts/quadrotor_remote_gpu.sh route-gauntlet \
  experiments/puffer_quadrotor_177901561690 frontier
```

Evaluate a policy on indoor navigation:

```bash
.venv/bin/python -m pufferlib.ocean.quadrotor.eval_hover \
  --task-mode indoor_nav \
  --action-mode velocity \
  --room-size 2.0 \
  --num-rays 16 \
  --spawn-radius 1.5 \
  --max-steps 512 \
  --episodes 8 \
  experiments/puffer_quadrotor_<run_id>/model_puffer_quadrotor_<update>.pt
```

Check the PD velocity-setpoint baseline:

```bash
.venv/bin/python -m pufferlib.ocean.quadrotor.eval_hover \
  --include-pd \
  --task-mode indoor_nav \
  --action-mode velocity \
  --room-size 2.0 \
  --num-rays 16 \
  --spawn-radius 1.5 \
  --max-steps 512
```

Check the obstacle-aware target/ray teacher on route tasks:

```bash
.venv/bin/python -m pufferlib.ocean.quadrotor.eval_hover \
  --include-raypd \
  --task-mode indoor_route \
  --action-mode velocity \
  --room-size 3.0 \
  --num-rays 24 \
  --ray-length 4.0 \
  --spawn-radius 2.5 \
  --max-steps 768 \
  --obstacle-mode random_blocks
```

Clone the obstacle-aware route teacher with DAgger:

```bash
.venv/bin/python -m pufferlib.ocean.quadrotor.train_nav_clone \
  --output experiments/quadrotor_route_clone/raypd_route_blocks_24ray_recurrent_dagger.pt \
  --load-model-path experiments/quadrotor_route_clone/raypd_route_blocks_24ray_recurrent_reset.pt \
  --teacher raypd \
  --rollout-policy mix \
  --student-rollout-prob 0.35 \
  --student-warmup-updates 100 \
  --task-mode indoor_route \
  --room-size 3.0 \
  --spawn-radius 2.5 \
  --max-steps 768 \
  --num-rays 24 \
  --ray-length 4.0 \
  --obstacle-mode random_blocks \
  --route-goal-count 3 \
  --rnn-name Recurrent
```

Run the stress gauntlet:

```bash
.venv/bin/python -m pufferlib.ocean.quadrotor.stress_eval --quick
```

The gauntlet is intentionally evaluation-only. It changes sensors, latency, dynamics, and geometry through explicit environment parameters; it does not change the reward to make a policy look better.

## 2026-05-16 Route Precision Findings

Current route policy:

```bash
experiments/quadrotor_precision_earlystop_20260516_1710/plain_075/model_puffer_quadrotor_000003.pt
```

The best relaxed route policy is stable for 1.0m and usable at 0.75m, but
strict 0.50m waypoint success is still effectively unsolved. Diagnostics show
the failure is not just hidden targets: one-goal 0.50m deterministic success is
only about 6-8%, and many failures stop just outside the radius with low inward
speed. Three-goal 0.50m success is therefore near zero because the policy must
hit that difficult event three times in one episode.

Important setup issue found: loaded checkpoints keep their learned
`policy.decoder_logstd`. The old source had very large action stds
(`exp(logstd) ~= [1.57, 0.71, 0.97, 3.25]`), so PPO training panels reported
success from noisy sampled actions while deterministic visual eval used only
the mean action. `scripts/quadrotor_expand_checkpoint.py --set-action-std`
now creates sanitized continuation checkpoints, for example:

```bash
.venv/bin/python scripts/quadrotor_expand_checkpoint.py \
  experiments/quadrotor_route_extra_obs_init/source_zero_unused.pt \
  experiments/quadrotor_route_extra_obs_init/source_zero_unused_std005.pt \
  --env puffer_quadrotor \
  --set env.route_extra_obs=1 \
  --zero-input-cols 20,21,23,24,25 \
  --set-action-std 0.05
```

Experiments run:

- Extra true-final-target observation features:
  `experiments/quadrotor_route_extra_obs_experiment_20260516_1740`.
  Result: no meaningful 0.50m improvement.
- Arrival shaping:
  `experiments/quadrotor_route_arrival_experiment_20260516_1800`.
  Result: modest 0.75m improvement, still no meaningful 0.50m success.
- One-goal precision curriculum with inherited high logstd:
  `experiments/quadrotor_route_precision_curriculum_20260516_1815`.
  Result: live training looked good but deterministic eval got worse, confirming
  the high-std mismatch.
- One-goal precision curriculum with sanitized std=0.05 source:
  `experiments/quadrotor_route_precision_curriculum_std005_20260516_1835`.
  Result: deterministic one-goal 0.50m improved only slightly, from 7.55% to
  8.70%, so low logstd is necessary but not sufficient.

Scientific conclusion: do not spend more blind PPO on three-goal 0.50m route
yet. The next fix should target the deterministic close-in controller/reward
directly, then re-chain route goals after one-goal 0.50m is reliably solved.

## 2026-05-16 Reference Route Parity Findings

The custom env now has an explicit `route_reward_mode=reference` path that
matches the upstream `puffer_drone_route` progress reward more closely, plus
`route_walls=0` so `room_size=3.0` is used for route sampling without adding
extra wall crashes. This exposed a real earlier mismatch: our first reference
ladder was unfairly training inside a hard 3m wall box, while upstream route
does not crash on those route sampling bounds unless room walls are enabled.

Corrected no-wall ladder:

```bash
experiments/quadrotor_route_reference_ladder_20260516_183832
```

It ran one-goal for 5M steps and three-goal for 10M steps on the GPU. The best
deterministic checkpoint was
`one_goal/model_puffer_quadrotor_000008.pt`, with about `12.2%` success on the
one-goal open reference case and `0%` success on the three-goal case. A focused
diagnostic of that checkpoint found `221/257` ground crashes, `32/257`
successes, and `4/257` divergence failures. Failed episodes often crossed near
the target but at high speed, then hit the ground. Visibility and obstacle
avoidance were not the bottleneck in this run.

Scientific conclusion: after removing the unfair wall mismatch, the current
custom direct-motor route setup is still failing at low-level flight control
under waypoint pursuit. This is not yet an exploration problem. The next
comparison should isolate whether the regression comes from our action/training
interface or from the route task: run a rate-action ablation, run a
hover-warm-start motor route ablation, and compare reset/action statistics
against upstream `puffer_drone_route` before adding obstacles or cameras.

Follow-up ablations:

- Existing good custom direct-motor route checkpoint
  `experiments/puffer_quadrotor_177893026034/model_puffer_quadrotor_000020.pt`
  scores `99.5%` one-goal and `97.1%` three-goal success under the new
  reference/no-wall evaluator. This proves the custom env can run the capable
  behavior; the broken piece is training/discovery, not environment
  expressiveness.
- Aggressive reference fine-tuning from that checkpoint with inherited high
  logstd and upstream-drone PPO settings destroyed the policy. The best first
  checkpoint in
  `experiments/quadrotor_route_reference_motor_warm_20260516_1852` was already
  down to `91.7%` one-goal and `77.9%` three-goal success, then later
  checkpoints collapsed to ground-crash behavior.
- Sanitizing the same checkpoint to `action_std=0.05` and using conservative
  Adam continuation preserved high performance. Best checkpoint:
  `experiments/quadrotor_route_reference_motor_conservative_20260516_1854/three_goals/model_puffer_quadrotor_000001.pt`,
  with `97.8%` one-goal and `91.4%` three-goal success.
- Legacy hover/rate warm-start is not a useful shortcut. Expanded checkpoint
  `experiments/quadrotor_reference_ablation_init/hover_rate_ref26.pt` trained
  with `action_mode=rate` stayed around `10%` one-goal and `0%` three-goal
  success, and a simple PD rate baseline also ground-crashed. Treat the old
  rate/hover stack as historical, not the path forward.

Updated conclusion: use the direct-motor route checkpoint as the seed, sanitize
loaded action std before PPO, and keep continuations conservative. The next
serious milestone is obstacle/depth/ray navigation from a strong motor policy,
not more work on the old rate controller.

## 2026-05-16 Ray Obstacle Continuation Findings

Strong direct-motor route seed expanded to 24 rays:

```bash
experiments/quadrotor_reference_ablation_init/route_good_std005_rays24.pt
```

The ray-expanded seed starts from the same motor behavior with zeroed ray input
weights and `action_std=0.05`. On the full local route benchmark with 16
evaluation rounds it scored mean success `52.5%`, open `89.5%`, dense random
blocks `47.2%`, and strict `0.50m` precision `0.1%`.

First conservative mixed-obstacle continuation:

```bash
experiments/quadrotor_rays24_conservative_mixed_20260516_2046
```

Best selected checkpoint:

```bash
experiments/quadrotor_rays24_conservative_mixed_20260516_2046/checkpoints/model_puffer_quadrotor_000008.pt
```

Full 16-round benchmark: mean success `56.2%`, open `93.4%`, dense random
blocks `52.3%`, mixed50 `69.5%`, mixed75 `58.7%`, and strict `0.50m`
precision `6.0%`. This is a real but modest improvement over the ray-expanded
seed. The final checkpoint was much worse, so checkpoint selection is mandatory.

A lower-learning-rate follow-up from checkpoint `000008`
(`experiments/quadrotor_rays24_low_lr_mixed_20260516_2125`) did not beat the
first run; its best fast-sweep checkpoint was around `55.2%` mean success.

Route diagnostics on the selected checkpoint in random-block rooms:

```bash
experiments/quadrotor_route_diagnostics_rays24_best_20260516_2125
```

It found `128/256` episode success, `122/256` obstacle crashes, `4/256` wall
crashes, and only `2/256` ground crashes. Waypoint reach rate was `80.1%`.
Failed waypoints almost never crossed the goal plane; `55.5%` were hidden for
most of the rollout and `98.4%` of failed waypoints belonged to obstacle or wall
crash episodes. This says the current bottleneck has shifted from basic flight
to obstacle-aware local navigation.

## 2026-05-16 Zero-To-Solve Gate Findings

The custom route env can now solve the Puffer-style open reference task from
scratch, but only when the reference gate actually matches the built-in drone
route assumptions. The critical mismatch was the floor: Puffer's no-wall route
lets agents move below `z=0` inside the broad `GRID_Z` bound, while our custom
route still terminated on ground contact even with `route_walls=0`. That made
early translation policies die while learning lateral flight.

Change made:

- `route_walls=0` now disables floor termination for `indoor_route`.
- `route_walls=1` still keeps floor, wall, ceiling, and obstacle collisions for
  the real indoor task.
- `motor_hover` was added as an experimental action mode where action zero maps
  to hover RPM and reset motors start spun at hover. It stabilizes the zero
  policy, but did not solve the open route gate under the tested PPO settings.

Failed gates:

- `experiments/quadrotor_zero_hover_action_20260516_231550`: `motor_hover`,
  Muon/upstream settings, `action_std=0.5`; best deterministic eval was only
  about `18%` one-goal and `0.3%` three-goal success.
- `experiments/quadrotor_zero_hover_adam_20260516_231725`: `motor_hover`,
  conservative Adam, `action_std=0.15`; best deterministic eval was about
  `17%` one-goal and `0%` three-goal success.

Successful gate:

```bash
experiments/quadrotor_zero_nofloor_raw_20260516_232011/three_goals/model_puffer_quadrotor_000098.pt
```

This was trained from scratch with raw `action_mode=motor`, `route_walls=0`,
`route_reward_mode=reference`, `route_direct_obs=1`, `goal_radius=1.0`, and
upstream-style PPO/Muon settings. Deterministic local reference eval:
`100.0%` one-goal and `100.0%` three-goal success. Later checkpoints degraded,
so checkpoint selection still matters.

The same checkpoint is not yet an indoor solution. Re-evaluated with
`route_walls=1` in the open room, it scored `51.2%` three-goal success with
`46.9%` wall hits and `2.0%` ground hits. That is the next honest curriculum
boundary: preserve the from-scratch open-route competence while reintroducing
floor/walls.

Updated conclusion: use the raw motor/no-floor route gate as the parity check
against Puffer's reference drone env. Reintroduce floor and walls as the next
curriculum step for indoor flight; do not judge indoor learning by the no-floor
score.

## 2026-05-17 Wall-Observation Route Gate

The first wall-enabled route gate succeeded from scratch after adding explicit
room-boundary observations. This keeps the raw motor policy and reference route
reward, but appends six normalized distances to the route observation:
`+x`, `-x`, `+y`, `-y`, ceiling, and floor. Old checkpoints still use the
original 26-input shape when `route_wall_obs=0`; wall-aware policies use 32
inputs when `num_rays=0`.

Successful wall-enabled gate:

```bash
experiments/quadrotor_zero_walls_wallobs_20260517_0050/three_goals/model_puffer_quadrotor_000115.pt
```

This was trained from scratch with raw `action_mode=motor`, `route_walls=1`,
`route_wall_obs=1`, `route_reward_mode=reference`, `route_direct_obs=1`,
`goal_radius=1.0`, and upstream-style PPO/Muon settings. The initial ladder
sweep found `000110` and `000115` tied on a small eval; a larger deterministic
wall-suite eval selected `000115`:

- `97.97%` one-goal success.
- `94.65%` three-goal success.
- `96.31%` mean success across one-goal and three-goal cases.
- `3.69%` mean timeout.
- `0.0%` obstacle hit in this open-room wall gate.

The training panel also showed about `97%` three-goal success with about `3%`
wall hits near the end, so the wall box is now learnable from scratch. Next
pressure should be obstacle walls/rays or a larger wall stress suite, not more
work on open-room wall following.

## 2026-05-17 Wall+Ray Obstacle Route Limit

The next pressure test expanded the wall-aware route policy to `24` range rays
while preserving the six wall-distance inputs. This required inserting zeroed
ray input columns before the existing wall-observation columns; blindly padding
the input layer would incorrectly reinterpret wall-distance weights as ray
weights. `scripts/quadrotor_expand_checkpoint.py` now supports
`--insert-zero-cols START:COUNT` for this case.

Expanded source:

```bash
experiments/quadrotor_wallobs_rays24_sources/model_puffer_quadrotor_000115_rays24_wallobs.pt
```

Baseline deterministic density eval with `route_walls=1`, `route_wall_obs=1`,
`route_direct_obs=1`, `route_reward_mode=reference`, and `num_rays=24`:

- mean density success: `70.84%`
- open: `95.31%`
- mixed 25/50/75: `81.85%`, `71.00%`, `58.89%`
- pure random blocks: `47.17%`
- mean obstacle hit: `26.52%`

Naive mixed-obstacle PPO from this checkpoint was destructive. With
`learning_rate=2e-4`, `action_std=0.1`, and mixed 50% blocks, every saved
checkpoint collapsed below `1%` mean density success. The live PPO panel showed
large initial KL and rapid wall/ground/timeouts, so this should be treated as a
bad optimizer/update setting, not a meaningful obstacle-learning result.

A conservative test with `action_std=0.03`, `learning_rate=1e-5`,
`clip_coef=0.02`, and no entropy avoided collapse but only made a tiny gain:

```bash
experiments/quadrotor_wallobs_rays24_mixed50_conservative_20260517_0109/model_puffer_quadrotor_000002.pt
```

Deterministic density eval:

- mean density success: `71.35%`
- open: `96.19%`
- mixed 25/50/75: `82.32%`, `71.71%`, `59.28%`
- pure random blocks: `47.27%`
- mean obstacle hit: `26.28%`

Continuing that checkpoint on pure random blocks did not improve the hard case.
The best pure-block continuation reached only `47.37%` random-block success and
lower mean density score than the mixed checkpoint, so it is not promoted.

Route diagnostics on the mixed-conservative checkpoint in pure random blocks:

- episode success: `47.27%`
- episode status counts: `121` success, `135` obstacle crashes
- waypoint reach rate: `79.44%`
- failed waypoints: `133`
- failed hidden-majority rate: `70.68%`
- failed near-miss rate: `13.53%`
- failed crossed-goal-plane rate: `1.50%`
- all failed waypoints occurred in obstacle-crash episodes

Current honest limit: with direct final-target observations plus 24 rays, the
policy mostly fails when the target is hidden and the direct route intersects
blocks. PPO fine-tuning can be made stable, but it does not discover robust
local navigation from sparse ray geometry under this setup. The next useful
push is not more low-LR PPO on the same observation/reward; it is either
planner/local-waypoint observations, a teacher/DAgger obstacle curriculum, or a
more structured representation of free-space direction.

## 2026-05-17 Target-Aligned Perception Probe

Added optional `route_perception_obs` features after the existing base, ray, and
wall observation columns. The 12 appended values are body-frame target delta,
target bearing, and target/front/side clearance probes. Existing policies remain
loadable when the flag is off; expanded policies keep old columns fixed and add
zero-initialized input weights for the new tail.

Local-waypoint observation (`route_direct_obs=0`) remains the best policy line,
but it is also a privileged obstacle-aware planner target. A 10M-step
conservative mixed continuation with the perception tail reached only a small
density gain:

- source expanded checkpoint:
  `experiments/quadrotor_perception_sources/model_puffer_quadrotor_000034_rays24_wallobs_perception.pt`
- best continuation:
  `experiments/quadrotor_rays24_perception_mixed50_20260517_1239/model_puffer_quadrotor_000026.pt`
- density mean: `77.22%` -> `77.48%`
- pure random blocks: `58.06%` -> `58.38%`
- masking rays, the perception tail, or both did not reduce block success
  (`~58%` in all cases)

The tail weights barely moved, so this is not evidence of learned visual
navigation. A 10x higher LR variant quickly damaged the policy, causing wall
hits and timeouts.

The cleaner direct-target test (`route_direct_obs=1`) did improve:

- source expanded checkpoint:
  `experiments/quadrotor_perception_sources/model_puffer_quadrotor_000002_direct_rays24_wallobs_perception.pt`
- best continuation:
  `experiments/quadrotor_direct_perception_mixed50_20260517_1244/model_puffer_quadrotor_000044.pt`
- density mean: `72.32%` -> `73.13%`
- pure random blocks: `48.40%` -> `50.23%`
- masking rays/perception still did not hurt (`~49%` block success)

Conclusion: the engineered perception values are available and sane, but PPO is
not yet using them behaviorally. Local-waypoint success mostly comes from the
planner target. Direct-target training can improve slightly, but not through
the new visual channels yet. The next push should make obstacle sensing
necessary and easier to optimize, for example by training from direct-target
observations with an auxiliary/behavior-cloning signal, initializing the visual
columns from a small avoidance controller, or temporarily removing the
privileged local waypoint from promoted "vision" benchmarks.

## 2026-05-17 Scratch Vision Curriculum

Added `route_blocked_target_prob`, a probability gate for blocked-line target
sampling when `route_target_mode=blocked`. This keeps the hard target generator
available, but lets scratch training start with a learnable mix instead of
requiring takeoff, wall avoidance, obstacle avoidance, and hidden-target
navigation on every first episode.

Named script scenarios now cover the two useful curriculum stages:
`route_vision_easy` and `route_vision_mid`.

The all-blocked scratch stress test was not learnable from zero under the
upstream direct-motor PPO settings: after 20M steps it stayed near `0%` success
and almost every episode crashed in roughly 20 env steps.

Two-stage scratch lineage:

1. `experiments/puffer_quadrotor_177901261222/model_puffer_quadrotor_000191.pt`
   trained from no checkpoint on `route_mixed` with `25%` obstacle episodes,
   `25%` blocked targets, one goal, direct target obs, 24 rays, wall obs, and
   target-aligned perception obs.
2. `experiments/puffer_quadrotor_177901281145/model_puffer_quadrotor_000191.pt`
   continued that policy on `50%` obstacle episodes, `50%` blocked targets, and
   three goals.

Deterministic stress eval for the final stage on pure random blocks with
`100%` blocked targets and three goals:

- success: `88.87%`
- obstacle hit: `7.03%`
- timeout: `3.81%`
- route goals: `2.78 / 3`

Ablations on the same pure-blocked three-goal stress case show the policy is
behaviorally using the new visual inputs:

- normal: `88.87%` success
- raw rays masked: `3.22%` success
- target-aligned perception tail masked: `2.20%` success

This is the first strong evidence that the custom env can learn indoor obstacle
navigation from a zero-start lineage without bootstrapping from the old blind
route checkpoint. It is still a curriculum, not a single-distribution solve:
pure all-blocked training from zero remains too harsh, and failures are still
mostly obstacle contacts.

## 2026-05-17 Variable-Height Route Targets

Added optional `route_target_z_min` and `route_target_z_max` route parameters.
The default `0.0, 0.0` preserves fixed-altitude behavior. When `max > min`,
each new route waypoint samples a safe target height from that range. Eval logs
now include `final_z_error` so vertical tracking can be separated from horizontal
route success.

Script aliases:

- `route_vision_z_easy`: mixed obstacles, 50% blocked targets, 3 goals,
  target heights from `1.25m` to `1.75m`.
- `route_vision_z_mid`: same setup, target heights from `1.0m` to `2.0m`.
- `scripts/quadrotor_visual_eval.sh route_z` and `route_z_mid` render the same
  variable-height scenarios.

Initial local checks on the current scratch-vision checkpoint
`experiments/puffer_quadrotor_177901281145/model_puffer_quadrotor_000191.pt`:

- `route_vision_z_easy`: `97.27%` success, `0.347m` final z error.
- `route_vision_z_mid`: `97.85%` success, `0.349m` final z error.
- Pure random blocks with `100%` blocked targets and z range `1.0m..2.0m`:
  `89.67%` success, `0.347m` final z error.
- Fixed-height pure random blocks with the same corrected observation alias:
  `91.21%` success, `0.323m` final z error.

Conclusion: variable-height waypoints are not a new bottleneck yet. The current
policy already generalizes surprisingly well in z, losing only about `1.5%`
absolute success on the hard pure-blocked route case. A tiny local CPU PPO
continuation reduced z error slightly but damaged route success, so do not draw
learning conclusions from it. Run a proper GPU continuation only if we decide
vertical precision itself matters before the next exploration push.

Follow-up GPU continuation on `beeblebrox`:

```bash
QUAD_ROUTE_TARGET_Z_MIN=1.0 QUAD_ROUTE_TARGET_Z_MAX=2.0 \
QUAD_ROUTE_TARGET_MODE=blocked QUAD_ROUTE_BLOCKED_TARGET_PROB=1.0 \
scripts/quadrotor_remote_gpu.sh train route_blocked 20000000 2701 \
  experiments/puffer_quadrotor_177901281145/model_puffer_quadrotor_000191.pt
```

The run produced `experiments/puffer_quadrotor_177901468696`. Deterministic
checkpoint sweep on pure random blocks, 100% blocked targets, and z range
`1.0m..2.0m` promoted:

```text
experiments/puffer_quadrotor_177901468696/model_puffer_quadrotor_000065.pt
```

Eval results:

- hard variable-height pure blocked: `96.29%` success, `3.71%` obstacle hits,
  `0.202m` final z error.
- fixed-height pure blocked: `96.98%` success, `3.02%` obstacle hits,
  `0.192m` final z error.
- mixed variable-height curriculum: `98.54%` success, `1.37%` obstacle hits,
  `0.216m` final z error.

Updated conclusion: the continuation substantially improved obstacle-route
success and vertical precision without an obvious fixed-height regression. It
has since been superseded by the robust combo checkpoint below.

Route frontier gauntlet added May 17, 2026:

```bash
QUAD_GAUNTLET_EPISODES=4 QUAD_GAUNTLET_NUM_ENVS=256 \
  scripts/quadrotor_remote_gpu.sh route-gauntlet CHECKPOINT frontier
```

Baseline frontier result for
`experiments/puffer_quadrotor_177901468696/model_puffer_quadrotor_000065.pt`:
mean success `76.43%`, mean obstacle hit `10.28%`, mean timeout `8.51%`.
Weak cases were:

- `delay_2`: `1.46%` success, mostly crashes.
- `precision_035`: `9.69%` success, mostly timeouts/near misses.
- `precision_050`: `39.36%` success.
- `combo`: `58.05%` success.

The first sim2real-relevant robust combo continuation trained from that
checkpoint with `0.75m` radius, `5` goals, z range `0.75m..2.25m`, mild
observation/action noise, one-step action delay, and fixed dynamics mismatch.
The run produced `experiments/puffer_quadrotor_177901561690`.
Checkpoint sweep on the exact combo case promoted:

```text
experiments/puffer_quadrotor_177901561690/model_puffer_quadrotor_000075.pt
```

Measured results for the promoted checkpoint:

- combo case: `87.89%` success, `11.52%` obstacle hits, no timeouts.
- full frontier mean: `88.49%` success, mean obstacle hit `7.79%`, mean
  timeout `2.42%`.
- `baseline`: `96.40%` success.
- `goals_8`: `88.57%` success.
- `precision_050`: `90.33%` success.
- `precision_035`: `57.62%` success.
- `delay_1`: `92.40%` success.
- `delay_2`: `32.71%` success, still mostly crashes.

Conclusion: this policy handled normal blocked routes, variable height, longer
routes, mild noise, one-step latency, and modest dynamics mismatch. It has since
been superseded by the randomized-latency checkpoint below.

A direct delay-2 continuation from the promoted checkpoint produced
`experiments/puffer_quadrotor_177901617596`. It proves the latency problem is
learnable, but it is not a general promotion:

- best `delay_2` checkpoint:
  `experiments/puffer_quadrotor_177901617596/model_puffer_quadrotor_000077.pt`
- isolated `delay_2`: `92.99%` success, `6.33%` obstacle hits.
- full frontier mean: `71.06%` success, worse than the promoted robust combo
  checkpoint.
- main regressions: `precision_035` fell to `1.73%`, `precision_050` to
  `20.21%`, `underpowered` to `26.12%`, and `combo` to `32.39%`.

Conclusion: fixed-delay training can solve the latency case by adopting a
slower, less precise control style. The next latency push should mix latency
with dynamics/precision pressure, or randomize latency per episode, instead of
promoting a pure delay-2 specialist.

Randomized-latency combo training did what we wanted. Added env support for
per-episode `action_delay_min/max`, then trained from the robust combo
checkpoint with latency sampled uniformly from `0..2`, while keeping the combo
pressures: `0.75m` radius, `5` goals, z range `0.75m..2.25m`, mild
observation/action noise, and fixed dynamics mismatch. The run produced
`experiments/puffer_quadrotor_177901683880` and promoted:

```text
experiments/puffer_quadrotor_177901683880/model_puffer_quadrotor_000077.pt
```

Measured full frontier result:

- mean success: `91.58%` versus `88.49%` for the previous promoted checkpoint.
- `delay_2`: `84.70%` versus `32.71%`.
- `combo`: `89.97%` versus `87.89%`.
- `baseline`: `95.90%` versus `96.40%`.
- `precision_050`: `91.41%` versus `90.33%`.
- `precision_035`: `56.15%` versus `57.62%`.
- `underpowered`: `95.61%` versus `97.66%`.

Conclusion: randomized latency is the right direction. It captures most of the
fixed-delay specialist's benefit without destroying general route behavior.
The small regressions in `precision_035`, baseline, and underpowered dynamics
are acceptable for now because frontier mean, combo, and latency improved. This
checkpoint is now the promoted route/vision visual default.

Diagnose exploration failures without rendering:

```bash
.venv/bin/python -m pufferlib.ocean.quadrotor.diagnose_explore --case blocks_edge --num-envs 256
.venv/bin/python -m pufferlib.ocean.quadrotor.diagnose_explore --case blocks_center --num-envs 256 --include-ray
```

Warm-start a policy from the velocity controller:

```bash
.venv/bin/python -m pufferlib.ocean.quadrotor.train_nav_clone \
  --output experiments/quadrotor_nav_clone/pd_velocity_clone.pt
```

## Next Milestones

1. Keep `experiments/puffer_quadrotor_177901683880/model_puffer_quadrotor_000077.pt`
   as the promoted general route checkpoint.
2. Continue the clean route ablation from scratch: final target in the base
   observation, 24 body-frame range rays, no wall-distance observation block,
   no target-aligned perception block, and reference route reward. The current
   limit is multi-goal obstacle chaining, not basic flight.
3. Do not promote the pure delay-2 specialist. Use randomized latency for
   sim2real robustness; it kept the robust combo and precision cases alive.
4. Train a precision ladder separately: `0.50m` is now good on the promoted
   checkpoint, `0.35m` still times out too often.
5. Add diagnostics for latency/precision tradeoffs: pre-crash speed, turn-rate
   lag, target direction, nearest obstacle normal, action saturation, and
   near-goal crossing/timeout rates.
6. Promote only by the route gauntlet, not by training-panel stats. Require no
   regression on baseline/variable-height/goal-count cases.
7. Return to exploration after route-with-rays passes the latency and precision
   stress cases; exploration should inherit stable flight and local navigation.
8. Then add exploration-specific stress cases: open room, random blocks,
   corridor, maze, sensor noise, action delay, and dynamics randomization.
9. Replace range rays with depth-like observations once ray-based navigation
   remains stable under the route gauntlet.
10. Model the real drone dimensions and thrust curve from measurements.
11. Sim2real deployment path: policy outputs velocity commands first; later
    action modes can descend toward attitude/rate/motor control.

## Clean Route Ablation

The orange local-planner render marker is now only drawn when the primary route
observation actually uses that planner target. The clean route profile uses only
the true final target in the base observation plus range rays.

First result:

```bash
scripts/quadrotor_remote_gpu.sh train route_clean_easy 50000000 1 none
scripts/quadrotor_remote_gpu.sh eval-checkpoints route_clean_easy 4 256 experiments/puffer_quadrotor_177901845082
```

Best clean hover-height easy checkpoint:

- `experiments/puffer_quadrotor_177901845082/model_puffer_quadrotor_000190.pt`
- `79.77%` success on the checkpoint sweep.
- `77.93%` success on a larger 8197-episode eval.
- Remaining failures are mostly crashes: obstacle `~8.8%`, wall `~6.4%`, ground
  `~7.5%`.

Important correction: the first clean-easy run used floor-height targets and
reached `92.82%`, but that was the wrong rung for indoor flight. It taught a
low-altitude reaching behavior and failed catastrophically when transferred to
variable-height, three-goal routing. The preserved alias is
`route_clean_flat_easy`; do not use it for promotion.

Current limit:

```bash
scripts/quadrotor_remote_gpu.sh train route_clean_mid 50000000 1 \
  experiments/puffer_quadrotor_177901814030/model_puffer_quadrotor_000191.pt
scripts/quadrotor_remote_gpu.sh eval-checkpoints route_clean_mid 4 256 experiments/puffer_quadrotor_177901825957
```

Clean-mid failed. Every checkpoint was around `0.1%` success, with `~99.9%`
crash rate and many ground crashes. The next honest step is not adding the
planner target back; it is a smoother height curriculum: one hover-height goal,
then wider height, then more blocked targets, then multiple goals.

May 17, 2026 follow-up: added smoother clean rungs and trained them from zero.
The observation contract stayed simple: 26 base values, 24 range rays, direct
current target, no wall obs, no target-aligned perception obs, reference reward.

Commands:

```bash
scripts/quadrotor_remote_gpu.sh train route_clean_wide_z 50000000 1 none
scripts/quadrotor_remote_gpu.sh train route_clean_blocked 50000000 1 none
scripts/quadrotor_remote_gpu.sh train route_clean_two_goal 50000000 1 none
scripts/quadrotor_remote_gpu.sh train route_clean_two_goal 100000000 1 none
```

Results:

- `route_clean_wide_z`: one goal, target height `1.0..2.0m`, `25%` blocked
  target sampling. Best checkpoint
  `experiments/puffer_quadrotor_177901919732/model_puffer_quadrotor_000170.pt`
  reached `94.14%` success, `5.86%` obstacle hits, and no timeouts.
- `route_clean_blocked`: one goal, target height `1.0..2.0m`, `50%` blocked
  target sampling. Best checkpoint
  `experiments/puffer_quadrotor_177901930443/model_puffer_quadrotor_000170.pt`
  reached `84.95%` success, `15.05%` obstacle hits, and no timeouts.
- `route_clean_two_goal` at `50M` steps was undertrained. Best checkpoint
  `experiments/puffer_quadrotor_177901942303/model_puffer_quadrotor_000190.pt`
  reached only `63.30%` success, but was still improving at the final checkpoint.
- `route_clean_two_goal` at `100M` steps from zero improved to `85.96%`
  success at
  `experiments/puffer_quadrotor_177901959824/model_puffer_quadrotor_000360.pt`,
  with `13.06%` obstacle hits and no timeouts.

Conclusion: the clean observation profile is viable from zero. The first real
limit is obstacle avoidance under chained targets. More steps close much of the
gap, but the remaining failures are obstacle collisions, not walls, altitude, or
timeouts. Next reward/diagnostic work should target obstacle-collision behavior
after reaching a previous goal, while keeping the observation profile clean.

May 17, 2026 clearance-shaping ablation: added gated reference-mode clearance
penalty knobs, defaulted off:

- `route_reference_clearance_radius = 0.45`
- `route_reference_clearance_scale = 0.0`

The diagnostic pass on the current two-goal baseline showed `87.9%` success over
`513` episodes, with `56` obstacle failures, `6` ground failures, and about half
of failures happening after the first goal. Failed waypoints usually had the
target hidden by an obstacle/wall and collided at roughly `15cm` clearance.

Sweep results from zero at `100M` steps:

- Baseline, no reference clearance shaping:
  `experiments/puffer_quadrotor_177901959824/model_puffer_quadrotor_000360.pt`
  re-evaluated at `85.96%` success, `13.06%` obstacle hits, no timeouts.
- Mild shaping, `route_reference_clearance_scale=0.01`:
  `experiments/puffer_quadrotor_177902093444/model_puffer_quadrotor_000370.pt`
  reached `74.73%` success, `24.88%` obstacle hits, no timeouts.
- Strong shaping, `route_reference_clearance_scale=0.03`:
  `experiments/puffer_quadrotor_177902116097/model_puffer_quadrotor_000382.pt`
  reached `82.63%` success, `16.49%` obstacle hits, no timeouts.

Conclusion: simple clearance shaping in the reference reward is not the next
lever. It makes behavior slower and sometimes visually safer, but does not beat
the clean baseline. Keep the knobs available for future stress tests, but leave
them off for the recommended path.

## 2026-05-17 Literature Check: Indoor Navigation Inputs

Quick pass over relevant quadrotor RL work:

- SimpleFlight / "What Matters in Learning a Zero-Shot Sim-to-Real RL Policy
  for Quadrotor Control?" focuses on input design, reward design, system ID, and
  selective domain randomization for real Crazyflie transfer:
  `https://github.com/thu-uav/SimpleFlight`.
- "Sim-to-(Multi)-Real" shows low-level neural policies can map state directly
  to motor outputs and transfer to several quadrotors when noise/model
  randomization are handled seriously: `https://arxiv.org/abs/1903.04628`.
- Flightmare separates fast quadrotor dynamics from richer rendering/sensors,
  and supports hundreds of parallel quadrotors:
  `https://github.com/uzh-rpg/flightmare`.
- "Learning High-Speed Flight in the Wild" and the UZH agile autonomy stack use
  simulation and onboard sensing, but rely on expert/privileged structure rather
  than expecting raw RL to invent navigation from weak observations:
  `https://github.com/uzh-rpg/agile_autonomy`.
- The Nature drone-racing system combines model-free RL with progress and a
  perception-aware objective, then narrows sim-to-real gaps with residual
  perception/dynamics models.
- PyFlyt and gym-pybullet-drones are useful references for Gymnasium task API,
  vehicle abstractions, and firmware/SITL interfaces, but are less useful than
  PufferLib for our current high-throughput iteration loop.

Takeaway: strong systems rarely ask a single policy to route around hidden
obstacles using only a direct target vector and an unordered local ray fan. They
either improve the observation representation, use privileged training, split
local control from planning, or add perception-aware objectives.

The repo already has the most direct minimal probe for this: `route_perception_obs`.
It appends body-frame target direction plus target/front/side clearance probes.
To keep the environment simple and avoid duplicate feature paths, the next clean
ablation is not a new C feature; it is a from-zero two-goal run with:

- `route_clean_two_goal_perception`
- direct target observation on
- 24 raw rays on
- target-aligned perception tail on
- wall observations off
- reference reward unchanged
- no clearance shaping

Promotion rule: only replace the current `route_clean_two_goal` visual checkpoint
if this beats the `85.96%` clean baseline on success without increasing obstacle
hits or relying on timeouts.

Result:

- Default PPO/LR `0.003`, `100M` steps:
  `experiments/puffer_quadrotor_177902191216/model_puffer_quadrotor_000100.pt`
  was the best checkpoint, but reached only `0.49%` success, `99.51%`
  crashes, and `11.80%` obstacle hits.
- Lower LR `0.001`, `50M` steps:
  `experiments/puffer_quadrotor_177902212589/model_puffer_quadrotor_000090.pt`
  was the best checkpoint, but reached only `3.60%` success, `96.40%`
  crashes, and `23.64%` obstacle hits.

Conclusion: simply appending the target-aligned perception tail to the scratch
two-goal task is actively harmful under the current PPO/action settings. Lower
LR reduces the initial KL spike, but does not recover useful learning. Do not
promote this checkpoint. The current best visual/eval baseline remains the clean
no-tail two-goal policy at `85.96%` success. The next perception attempt should
start on an easier rung and test normalized/gated compact features, not add a
larger observation tail directly to the hardest current task.

May 17, 2026 compact perception follow-up: changed `route_perception_obs` from a
boolean to an observation mode:

- `0`: no target-aligned perception tail.
- `1`: legacy 12-value tail from the failed ablation above.
- `2`: compact 8-value tail with unit body-frame target direction, normalized
  target distance, target/left/right clearance probes, and forward clearance.

The compact mode was added to preserve the original failing result while testing
a smaller, more normalized representation. New scenarios:

```bash
scripts/quadrotor_remote_gpu.sh train route_clean_wide_z_compact 50000000 1 none
scripts/quadrotor_remote_gpu.sh train route_clean_two_goal_compact 50000000 1 none
QUAD_LEARNING_RATE=0.001 scripts/quadrotor_remote_gpu.sh train route_clean_two_goal_compact 50000000 1 none
```

Results:

- One-goal wide-Z compact, `50M` steps:
  `experiments/puffer_quadrotor_177902292267/model_puffer_quadrotor_000190.pt`
  reached `92.91%` success, `6.99%` obstacle hits, no timeouts. This is viable,
  but still worse than the clean one-goal wide-Z baseline at `94.14%` success
  and `5.86%` obstacle hits.
- Two-goal compact, default LR `0.003`, `50M` steps:
  `experiments/puffer_quadrotor_177902304541/model_puffer_quadrotor_000170.pt`
  was best at `0.49%` success, `99.51%` crashes, and `15.04%` obstacle hits.
  The first PPO update produced a very large KL spike, mirroring the legacy-tail
  failure mode.
- Two-goal compact, lower LR `0.001`, `50M` steps:
  `experiments/puffer_quadrotor_177902317339/model_puffer_quadrotor_000120.pt`
  was best at `2.63%` success, `97.37%` crashes, and `15.79%` obstacle hits.
  The lower LR removed the violent first-update KL spike, but did not recover
  useful two-goal learning.

Conclusion: compact target-clearance features are not enough. They can learn the
easier one-goal rung, but they regress that baseline and fail badly on the
two-goal task. Do not promote perception tails yet. The clean no-tail route
policy remains the mainline. The next useful question is no longer "add a better
tail"; it is why extra observation channels destabilize scratch learning on
multi-goal routing, and whether this should be solved by normalization inside the
policy, a cleaner target encoding that replaces old target slots instead of
appending to them, or a proper planner/control split.

## 2026-05-17 Target Encoding And Exploration Scale Sweep

Added `route_target_obs_mode` to test fixed-width replacements for the legacy
target slots without appending more observation channels:

- `legacy`: old target slots, preserving checkpoint compatibility.
- `body`: normalized body-frame target delta/direction/bearing features.
- `minimal`: compact unit-direction/distance/closing-speed/bearing features.

These did not solve the two-goal clean task from zero:

- `body`, `50M`: `3.69%` success, `96.31%` crashed.
- `minimal`, `50M`: `60.53%` success, `39.47%` crashed.
- legacy with `route_extra_obs=1`, `50M`: `59.71%` success, `40.29%` crashed.

Reference-clearance and wall-observation variants also failed to beat the
baseline:

- clearance scale `0.02`, `50M`: `65.66%` success.
- clearance scale `0.02`, `100M`: `78.63%` success.
- clearance scale `0.05`, `50M`: collapse to `0.39%` success.
- `route_wall_obs=1`, `50M`: `64.75%` success.

The first clear win was increasing initial policy exploration for this direct
motor two-goal route task:

```bash
QUAD_ACTION_STD=0.2 QUAD_CHECKPOINT_INTERVAL=20 \
  scripts/quadrotor_remote_gpu.sh train route_clean_two_goal 100000000 1 none
```

Best checkpoint:

```text
experiments/puffer_quadrotor_177902519170/model_puffer_quadrotor_000380.pt
```

Deterministic eval on the same clean two-goal blocked route profile:

- success: `93.07%`
- crashed: `6.93%`
- obstacle hit: `5.27%`
- timeout: `0.0%`
- average route goals: `1.91 / 2`

This beats the previous clean two-goal baseline
`experiments/puffer_quadrotor_177901959824/model_puffer_quadrotor_000360.pt`
at `85.96%` success and `13.06%` obstacle hits. The most likely root cause was
insufficient early exploration in the direct-motor action distribution, not
broken ray sensing or missing target features.

Promoted defaults:

- `route_clean_two_goal` training now defaults to `action_std=0.2`.
- `scripts/quadrotor_visual_eval.sh route_clean_two_goal` now points at the new
  checkpoint and passes `--action-std 0.2`.

Next honest limit: the task is not solved yet. The remaining failures are still
collisions, so the next sweep should bracket `action_std` around `0.2`, then
increase pressure with three goals, harder blocked-target probability, or a
route gauntlet while keeping the clean observation profile.

Follow-up three-goal pressure test:

```bash
QUAD_ACTION_STD=0.2 QUAD_CHECKPOINT_INTERVAL=20 \
  scripts/quadrotor_remote_gpu.sh train route_clean_mid 100000000 1 none
```

Best checkpoint:

```text
experiments/puffer_quadrotor_177902592213/model_puffer_quadrotor_000380.pt
```

Deterministic eval on the clean three-goal blocked route profile:

- success: `92.09%`
- crashed: `7.91%`
- obstacle hit: `7.42%`
- timeout: `0.0%`
- average route goals: `2.83 / 3`

This is a major correction to the earlier `route_clean_mid` failure. The old
three-goal run failed because the exploration scale was wrong, not because the
clean observation profile was fundamentally unable to chain goals. `route_clean`
and `route_clean_mid` now also default to `action_std=0.2`.

## 2026-05-17 Route Frontier And Dynamics Ramp

Added `scripts/quadrotor_route_gauntlet.py` as a repeatable route pressure
test. The clean-observation frontier includes nominal pure-blocked routes,
precision radius cuts, longer waypoint chains, z-range widening, smaller rooms,
latency/noise, and dynamics mismatch. Two extra dynamics rungs were added after
the first run:

- `underpowered_mild`: mass `1.05`, thrust `0.98`, drag `1.05`, motor lag
  `1.10`.
- `underpowered_medium`: mass `1.10`, thrust `0.97`, drag `1.07`, motor lag
  `1.15`.
- `underpowered`: mass `1.15`, thrust `0.95`, drag `1.10`, motor lag `1.20`.

The three-goal clean checkpoint
`experiments/puffer_quadrotor_177902592213/model_puffer_quadrotor_000380.pt`
was strong on its train distribution but weak under the frontier gauntlet:

- frontier mean success: `46.00%`
- nominal pure-blocked success: `70.80%`
- mixed50 success: `91.31%`
- hard underpowered success: `0.00%`, mostly ground hits
- combo success: `0.00%`, mostly ground hits

A naive pure-blocked continuation with the old remote wrapper action std
collapsed immediately with huge KL. The wrapper now stops overriding scenario
defaults unless `QUAD_ACTION_STD` is explicitly set. A conservative pure-blocked
continuation promoted:

```text
experiments/puffer_quadrotor_177902669986/model_puffer_quadrotor_000032.pt
```

This improved the clean frontier slightly:

- frontier mean success: `48.23%`
- nominal pure-blocked success: `72.85%`
- mixed50 success: `93.46%`
- hard underpowered success: `0.00%`

Direct hard-underpowered adaptation from that checkpoint learned slowly and only
reached `12.29%` deterministic success on the hard dynamics case after 20M
steps. A staged dynamics ramp worked much better:

1. Mild dynamics continuation:
   `experiments/puffer_quadrotor_177902720686/model_puffer_quadrotor_000075.pt`
   - nominal: `76.68%`
   - mild: `71.90%`
   - medium: `43.46%`
   - hard: `0.10%`
2. Medium dynamics continuation:
   `experiments/puffer_quadrotor_177902741363/model_puffer_quadrotor_000075.pt`
   - nominal: `74.80%`
   - mild: `73.24%`
   - medium: `64.88%`
   - hard: `16.50%`
3. Hard dynamics continuation:
   `experiments/puffer_quadrotor_177902756643/model_puffer_quadrotor_000077.pt`
   - nominal: `67.87%`
   - mild: `68.75%`
   - medium: `64.06%`
   - hard: `42.80%`

Conclusion: gradual dynamics pressure is a real training signal, but fixed-case
specialization is starting to trade nominal obstacle performance for hard
underpowered recovery. Do not promote the hard specialist as the default indoor
route policy. The next cleaner step is per-episode dynamics randomization, with
the actual sampled dynamics exposed only if we intentionally want system-ID
style conditioning. Without that, the policy must infer mismatch from motor
speed, vertical velocity, and failed thrust response.

## 2026-05-17 Per-Episode Dynamics Randomization

Added per-episode randomization ranges for mass, inertia, thrust, torque, drag,
and motor response. Fixed-scale behavior is unchanged when the min/max ranges
are left at `0.0`. When ranges are set, each reset samples a dynamics scale and
applies it before episode initialization. Logs/debug output now include the
sampled scales, which makes it possible to confirm the training distribution
without exposing those values to the policy observation.

The first randomized-dynamics continuation started from the balanced medium
dynamics checkpoint:

```bash
QUAD_TAG=route_clean_mid_dynrand_50m_20260517 \
QUAD_OBSTACLE_BLOCK_PROB=1.0 \
QUAD_ROUTE_BLOCKED_TARGET_PROB=1.0 \
QUAD_ACTION_STD=0.2 \
QUAD_LEARNING_RATE=0.00005 \
QUAD_CLIP_COEF=0.03 \
QUAD_ENT_COEF=0.0 \
QUAD_CHECKPOINT_INTERVAL=10 \
QUAD_MASS_SCALE_MIN=1.0 \
QUAD_MASS_SCALE_MAX=1.15 \
QUAD_THRUST_SCALE_MIN=0.95 \
QUAD_THRUST_SCALE_MAX=1.02 \
QUAD_DRAG_SCALE_MIN=1.0 \
QUAD_DRAG_SCALE_MAX=1.10 \
QUAD_MOTOR_TAU_SCALE_MIN=1.0 \
QUAD_MOTOR_TAU_SCALE_MAX=1.20 \
scripts/quadrotor_remote_gpu.sh train route_clean_mid 50000000 8 \
  experiments/puffer_quadrotor_177902741363/model_puffer_quadrotor_000075.pt
```

Remote throughput was roughly `0.75M-0.80M SPS` with the current multiprocessing
vector setup. The NVML utilization thread on `beeblebrox` still reports an
NVML/RM mismatch, but training itself completed.

Best checkpoint:

```text
experiments/puffer_quadrotor_177902839697/model_puffer_quadrotor_000191.pt
```

Four-case dynamics check, 4x256 deterministic eval episodes:

- nominal: `80.18%`
- underpowered mild: `80.59%`
- underpowered medium: `79.16%`
- underpowered hard: `65.92%`
- mean: `76.46%`

This is a clear improvement over both the balanced fixed-dynamics checkpoint
and the hard fixed-dynamics specialist. The hard specialist reached only
`42.80%` on hard underpowered and paid a nominal-performance cost. Randomized
dynamics instead improved hard robustness while keeping nominal performance
above `80%`.

Full frontier on the best checkpoint:

- mean success: `64.03%`
- baseline: `80.18%`
- mixed50: `92.31%`
- room 2.50m: `86.55%`
- z wide: `79.94%`
- z extreme: `75.78%`
- underpowered hard: `65.92%`
- overpowered: `68.55%`
- precision radius 0.50m: `61.56%`
- precision radius 0.35m: `30.86%`
- delay 1: `73.93%`
- delay 2: `10.92%`
- combo radius 0.75, five goals, z 0.75..2.25, delay 1, noise, dynamics:
  `16.78%`

Conclusion: dynamics randomization is now a win and should stay in the
training toolbox. The next honest limit is not static dynamics mismatch. It is
latency and combined disturbances. Two-step action delay destroys the current
policy, mostly through obstacle and ground collisions. The next run should use
random train-time action delay, probably `0..1` first and then `0..2`, with
dynamics randomization kept on. After latency stabilizes, revisit precision
radius and combo cases.

## 2026-05-17 Latency Curriculum

The first latency curriculum kept the randomized dynamics distribution above
and added per-episode action delay sampled from `0..1`:

```bash
QUAD_TAG=route_clean_mid_dynrand_delay01_50m_20260517 \
QUAD_OBSTACLE_BLOCK_PROB=1.0 \
QUAD_ROUTE_BLOCKED_TARGET_PROB=1.0 \
QUAD_ACTION_STD=0.2 \
QUAD_LEARNING_RATE=0.00005 \
QUAD_CLIP_COEF=0.03 \
QUAD_ENT_COEF=0.0 \
QUAD_CHECKPOINT_INTERVAL=10 \
QUAD_ACTION_DELAY_MIN=0 \
QUAD_ACTION_DELAY_MAX=1 \
QUAD_MASS_SCALE_MIN=1.0 \
QUAD_MASS_SCALE_MAX=1.15 \
QUAD_THRUST_SCALE_MIN=0.95 \
QUAD_THRUST_SCALE_MAX=1.02 \
QUAD_DRAG_SCALE_MIN=1.0 \
QUAD_DRAG_SCALE_MAX=1.10 \
QUAD_MOTOR_TAU_SCALE_MIN=1.0 \
QUAD_MOTOR_TAU_SCALE_MAX=1.20 \
scripts/quadrotor_remote_gpu.sh train route_clean_mid 50000000 9 \
  experiments/puffer_quadrotor_177902839697/model_puffer_quadrotor_000191.pt
```

Best checkpoint:

```text
experiments/puffer_quadrotor_177902913617/model_puffer_quadrotor_000191.pt
```

Full frontier:

- mean success: `68.69%`
- baseline: `83.69%`
- mixed50: `93.26%`
- delay 1: `78.32%`
- delay 2: `23.73%`
- underpowered medium: `83.43%`
- underpowered hard: `79.41%`
- precision radius 0.35m: `37.30%`
- goals 8: `58.11%`
- combo radius 0.75, five goals, z 0.75..2.25, delay 1, noise, dynamics:
  `36.78%`

This is the current balanced `route_clean_mid` default. It improved the full
frontier mean from `64.03%` to `68.69%`, improved baseline success, and made
the combo case much less brittle. Two-step delay is still weak, but the policy
is no longer only solving the nominal room.

A second curriculum sampled action delay from `0..2`, starting from the
delay-0..1 checkpoint:

```bash
QUAD_TAG=route_clean_mid_dynrand_delay02_50m_20260517 \
QUAD_OBSTACLE_BLOCK_PROB=1.0 \
QUAD_ROUTE_BLOCKED_TARGET_PROB=1.0 \
QUAD_ACTION_STD=0.2 \
QUAD_LEARNING_RATE=0.00005 \
QUAD_CLIP_COEF=0.03 \
QUAD_ENT_COEF=0.0 \
QUAD_CHECKPOINT_INTERVAL=10 \
QUAD_ACTION_DELAY_MIN=0 \
QUAD_ACTION_DELAY_MAX=2 \
QUAD_MASS_SCALE_MIN=1.0 \
QUAD_MASS_SCALE_MAX=1.15 \
QUAD_THRUST_SCALE_MIN=0.95 \
QUAD_THRUST_SCALE_MAX=1.02 \
QUAD_DRAG_SCALE_MIN=1.0 \
QUAD_DRAG_SCALE_MAX=1.10 \
QUAD_MOTOR_TAU_SCALE_MIN=1.0 \
QUAD_MOTOR_TAU_SCALE_MAX=1.20 \
scripts/quadrotor_remote_gpu.sh train route_clean_mid 50000000 10 \
  experiments/puffer_quadrotor_177902913617/model_puffer_quadrotor_000191.pt
```

Best key-case checkpoint:

```text
experiments/puffer_quadrotor_177902958735/model_puffer_quadrotor_000170.pt
```

Full frontier:

- mean success: `67.93%`
- baseline: `76.17%`
- mixed50: `91.33%`
- delay 1: `75.29%`
- delay 2: `63.18%`
- underpowered medium: `80.60%`
- underpowered hard: `81.85%`
- overpowered: `52.78%`
- precision radius 0.50m: `59.12%`
- precision radius 0.35m: `40.10%`
- goals 8: `48.29%`
- combo radius 0.75, five goals, z 0.75..2.25, delay 1, noise, dynamics:
  `49.71%`

This is not a simple promotion. The delay-0..2 run fixed most of the two-step
latency failure, but it traded away nominal and long-horizon route performance.
Use it as a latency specialist and as evidence that the action-delay failure is
trainable, not as the default visual policy yet.

Next limit: the frontier is no longer dominated by static dynamics or one-step
latency. The remaining failures are split across tight precision, long waypoint
chains, overpowered dynamics, and combined disturbances. The next experiment
should either add a minimal action-history/motor-command observation for
latency robustness, or run a mixed frontier curriculum that samples delay,
dynamics, precision radius, and longer routes without overfitting to delay-2.

## 2026-05-17 Action-History Observation Probe

Added optional `action_history_obs` to the custom quadrotor env. It appends the
last `N` policy commands, newest first, after the existing observation tail.
The default is `0`, so old checkpoints and the clean observation contract are
unchanged. With `action_history_obs=2`, the route policy sees eight extra
values: the last two four-motor command vectors. Observation noise does not
perturb these command-history values.

The source checkpoint was expanded with zero weights on the new input columns:

```bash
QUAD_NUM_RAYS=24 \
QUAD_ROUTE_DIRECT_OBS=1 \
QUAD_ROUTE_WALL_OBS=0 \
QUAD_ROUTE_PERCEPTION_OBS=0 \
QUAD_ACTION_HISTORY_OBS=2 \
scripts/quadrotor_remote_gpu.sh expand-quadrotor \
  experiments/puffer_quadrotor_177902913617/model_puffer_quadrotor_000191.pt \
  experiments/puffer_quadrotor_177902913617/model_puffer_quadrotor_000191_actionhist2.pt
```

Then we repeated the delay-0..2 randomized-dynamics curriculum:

```bash
QUAD_TAG=route_clean_mid_actionhist2_delay02_50m_20260517 \
QUAD_OBSTACLE_BLOCK_PROB=1.0 \
QUAD_ROUTE_BLOCKED_TARGET_PROB=1.0 \
QUAD_ACTION_HISTORY_OBS=2 \
QUAD_ACTION_STD=0.2 \
QUAD_LEARNING_RATE=0.00005 \
QUAD_CLIP_COEF=0.03 \
QUAD_ENT_COEF=0.0 \
QUAD_CHECKPOINT_INTERVAL=10 \
QUAD_ACTION_DELAY_MIN=0 \
QUAD_ACTION_DELAY_MAX=2 \
QUAD_MASS_SCALE_MIN=1.0 \
QUAD_MASS_SCALE_MAX=1.15 \
QUAD_THRUST_SCALE_MIN=0.95 \
QUAD_THRUST_SCALE_MAX=1.02 \
QUAD_DRAG_SCALE_MIN=1.0 \
QUAD_DRAG_SCALE_MAX=1.10 \
QUAD_MOTOR_TAU_SCALE_MIN=1.0 \
QUAD_MOTOR_TAU_SCALE_MAX=1.20 \
scripts/quadrotor_remote_gpu.sh train route_clean_mid 50000000 11 \
  experiments/puffer_quadrotor_177902913617/model_puffer_quadrotor_000191_actionhist2.pt
```

Run:

```text
experiments/puffer_quadrotor_177903073716
```

Best key-case checkpoint:

```text
experiments/puffer_quadrotor_177903073716/model_puffer_quadrotor_000150.pt
```

Full frontier on that checkpoint:

- mean success: `68.94%`
- baseline: `77.34%`
- mixed50: `91.80%`
- delay 1: `75.78%`
- delay 2: `64.65%`
- underpowered hard: `80.47%`
- overpowered: `58.77%`
- precision radius 0.50m: `63.61%`
- precision radius 0.35m: `43.55%`
- goals 8: `47.75%`
- combo radius 0.75, five goals, z 0.75..2.25, delay 1, noise, dynamics:
  `46.88%`

This is a small improvement, not a breakthrough. It is the best full-frontier
mean so far, but only by `0.25` percentage points over the delay-0..1 default
and `1.01` points over the no-history delay-0..2 specialist. It improves
two-step delay slightly (`64.65%` vs `63.18%`) and tight precision
(`43.55%` vs `40.10%`), but the combo case is worse than the no-history
delay-0..2 checkpoint (`46.88%` vs `49.71%`) and nominal baseline remains well
below the delay-0..1 default (`77.34%` vs `83.69%`).

Do not replace the default `route_clean_mid` checkpoint yet. Use
`scripts/quadrotor_visual_eval.sh route_clean_mid_actionhist` to inspect the
action-history policy. The next bigger push should train against a mixed
frontier distribution instead of only delay-0..2, because the best remaining
frontier failures are now precision, long waypoint chains, overpowered
dynamics, and combo robustness.

## 2026-05-17 Mixed Frontier Curriculum

Trained from the action-history checkpoint into a harder mixed-frontier
distribution instead of specializing only on delay. The curriculum kept
`action_history_obs=2`, direct route target observations, and 24 rays, then
sampled:

- route radius `0.75`
- five route targets
- target height `0.75..2.25`
- action delay `0..2`
- mass `0.85..1.15`
- thrust `0.95..1.05`
- drag `0.90..1.10`
- motor lag `0.80..1.20`

Training command:

```bash
QUAD_TAG=route_clean_mid_actionhist2_combo_50m_20260517 \
QUAD_OBSTACLE_BLOCK_PROB=1.0 \
QUAD_ROUTE_BLOCKED_TARGET_PROB=1.0 \
QUAD_ACTION_HISTORY_OBS=2 \
QUAD_ACTION_STD=0.2 \
QUAD_LEARNING_RATE=0.00005 \
QUAD_CLIP_COEF=0.03 \
QUAD_ENT_COEF=0.0 \
QUAD_CHECKPOINT_INTERVAL=10 \
QUAD_ACTION_DELAY_MIN=0 \
QUAD_ACTION_DELAY_MAX=2 \
QUAD_ROUTE_GOAL_RADIUS=0.75 \
QUAD_ROUTE_GOAL_COUNT=5 \
QUAD_MAX_STEPS=1706 \
QUAD_ROUTE_TARGET_Z_MIN=0.75 \
QUAD_ROUTE_TARGET_Z_MAX=2.25 \
QUAD_MASS_SCALE_MIN=0.85 \
QUAD_MASS_SCALE_MAX=1.15 \
QUAD_THRUST_SCALE_MIN=0.95 \
QUAD_THRUST_SCALE_MAX=1.05 \
QUAD_DRAG_SCALE_MIN=0.90 \
QUAD_DRAG_SCALE_MAX=1.10 \
QUAD_MOTOR_TAU_SCALE_MIN=0.80 \
QUAD_MOTOR_TAU_SCALE_MAX=1.20 \
scripts/quadrotor_remote_gpu.sh train route_clean_mid 50000000 12 \
  experiments/puffer_quadrotor_177903073716/model_puffer_quadrotor_000150.pt
```

Run:

```text
experiments/puffer_quadrotor_177903264205
```

Best checkpoint:

```text
experiments/puffer_quadrotor_177903264205/model_puffer_quadrotor_000191.pt
```

Full frontier:

- mean success: `78.49%`
- weakest case: `precision_035` at `59.02%`
- baseline: `85.67%`
- mixed50: `95.51%`
- delay 1: `83.89%`
- delay 2: `80.18%`
- underpowered hard: `85.55%`
- overpowered: `77.56%`
- precision radius 0.50m: `75.10%`
- precision radius 0.35m: `59.02%`
- goals 8: `66.31%`
- room 2.25m: `71.48%`
- room 2.50m: `86.44%`
- combo radius 0.75, five goals, z 0.75..2.25, delay 1, noise, dynamics:
  `59.96%`

This is the first clearly meaningful frontier improvement. Compared with the
previous action-history run, full-frontier mean improved from `68.94%` to
`78.49%`; delay-2 improved from `64.65%` to `80.18%`; overpowered dynamics
from `58.77%` to `77.56%`; tight precision from `43.55%` to `59.02%`; and
combo from `46.88%` to `59.96%`.

The policy still fails mostly through obstacle hits, not timeouts. On the full
frontier, mean obstacle hit rate is `18.38%` and mean timeout rate is `2.59%`.
The remaining hard cases are tight target precision, compact rooms, long
multi-goal routes, and the combined disturbance case.

Promote this as the current visual/default action-history policy:

```bash
scripts/quadrotor_visual_eval.sh route_frontier
```

The next frontier should stop relying only on aggregate route success. We need
failure-mode diagnostics for precision versus collision: near-target crashes,
corner clipping, and whether target acquisition becomes unstable when the goal
radius shrinks below `0.5m`.

## 2026-05-17 Frontier Continuation

After the first mixed-frontier win, we tried two follow-up pushes from:

```text
experiments/puffer_quadrotor_177903264205/model_puffer_quadrotor_000191.pt
```

### Narrow Precision/Compact-Room Probe

This run intentionally pushed the weak cases too directly:

- room size `2.5m`
- spawn radius `2.0m`
- route radius `0.50m`
- five route targets
- target height `0.75..2.25`
- action delay `0..2`
- the same broad mass/thrust/drag/motor-lag ranges as the mixed-frontier run

Training command:

```bash
QUAD_TAG=route_frontier_precision_room_50m_20260517 \
QUAD_OBSTACLE_BLOCK_PROB=1.0 \
QUAD_ROUTE_BLOCKED_TARGET_PROB=1.0 \
QUAD_ACTION_HISTORY_OBS=2 \
QUAD_ACTION_STD=0.2 \
QUAD_LEARNING_RATE=0.00005 \
QUAD_CLIP_COEF=0.03 \
QUAD_ENT_COEF=0.0 \
QUAD_CHECKPOINT_INTERVAL=10 \
QUAD_ROOM_SIZE=2.5 \
QUAD_SPAWN_RADIUS=2.0 \
QUAD_ACTION_DELAY_MIN=0 \
QUAD_ACTION_DELAY_MAX=2 \
QUAD_ROUTE_GOAL_RADIUS=0.50 \
QUAD_ROUTE_GOAL_COUNT=5 \
QUAD_MAX_STEPS=1706 \
QUAD_ROUTE_TARGET_Z_MIN=0.75 \
QUAD_ROUTE_TARGET_Z_MAX=2.25 \
QUAD_MASS_SCALE_MIN=0.85 \
QUAD_MASS_SCALE_MAX=1.15 \
QUAD_THRUST_SCALE_MIN=0.95 \
QUAD_THRUST_SCALE_MAX=1.05 \
QUAD_DRAG_SCALE_MIN=0.90 \
QUAD_DRAG_SCALE_MAX=1.10 \
QUAD_MOTOR_TAU_SCALE_MIN=0.80 \
QUAD_MOTOR_TAU_SCALE_MAX=1.20 \
scripts/quadrotor_remote_gpu.sh train route_clean_mid 50000000 13 \
  experiments/puffer_quadrotor_177903264205/model_puffer_quadrotor_000191.pt
```

Run:

```text
experiments/puffer_quadrotor_177903628229
```

Best 8-case key checkpoint:

```text
experiments/puffer_quadrotor_177903628229/model_puffer_quadrotor_000010.pt
```

Key-suite mean was only `70.47%`, below the previous frontier checkpoint on
the same case mix. Later checkpoints improved compact-room success, reaching
`82.85%` on `room_225`, but they badly damaged baseline, long-horizon, delay,
and combo behavior.

Conclusion: fixed tiny-radius compact-room fine-tuning is too narrow. It
causes frontier forgetting instead of a clean precision improvement.

### Broad Mixed-Frontier Continuation

The successful follow-up simply continued the previous mixed-frontier
distribution:

```bash
QUAD_TAG=route_frontier_combo_continue_50m_20260517 \
QUAD_OBSTACLE_BLOCK_PROB=1.0 \
QUAD_ROUTE_BLOCKED_TARGET_PROB=1.0 \
QUAD_ACTION_HISTORY_OBS=2 \
QUAD_ACTION_STD=0.2 \
QUAD_LEARNING_RATE=0.00005 \
QUAD_CLIP_COEF=0.03 \
QUAD_ENT_COEF=0.0 \
QUAD_CHECKPOINT_INTERVAL=10 \
QUAD_ACTION_DELAY_MIN=0 \
QUAD_ACTION_DELAY_MAX=2 \
QUAD_ROUTE_GOAL_RADIUS=0.75 \
QUAD_ROUTE_GOAL_COUNT=5 \
QUAD_MAX_STEPS=1706 \
QUAD_ROUTE_TARGET_Z_MIN=0.75 \
QUAD_ROUTE_TARGET_Z_MAX=2.25 \
QUAD_MASS_SCALE_MIN=0.85 \
QUAD_MASS_SCALE_MAX=1.15 \
QUAD_THRUST_SCALE_MIN=0.95 \
QUAD_THRUST_SCALE_MAX=1.05 \
QUAD_DRAG_SCALE_MIN=0.90 \
QUAD_DRAG_SCALE_MAX=1.10 \
QUAD_MOTOR_TAU_SCALE_MIN=0.80 \
QUAD_MOTOR_TAU_SCALE_MAX=1.20 \
scripts/quadrotor_remote_gpu.sh train route_clean_mid 50000000 14 \
  experiments/puffer_quadrotor_177903264205/model_puffer_quadrotor_000191.pt
```

Run:

```text
experiments/puffer_quadrotor_177903678332
```

Best checkpoint:

```text
experiments/puffer_quadrotor_177903678332/model_puffer_quadrotor_000140.pt
```

Full frontier:

- mean success: `80.54%`
- weakest case: `precision_035` at `60.68%`
- mean obstacle hit: `16.18%`
- mean timeout: `2.89%`
- baseline: `86.82%`
- mixed50: `95.80%`
- delay 1: `84.67%`
- delay 2: `82.55%`
- underpowered hard: `87.32%`
- overpowered: `82.24%`
- precision radius 0.50m: `75.39%`
- precision radius 0.35m: `60.68%`
- goals 8: `71.68%`
- room 2.25m: `69.82%`
- room 2.50m: `85.55%`
- combo radius 0.75, five goals, z 0.75..2.25, delay 1, noise, dynamics:
  `64.16%`

Compared with the previous promoted checkpoint:

- full-frontier mean improved from `78.49%` to `80.54%`
- combo improved from `59.96%` to `64.16%`
- goals 8 improved from `66.31%` to `71.68%`
- delay 2 improved from `80.18%` to `82.55%`
- overpowered improved from `77.56%` to `82.24%`
- obstacle-hit rate dropped from `18.38%` to `16.18%`

The tradeoff is compact-room performance: `room_225` fell from `71.48%` to
`69.82%`. This is acceptable for promotion because the full frontier improved
and the narrow compact-room run showed that directly optimizing that case is
too destructive.

Promote this as the current frontier policy:

```bash
scripts/quadrotor_visual_eval.sh route_frontier
```

Next recommended push: add true distribution randomization for route radius and
room size instead of fixed hard fine-tuning. The failed narrow probe is evidence
that we need mixed difficulty in the environment, not a single harder setting.

## 2026-05-17 Custom Quadrotor Zero-Start Findings

Current simple benchmark:

```bash
QUAD_RNN_NAME=None \
QUAD_ACTION_STD=0.2 \
QUAD_LEARNING_RATE=0.0022 \
QUAD_ENT_COEF=0.001 \
QUAD_ANNEAL_LR=false \
QUAD_CHECKPOINT_INTERVAL=10 \
scripts/quadrotor_remote_gpu.sh train route_center_block_waypoint_flat 8000000 88 none

QUAD_GPU_SYNC=0 QUAD_EVAL_RNN_NAME=None QUAD_ACTION_STD=0.2 \
scripts/quadrotor_remote_gpu.sh eval-checkpoints route_center_block_waypoint_flat 4 512 latest
```

This trains the custom velocity-control quadrotor from zero on the flat
center-block waypoint task. It is not yet fully reliable, but it is the cleanest
first benchmark so far because it learns without a teacher or blind checkpoint.

Key results:

- `action_std=0.1` is too brittle. Seed 88 can collapse to `0%` success.
- `action_std=0.2` is the important fix. It turns the same bad seed into a
  viable deterministic policy.
- Deterministic eval remains the pass/fail signal. Sampled PPO rollouts can
  report near-zero success while the mean-action checkpoint is already good.
- The built-in Puffer drone hyperparameters do not transfer directly to this
  velocity-control task. They produced only `5.27%` best deterministic success
  on `experiments/puffer_quadrotor_177904635263`.
- Lowering `max_vel` to `0.8` and increasing the horizon to `768` made the task
  worse, not safer: `experiments/puffer_quadrotor_177904697181` peaked at only
  `2.14%`.

Best high-exploration long probe:

```text
experiments/puffer_quadrotor_177904667062/model_puffer_quadrotor_000030.pt
success=96.24%, obstacle_hit=3.76%, timeout=0.00%
```

Short fixed-LR confirmation:

```text
experiments/puffer_quadrotor_177904685028/model_puffer_quadrotor_000031.pt
seed=88, success=94.29%, obstacle_hit=1.66%, timeout=3.56%
```

Second seed with the same short recipe:

```text
experiments/puffer_quadrotor_177904691025/model_puffer_quadrotor_000020.pt
seed=89, success=83.01%, obstacle_hit=1.76%, timeout=3.37%
```

The benchmark is therefore learnable from zero, but not robustly solved. The
next harder limit is seed-to-seed reliability and late-training collapse:
policies can reach `90%+`, then degrade within another 10 checkpoints. Do not
claim the custom env is solved until a fixed command gives a good final
checkpoint across multiple seeds, or until deterministic checkpoint selection is
made an explicit part of the benchmark protocol.

## 2026-05-21 Procedural Exploration Correction

`proc95` must use reachable-component coverage. Counting every valid cell in a
procedural layout made the target partly impossible when the spawn component did
not connect to all free cells. The corrected flag is:

```bash
--explore-reachable-coverage
```

Current corrected baseline:

```bash
scripts/quadrotor4.sh explore-v0-target-distill-eval \
  experiments/qdr224/qdr224_qdr221_vector_dist_row_only_s88.pt \
  --deploy-patch-obs --explore-reachable-coverage \
  --mode procedural_indoor --seed 88 --episodes 4096 \
  --total-agents 1024 --num-buffers 2 --num-threads 16 \
  --max-steps 2048 --speed-fraction 0.35 --spawn-radius-min 0.0 \
  --explore-goal-fraction 0.95 \
  --safety-cone-ray-stop 0.25 --safety-cone-deg 60 \
  --safety-slow-ray 0.70
```

Result: QDR-224 reaches `68.71%` corrected proc95 success. The oracle reaches
`80.90%`, so the task is now valid but not solved. Continuation attempts
QDR-232/QDR-233 regressed, which means the next step is a more stable
distillation/selection method or stronger deployable spatial memory, not more
blind continuation from QDR-224.

## 2026-05-21 QDR-282 Onboard-Target Scaffold

The current hard-procedural exploration candidate is:

```bash
experiments/qdr282/qdr282_onboard_target_obs_qdr251_s88.pt
```

QDR-280 showed a privileged region teacher can solve hard procedural
exploration. QDR-281 showed that simply distilling those privileged targets into
the old deployable recurrent actor does not transfer to hard-p1. QDR-282 makes
the more research-aligned move: expose a deployable onboard frontier target to
the actor with `--deploy-onboard-target-obs`, while rejecting privileged target
modes. This keeps actor inputs real-drone-feasible because the target comes from
the onboard explored/blocked grid and rays, not the hidden map.

Use corrected evals. High-parallel eval without draining active episodes can
censor slow timeouts and report a false `100%`. Add:

```bash
--drain-initial-episodes
```

Current corrected hard-p1 results:

- QDR-251 baseline seed `88`: `28.16%` success, `71.84%` timeout, `0%` obstacle.
- QDR-282 seed `88`: `75.51%` success, `24.49%` timeout, `0%` obstacle.
- QDR-282 seeds `90/91`: `76.46%` / `77.82%`, `0%` obstacle.

Procedural seeds `88/90/91` and fixed-layout seed `88` retained `100%` in the
quick checks, but visual/no-render still finds near-wall stall timeouts. The
next roadmap item is therefore not more privileged-label distillation; it is
recovering from onboard-target stalls or learning a residual around the onboard
planner while preserving the deployable observation boundary.

## 2026-05-22 QDR-283 Exploration Efficiency Reframe

QDR-283 shows the QDR-282 timeout tail is mostly an efficiency/horizon issue,
not a basic perception or safety failure. Hard-p1 success at the strict `2048`
budget is still only around `68-76%` depending on eval size, but the same
checkpoint reaches about `97-99%` over seeds `88/90/91` when the horizon is
`4096`, with zero contacts.

This aligns the project more closely with FUEL/TARE/Active-Neural-SLAM style
exploration: the next benchmark should report coverage over time, final
coverage, episode length, new-cell rate, and contacts. A short horizon remains
useful, but as a fast-exploration metric rather than proof that the actor
cannot explore.

New probe command:

```bash
scripts/quadrotor4.sh explore-v0-target-horizon-probe experiments/qdr282/qdr282_onboard_target_obs_qdr251_s88.pt --horizons 2048,4096 --seeds 88,90,91 --episodes 256
```

Next roadmap item: optimize exploration rate while preserving the current
zero-contact behavior. Candidate levers are frontier/path ordering, avoiding
low-value local viewpoint pauses, speed scheduling based on clearance and
frontier distance, and eventually a learned selector over onboard planner
candidates. Do not return to pure label distillation unless a diagnostic shows
the actor is ignoring a good deployable target.

## 2026-05-22 QDR-284 Fast-Safe Controller Profile

QDR-284 turns part of the QDR-283 efficiency finding into a stronger controller
profile. The current best hard-p1 `2048` eval/visual recipe is the QDR-282
checkpoint with:

```bash
--speed-fraction 0.50 --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80
```

This improves hard-p1 seeds `88/90/91` to `88.76%` / `93.54%` / `90.27%` with
zero contacts, versus the old QDR-282 `2048` profile around `75-78%`. It also
kept procedural/fixed-layout smoke checks near `99-100%` with zero contacts and
passed a `3/3` no-render hard-p1 visual smoke.

Interpretation: faster exploration requires the deployable local slide safety
layer. Raising speed without slide either barely helps or creates contacts. The
next roadmap item is to make the remaining timeout tail smaller by improving
frontier/path efficiency or by training with this fast-safe controller in the
loop; do not push raw speed above this profile as the main line.

## 2026-05-22 QDR-285 Fast-Safe Rollout Checkpoint

QDR-285 completes the next step from QDR-284: train with the fast-safe
controller in the rollout loop. The current checkpoint is:

```bash
experiments/qdr285/qdr285_fast_safe_rollout_qdr282_s88.pt
```

Evaluate and visualize it with:

```bash
--speed-fraction 0.50 --safety-slide --safety-cone-ray-stop 0.30 --safety-cone-deg 60 --safety-slow-ray 0.80
```

Hard-p1 `2048` drained success over seeds `88/90/91` is now `94.79%` /
`92.70%` / `92.12%`, all zero contacts. This improves the three-seed mean over
QDR-284's eval-only profile (`93.20%` vs `90.86%`) and reduces the strict
timeout tail without turning it into obstacle contacts.

The tradeoff is retention: procedural aggregate/layout_mixed/corridor remain
near `99%`, but maze falls to `97.90%` in the seed-88 smoke. Treat QDR-285 as
the current hard-p1 fast-exploration baseline, not as the final explorer.

Next roadmap item: attack the remaining slow tail directly. The visual smoke
still shows recoverable near-wall stalls. The most promising build is a
trajectory-level local motion/recovery controller or learned residual that uses
deployable lidar/odometry/history and optimizes progress under safety, rather
than another one-step speed or cone tweak. Keep QDR-285 as the regression
checkpoint while building that next layer.

## 2026-05-23 QDR-303 Result: Approach Viewpoints Are Not Enough

QDR-303 tried a native planner-side approach-viewpoint flag for shallow onboard
frontiers. This was the simplest version of the region-approach idea: when a
selected frontier is one or two cells away, replace the immediate frontier with
a nearby local viewpoint that has better clearance.

The result was negative on the hard-p1 oracle smoke. Baseline QDR-301/QDR-300
with `--cell-size 0.30` solved the local 256-episode check at `100%` success.
Approach viewpoints increased min clearance from `0.3836m` to about
`0.40m`, but made paths roughly `20m` longer and introduced `1.9-2.7%`
timeouts. The existing late-stall viewpoint trigger remained neutral.

Roadmap implication: do not keep pushing blanket local-viewpoint substitution.
The near-wall tail needs a planner/controller that scores candidate actions or
approach states by predicted progress through the doorway/region, ideally using
short closed-loop rollouts or a small topological state machine. Keep
QDR-300 u100 with explicit `--cell-size 0.30` as the hard-p1 baseline until a
candidate beats it on oracle and policy eval.

## 2026-05-23 QDR-304 Result: Local Trajectory Scoring Is Saturated Too

QDR-304 tried the least repetitive version of a trajectory-valued controller:
no hard-state replay, no label generation, no learned recovery sidecar. The
runtime controller used only deployable current lidar and target state, sampled
short constant-velocity headings, and scored progress, rollout clearance, wall
escape, and smoothness.

The result was neutral/negative. Standalone trajectory scoring was safe but
made local hard-p1 paths much longer. A strict near-wall-only version stacked on
top of barrier/slide was almost indistinguishable from baseline: tiny path gain
on seed `88`, tiny path loss on seed `90`, with both at `100%` oracle success
and zero contacts.

Roadmap implication: current-ray local trajectory scoring is not enough. It
collapses back into the same behavior as the barrier/slide controller. The next
serious improvement should move up one level: persistent topological/frontier
state, region-progress value, or learned/global memory that can say which
doorway/region is worth committing to, not just which immediate heading is
safe.

## 2026-05-23 QDR-305 Result: Make Region Memory Passive First

QDR-305 tried to jump straight to an active persistent frontier-region manager.
It added stable memory slots for frontier clusters and let those slots affect
target choice, attempts, failures, progress, and stale decisions.

That was too large a step. The active selector badly regressed the hard-p1
oracle: the baseline solved the local seed-88 check at `100%`, while active
persistent variants ranged from `62%` down to `14%` success depending on how
progress/stale timing was handled. The failure mode is clear: region memory can
easily suppress reachable frontiers or stale-mark useful regions if it is
allowed to act before its predictions are validated.

Roadmap implication: the next build should be a passive topological tracker,
not an active planner. It should compute and display stable region IDs, region
switches, active-region age, coverage gain while active, and candidate stale
decisions, while leaving the current target stream unchanged. If that passive
tracker can reliably identify the doorway stalls seen in visual eval, then use
it as a scoring feature. Do not let it own target selection yet.

## 2026-05-23 QDR-306 Result: Passive Tracker Is Safe

QDR-306 built that passive topological tracker behind:

```bash
--explore-onboard-region-tracker
```

It is deliberately diagnostic-only: it reports frontier-region count, active
region ID/age, active-region no-progress age, region switch rate, and stale
prediction rate in eval logs and the visual HUD, while leaving target selection
and stale marking unchanged.

The hard-p1 behavior-equivalence gate passed exactly against the QDR-301/QDR-300
baseline. On seed `88`, `256` episodes, baseline and tracker both reached
`100%` success, `0%` contacts/timeouts, coverage `0.950574`, path `168.063m`,
and the same action/clearance metrics. Tracker diagnostics on that run were:
regions `15.4784`, active age `386.797`, since-progress `34.1852`, switch rate
`0.0428793`, stale rate `0.0236302`. A no-render visual smoke also passed, and
the current QDR-300 policy solved a `32/32` no-render visual smoke with no
contacts or timeouts. It still logged `3` near-wall stalls; the visual summary
reported tracker regions `14.6912`, active age `361.2634`, since-progress
`33.8738`, switch rate `0.0473`, and stale rate `0.0240`.

Roadmap implication: we now have the safe debug view needed before touching the
planner again. The next step is not GPU training; it is visual/stall diagnosis
with the tracker enabled. If stalls correlate with region switching or
long active-region no-progress, build the next active planner from that
measured failure. If they do not, the blocker is lower-level local control or
viewpoint execution rather than topological region identity.

## 2026-05-23 QDR-307/QDR-308 Result: Slide Projection Is The Local Tail

QDR-307 added the missing local-control diagnosis to no-render visual stall
logs: raw desired action, barrier-only action, slide-after-barrier action,
target cone scale, progress ratios, and turn angles. This showed that the
remaining near-wall stalls are not primarily missing planner targets. In two
representative QDR-300 policy stalls, the target cone was still moderately
valid (`0.698`/`0.761` scale), barrier kept some progress
(`0.181`/`0.197`), but slide reduced progress to almost zero
(`0.006`/`0.042`) and turned the command nearly sideways.

Removing slide is not the answer. On the oracle controller, barrier without
slide stayed safe but had worse path (`172.609m` vs `168.063m`) and lower min
clearance (`0.344m` vs `0.384m`). Slide is doing useful global work; it just has
a local over-projection tail.

QDR-308 tested the existing post-safety progress floor as a minimal correction.
It was not enough. Oracle path improved slightly, but QDR-300 policy eval
regressed slightly in new-cell rate, path, and min clearance, and visual stall
count did not improve. Do not promote this.

Roadmap implication: the next controller build should be smarter than a scalar
floor. Good candidates are a progress-aware slide projection that only restores
forward motion when the target cone has enough clearance, or a short closed-loop
local controller that optimizes progress and clearance together. Keep the
QDR-307 diagnostics as the regression loop for this work.

## 2026-05-23 QDR-309 Result: Scalar Progress-Aware Slide Is Not Enough

QDR-309 implemented the conservative progress-aware slide probe behind:

```bash
--safety-slide-progress-aware
```

It only interpolates back toward the pre-slide command, and only when the target
cone is passable. The diagnostic effect was real: the worst slide-erased
progress ratios improved from roughly zero to about `0.11`. But the benchmark
did not improve. The 32-episode visual still logged `3` stalls, new-cell rate
fell, and a stronger restoration fraction caused timeouts. Policy eval at the
safe fraction also regressed slightly on new-cell rate, path, and clearance.

Roadmap implication: do not spend more time on scalar progress restoration. The
next local-control attempt needs a richer decision rule: short-horizon
closed-loop action search, or a slide/controller model that understands
doorway passage/tangent geometry and only accepts a progress-preserving action
when it predicts real clearance and progress.

## 2026-05-23 QDR-310 Result: Binary Passability Is Too Crude

QDR-310 tried the user-proposed passability abstraction directly: inflate known
blocked cells in the onboard grid before BFS/frontier traversal, behind:

```bash
--explore-onboard-clearance-passable --explore-onboard-passable-radius <m>
```

It should stay disabled. At the current `0.30m` hard-p1 grid, radius `0.30`
does nothing, while `0.31m+` closes too much graph structure and turns a
`100%` oracle into roughly `30-35%` success with many timeouts. A finer
`0.20m` grid plus clearance was still far worse than the current `0.30m`
baseline.

Roadmap implication: we should stop thinking of the doorway problem as a
binary "try gap / do not try gap" decision. The next promising planner build is
passage quality: detect narrow passages in the onboard known/free/blocked grid,
target the centerline or a clearance-weighted waypoint through them, and avoid
emitting wall-edge frontier targets while still preserving real passable gaps.

## 2026-05-24 QDR-311 Result: Passage Quality Works

QDR-311 is the current best hard-p1 runtime profile. It validates the corrected
doorway abstraction: preserve passable gaps, but bias targets and local
viewpoints toward clearance-centered passage cells.

Use:

```bash
--explore-onboard-passage-quality --explore-onboard-passage-weight 0.35
```

with the QDR-300/QDR-301 hard-p1 stack. The learned QDR-300 policy improved
without retraining: hard-p1 seeds `88/90/91` all reached `100%` success with
zero contacts/timeouts, while path length dropped to about `146-148m` and
new-cell rate rose to about `0.91-0.93`. The 32-episode no-render visual smoke
had `wall_stall_logs=0`. Broader hard-family validation also passed:
p0/p1/p2/p3 across seeds `88/90/91` were all `100%` with zero contacts/timeouts,
and random `procedural_indoor_hard` seeds `88/90/91` were also `100%`.

Roadmap implication: this is worth taking seriously. The next work should be:

1. Make passage-quality targets part of the training/distillation
   distribution instead of only an eval-time planner wrapper.
2. Then push to the next harder benchmark: larger/harder layouts where passage
   quality may need a true local passage graph or learned spatial memory.

## 2026-05-24 QDR-312 Result: Distillation Helps, But Hard-p1 Is Saturated

QDR-312 trained the QDR-311 passage-quality target distribution into the
deployable target policy. This is useful but not a large conceptual step.

Best trained checkpoint:

```text
experiments/qdr312/checkpoints/qdr312_passage_quality_distill_s88_u000400.pt
```

Do not use the final u800 checkpoint as the reference. Late training pushed
student rollout probability to `0.5` and reintroduced timeout drift. The u400
checkpoint is the useful early-stop point.

Against QDR-300 plus the QDR-311 runtime passage-quality flags on hard-p1 seeds
`88/90/91`, u400 slightly improves efficiency while keeping zero contacts:
paths moved from `149.445/147.239/149.940m` to
`148.552/145.010/148.999m`, and new-cell rate moved from
`0.899/0.917/0.896` to `0.907/0.933/0.907`. Success is basically tied and
still has tiny timeout tails under larger drained evals.

Visual eval is mixed: u400 solved `32/32` with zero contacts/timeouts and a
higher new-cell rate than QDR-300+QDR-311, but it logged one recoverable
near-wall stall. QDR-300+QDR-311 remains the cleaner visual reference with
`wall_stall_logs=0`.

Roadmap implication: hard-p1 is now too saturated to guide the next big move.
Use it as a regression gate, not as the main research target. The next useful
work is a harder procedural suite:

1. Larger rooms with more rooms and bottlenecks.
2. Randomized doorway widths and offset doors.
3. Longer coverage requirement or coverage-AUC/time-to-coverage scoring.
4. Side-by-side comparison of QDR-300+QDR-311 runtime planning vs QDR-312 u400
   trained passage-quality policy.

Only after that failure pattern is clear should we build the next planner or
learning component. Likely candidates are local passage graphs, explicit
coverage-AUC training targets, or learned spatial memory over larger layouts.

## 2026-05-24 QDR-313 Result: Tight Topology Is The Next Limit

QDR-313 answered the "are we just making the room too easy?" question. Enlarging
hard-p1 to an `8.0m` room and even raising coverage from `0.95` to `0.98` did
not create a useful next benchmark. The current stack still solves it almost
perfectly with zero contacts. The old hard generator was saturated because its
topology was too forgiving, not because the room was too small.

The new useful benchmark is `procedural_indoor_tight`, especially
`procedural_indoor_tight_p1`. It narrows doorway/gap widths and adds more
clutter while keeping the same deployable observation/control stack. Tight-p1
is now the best next regression target because failures are mostly safe
timeouts, not crashes:

- oracle seeds `88/90/91`: `92.62/91.89/92.22%`, `0%` contacts;
- QDR-300+QDR-311 runtime stack: `93.84/93.42/94.89%`, `0%` contacts;
- QDR-312 u400: `94.27/94.61/94.19%`, `0%` contacts.

Roadmap implication: stop launching more student training against the old
hard-p1 suite as the main progress signal. The next build should raise the
tight-p1 planner/oracle ceiling first. If the oracle remains around `92%`, more
distillation or PPO is mostly teaching the policy to imitate a planner that is
already failing late exploration completion.

Recommended next QDR:

1. Diagnose tight-p1 oracle timeouts with no-render visual/stall logs.
2. Improve late-frontier completion in the onboard planner: passage graph,
   doorway centerline waypoints, persistent region search, or a late-frontier
   objective that prefers reachable completion over nearest wall-edge cells.
3. Gate the planner change on oracle tight-p1 seeds `88/90/91`.
4. Only then distill/train QDR-312-style if the oracle ceiling moves.

## 2026-05-24 QDR-314 Result: Cone Alignment Is Not The Missing Piece

QDR-314 reproduced tight-p1 oracle timeouts in a small no-render visual loop and
falsified the most obvious planner/safety mismatch.

Baseline no-restart tight-p1 seed `88`, `32` episodes, `cell_size=0.30`,
QDR-311 passage-quality stack:

- success `90.62%`, timeout `9.38%`, contacts `0%`;
- timeout episodes occurred at `48.21%`, `46.02%`, and `62.60%` coverage;
- `wall_stall_logs=101`.

I added `--explore-onboard-clearance-cone-deg` so the C onboard planner can
score candidate directions with a configurable cone instead of a hard-coded
30 degree side sample. Matching the planner to the deployed safety cone
(`60deg`) regressed the same trace to `84.38%` success and `15.62%` timeouts,
still with `0%` contacts.

Roadmap implication: do not spend the next cycle training a student against
this as if the teacher improved. The current tight-p1 issue is not simply that
the planner is too optimistic about side clearance. The better next levers are:

1. Representation scale/passability: test whether `cell_size=0.30` is the wrong
   scale for `0.34-0.50m` tight gaps.
2. Target lifecycle: expose stale ticks/min-progress as configurable diagnostics
   if cell size does not move the oracle ceiling.
3. Topological completion: use persistent region state only after the diagnostic
   predicts which regions are actually causing timeouts.

## 2026-05-24 QDR-315 Result: Tight Topology Needed A Coarser Grid

QDR-315 found a real lever. The tight-p1 plateau was mostly a representation
scale problem.

Tight procedural gaps are `0.34-0.50m`. The old `0.30m` onboard exploration grid
was not simply "too coarse" or "too fine" in the abstract; it was the wrong
scale for this gap distribution. It preserved narrow passages but generated too
many unstable one-cell/stale frontier objectives. Smaller `0.25m` made the
churn much worse. Larger `0.40m` hid too much structure. `0.35m` is the useful
middle.

Oracle tight-p1 seed `88`:

- `0.25m`: `73.33%` success, `26.67%` timeout;
- `0.30m`: `88.63%` success, `11.37%` timeout;
- `0.35m`: `96.63%` success, `3.37%` timeout;
- `0.40m`: `86.15%` success, `13.85%` timeout.

Held-out oracle `0.35m` seeds `90/91`: `96.89/96.84%`, `0%` contacts.

The current best checkpoint also benefits from the same scale. QDR-312 u400
with explicit `cell_size=0.35` override:

- tight-p1 seeds `88/90/91`: `97.51/97.41/97.83%`, all `0%` contacts;
- p0 seeds `88/90/91`: `99.62/99.57/99.67%`, all `0%` contacts;
- p2 seeds `88/90/91`: `99.80/99.78/99.77%`, all `0%` contacts;
- p3 seeds `88/90/91`: `99.93/99.90/99.95%`, all `0%` contacts.

Roadmap implication: promote QDR-312 u400 plus `cell_size=0.35` as the current
tight-topology baseline. The remaining hard case is specifically p1's `2-3%`
safe-timeout tail, not collision avoidance or generic exploration failure.

Next useful work:

1. Run random `procedural_indoor_tight` at `0.35m` to make sure p1/p0/p2/p3
   transfer covers the generator distribution.
2. Distill/train with `cell_size=0.35` directly instead of evaluating a
   `0.30m` checkpoint under an override.
3. Then attack remaining p1 timeouts with target lifecycle or topological
   completion diagnostics, not another safety-cone tweak.

## 2026-05-24 QDR-316 Result: Promote Direct 0.35m Training, Then Raise The Bar

QDR-316 completed item 2 above. The direct `0.35m` training baseline is now
better than keeping QDR-312 as a `0.30m` checkpoint and overriding the env at
eval time.

Promoted checkpoint:

```text
experiments/qdr316/checkpoints_preserve/qdr316_cell035_preserve_distill_s88_u000200.pt
```

What changed:

1. Added train-time support for `--ignore-checkpoint-env-config` during
   init-checkpoint loading. This matters because the invalid first run silently
   restored QDR-312's old `cell_size=0.30`.
2. Trained the QDR-312 passage-quality target policy directly under
   `cell_size=0.35`.
3. Preserved the tight-family safety profile while reducing visible near-wall
   stall churn.

Main evidence:

- tight-p1 seeds `88/90/91`: `97.96/98.16/97.24%` success, all `0%` contacts;
- p0 seeds `90/91`: `99.70/99.74%`, all `0%` contacts;
- p2 seeds `90/91`: `99.80/99.85%`, all `0%` contacts;
- p3 seeds `90/91`: `99.91/99.93%`, all `0%` contacts;
- no-render visual tight-p1 seed `88`: `32/32`, `0%` contacts/timeouts,
  `wall_stall_logs=5`.

Roadmap implication: this is progress, but it also means the current tight-p1
suite is close to saturated. The next roadmap step should be limit-finding, not
more PPO-style tuning on the same distribution.

Recommended next QDR:

1. Evaluate QDR-316 u200 on random `procedural_indoor_tight` mixtures and higher
   completion thresholds, especially `0.98`.
2. Add a larger connected-map exploration benchmark that keeps actor inputs
   deployable and removes any easy fixed-pattern memorization.
3. Measure revisit inefficiency and late-frontier completion, not just final
   success.
4. Only build a new planner/memory component after the harder benchmark exposes
   a specific failure mode.

## 2026-05-24 QDR-317 Result: House v0 Is Built, But Already Solved

QDR-317 implemented item 2 as a PufferLib-native `procedural_house` generator.
This is the right architectural direction: realistic indoor topology inside the
fast C loop, not a slow simulator bridge.

The first version is single-floor and box-based:

- central-hall house;
- side-corridor house;
- loop/courtyard house;
- irregular open-plan house;
- randomized door positions and furniture/clutter;
- forced variants `procedural_house_p0..p3`;
- existing lidar/ray, onboard planner, safety, and visual harness.

The good news: the benchmark is connected, reachable, and fast enough for GPU
eval. The bad news: it is not yet a new learning limit.

GPU result, room `8.0m`, `cell_size=0.35`, `4096` steps:

- goal `0.98`: oracle `99.82%`, QDR-316 `99.84%`, all `0%` contacts;
- goal `0.99`: oracle `99.68%`, QDR-316 `99.74%`, all `0%` contacts.

Roadmap implication: do not train on house v0. It is a useful regression and
visualization scaffold, but it does not push beyond QDR-316.

Next roadmap step:

1. Build `procedural_house_multizone` or `procedural_house_multifloor`.
2. Keep it PufferLib-native and inspectable: stacked floor layers, stairwell or
   shaft connector volumes, and room-discovery metrics before photorealism.
3. Add metrics beyond coverage: rooms discovered, floor transitions, revisit
   ratio, late-frontier completion, and connector discovery time.
4. Gate the oracle first. If the oracle ceiling is poor, fix the benchmark. If
   the oracle is high but QDR-316 fails, then train/distill against the new
   failure mode.

## 2026-05-24 QDR-318 to QDR-322: House Scale Is Solved, Fast Exploration Is Not

QDR-318 implemented `procedural_house_multizone` and forced variants
`procedural_house_multizone_p0..p3`. After fixing the first malformed/hard
draft, the oracle ceiling was high and QDR-316 matched it: at room `8m`,
`cell_size=0.35`, `4096` steps, goal `0.99`, QDR-316 reached `99.03%` with
`0%` contacts.

QDR-319 scaled the same family to larger rooms. It still did not expose a
policy gap:

- `10m`, `0.35m` grid, `4096` or `3072` steps: oracle and QDR-316 `100%`;
- `12m`, `0.40m` grid, `4096` steps: oracle and QDR-316 `100%`;
- all checked cases had `0%` contacts/timeouts.

This means single-floor 2D coverage with a loose horizon is no longer the right
main benchmark. It is useful as regression infrastructure, but not as the next
research push.

QDR-320 changed the axis to fast exploration: same `12m` multizone house,
`0.40m` grid, goal `0.99`, but only `1200` steps.

- speed `0.50`: safe but too slow, QDR-316 `26.11%`;
- speed `0.70`: useful improvement, QDR-316 `40.24%`, but `0.231%` contacts;
- speed `0.90`: unsafe, QDR-316 `41.14%` with `34.44%` contacts.

QDR-321 tested a simple global clearance schedule at speed `0.70`. It restored
near-zero contacts but dropped success to about `36%`, so it is not promotable.

QDR-322 tested the existing short-horizon trajectory safety wrapper. It is also
not promotable: speed `0.70` did not improve the success/contact tradeoff, and
speed `0.90` still crashed far too often. The current wrapper is not enough to
turn high-speed house exploration into a safe benchmark.

Roadmap update:

1. Treat QDR-320 as the current house-limit benchmark: fast high-coverage
   exploration, not eventual coverage.
2. Diagnose QDR-320/QDR-322 contact states and then build adaptive speed/local
   control using only deployable local signals: rays, passage quality, target
   direction, and short-horizon clearance.
3. Keep QDR-316 u200 as the current checkpoint; the student is not the main
   bottleneck until the oracle/controller ceiling moves.
4. After adaptive local speed is solved, add explicit room/connector discovery
   metrics or a true 2.5D/multifloor exploration grid.

## 2026-05-24 QDR-323 to QDR-324: Fast-House p3 Is the Current Blocker

QDR-323 diagnosed the fast-house contact tail instead of guessing. The forced
`procedural_house_multizone_p3` failures are not mainly bad target prediction:
student and oracle target directions are close, the commanded/action rays often
look open, and the common failure is side-clearance collapse near passages.

QDR-324 tested the first deployable controller response: a lidar min-ray speed
cap applied after target/barrier/slide action selection. It is useful but not
enough.

What changed in the current best fast-house profile:

- aggregate `12m` multizone, speed `0.70`, goal `0.99`, max steps `1200`;
- QDR-316 with light cap `0.18/0.45`: `41.85%` success, `0.021%` contacts;
- previous QDR-320 speed `0.70`: `40.24%` success, `0.231%` contacts.

What still fails:

- forced p3 at speed `0.70` is safe but mostly timeout;
- forced p3 at speed `0.90` remains unsafe even with stronger min-ray caps;
- stronger global caps convert contacts into timeouts rather than producing
  deliberate fast passage traversal.

Roadmap implication: stay in 2D house exploration for now. Full 3D/multifloor
houses would multiply complexity before the local passage controller is solved.
The next meaningful build is a passage-aware local controller or learned
hard-state controller for p3-like passages, using deployable inputs only. Good
candidate signals are side-clearance trend, front/cone clearance, corridor
width estimate, target bearing, and whether recent progress is dropping while
the target ray stays open.

QDR-325 rejected the simplest version of that idea. Stateless left/right
passage-centering made p3 worse:

- centering-only speed `0.70`: about `13-15%` contacts;
- centering plus QDR-324 cap speed `0.70`: about `1-2%` contacts and no
  success gain;
- centering plus cap speed `0.90`: oracle contacts about `41%`.

Do not revisit naive side-imbalance steering. QDR-326 also tested the existing
resettable constant-action MPC replay on p3 stall states. It did not beat
baseline: baseline gain `0.1516`, MPC gain `0.1401`, with no successes or
crashes in either replay. This means "compact local MPC" is not the right
recommendation unless it becomes a materially different closed-loop training
task.

The least-saturated next step is now narrower and stronger: build a resettable
p3 passage skill benchmark. It should restore real p3 hard states, optimize or
train over full closed-loop outcomes, and only promote a skill if it beats
QDR-324 baseline replay on progress, success, and contacts. The actor side of
that skill must stay deployable; privileged map/geometry is allowed only for
teacher labels, critic/value, or diagnostics.

## 2026-05-24 QDR-327 to QDR-330: Current p3 Direction

The resettable passage-skill branch was tested and partially rejected.
QDR-327 proved the harness works, but the existing recovery-PG sidecar is worse
than base replay. QDR-328 then showed why: most p3 hard states are not
unrecoverable traps. Base replay keeps making safe progress if allowed more
time, but the `1200`-step fast-house budget expires before enough coverage is
completed.

The current blocker is therefore time-efficient passage traversal, not a
standalone escape maneuver.

QDR-329 found the first useful planner lever:

- onboard path lookahead `2` is too local for p3;
- lookahead `4` roughly triples forced-p3 success for QDR-316
  (`4.54% -> 13.82%`) while improving aggregate success
  (`41.85% QDR-324 -> 42.23%`) at the cost of a larger contact tail;
- lookahead `6` is too far and regresses badly.

QDR-330 tried stronger min-ray caps around lookahead `4`. It improved forced
p3 slightly but hurt aggregate throughput, so it is not the default.

Current recommendation:

1. Keep two fast-house profiles:
   - QDR-331 safety baseline: speed `0.70`, zero-contact aggregate in the
     latest gate.
   - QDR-332 pushed profile: speed `0.72`, aggregate QDR-316 `47.91%`
     success / `0.046%` contacts / `90.38%` coverage.
2. Do not return to recovery-PG, constant-action MPC, naive side-centering, or
   lookahead `6` without a new diagnostic.
3. QDR-331 is a real promotion over the fixed lookahead branch: forced p3
   QDR-316 `16.85%` success / `0%` contacts / `86.86%` coverage, aggregate
   `45.12%` success / `0%` contacts / `89.76%` coverage.
4. QDR-332 shows raw speed is not the full solution. Speed `0.75` improves
   aggregate to `48.54%` but with `0.182%` contacts, and p3 becomes unsafe
   quickly above that.
5. QDR-333 through QDR-335 narrowed the failure:
   - passage-risk speed capping helped the oracle but not the fixed student;
   - velocity-aware momentum capping was a no-op without activation diagnostics;
   - hard-weighted target-student fine-tuning did not beat QDR-332.
6. QDR-336 through QDR-338 clarified the current blocker:
   - corridor-width-aware local velocity selection failed to beat QDR-332;
   - p3 success jumps from `16.36%` at `1200` steps to `90.52%` at `2048`
     with QDR-316 and zero contacts, so the dominant issue is coverage-rate
     under a tight budget;
   - adaptive max-6 lookahead worsened p3 target quality and should not replace
     QDR-331's max-4 adaptive target horizon.
7. Do not spend the next cycle on another local speed cap, side-clearance
   selector, or target-student fine-tune. The highest-potential branch is
   planner-level exploration efficiency: frontier clustering, expected
   new-cell gain per travel time, target commitment, and budget-aware ordering.
   Use `2048` steps as the solvability sanity check, `1600` as the near-term
   benchmark, and `1200` as the stress gate.
8. Keep a strict promotion rule: beat QDR-332 speed `0.72` aggregate and p3,
   while not turning timeouts into contacts. If a new method only shifts along
   that tradeoff curve, record it and move on.

## 2026-05-25 Privileged Unexplored Direction

QDR-339 through QDR-341 changed the roadmap again.

The budget-aware deployable frontier scorer was a dead end. It reduced
new-cell rate and p3 success rather than improving budget efficiency. The
strong diagnostic was QDR-340: privileged nearest-unexplored target selection
nearly solves forced p3 under the same `1200`-step budget (`98.34%` success,
`98.34%` coverage, about `0.1%` contacts), while the current onboard oracle is
only `20.07%` success and `88.77%` coverage. That proves the current stress
gate is not impossible; the missing capability is target selection/search over
unseen space.

QDR-341 also ruled out the simplest deployable shortcut. Treating all unknown
cells as tentatively traversable collapses p3 to about `27%` coverage and `0%`
success, even with almost no contacts. Unknown space cannot simply be planned
through as free space.

Roadmap implication:

1. Stop tuning hand-coded frontier cluster scores unless a new diagnostic shows
   they explain the privileged-unexplored gap.
2. Use privileged unexplored as a training/diagnostic signal, not as actor
   input.
3. The current highest-potential line is QDR-342-style privileged-label
   learning: recurrent deployable target policies that learn from privileged
   nearest-unexplored labels while receiving only rays, state/history, coverage,
   and local map patches.
4. If privileged-label learning cannot transfer, the next engineering build is
   a constrained topology/visibility planner that preserves region identity and
   only reasons through unknown space when supported by current sensor evidence
   or accumulated map topology.

QDR-342/QDR-343 now put a bound on item 3. The privileged-label target-vector
policy did not transfer:

- QDR-342 from-zero, teacher-assisted DAgger: final p3 `0%` success,
  `28.56%` coverage, `0.83%` contacts.
- QDR-343 aggressive on-policy DAgger from QDR-342 u200: final p3 `0%`
  success, `30.97%` coverage, `3.90%` contacts.

This does not invalidate privileged learning, but it does reject the current
thin interface: a four-value target direction head trained from local
rays/patch/coverage is not enough to reproduce privileged unexplored search.

Next roadmap item: build a deployable spatial state interface before more
imitation training. Good candidates are a local egocentric occupancy/frontier
map head with persistent memory, a topological frontier graph exposed through
deployable features, or an auxiliary loss that predicts reachable unexplored
regions/frontier age from onboard map history. After that, retry privileged
unexplored labels as supervision for the richer state, not as a bare target
vector.

QDR-344 is the active first step on that roadmap: a deployable frontier graph
selector with persistent region identity and unknown-component scoring. It is a
planner-side diagnostic, not a promoted baseline yet. If its p3 oracle improves
meaningfully over QDR-337/QDR-340 onboard oracle without contacts, then train or
distill against the graph-selected target interface. If it does not, the next
engineering build should expose a richer spatial memory/map representation to
the policy instead of continuing hand-coded frontier scoring.

QDR-344 came back positive but bounded. It improves forced p3 over the old
onboard selector at both `1200` and `1600` steps, including with the fixed
QDR-316 student, and does not create a contact tradeoff. It still remains far
below privileged unexplored search. Roadmap implication: keep QDR-344 as the
current deployable planner-side stepping stone, but do not keep tuning scalar
frontier scores for long. The next high-potential branch is either:

1. turn the graph state into a policy-facing spatial-memory interface and train
   against privileged unexplored labels again, or
2. make the graph planner closer to the privileged BFS by adding deployable
   topological loop/room identity and explicit "go inspect unknown boundary"
   state, not just unknown-component size.

QDR-345 refines this: unknown-component gain is the useful part of the graph
selector, while sticky persistence penalties hurt. The next planner profile is
`no_persistence`: component gain on, active/attempt/failure/stale penalties off.
This should be used as the graph baseline before another training run. Do not
spend many cycles hand-tuning those scalar weights; the remaining gap to
privileged unexplored is still too large. The next higher-leverage build is a
policy-facing deployable spatial-memory/topology interface that lets learning
operate on frontier regions, unknown-component size, age, and reachability
directly.

QDR-346 promotes `graph_no_persistence` as the current deployable planner
baseline. It holds on aggregate house at both `1200` and `1600` steps and keeps
contacts low. This is the strongest non-privileged exploration target selector
so far, but the conclusion is still the same: the policy needs graph/spatial
state, not just a single target vector. QDR-347 should expose a compact set of
frontier candidates to the actor: body-frame direction, normalized path depth,
unknown-component gain, cluster size, passage/clearance score, and selected
candidate marker.

QDR-347 tested that exact compact graph-candidate observation. It was useful as
a diagnosis but not as a policy solution. The ABI stayed stable by writing four
candidates into the tail of the existing exploration patch observation, and
`--deploy-graph-candidate-obs` can expose only those candidate features plus
base deployable state. The closed-loop policy still failed: u150 deterministic
p3 eval was `0%` success, `30.57%` coverage, and `99.80%` timeout. The failure
is now clearer: the actor predicts that a target exists, but chooses directions
that disagree with the privileged oracle by about `116deg` and point into much
lower clearance.

Roadmap implication: do not keep treating graph candidates as passive features
for a target-vector regressor. The next high-potential build is candidate
selection as the task:

1. cache top-K graph candidates from the planner so training throughput recovers;
2. add a candidate-index or candidate-scoring policy head;
3. label the correct candidate with privileged unexplored progress or train it
   with a privileged critic/reward;
4. convert the selected candidate into the existing velocity target/action path
   for eval and visual debugging.

This keeps the deployed actor real-drone-feasible while making the learned
decision match the structure of the exploration problem.

Agile flight roadmap update: use AFL-016 as the current open-space direct-motor
baseline for the custom env. The important correction was to stop mixing the
native `ocean/drone` control task with conservative route PPO settings. The
promoted recipe is:

- `action_mode=motor_ref_hover`
- `dynamics_profile=drone_reference`
- `physics_dt=0.002`, `physics_substeps=5`
- `route_target_obs_mode=3`
- `route_step_cost=0.0`
- `route_reward_mode=4` for controlled arrival, with `route_goal_speed=2.0`,
  `route_arrival_radius=1.5`, and `route_arrival_vel_scale=0.08`
- AFL-008 native-drone PPO hparams (`lr=0.00975033`, `gamma=0.981094`,
  `gae_lambda=0.883828`, `replay_ratio=2.25498`, `clip=0.067834`,
  `vf_coef=4.42222`, `ent_coef=9.62597e-05`, `action_std=1.0`)

AFL-013 solved route-mode reach-hover at `99.41%`, AFL-014 solved one
same-altitude waypoint at `100%`, and AFL-015 solved open 3-waypoint 3D flight
at `98.37%`. AFL-016 then solved controlled-arrival open 3-waypoint flight at
`100%`, with `0%` contacts and final speed reduced to `1.85m/s`. Do not spend
more time on conservative PPO variants for this rung. The next harder limit is
geometry: simple gates/obstacles, path efficiency, and action saturation on top
of AFL-016.

AFL-017 through AFL-019 pushed that geometry limit. The new `gate_slalom`
obstacle family deliberately includes offset gates, pillar slalom, cross gates,
and chicanes. Mixed gates failed from zero (`35.0%` success, `53.7%` obstacle
hits), and forced-pattern eval showed the wall/gate variants are still too hard
for direct target pursuit plus rays. Isolating the easiest P1 pillar slalom
helped but still collided (`53.7%` success, `45.9%` obstacle hits). Adding
stronger clearance shaping (`radius=0.80`, `scale=0.60`) produced the first
safe creative-obstacle baseline:

```text
experiments/agile/afl019/reference_waypoint_pillar_slalom_clearance/quadrotor/1781033504492/0000000041353216.bin
```

At 1024 steps it is cautious: `76.6%` success, `0%` obstacle hits, and `23.1%`
timeouts. At 1536 steps it reaches `99.7%` success with `0%` obstacle hits and
no timeouts. This is the current obstacle-flight baseline. The next roadmap item
is to recover speed/path efficiency without reintroducing contacts, then use a
curriculum from pillar slalom back toward true gate passages.

AFL-020 through AFL-022 show that simple global time pressure is not that
recovery mechanism. Training at `1536` with a tiny step cost reached only
`68.46%` success; training at `1536` without the step cost reached `74.57%`;
training at `1024` with the step cost reached `66.28%`. All three stayed mostly
safe but increased timeouts and lost to AFL-019. Keep AFL-019 as the promoted
safe obstacle baseline and do not run more small step-cost sweeps on this
reward. The higher-potential next branch is structured speed curriculum:
teach fast passage with looser clearance/easier P1, then restore AFL-019
clearance, or make speed reward explicitly clearance-conditioned rather than
global.

AFL-039/AFL-040 pushed harder fixed geometry with a new `agile_geometry`
family. This was the right kind of stress test, but the first curriculum was
not right. The intro mix failed at `28.3%` success, mostly because the dense
pillar-weave variant was much harder than the old solved P1. Isolated P3
switchback baffles were already good from AFL-038 (`~92%` success), and
stronger clearance fine-tuning reduced contacts only marginally (`7.6%` to
`6.9%`). Conclusion: scalar reward tuning on the current fixed policy is
saturated for this branch.

Next agile-flight roadmap item: build a graded local-geometry curriculum rather
than another mixed random course. Start with pillar density/gap width levels
that interpolate from old `gate_slalom_p1` to the failed dense weave, then add
explicit diagnostics for near-obstacle corner clips. If reward-only learning
still leaves a contact tail, add a deployable local passage affordance or
recovery-label objective for near-corner states before trying broader mixes or
dynamics randomization.

June 23, 2026 update: AFL-046/AFL-047 exposed a harness bug before a learning
result. The AFL009 wrapper rebuilt the backend in bf16, which made direct-motor
float checkpoints saturate actions and crash immediately. AFL009 now builds
float32 by default. AFL-048 confirmed stable fixed-logstd training, but AFL-049
showed that a fair 20M-step float32 continuation from AFL-038 does not improve
the bottleneck curriculum: best last-5 eval was `42.17%` success, `46.84%`
timeout, and `10.81%` obstacle hits. The next lever should not be more
conservative PPO continuation; it should be a passage-specific mechanism such
as hard-state replay, deployable passage-affordance/recovery labels, or a local
commit-through-gap action/waypoint rung.
