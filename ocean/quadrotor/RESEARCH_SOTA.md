# Quadrotor RL / Sim-to-Real Research Notes

Last researched: 2026-05-21.

Purpose: keep a practical map of open-source drone-RL and sim-to-real work that
should influence this custom PufferLib quadrotor environment. This is not a
survey paper. It is a "what can we learn or reuse next" note.

## Executive Takeaway

May 21, 2026 reassessment after QDR-275/QDR-276: we were partly going
sideways. The local wall-stall probes are useful diagnostics, but the public
exploration literature does not solve this class of problem by asking a
reactive local policy or a scalar-weighted local MPC controller to infer
exploration structure from rays alone. Strong systems keep persistent spatial
state and decompose the task:

- map or local occupancy/frontier memory;
- global/topological frontier or viewpoint selection;
- local collision-safe trajectory or waypoint generation;
- low-level flight control;
- optional learned modules trained with privileged labels, critics, or
  imitation, while the deployed actor stays onboard-feasible.

May 22, 2026 QDR-283/QDR-284 update: this architectural reading now matches our
own data. The QDR-282 onboard-target scaffold is safe and mostly succeeds when
the horizon is doubled from `2048` to `4096`; hard-p1 rises to about `97-99%`
with zero contacts over seeds `88/90/91`. QDR-284 then shows that a deployable
local slide-safety layer lets the same checkpoint run faster at `2048` steps:
speed `0.50` with earlier slide/slowdown reaches about `89-94%` hard-p1
success with zero contacts. The real near-term problem is exploration rate:
time-to-coverage, low-value local target pauses, and path/viewpoint efficiency.
Treat a short-horizon failure as a fast-exploration metric, not automatically
as a perception failure.

This does not mean abandoning PufferLib. It means porting the architectural
ideas, not the ROS stacks. The next custom-env build should introduce explicit
frontier/viewpoint candidates or a small topological graph in the PufferLib C
env, add a debug view that proves candidate correctness, then train a learned
high-level selector or student against privileged labels. Fast indoor flight
remains the Swift-style track; indoor exploration should become a
map/frontier/graph task first, then later merge with fast flight.

May 19, 2026 pivot: the custom quadrotor work should now follow this Swift-style
structure instead of continuing to optimize the current waypoint/velocity
benchmark. The old benchmark is a regression harness. The main research track is
privileged learning plus deployable onboard observations and thrust/body-rate
control inside PufferLib.

Swift is the right north star, but it is not an open-source codebase we can port.
The Nature paper explicitly releases pseudocode and racing data, not full source
code. The useful pieces to copy conceptually are:

- low-level command interface: collective thrust plus body rates, with an
  accurate low-level controller model;
- observation design: state/gate context, previous action, normalized inputs,
  and a privileged critic during training;
- reward design: progress plus perception-awareness plus command smoothness plus
  terminal crash penalty;
- sim-to-real strategy: start with a simple simulator, then identify residual
  observation and dynamics errors from real flights and fine-tune;
- scientific evaluation: controlled stress cases for observation shift,
  dynamics shift, crash rate, margin, completion, and speed.

For our repo, PufferLib is not optional infrastructure. It is the core research
loop because it gives us the iteration speed needed to debug RL honestly. The
best next move is to keep PufferLib as the main environment and training stack,
then borrow benchmark structure from other systems: gate/race-style indoor
courses, body-rate/thrust action rungs, previous-action observation, privileged
critic or diagnostics, residual dynamics hooks, and harder train/eval splits.

## Source Map

| Source | What is open | What matters for us | Portability |
| --- | --- | --- | --- |
| [Swift Nature paper](https://www.nature.com/articles/s41586-023-06419-4) | Paper, figures, methods. Full source intentionally not public. | Defines the modern champion-level recipe: PPO in sim, thrust/body-rate actions, gate/state observations, previous action, privileged critic, progress/perception/smoothness/crash reward, residual obs/dynamics fine-tuning. | High conceptual value, low code reuse. |
| [Swift supplementary data](https://zenodo.org/records/7955278) | `pseudocode.zip` and `racing_data.zip`. | Pseudocode covers gate detection, VIO drift estimation, policy training, and residual model identification. Racing data is useful for analysis style, not our immediate training. | High conceptual value. Small enough to inspect when implementing Swift-lite. |
| [uzh-rpg/sim2real_drone_racing](https://github.com/uzh-rpg/sim2real_drone_racing) | Older MIT UZH zero-shot sim2real racing repo. ROS Melodic, Python 2.7, TensorFlow 1.13. | Good for domain randomization, gate perception, sim data collection, and sim-to-real workflow. Too old/heavy for direct adoption. | Medium conceptual value; low direct portability. |
| [uzh-rpg/agilicious](https://github.com/uzh-rpg/agilicious) | Open-source/open-hardware agile quadrotor stack. | Best hardware/software reference for the real drone direction: estimator/controller architecture, high thrust-to-weight design, onboard compute, Betaflight-style deployment assumptions. | High design value; avoid importing stack now. |
| [uzh-rpg/flightmare](https://github.com/uzh-rpg/flightmare) | MIT simulator with Unity rendering, physics, sensors, RL API. | Reference for richer sensors and hundreds of parallel quadrotors. Shows the value of decoupling physics and rendering. | Medium. Useful to compare sensor abstractions; too heavy for our fast C/PufferLib loop. |
| [uzh-rpg/acmpc_public](https://github.com/uzh-rpg/acmpc_public) | GPL implementation of Actor-Critic MPC for agile flight. | Shows how UZH combines RL with differentiable MPC for better OOD robustness and sample efficiency. | Conceptual only unless license/stack fit. |
| [uzh-rpg/cl_initial_buffer](https://github.com/uzh-rpg/cl_initial_buffer) | GPL contrastive initial-state buffer library, not full drone-racing training code. | Relevant if exploration gets stuck: reuse prior visited states to initialize more informative training episodes. | Medium concept; code is small but GPL. |
| [uzh-rpg/learned_inertial_model_odometry](https://github.com/uzh-rpg/learned_inertial_model_odometry) | GPL learned inertial odometry code. | Relevant later for real IMU/VIO modelling and observation noise. Not needed for current sim-only learning. | Later-stage sim-to-real reference. |
| [uzh-rpg/learning_on_the_fly](https://github.com/uzh-rpg/learning_on_the_fly) | GPL 2026 rapid adaptation code with checkpoints and notebooks. | Very relevant for future residual dynamics learning and fast policy adaptation under payload/wind/model mismatch. | High conceptual value; later-stage code reference. |
| [uzh-rpg/rpg_flightning](https://github.com/uzh-rpg/rpg_flightning) | GPL differentiable-simulation quadrotor-control code. | Suggests an alternative to PPO for low-level control: differentiable sim can learn recovery/control very fast from state or visual features. | Good future branch; not the next PufferLib PPO benchmark. |
| [uzh-rpg/foresight-nav](https://github.com/uzh-rpg/foresight-nav) | GPL indoor exploration / scene-imagination code. | Useful for the high-level exploration roadmap: global exploration is usually not just reactive local control. Predict occupancy/semantics for unseen regions, then pick long-term goals. | Conceptual for exploration, not quadrotor dynamics. |
| [ntnu-arl/aerial_gym_simulator](https://github.com/ntnu-arl/aerial_gym_simulator) | BSD Isaac Gym aerial simulator. | Strong reference for high-throughput depth/LiDAR/camera sensors, modular robots/tasks/controllers, and GPU training of vision navigation. | High reference value if we later need Isaac-style depth/LiDAR. |
| [emNavi/AirGym](https://github.com/emNavi/AirGym) | BSD IsaacGym quadrotor DRL + sim-to-real toolchain. | Very relevant because it has control modes from position to single-rotor thrust, depth-based planning, PX4-aligned controller, and sim-to-real deployment modules. | High reference value; direct port would be large. |
| [microsoft/AirSim-Drone-Racing-Lab](https://github.com/microsoft/AirSim-Drone-Racing-Lab) | MIT AirSim drone-racing benchmark and baselines. | Good older benchmark reference and API examples. Less useful for our current high-throughput RL loop. | Medium-low. |
| [learnsyslab/gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) | MIT Gymnasium/PyBullet quadcopter envs with Betaflight/Crazyflie SITL examples. | Useful for API conventions and firmware/SITL bridge ideas. Not a high-throughput racing SOTA stack. | Medium for interfaces, low for policy performance. |
| [jjshoots/PyFlyt](https://github.com/jjshoots/PyFlyt) | MIT Gymnasium/PettingZoo UAV environments. | Useful for clean Gymnasium task design and simple baselines. Not the SOTA path for aggressive indoor nav. | Low-medium. |
| [FUEL](https://arxiv.org/abs/2010.11561) / [HKUST-Aerial-Robotics/FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL) | Open-source fast UAV exploration framework. | Maintains an incremental frontier information structure, plans global coverage, refines local viewpoints, then generates minimum-time trajectories. This is the clearest template for our frontier/viewpoint layer. | High conceptual value; port the data structures/metrics, not ROS. |
| [TARE](https://publications.ri.cmu.edu/tare-a-hierarchical-framework-for-efficiently-exploring-complex-3d-environments) | Paper and public exploration code/environment links. | Uses dense local planning plus sparse global/topological planning to explore complex 3D spaces efficiently. Reinforces that local detail and global coverage should be separate layers. | High conceptual value for our topological frontier manager. |
| [Active Neural SLAM](https://arxiv.org/abs/2004.05155) / [devendrachaplot/Neural-SLAM](https://github.com/devendrachaplot/Neural-SLAM) | PyTorch implementation and pretrained models. | Explicitly argues against expensive end-to-end exploration; uses learned mapping/global/local modules with an analytical planner in between. | High architecture value for map-state plus learned goal selection. |
| [FAR Planner](https://arxiv.org/abs/2110.09460) / [MichaelFYang/far_planner](https://github.com/MichaelFYang/far_planner) | Open-source visibility-graph planner. | Dynamically updates visibility graph edges as unknown environments are observed. Relevant to our repeated problem of stale/invalid local targets. | Medium-high: adapt the visibility/candidate idea in 2D. |
| [FASTER](https://arxiv.org/abs/2001.04420) / [mit-acl/faster](https://github.com/mit-acl/faster) | Open-source trajectory planner. | Handles high-speed navigation in unknown clutter by planning in known and unknown space while preserving a safe backup trajectory. Useful for future fast exploration and safety constraints. | Medium: later once we move past slow exploration. |
| [Learning High-Speed Flight in the Wild](https://arxiv.org/abs/2110.05113) / [uzh-rpg/agile_autonomy](https://github.com/uzh-rpg/agile_autonomy) | Open-source code for privileged-learning agile navigation. | Trains a deployable perception-to-trajectory policy from a privileged expert. This is the right learning pattern for short-horizon clutter flight, not a complete exploration planner. | High for teacher/student design and trajectory labels. |
| [Massive Parallel DRL for Active SLAM](https://arxiv.org/abs/2603.25834) | 2026 active-SLAM DRL paper, released as open source per abstract. | Useful reminder that RL exploration should be framed as active SLAM with scalable parallel training and continuous actions, not only local reward shaping. | Watch/inspect when code URL is easy to retrieve; conceptual fit with PufferLib throughput. |

## What Swift Actually Did

The important Swift details for our implementation are:

1. The deployed controller is not direct motor PWM from a neural net. The policy
   outputs mass-normalized collective thrust and body rates. A modeled Betaflight
   PID/ESC stack translates those commands to motors.
2. The actor observation includes platform state, the next gate observation, and
   the previous action. The value network can access privileged state during
   training.
3. The reward is dense and task-level: progress toward the next gate, a
   perception objective that keeps the gate in camera view, command smoothness,
   and a crash penalty that terminates the episode.
4. PPO training used 100 parallel agents and 100M interactions. Reported train
   time was about 50 minutes on an i9-12900K plus RTX 3090.
5. Sim-to-real did not rely on blind domain randomization alone. Swift collected
   a small amount of real flight data, identified residual observation and
   dynamics models, and fine-tuned the policy in simulation.
6. Their simulator models details that matter at high speed: propeller thrust,
   aerodynamic effects, battery/ESC behavior, and Betaflight controller behavior.

Our current custom env has only part of this recipe. We have fast PPO, simple
velocity-level control, synthetic local range observations, waypoint scaffolding,
and stress evaluation. We do not yet have Swift-like action interface, real
vehicle ID, perception residuals, gate/camera observation, or racing-speed
benchmarks.

## What This Means For Our Repo

Keep these constraints:

- Preserve fast PufferLib iteration. Do not switch the main loop to ROS,
  Flightmare, AirSim, or Isaac. External stacks are references or later
  validation tools, not replacements for this repo's learning loop.
- Keep actor observations real-drone-feasible. Privileged information belongs in
  the critic, diagnostics, reset curriculum, or teacher, not in the deployed
  actor.
- Avoid one-off reward tricks. Rewards should map to measurable objectives:
  progress, clearance/margin, visible target/gate, smooth commands, energy, and
  crash.
- Every new result needs a named train/eval split and a hard stress suite. A
  single easy layout at 99% is not evidence of indoor navigation.

## Recommended Benchmark Ladder

### 1. Robust Current Rung

Promote or reject QDR-113 with a 3-seed gate:

- final-target observation on;
- 1..5 route-length curriculum;
- fixed train/eval/stress commands;
- success, obstacle, timeout, and per-layout failure metrics.

This tells us whether the current best result is robust or just a good seed.

### 2. Swift-Lite State Racing

Build a small gate/race benchmark inside our C env:

- 3D gate centers/corners instead of only point targets;
- actor sees state plus next gate/corners in body frame plus previous action;
- action mode: collective thrust plus body rates;
- reward: gate progress, camera/gate alignment, smooth command, crash;
- value/diagnostic path can see full privileged state;
- eval reports lap completion, gate margin, crash type, lap time, speed.

This is the closest scientifically useful bridge from our current env to Swift.
It stays simple, but it tests the same control-and-observation class.

### 3. Indoor Nav Without Waypoint Actor Crutch

Once Swift-lite state racing works, reduce the navigation crutch:

- actor gets IMU/proprioception plus LiDAR/depth-like observations plus final
  goal, not planner waypoints;
- planner/waypoints may remain as privileged critic input, reset generator, or
  diagnostic label;
- compare actor-with-waypoint vs actor-without-waypoint on the same courses.

The pass condition should not be "learns one easy room". It should beat a
named stress suite with hidden target, multi-goal, delay, noise, and dynamics
shift.

### 4. Real Drone Model Hook

Before claiming sim-to-real relevance:

- define a drone config file: mass, inertia, arm length, thrust coefficient,
  drag coefficient, motor time constant, max thrust, battery/voltage model;
- add randomization knobs around measured parameters;
- add residual-dynamics hooks that can later consume real logs;
- decide the action interface we actually expect to fly: velocity, body-rate
  plus thrust, or motor thrust.

Swift and AirGym both point in the same direction: sim-to-real gets easier when
the simulator and deployment controller speak the same control language.

### 4.1 What Today's Negative Results Imply

The QDR-131..QDR-133 failures line up with the imitation-learning literature:
supervised cloning can look good on mixed/expert rollouts while failing under
the learner's own induced state distribution. DAgger's fix is to train on states
visited by the learner, and recent agile-flight work goes further: train a
privileged teacher, distill to a deployable student, then use RL fine-tuning so
the student can exceed the teacher's state distribution instead of only copying
it.

For this repo, the next research-grade rung is therefore:

- keep PufferLib as the simulator/trainer loop;
- preserve the deployable actor observation contract: proprioception, rays,
  direct target, wall distances, and previous action;
- train or initialize from the QDR-130 privileged student;
- fine-tune with the actual route reward, ideally with a privileged critic or
  at least a deployable-observation PPO fine-tune;
- only after that, move the action interface toward thrust/body-rate and add
  measured real-drone noise/dynamics residuals.

QDR-134 confirms the first half of this bridge: previous action is now a native
deployable observation and the converted QDR-130 student keeps its `~91%`
deterministic success. It also rules out naive PPO as the complete answer. The
student is sensitive to exploration noise, and plain PPO either preserves the
base policy briefly or drifts lower. Treat actor regularization or a privileged
critic as the next SOTA-aligned ingredient, not optional polish.

QDR-135 confirms the second half directionally: a privileged critic with a
deployable actor improves the direct actor by about one percentage point across
three eval seeds. This is still far from solved, but it breaks the previous
loop of imitation-only and symmetric-PPO failures. The SOTA-aligned next step is
not "more steps"; it is a better asymmetric update: cleaner recurrent state
handling, privileged value warmup, early stopping on actor drift, and possibly a
KL/BC schedule rather than a fixed coefficient.

### 5. Exploration Layer

Do not expect PPO over local rays to invent full indoor exploration by itself.
For exploration:

- keep a low-level collision-free flight policy;
- add a high-level frontier / information-gain / learned-goal module;
- reward observed coverage, not just physical visitation;
- use coverage curve/AUC, final coverage, collision rate, time-to-cover,
  revisit/stagnation, return-to-home, and map quality as metrics;
- eventually use depth/LiDAR/occupancy, not just synthetic planar rays.

ForesightNav is a reminder that high-level exploration is a separate problem
from low-level agile flight. We can still train end-to-end later, but the first
research-grade system should expose the decomposition clearly.

Useful exploration references for the custom PufferLib track:

- Frontier-based exploration: Yamauchi-style frontiers are still the clean
  baseline for unknown-space coverage. Use them as a privileged planner/teacher
  or diagnostic ceiling, not as deployable actor input.
- Active Neural SLAM: maintain a map and pose estimate, choose long-term
  exploration goals, and use a local policy/planner for motion. The lesson is
  to separate mapping, goal selection, and local control before recombining.
- RLeXplore and URLB: intrinsic rewards such as RND, ICM, RE3, NGU, and APT
  are useful method families once the benchmark has honest coverage metrics.
- Go-Explore: our native snapshot/restore support makes archive-and-return
  style exploration plausible for hard frontier states.
- FUEL/fast UAV exploration: for the eventual fast exploration track, the
  strong robotics baseline is frontier/viewpoint selection plus time-aware
  trajectory generation, not hidden-target PPO.

## Concrete Next Tests

Use these as candidate QDR rows, not all at once:

1. `QDR-114`: three-seed robustness gate for QDR-113 final-target plus 1..5
   curriculum. This is the current promotion check.
2. `QDR-115`: Swift-lite observation ablation on current route task: add one
   previous action only, with final-target context, and evaluate specifically
   under delay/dynamics stress. This is not a repeat of QDR-099 because QDR-099
   used two actions, no final-target context, and the old stress regime.
3. `QDR-116`: first thrust/body-rate action rung on hover and one-gate route.
   Goal is to verify controllability and throughput, not navigation.
4. `QDR-117`: gate/corner observation benchmark with progress plus view-alignment
   reward, velocity action first if needed, then thrust/body-rate.
5. `QDR-118`: no-waypoint actor ablation: final-goal plus local range only,
   waypoint available only to critic/diagnostics if the trainer supports it.

## Do Not Do Next

- Do not port Swift wholesale. The full source is not released.
- Do not move the whole project into ROS/Flightmare/AirSim/Isaac. That would
  slow the current diagnosis loop and work against the reason this repo exists.
- Do not make the actor depend on privileged planner breadcrumbs if the goal is
  robust indoor navigation. Use them only as scaffolding, critic input, or a
  baseline to beat.
- Do not treat visual success on one layout as progress. Use named stress cases
  and per-failure diagnostics.

## Source Links

- Swift paper: https://www.nature.com/articles/s41586-023-06419-4
- Swift PDF: https://rpg.ifi.uzh.ch/docs/Nature23_Kaufmann.pdf
- Swift supplementary data: https://zenodo.org/records/7955278
- UZH drone-racing research page: https://rpg.ifi.uzh.ch/research_drone_racing.html
- Sim2Real Drone Racing: https://github.com/uzh-rpg/sim2real_drone_racing
- Agilicious: https://github.com/uzh-rpg/agilicious
- Flightmare: https://github.com/uzh-rpg/flightmare
- Actor-Critic MPC: https://github.com/uzh-rpg/acmpc_public
- Contrastive Initial State Buffer: https://github.com/uzh-rpg/cl_initial_buffer
- Learned Inertial Model Odometry: https://github.com/uzh-rpg/learned_inertial_model_odometry
- Learning on the Fly: https://github.com/uzh-rpg/learning_on_the_fly
- Flightning: https://github.com/uzh-rpg/rpg_flightning
- ForesightNav: https://github.com/uzh-rpg/foresight-nav
- Aerial Gym Simulator: https://github.com/ntnu-arl/aerial_gym_simulator
- AirGym: https://github.com/emNavi/AirGym
- AirSim Drone Racing Lab: https://github.com/microsoft/AirSim-Drone-Racing-Lab
- gym-pybullet-drones: https://github.com/learnsyslab/gym-pybullet-drones
- PyFlyt: https://github.com/jjshoots/PyFlyt
