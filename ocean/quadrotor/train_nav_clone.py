import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pufferlib.ocean.quadrotor import Quadrotor
from pufferlib.ocean.quadrotor.eval_hover import (
    GRAVITY,
    initial_policy_state,
    load_policy,
    pd_velocity_action,
    ray_pd_action,
    reset_done_policy_state,
)


def make_env(args):
    return Quadrotor(
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        action_repeat=args.action_repeat,
        max_rate=args.max_rate,
        max_vel=args.max_vel,
        max_z_vel=args.max_z_vel,
        spawn_radius=args.spawn_radius,
        spawn_height=args.spawn_height,
        init_vel_radius=args.init_vel_radius,
        init_omega_radius=args.init_omega_radius,
        room_size=args.room_size,
        ceiling_height=args.ceiling_height,
        num_rays=args.num_rays,
        ray_length=args.ray_length,
        hover_radius=args.hover_radius,
        obstacle_mode=args.obstacle_mode,
        route_goal_count=args.route_goal_count,
        task_mode=args.task_mode,
        action_mode="velocity",
        report_interval=args.report_interval,
        render_mode=None,
        seed=args.seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Clone the indoor velocity controller")
    parser.add_argument("--output", required=True)
    parser.add_argument("--load-model-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--print-interval", type=int, default=250)
    parser.add_argument("--reset-interval", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--max-rate", type=float, default=1.0)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--max-z-vel", type=float, default=0.75)
    parser.add_argument("--spawn-radius", type=float, default=1.5)
    parser.add_argument("--spawn-height", type=float, default=1.5)
    parser.add_argument("--init-vel-radius", type=float, default=0.0)
    parser.add_argument("--init-omega-radius", type=float, default=0.0)
    parser.add_argument("--room-size", type=float, default=2.0)
    parser.add_argument("--ceiling-height", type=float, default=3.0)
    parser.add_argument("--num-rays", type=int, default=16)
    parser.add_argument("--ray-length", type=float, default=3.0)
    parser.add_argument("--hover-radius", type=float, default=0.25)
    parser.add_argument("--obstacle-mode", default="none")
    parser.add_argument("--route-goal-count", type=int, default=3)
    parser.add_argument("--task-mode", default="indoor_nav")
    parser.add_argument("--report-interval", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--action-std", type=float, default=0.1)
    parser.add_argument("--teacher", choices=("pd", "raypd"), default="pd")
    parser.add_argument("--rollout-policy", choices=("teacher", "student", "mix"), default="teacher")
    parser.add_argument("--student-rollout-prob", type=float, default=0.0)
    parser.add_argument("--student-warmup-updates", type=int, default=0)
    parser.add_argument("--rnn-name", default=None)
    parser.add_argument("--rnn-input-size", type=int, default=128)
    parser.add_argument("--rnn-hidden-size", type=int, default=128)
    parser.add_argument("--pd-kp-pos", type=float, default=2.0)
    parser.add_argument("--pd-max-vel", type=float, default=1.0)
    parser.add_argument("--pd-kp-vel", type=float, default=6.0)
    parser.add_argument("--pd-max-tilt-acc", type=float, default=GRAVITY * 0.364)
    parser.add_argument("--pd-max-z-acc", type=float, default=GRAVITY * 0.5)
    parser.add_argument("--pd-kp-att", type=float, default=20.0)
    parser.add_argument("--pd-kd-att", type=float, default=5.0)
    parser.add_argument("--pd-avoid-radius", type=float, default=1.0)
    parser.add_argument("--pd-avoid-gain", type=float, default=1.0)
    parser.set_defaults(action_mode="velocity")
    return parser.parse_args()


def detach_policy_state(state):
    for key, value in list(state.items()):
        if torch.is_tensor(value):
            state[key] = value.detach()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    policy = load_policy(env, args.load_model_path, args).to(args.device)
    policy.train()
    state = initial_policy_state(args)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    last_loss = None

    for update in range(1, args.updates + 1):
        if args.teacher == "raypd":
            teacher_actions = ray_pd_action(obs, env, args)
        else:
            teacher_actions = pd_velocity_action(obs, env, args)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
        teacher_tensor = torch.as_tensor(teacher_actions, dtype=torch.float32, device=args.device)

        logits, _ = policy.forward_eval(obs_tensor, state)
        if not isinstance(logits, torch.distributions.Normal):
            raise TypeError("Controller cloning expects a continuous-action Normal policy")

        loss = F.mse_loss(logits.mean, teacher_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        step_actions = teacher_actions
        if update > args.student_warmup_updates and args.rollout_policy != "teacher":
            student_actions = logits.mean.detach().cpu().numpy()
            if args.rollout_policy == "student":
                step_actions = student_actions
            else:
                prob = np.clip(args.student_rollout_prob, 0.0, 1.0)
                use_student = np.random.random(env.num_agents) < prob
                step_actions = teacher_actions.copy()
                step_actions[use_student] = student_actions[use_student]

        obs, _, terminals, truncations, _ = env.step(step_actions)
        reset_done_policy_state(state, np.logical_or(terminals, truncations))
        detach_policy_state(state)
        if update % args.reset_interval == 0:
            obs, _ = env.reset(seed=args.seed + update)
            state = initial_policy_state(args)

        if update % args.print_interval == 0 or update == args.updates:
            print(f"update={update} loss={last_loss:.8f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output)
    env.close()
    print(f"wrote {output} final_loss={last_loss:.8f}")


if __name__ == "__main__":
    main()
