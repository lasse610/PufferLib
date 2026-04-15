import torch
import numpy as np
import pufferlib
import pufferlib.vector
import pufferlib.models
import pufferlib.pufferl

from pufferlib.ocean.snake_game.snake_game import SnakeGame

def env_creator(num_envs=1, **kwargs):
    return SnakeGame(num_envs=num_envs, **kwargs)

if __name__ == '__main__':
    vecenv = pufferlib.vector.make(
        env_creator,
        env_kwargs=dict(num_envs=128, width=10, height=10, vision=5),
        num_envs=6,               # 6 processes, one per core
        num_workers=6,
        batch_size=6,             # collect from all workers each batch
        backend=pufferlib.vector.Multiprocessing,
    )

    policy = pufferlib.models.Default(vecenv, hidden_size=256)

    config = dict(
        seed=42,
        torch_deterministic=True,
        cpu_offload=False,
        device='cpu',
        optimizer='adam',
        precision='float32',
        total_timesteps=10_000_000,
        learning_rate=0.0003,
        anneal_lr=True,
        min_lr_ratio=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        update_epochs=4,
        clip_coef=0.2,
        vf_coef=0.5,
        vf_clip_coef=0.2,
        max_grad_norm=0.5,
        ent_coef=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        batch_size=768 * 32,  # total_agents * bptt_horizon
        minibatch_size=2048,
        max_minibatch_size=32768,
        bptt_horizon=32,
        compile=False,
        compile_mode='default',
        compile_fullgraph=True,
        vtrace_rho_clip=1.0,
        vtrace_c_clip=1.0,
        prio_alpha=0.0,
        prio_beta0=0.0,
        data_dir='experiments',
        checkpoint_interval=200,
        use_rnn=False,
        env='snake_game',
    )

    trainer = pufferlib.pufferl.PuffeRL(config, vecenv, policy)

    while trainer.global_step < config['total_timesteps']:
        stats = trainer.evaluate()
        logs = trainer.train()
        if logs:
            score = logs.get('environment/snake_length', '?')
            food = logs.get('environment/food_eaten', '?')
            ep_len = logs.get('environment/episode_length', '?')
            print(f"Step {trainer.global_step:>8} | Length: {score} | Food: {food} | Ep len: {ep_len}")

    trainer.close()
    print("Done!")
