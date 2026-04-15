import torch
import numpy as np
import pufferlib
import pufferlib.vector
import pufferlib.models
import pufferlib.pufferl

from pufferlib.ocean.coinflip.coinflip import CoinFlip

def env_creator(num_envs=1, **kwargs):
    return CoinFlip(num_envs=num_envs, **kwargs)

if __name__ == '__main__':
    vecenv = pufferlib.vector.make(
        env_creator,
        env_kwargs=dict(num_envs=128),
        num_envs=1,
        backend=pufferlib.vector.Serial,
    )

    policy = pufferlib.models.Default(vecenv)

    config = dict(
        seed=42,
        torch_deterministic=True,
        cpu_offload=False,
        device='cpu',
        optimizer='adam',
        precision='float32',
        total_timesteps=500_000,
        learning_rate=0.001,
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
        batch_size=4096,
        minibatch_size=512,
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
        env='coinflip',
    )

    trainer = pufferlib.pufferl.PuffeRL(config, vecenv, policy)

    while trainer.global_step < config['total_timesteps']:
        stats = trainer.evaluate()
        logs = trainer.train()
        if logs and 'environment/score' in logs:
            score = logs['environment/score']
            print(f"Step {trainer.global_step:>8} | Score: {score:.3f}")

    trainer.close()
    print("Done!")
