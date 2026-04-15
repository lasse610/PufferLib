"""Watch a trained snake agent play. Press ESC to quit."""

import torch
import numpy as np
import pufferlib
import pufferlib.vector
import pufferlib.models

from pufferlib.ocean.snake_game.snake_game import SnakeGame

CHECKPOINT = "experiments/snake_game_177495990021/model_snake_game_000407.pt"
WIDTH = 10
HEIGHT = 10

if __name__ == '__main__':
    # Create a single env for visualization
    env = SnakeGame(num_envs=1, width=WIDTH, height=HEIGHT, start_length=1, vision=5)

    # Build the same policy architecture as training
    vecenv = pufferlib.vector.make(
        lambda num_envs=1, **kw: SnakeGame(num_envs=num_envs, width=WIDTH, height=HEIGHT, start_length=1, vision=5, **kw),
        env_kwargs=dict(num_envs=1),
        num_envs=1,
        backend=pufferlib.vector.Serial,
    )
    policy = pufferlib.models.Default(vecenv, hidden_size=256)
    policy.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True))
    policy.eval()
    vecenv.close()

    # Play loop
    obs, _ = env.reset()
    env.render()

    while True:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
            logits, value = policy.forward_eval(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).numpy()

        obs, rewards, terminals, truncations, info = env.step(action)
        env.render()

        if terminals[0]:
            print(f"Died! Length: {info}")
            obs, _ = env.reset()
