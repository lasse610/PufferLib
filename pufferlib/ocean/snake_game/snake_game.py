import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.snake_game import binding

class SnakeGame(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=25, height=25, start_length=0, vision=5,
            render_mode=None, log_interval=128, buf=None, seed=0):
        window = 2 * vision + 1
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(window * window * 4,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions,
            self.rewards, self.terminals, self.truncations, num_envs, seed,
            width=width, height=height, start_length=start_length, vision=vision)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)
