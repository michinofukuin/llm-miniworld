import gym
import numpy
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

class FrozenLakeEnvRM(gym.Wrapper):
    def __init__(self):
        super().__init__(FrozenLakeEnv(desc=None, map_name="4x4", is_slippery=True))

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        return next_obs, original_reward, env_done, info

    def get_events(self):
        event = ''