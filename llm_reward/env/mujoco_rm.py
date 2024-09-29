import gym
import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


class HalfCheetahEnvRM(gym.Wrapper):
    def __init__(self):
        super().__init__(HalfCheetahEnv(exclude_current_positions_from_observation=False))

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        return next_obs, original_reward, env_done, info

    def get_events(self):
        events = ''
        if self.info['x_position'] < -10:
            events+='b'
        if self.info['x_position'] > 10:
            events+='a'
        if self.info['x_position'] < -2:
            events+='d'
        if self.info['x_position'] > 2:
            events+='c'
        if self.info['x_position'] > 4:
            events+='e'
        if self.info['x_position'] > 6:
            events+='f'
        if self.info['x_position'] > 8:
            events+='g'
        return events


if __name__ == "__main__":
    test_env = HalfCheetahEnvRM()
    o = test_env.reset()
    n_o, r, done, info = test_env.step(test_env.action_space.sample())
    print(test_env.info)
