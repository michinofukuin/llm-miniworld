import gym
import sys
sys.path.append("/home/ubuntu/llm_reward/reward_machines")

import numpy as np
from atariari.benchmark.wrapper import AtariARIWrapper
import env_with_rm
from env_with_rm import RewardMachineEnv, AtariRewardMachineEnv

class FreewayEnvRM(gym.Wrapper):
    def __init__(self, failure_reward=0):
        super().__init__(AtariARIWrapper(gym.make("FreewayDeterministic-v4")))
        # self.env.env.env.mode = 3
        self.last_score = 0
        self.info = {}
        self.last_plaer_y = 0
        self.failure_reward = failure_reward

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        # if self.failure_reward > 0:
        #     if info['labels']['player_y'] < self.last_plaer_y - 100 and original_reward == 0:
        #         original_reward -= self.failure_reward
        if env_done:
            print(info, env_done)
        return next_obs, original_reward, env_done, info

    def get_events(self):
        events = ''
        if self.info['labels']['player_y'] >= 6:
            events+='a'
        if self.info['labels']['player_y'] >= 22:
            events+='b'
        if self.info['labels']['player_y'] >= 38:
            events+='c'
        if self.info['labels']['player_y'] >= 54:
            events+='d'
        if self.info['labels']['player_y'] >= 70:
            events+='e'
        if self.info['labels']['player_y'] >= 86:
            events+='f'
        if self.info['labels']['player_y'] >= 102:
            events+='g'
        if self.info['labels']['player_y'] >= 118:
            events+='h'
        if self.info['labels']['player_y'] >= 134:
            events+='i'
        if self.info['labels']['player_y'] >= 150:
            events+='j'
        if self.info['labels']['player_y'] >= 166:
            events+='k'
        if self.info['labels']['score'] > self.last_score:
            events += 'bcdefghijkl'
        # print(events)
        self.last_plaer_y = self.info['labels']['player_y']
        return events


class FreewayRMEnvEmptyRM(AtariRewardMachineEnv): # RM
    def __init__(self):
        env = FreewayEnvRM()
        rm_files = ["./env/reward_machines/freeway.txt"]
        super().__init__(env, rm_files, use_reward_machine=True)

class FreewayRMEnvRM(AtariRewardMachineEnv): # RM
    def __init__(self):
        env = FreewayEnvRM(failure_reward=1000)
        rm_files = ["./env/reward_machines/transferred_freeway.txt"]
        super().__init__(env, rm_files, use_reward_machine=True, add_rs=True)

class FreewayRMEnvEvaluate(AtariRewardMachineEnv): # Evaluate
    def __init__(self):
        env = FreewayEnvRM()
        rm_files = ["./env/reward_machines/freeway.txt"]
        super().__init__(env, rm_files, use_reward_machine=False, origianl_task=True, is_evaluate_env=True)


test_env = FreewayRMEnvRM()
obs = test_env.reset()
next_obs, reward, done, info = test_env.step(1)
print(obs['features'].shape)