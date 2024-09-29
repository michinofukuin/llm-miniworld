import gymnasium as gym
import env
import torch.nn as nn
import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
import math
from copy import deepcopy

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.set_device(6)

if __name__ == "__main__":
    env = gym.make("Sign-RM-v0")
    print('--------------------')
    model = PPO.load("/home/ubuntu/llm_miniworld/source/res/2e-5-0.2",env=env)
    reward_machine_states_set = deepcopy(env.reward_machines[0].U)
    reward_machine_states_set.append(-1)
    reward_machine_values = {i : [] for i in reward_machine_states_set}
    long_term_reward = []
    last_value = 0
    for i in range(5):
        done = False
        obs = env.reset()
        value = model.predict_values(obs).data.to('cpu').numpy()[0]
        reward_machine_state = env.current_u_id
        while not done:
            print(obs[0], reward_machine_state, value)
            reward_machine_values[reward_machine_state].append(value)
            action, _state = model.predict(obs, deterministic=False)
            value = model.predict_values(obs).data.to('cpu').numpy()[0]
            obs, reward, done, info = env.step(action)
            if done:
                last_value = model.predict_values(obs).data.to('cpu').numpy()[0]
                print(last_value)
                print('dddddddddddddddddddddddddddddddddddddddddddddddd')
            # print(obs)
            # print(reward_machine_state, info['cart_position'], reward)
            reward_machine_state = env.current_u_id
            # print(reward_machine_state)
            # long_term_reward.append(reward)
    reward_machine_value = {}
    for i in reward_machine_states_set:
        if i == -1:
            reward_machine_value[i] = 0
        else:
            reward_machine_value[i] = np.mean(reward_machine_values[i])
    # print(np.sum(long_term_reward), len(long_term_reward))
    reward_machine_delta_r = {i: {} for i in env.reward_machines[0].U}
    # print(reward_machine_value)
    reward_machine_discount = 0.999
    for i in env.reward_machines[0].U:
        for key in env.reward_machines[0].delta_r[i]:
            reward_machine_delta_r[i][key] = (reward_machine_discount * reward_machine_value[key] - reward_machine_value[i])
            if key == -1:
                reward_machine_delta_r[i][key] = last_value
    print(reward_machine_delta_r)