import gym
import env
import torch.nn as nn
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from atariari.benchmark.wrapper import AtariARIWrapper
import math

# info['labels']['player_y']
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.set_device(2)

if __name__ == "__main__":
    # env = AtariARIWrapper(gym.make("Frostbite-ramDeterministic-v4"))
    # obs = env.reset()
    # for i in range(100):
    #     next_obs, reward, done, info = env.step(env.action_space.sample())
    #     print(info)
    vec_env = make_vec_env("Freeway-RM-v0", n_envs=1, seed=0)
    # env = VecFrameStack(vec_env, n_stack=4)
    # vec_env = make_atari_env("Freeway-RM-Evaluate-v0", n_envs=1, seed=0)
    # vec_env = VecFrameStack(vec_env, n_stack=4)
    model = PPO.load("Freeway-PPO/modelRM_and_Transfered_Reward/best_model")
    done = False
    obs = vec_env.reset()
    long_term_reward = []
    long_term_player_loc = []
    long_term_player_score = []
    labels = []
    u_ids = []
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print(reward[0])
        # print(info[0]['events'])
        long_term_reward.append(reward)
        # print(info[0]['labels']['player_y'])
        # old_info = info
        # labels.append(info[0]['labels']['player_y'])
        # u_ids.append(info[0]['rm_state'])
        # # print(info[0]['labels']['score'])
        # # print("player_loc>>>", info[0]['labels']['player_y'])
        # long_term_player_loc.append(info[0]['labels']['player_y'])
        # long_term_player_score.append(info[0]['labels']['score'])
    # state_rm_state_map = {}
    # labels_set = set(labels)
    # for i in labels_set:
    #     state_rm_state_map[i] = []
    # for index in range(len(labels)):
    #     state_rm_state_map[labels[index]].append(u_ids[index])
    # for i in labels_set:
    #     state_rm_state_map[i] = set(state_rm_state_map[i])
    print(sum(long_term_reward))
    print(info[0]['labels']['score'])
