import miniworld
import gymnasium as gym
import gymnasium
import sys
sys.modules["gym"] = gymnasium
import env
import torch.nn as nn
import torch
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
import argparse
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import NatureCNN
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces

class CustomDictCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super(CustomDictCNN, self).__init__(observation_space, features_dim)
        
        # 提取各个子空间
        obs_space = observation_space.spaces['obs']
        goal_space = observation_space.spaces['goal']
        rm_state_space = observation_space.spaces.get('rm-state', None)
        
        # 定义 CNN 层用于 'obs'
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 32, kernel_size=3, stride=2, padding=1),  # [32, 40, 30]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [64, 20, 15]
            nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [128, 10, 8]
            # nn.ReLU(),
            nn.Flatten()  # [128 * 10 * 8 = 10240]
        )
        
        # 计算 CNN 输出的维度
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, *obs_space.shape)).shape[1]
        
        # 定义全连接层用于 'goal'
        self.fc_goal = nn.Sequential(
            nn.Linear(goal_space.shape[0], 32),
            nn.ReLU()
        )
        
        # 如果有 'rm-state'，定义全连接层
        if rm_state_space:
            self.fc_rm_state = nn.Sequential(
                nn.Linear(rm_state_space.shape[0], 32),
                nn.ReLU()
            )
        
        # 计算合并后的特征维度
        combined_size = n_flatten + 32
        if rm_state_space:
            combined_size += 32
        
        # 定义最终的全连接层
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # 处理 'obs' 部分
        obs_features = self.cnn(observations['obs'])
        
        # 处理 'goal' 部分
        goal_features = self.fc_goal(observations['goal'])
        
        # 处理 'rm-state' 部分，如果存在
        if 'rm-state' in observations:
            rm_state_features = self.fc_rm_state(observations['rm-state'])
        else:
            rm_state_features = th.zeros((observations['goal'].shape[0], 32)).to(observations['goal'].device)
        
        if 'rm-state' in observations:
            combined_features = th.cat([obs_features, goal_features, rm_state_features], dim=1)
        else:
            combined_features = th.cat([obs_features, goal_features], dim=1)
        
        return self.fc_combined(combined_features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose wheather to use transferred reward machine for half-cheetah task!")
    parser.add_argument('--use_reward_machine', action='store_true', help='Use reward machine for half-cheetah task!')
    parser.add_argument('--use_transfered_reward', action='store_true',
                        help='Use reward machine for half-cheetah task!')
    parser.add_argument('--seed', type=int, default=0, help='the random seed for training')
    parser.add_argument('--model_dir', type=str, default="./llm_sign_1/model", help='model_dir')
    parser.add_argument('--log_dir', type=str, default="./llm_sign_1/Sign_PPO_log/", help='log_dir')
    parser.add_argument('--gpu_id', type=int, default=0, help='')
    args = parser.parse_args()
    gpu_id = args.gpu_id
    if torch.cuda.is_available():
        print(torch.cuda.current_device())
        torch.cuda.set_device(gpu_id)
    is_use_reward_machine = False
    print(is_use_reward_machine)
    is_use_transfered_reward = True
    print(is_use_transfered_reward)
    log_dir = args.log_dir
    seed = args.seed
    if is_use_reward_machine:
        if is_use_transfered_reward:
            method = "RM_and_Transfered_Reward"
            train_env = gym.make("Sign-RM-v1", render_mode="human")
            # train_env = gym.make("Sign-RM-v1", render_mode="human")
        else:
            method = "RM_and_Empty_Reward"
            train_env = gym.make("Sign-RM-v0")
            # train_env = gym.make("Half-Cheetah-RM-v1")
        test_env = gym.make("Sign-RM-Evaluate-v0", render_mode="human")
        # test_env = gym.make("Half-Cheetah-RM-Evaluate-v0")
    else:
        if is_use_transfered_reward:
            method = "Only_Transfered_Reward"
            train_env = gym.make("Sign-v1")
            # train_env = gym.make("Half-Cheetah-v0")
        else:
            method = "Original"
            train_env = gym.make("Sign-v0")
            # train_env = gym.make("Half-Cheetah-v1")
        test_env = make_vec_env("Sign-Evaluate-v0")
        # test_env = gym.make("Half-Cheetah-Evaluate-v0")
    model_dir = args.model_dir + method + "/"
    log_dir = args.log_dir + method + "/"
    print(method)
    batch_size = 128
    num_cpu = 16
    eval_callback = EvalCallback(eval_env=test_env, best_model_save_path=model_dir, log_path=log_dir+"evaluate/",
                                 eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False)
    policy_kwarg = dict(
                    features_extractor_class=CustomDictCNN,)
    model = PPO("MultiInputPolicy",
                train_env,
                policy_kwargs=policy_kwarg,
                n_steps=512,
                device='cuda:1',
                batch_size=batch_size,
                n_epochs=20,
                learning_rate=2e-5,
                gamma=0.99,
                clip_range=0.2,
                ent_coef=0.0,
                gae_lambda=0.9,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=log_dir+"train/")
    # model = DDPG("MlpPolicy",
    #              train_env,
    #              policy_kwargs=dict(net_arch=[256, 256]),
    #              learning_rate=5e-4,
    #              gamma=0.99,
    #              tau=0.01,
    #              batch_size=batch_size,
    #              tensorboard_log=log_dir + "train/"
    #              )
    print(">>>>>>>Training Start")
    model.learn(total_timesteps=int(1e6), callback=eval_callback)

#  xvfb-run -a python /home/ubuntu/llm_reward/test.py 

# pip install stable-baselines3==1.3.0
