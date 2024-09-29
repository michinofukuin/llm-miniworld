import gym
import env
import torch.nn as nn
import torch
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
import argparse
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose wheather to use transferred reward machine for half-cheetah task!")
    parser.add_argument('--use_reward_machine', action='store_true', help='Use reward machine for half-cheetah task!')
    parser.add_argument('--use_transfered_reward', action='store_true',
                        help='Use reward machine for half-cheetah task!')
    parser.add_argument('--seed', type=int, default=0, help='the random seed for training')
    parser.add_argument('--model_dir', type=str, default="./mujuco_log/Half-Cheetah-RM_PPO/model", help='model_dir')
    parser.add_argument('--log_dir', type=str, default="./mujuco_log/Half-Cheetah_PPO/", help='log_dir')
    parser.add_argument('--gpu_id', type=int, default=1, help='')
    args = parser.parse_args()
    gpu_id = args.gpu_id
    if torch.cuda.is_available():
        print(torch.cuda.current_device())
        torch.cuda.set_device(gpu_id)
    is_use_reward_machine = args.use_reward_machine
    print(is_use_reward_machine)
    is_use_transfered_reward = args.use_transfered_reward
    print(is_use_transfered_reward)
    log_dir = args.log_dir
    seed = args.seed
    if is_use_reward_machine:
        if is_use_transfered_reward:
            method = "RM_and_Transfered_Reward"
            # train_env = make_vec_env("Half-Cheetah-RM-v0", n_envs=4, seed=seed)
            train_env = gym.make("Half-Cheetah-RM-v0")
        else:
            method = "RM_and_Empty_Reward"
            train_env = make_vec_env("Half-Cheetah-RM-v1")
            # train_env = gym.make("Half-Cheetah-RM-v1")
        test_env = make_vec_env("Half-Cheetah-RM-Evaluate-v0")
        # test_env = gym.make("Half-Cheetah-RM-Evaluate-v0")
    else:
        if is_use_transfered_reward:
            method = "Only_Transfered_Reward"
            train_env = make_vec_env("Half-Cheetah-v0")
            # train_env = gym.make("Half-Cheetah-v0")
        else:
            method = "Original"
            train_env = make_vec_env("Half-Cheetah-v1")
            # train_env = gym.make("Half-Cheetah-v1")
        test_env = make_vec_env("Half-Cheetah-Evaluate-v0")
        # test_env = gym.make("Half-Cheetah-Evaluate-v0")
    model_dir = args.model_dir + method + "/"
    log_dir = args.log_dir + method + "/"
    print(method)
    batch_size = 128
    num_cpu = 16
    eval_callback = EvalCallback(eval_env=test_env, best_model_save_path=model_dir, log_path=log_dir+"evaluate/",
                                 eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False)
    model = PPO("MlpPolicy",
                train_env,
                policy_kwargs=dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU, net_arch=[dict(pi=[256,256], vf=[256, 256])]),
                n_steps=512,
                batch_size=batch_size,
                n_epochs=20,
                learning_rate=3e-5,
                gamma=0.99,
                clip_range=0.4,
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
    model.learn(total_timesteps=int(2e6), callback=eval_callback)
