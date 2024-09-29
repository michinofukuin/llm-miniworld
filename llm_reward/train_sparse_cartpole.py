import gym
import env
import torch.nn as nn
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose wheather to use transferred reward machine for half-cheetah task!")
    parser.add_argument('--use_reward_machine', action='store_true', help='Use reward machine for half-cheetah task!')
    parser.add_argument('--use_transfered_reward', action='store_true',
                        help='Use reward machine for half-cheetah task!')
    parser.add_argument('--seed', type=int, default=0, help='the random seed for training')
    parser.add_argument('--model_dir', type=str, default="Cartpole_PPO/model", help='model_dir')
    parser.add_argument('--log_dir', type=str, default="Cartpole_PPO/", help='log_dir')
    parser.add_argument('--gpu_id', type=int, default=6, help='')
    args = parser.parse_args()
    gpu_id = args.gpu_id
    seed = args.seed
    if torch.cuda.is_available():
        print(torch.cuda.current_device())
        torch.cuda.set_device(gpu_id)
    is_use_reward_machine = args.use_reward_machine
    print(is_use_reward_machine)
    is_use_transfered_reward = args.use_transfered_reward
    print(is_use_transfered_reward)
    model_dir = args.model_dir
    log_dir = args.log_dir
    batch_size = 64
    num_cpu = 4
    if is_use_reward_machine:
        if is_use_transfered_reward:
            vec_env = make_vec_env("Cartpole-RM-v2", n_envs=4, seed=seed)
            method = "RM_and_Reward"
        else:
            vec_env = make_vec_env("Cartpole-RM-v1", n_envs=4, seed=seed)
            method = "RM"
        test_env = make_vec_env("Cartpole-RM-v1", n_envs=4, seed=seed)
    else:
        if is_use_transfered_reward:
            vec_env = make_vec_env("Cartpole-NoRM-v2", n_envs=4, seed=seed)
            method = "Reward"
        else:
            vec_env = make_vec_env("Cartpole-NoRM-v1", n_envs=4, seed=seed)
            method = "Original"
        test_env = make_vec_env("Cartpole-NoRM-v1", n_envs=4, seed=seed)
    log_dir = log_dir + method + "/"
    model_dir = model_dir + method + "/"
    eval_callback = EvalCallback(eval_env=test_env, best_model_save_path=model_dir, log_path=log_dir+"evaluate/",
                                 eval_freq=1000, n_eval_episodes=5, deterministic=True, render=False)
    model = PPO("MlpPolicy",
                vec_env,
                policy_kwargs=dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU, net_arch=[dict(pi=[256,256], vf=[256, 256])]),
                n_steps=512,
                batch_size=batch_size,
                n_epochs=20,
                learning_rate=1e-4,
                gamma=0.99,
                clip_range=0.1,
                ent_coef=0.000401762,
                gae_lambda=0.92,
                max_grad_norm=0.8,
                verbose=1,
                tensorboard_log=log_dir + "train/")
    print(">>>>>>>Training Start")
    model.learn(total_timesteps=500000, callback=eval_callback)
    del model  # remove to demonstrate saving and loading
