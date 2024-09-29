import gym
import env
import torch.nn as nn
import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose wheather to use transferred reward machine for Freeway task!")
    parser.add_argument('--gpu_id', type=int, default=1, help='')
    parser.add_argument('--seed', type=int, default=0, help='the random seed for training')
    args = parser.parse_args()
    gpu_id = args.gpu_id
    seed = args.seed
    if torch.cuda.is_available():
        print(torch.cuda.current_device())
        torch.cuda.set_device(gpu_id)
    batch_size = 256
    num_cpu = 4
    vec_env = make_atari_env("FreewayDeterministic-v4", n_envs=4, seed=seed)
    model_dir = "./Freeway_log/Freeway-PPO/model"
    logdir = "./Freeway_log/Freeway-PPO/"
    vec_env = VecFrameStack(vec_env, n_stack=4)
    test_env = make_atari_env("FreewayDeterministic-v4", n_envs=4, seed=seed)
    eval_callback = EvalCallback(eval_env=test_env, best_model_save_path=model_dir, log_path=logdir+"evaluate/",
                                 eval_freq=5000, n_eval_episodes=5, deterministic=True, render=False)
    model = PPO("CnnPolicy",
                vec_env,
                n_steps=128,
                batch_size=batch_size,
                n_epochs=20,
                learning_rate=2.5e-04,
                gamma=0.98,
                clip_range=0.1,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log="./Freeway_log/Freeway-PPO/train")
    print(">>>>>>>Training Start")
    model.learn(total_timesteps=1000000)
    model.save("Freeway_PPO/model")
    del model  # remove to demonstrate saving and loading
