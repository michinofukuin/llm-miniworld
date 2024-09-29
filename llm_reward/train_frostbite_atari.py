import gym
import env
import torch.nn as nn
import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.set_device(6)

if __name__ == "__main__":
    batch_size = 256
    num_cpu = 4
    vec_env = make_atari_env("FrostbiteDeterministic-v4", n_envs=4, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model = PPO("MultiInputPolicy",
                vec_env,
                n_steps=128,
                batch_size=batch_size,
                n_epochs=20,
                learning_rate=2.5e-04,
                gamma=0.98,
                clip_range=0.1,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log="Frostbite_PPO/")
    print(">>>>>>>Training Start")
    model.learn(total_timesteps=10000000)
    model.save("Frostbite_PPO/model")
    del model  # remove to demonstrate saving and loading
