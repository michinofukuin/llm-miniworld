import miniworld
import gymnasium as gym
import sys
import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import PPO, DQN
import torch
import random
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn
# Initialize your custom environment
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def make_eval_env():
    return gym.make("MiniWorld-Sn-v0", render_mode="human")
env = gym.make("MiniWorld-Sign-v0", render_mode="human")
test_env = gym.make("MiniWorld-Sign-v0", render_mode="human")
log_dir = "logs_llm/big_sign/"
model_dir = "logs_llm/eval_model/big_sign/"
# Create the PPO model
eval_callback = EvalCallback(eval_env=test_env, best_model_save_path=model_dir, log_path=log_dir+"evaluate/",
                                 eval_freq=5000, n_eval_episodes=2, deterministic=False, render=False)
model = PPO("MultiInputPolicy",
            env,
            device='cuda:9',
            policy_kwargs=dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU, net_arch=dict(pi=[256,256], vf=[256, 256])),
            n_steps=512,
            batch_size=128,
            n_epochs=20,
            learning_rate=2e-5,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.0,
            gae_lambda=0.9,
            max_grad_norm=0.5,
            verbose=1,
            seed=seed,
            tensorboard_log=log_dir+"train/")
# model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir+"train/", learning_rate=2e-5, batch_size=128, device="cuda:1")
# Train the model
# model = PPO.load("/mnt/data/chenhaosheng/llm-miniworld/source/logs_llm/5_5_3/new/2e-5-0.2")
model.learn(total_timesteps=200000, callback=eval_callback)
# model.learn(total_timesteps=200000)
# Save the model
model.save("logs_llm/big_sign/2e-5-0.2")
# # Evaluate the model
# obs = env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)


# env.plot_agent_path()  # 显示agent的运动路径
# env.close()


# xvfb-run -a python /home/ubuntu/llm-miniworld/s.py