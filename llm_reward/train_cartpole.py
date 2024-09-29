import gym
import env
import torch.nn as nn
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.set_device(6)

if __name__ == "__main__":
    batch_size = 256
    num_cpu = 4
    vec_env = gym.make("Cartpole-RM-v0")
    vec_env.env.set_original_task()

    eval_callback = EvalCallback(eval_env=vec_env, best_model_save_path="Cartpole_Original/model", log_path="Cartpole_Original/",
                                 eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False)
    model = DQN("MlpPolicy", vec_env, verbose=1, tensorboard_log="Cartpole_Original/", learning_rate=1e-4, batch_size=batch_size)
    print(">>>>>>>Training Start")
    model.learn(total_timesteps=2000000, callback=eval_callback)
    del model  # remove to demonstrate saving and loading

