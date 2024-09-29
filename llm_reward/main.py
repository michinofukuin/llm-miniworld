import gym
import env
import torch.nn as nn
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.set_device(6)

if __name__ == "__main__":
    batch_size = 64
    num_cpu = 4
    # vec_env = make_vec_env("Half-Cheetah-RM-v0", n_envs=4, seed=0)
    vec_env = make_vec_env("HalfCheetah-v3", n_envs=4, seed=0)
    model = PPO("MlpPolicy",
                vec_env,
                policy_kwargs=dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU, net_arch=[dict(pi=[256,256], vf=[256, 256])]),
                n_steps=512,
                batch_size=batch_size,
                n_epochs=20,
                learning_rate=2.0633e-05,
                gamma=0.98,
                clip_range=0.1,
                ent_coef=0.000401762,
                gae_lambda=0.92,
                max_grad_norm=0.8,
                verbose=1,
                tensorboard_log="Half-Cheetah_PPO/")
    print(">>>>>>>Training Start")
    # model.learn(total_timesteps=int(10e5))
    # model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100_0000)
    model.save("Half-Cheetah-RM_PPO/model")
