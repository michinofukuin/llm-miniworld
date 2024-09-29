import gym
import env
import torch.nn as nn
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
import argparse
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from gym import ObservationWrapper
from gym.spaces import Box


class FlattenObservationWrapper(ObservationWrapper):
    def __init__(self, env, n_stack):
        super().__init__(env)
        self.n_stack = n_stack
        self.observation_space = spaces.Dict({
            'features': Box(low=0, high=255, shape=(3 * n_stack, 210, 160), dtype=np.float32),
            'u-id': Box(low=0, high=1, shape=(env.num_rm_states,), dtype=np.float32)
        })

    def observation(self, observation):
        image = observation['features']
        rm_state = observation['u-id']
        stacked_image = np.concatenate([image] * self.n_stack, axis=2)
        return {'features': np.moveaxis(stacked_image, 2, 0), 'u-id': rm_state}


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_rm_states, n_stack, **kwargs):
        self.image_space = spaces.Box(low=0, high=255, shape=(3 * n_stack, 210, 160), dtype=np.uint8)
        self.n_stack = n_stack
        image_extractor = NatureCNN(self.image_space, **kwargs)
        rm_state_extractor = nn.Linear(num_rm_states, 64)

        # Calculate the features_dim before calling the super().__init__() method
        dummy_input = torch.zeros((1, 3 * n_stack, 210, 160))
        dummy_output = image_extractor(dummy_input)
        features_dim = dummy_output.shape[1] + 64

        super().__init__(observation_space, features_dim=features_dim)

        self.image_extractor = image_extractor
        self.rm_state_extractor = rm_state_extractor

    def forward(self, observations):
        image_features = self.image_extractor(observations['features'].view(-1, 3 * self.n_stack, 210, 160))
        rm_state_features = self.rm_state_extractor(observations['u-id'])
        return torch.cat((image_features, rm_state_features), dim=1)





class MultiInputPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        num_rm_states = kwargs.pop("num_rm_states", None)
        n_stack = kwargs.pop("n_stack", None)
        features_extractor_kwargs = dict(num_rm_states=num_rm_states, n_stack=n_stack)
        super().__init__(*args, features_extractor_class=CustomFeatureExtractor, features_extractor_kwargs=features_extractor_kwargs, **kwargs)

def make_wrapped_env(env_id, n_stack):
    def _init_env():
        env = gym.make(env_id)
        wrapped_env = FlattenObservationWrapper(env, n_stack)
        return wrapped_env
    return _init_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose wheather to use transferred reward machine for Freeway task!")
    parser.add_argument('--use_reward_machine', action='store_true', help='Use reward machine for Freeway task!')
    parser.add_argument('--use_transfered_reward', action='store_true',
                        help='Use reward machine for Freeway task!')
    parser.add_argument('--seed', type=int, default=0, help='the random seed for training')
    parser.add_argument('--model_dir', type=str, default="./Freeway_log/Freeway-PPO/model", help='model_dir')
    parser.add_argument('--log_dir', type=str, default="./Freeway_log/Freeway-PPO-NRT/", help='log_dir')
    parser.add_argument('--gpu_id', type=int, default=6, help='')
    args = parser.parse_args()
    gpu_id = args.gpu_id
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        print(torch.cuda.current_device())
    is_use_reward_machine = args.use_reward_machine
    print(is_use_reward_machine)
    is_use_transfered_reward = args.use_transfered_reward
    print(is_use_transfered_reward)
    log_dir = args.log_dir
    seed = args.seed
    if is_use_transfered_reward:
        method = "RM_and_Transfered_Reward"
        train_env = make_vec_env(make_wrapped_env("Freeway-RM-v0", 4), n_envs=8, seed=0)
    else:
        method = "RM_and_Transfered_Reward"
        train_env = make_vec_env(make_wrapped_env("Freeway-RM-v1", 4), n_envs=8, seed=0)
    test_env = make_vec_env(make_wrapped_env("Freeway-RM-Evaluate-v0", 4), n_envs=8, seed=0)
    # test_env = VecFrameStack(test_env, n_stack=4)
    model_dir = args.model_dir + method + "/"
    log_dir = args.log_dir + method + "/"
    print(method)
    batch_size = 64
    num_cpu = 4
    eval_callback = EvalCallback(eval_env=test_env, best_model_save_path=model_dir, log_path=log_dir+"evaluate/",
                                 eval_freq=5000, n_eval_episodes=5, deterministic=True, render=False)
    print(eval_callback.last_mean_reward)
    # model = PPO("CnnPolicy",
    #             train_env,
    #             n_steps=128,
    #             batch_size=batch_size,
    #             n_epochs=20,
    #             learning_rate=2.5e-05,
    #             clip_range_vf=1,
    #             gamma=0.98,
    #             clip_range=0.01,
    #             ent_coef=0.1,
    #             verbose=1,
    #             vf_coef=0.1,
    #             tensorboard_log=log_dir + "train/")
    model = PPO(MultiInputPolicy, train_env, verbose=1, policy_kwargs=dict(num_rm_states=12, n_stack=4),
                n_steps=128, n_epochs=4, batch_size=256, learning_rate=2.5e-4, clip_range=0.1, vf_coef=0.5,
                ent_coef=0.01)
    print(">>>>>>>Training Start")
    model.learn(total_timesteps=1000000)
