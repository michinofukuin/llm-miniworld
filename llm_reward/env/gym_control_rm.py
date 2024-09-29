import sys, os
sys.path.append("/home/ubuntu/llm_reward/reward_machines")
import gymnasium as gym
import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.classic_control import AcrobotEnv, CartPoleEnv, MountainCarEnv, PendulumEnv
from miniworld.envs.sign import Sign
from reward_machines.env_with_rm import RewardMachineEnv

class SignEnvRM(gym.Wrapper):
    def __init__(self):
        super().__init__(Sign(render_mode='human'))
        self.info = {}

    def step(self, action):
        next_obs, original_reward, env_done, t, info = self.env.step(action)
        self.info = info['stage']
        return next_obs, original_reward, env_done, t, info 
    
    def get_events(self):
        events = ''
        if self.info == 0:
            events+='e'
        if self.info == 1:
            events+='a'
        if self.info == 2:
            events+='b'
        if self.info == 3:
            events+='c'
        if self.info == 4:
            events+='d'
        return events

class AcrobotEnvRM(gym.Wrapper):
    def __init__(self):
        super().__init__(AcrobotEnv())
        self.info = None

    def step(self, action):
        next_obs, original_reward, env_done, t, info = self.env.step(action)

        self.info = info
        return next_obs, original_reward, env_done, t, info

    def get_events(self):
        events = ''

class CartPoleEnvRM(gym.Wrapper):
    def __init__(self):
        super().__init__(CartPoleEnv())
        self.info = {}
        self._original_task = False

    def set_original_task(self):
        self._original_task = True

    def reset(self):
        obs = self.env.reset()
        self.info['cart_position'] = obs[0]
        return obs

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)
        reward_ctrl = original_reward
        reward_fun = next_obs[0] - self.info['cart_position']
        self.info = info
        if not self._original_task:
            self.info['reward_ctrl'] = 0
            if next_obs[0] < 2.4 and env_done:
                self.info['reward_ctrl'] = -100
            self.info['reward_run'] = reward_fun * 0
        else:
            self.info['reward_run'] = reward_fun * 50
            self.info['reward_ctrl'] = 1
        self.info['cart_position'] = next_obs[0]
        return next_obs, original_reward, env_done, info

    def get_events(self):
        events = ''
        if self.info['cart_position'] < -4.8:
            events += 'b'
        if self.info['cart_position'] > 2.4:
            events += 'a'  # termination event
        if self.info['cart_position'] < -2.4:
            events += 'd'
        if self.info['cart_position'] >= 0.1:
            events += 'c'
        if self.info['cart_position'] > 0.3:
            events += 'e'
        if self.info['cart_position'] > 0.7:
            events += 'f'
        if self.info['cart_position'] > 1.5:
            events += 'g'
        return events

class MountainCarEnvRM(gym.Wrapper):
    def __init__(self):
        super().__init__(MountainCarEnv)
        self.info = {}

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        self.info['car_position'] = next_obs[0]
        return next_obs, original_reward, env_done, info

    def get_events(self):
        events = ''
        if self.info['car_position'] >= -0.4:
            events += 'a'
        if self.info['car_position'] >= -0.2:
            events += 'b'
        if self.info['car_position'] >= -0:
            events += 'c'
        if self.info['car_position'] >= 0.2:
            events += 'd'
        if self.info['car_position'] >= 0.3:
            events += 'e'
        if self.info['car_position'] >= 0.4:
            events += 'f'
        if self.info['car_position'] >= 0.5:
            events += 'g'
        return events


class HalfCheetahEnvRM(gym.Wrapper):
    def __init__(self):
        super().__init__(HalfCheetahEnv(exclude_current_positions_from_observation=False))

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        original_reward = info['reward_run']
        return next_obs, original_reward, env_done, info

    def get_events(self):
        events = ''
        if self.info['x_position'] < -10:
            events+='b'
        if self.info['x_position'] > 5:
            events+='a'
        if self.info['x_position'] < -2:
            events+='d'
        if self.info['x_position'] > 2:
            events+='c'
        if self.info['x_position'] > 4:
            events+='e'
        if self.info['x_position'] > 6:
            events+='f'
        if self.info['x_position'] > 8:
            events+='g'
        return events

class HumanoidEnvRM(gym.Wrapper):
    def __init__(self):
        super().__init__(HumanoidEnv(exclude_current_positions_from_observation=False))

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        return next_obs, original_reward, env_done, info

    def get_events(self):
        events = ''
        if self.info['x_position'] < -10:
            events+='b'
        if self.info['x_position'] > 10:
            events+='a'
        if self.info['x_position'] < -2:
            events+='d'
        if self.info['x_position'] > 2:
            events+='c'
        if self.info['x_position'] > 4:
            events+='e'
        if self.info['x_position'] > 6:
            events+='f'
        if self.info['x_position'] > 8:
            events+='g'
        return events

class HopperEnvRM(gym.Wrapper):
    def __init__(self):
        super().__init__(HopperEnv(exclude_current_positions_from_observation=False))

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        return next_obs, original_reward, env_done, info

    def get_events(self):
        events = ''
        if self.info['x_position'] < -10:
            events+='b'
        if self.info['x_position'] > 10:
            events+='a'
        if self.info['x_position'] < -2:
            events+='d'
        if self.info['x_position'] > 2:
            events+='c'
        if self.info['x_position'] > 4:
            events+='e'
        if self.info['x_position'] > 6:
            events+='f'
        if self.info['x_position'] > 8:
            events+='g'
        return events

class CartPoleRMEnv(RewardMachineEnv):
    def __init__(self):
        env = CartPoleEnvRM()
        rm_files = ["./env/reward_machines/cartpole.txt"]
        super().__init__(env, rm_files, use_reward_machine=True, origianl_task=False, add_rs=False)

class CartPoleRMEnvSparse(RewardMachineEnv):
    def __init__(self):
        env = CartPoleEnvRM()
        rm_files = ["./env/reward_machines/sparse_cartpole.txt"]
        super().__init__(env, rm_files, use_reward_machine=True, origianl_task=False, add_rs=False)

class CartPoleRMEnvTransfer(RewardMachineEnv):
    def __init__(self):
        env = CartPoleEnvRM()
        rm_files = ["./env/reward_machines/transferred_cartpole.txt"]
        super().__init__(env, rm_files, use_reward_machine=True, origianl_task=False, add_rs=False)

class CartPoleNoRMEnvSparse(RewardMachineEnv):
    def __init__(self):
        env = CartPoleEnvRM()
        rm_files = ["./env/reward_machines/sparse_cartpole.txt"]
        super().__init__(env, rm_files, use_reward_machine=False, origianl_task=False, add_rs=False)

class CartPoleNoRMEnvTransfer(RewardMachineEnv):
    def __init__(self):
        env = CartPoleEnvRM()
        rm_files = ["./env/reward_machines/transferred_cartpole.txt"]
        super().__init__(env, rm_files, use_reward_machine=False, origianl_task=False, add_rs=False)

class HalfCheetahRMEnv(RewardMachineEnv): # RM+reward
    def __init__(self):
        env = HalfCheetahEnvRM()
        rm_files = ["./env/reward_machines/transferred_halfcheetah.txt"]
        super().__init__(env, rm_files, use_reward_machine=True)

class SignRMEnv_r(RewardMachineEnv):
    metadata = {  # type: ignore
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    def __init__(self,render_mode=None):
        env = SignEnvRM()
        rm_files = ["/home/ubuntu/llm_reward/env/reward_machines/transferred_sign.txt"]
        super().__init__(env, rm_files, use_reward_machine=True)

    def reset(self, seed=None, options=None):
        # Optionally handle the seed
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Call the reset function of the RewardMachineEnv (parent class)
        return super().reset()

class SignRMEnv(RewardMachineEnv):
    metadata = {  # type: ignore
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    def __init__(self, render_mode=None):
        env = SignEnvRM()
        rm_files = ["/home/ubuntu/llm_reward/env/reward_machines/sign.txt"]
        super().__init__(env, rm_files, use_reward_machine=True)
    def reset(self, seed=None, options=None):
        # Optionally handle the seed
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Call the reset function of the RewardMachineEnv (parent class)
        return super().reset()


class SignRMEnvEmpty(RewardMachineEnv):
    metadata = {  # type: ignore
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    def __init__(self, render_mode=None):
        env = SignEnvRM()
        rm_files = ["/home/ubuntu/llm_reward/env/reward_machines/sign.txt"]
        super().__init__(env, rm_files)
    def reset(self, seed=None, options=None):
        # Optionally handle the seed
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Call the reset function of the RewardMachineEnv (parent class)
        return super().reset()

class SignRMEnvEmpty_r(RewardMachineEnv):
    metadata = {  # type: ignore
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    def __init__(self, render_mode=None):
        env = SignEnvRM()
        rm_files = ["/home/ubuntu/llm_reward/env/reward_machines/transferred_sign.txt"]
        super().__init__(env, rm_files)
    def reset(self, seed=None, options=None):
        # Optionally handle the seed
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Call the reset function of the RewardMachineEnv (parent class)
        return super().reset()

class SignRMEnvEvaluate(RewardMachineEnv): # RM Evaluate
    metadata = {  # type: ignore
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
        }
    def __init__(self, render_mode=None):
        env = SignEnvRM()
        rm_files = ["/home/ubuntu/llm_reward/env/reward_machines/sign.txt"]
        super().__init__(env, rm_files, is_evaluate_env=True, use_reward_machine=True)

    def reset(self, seed=None, options=None):
        # Optionally handle the seed
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Call the reset function of the RewardMachineEnv (parent class)
        return super().reset()

class SignEnvEvaluate(RewardMachineEnv): # R Evaluate
    metadata = {  # type: ignore
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    def __init__(self, render_mode=None):
        env =SignEnvRM()
        rm_files = ["/home/ubuntu/llm_reward/env/reward_machines/sign.txt"]
        super().__init__(env, rm_files, is_evaluate_env=True)
    
    def reset(self, seed=None, options=None):
        # Optionally handle the seed
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Call the reset function of the RewardMachineEnv (parent class)
        return super().reset()

class HalfCheetahRMEnvEmptyRM(RewardMachineEnv): # RM
    def __init__(self):
        env = HalfCheetahEnvRM()
        rm_files = ["./env/reward_machines/halfcheetah.txt"]
        super().__init__(env, rm_files)

class HalfCheetahRMEnvEvaluate(RewardMachineEnv): # RM Evaluate
    def __init__(self):
        env = HalfCheetahEnvRM()
        rm_files = ["./env/reward_machines/halfcheetah.txt"]
        super().__init__(env, rm_files, is_evaluate_env=True, use_reward_machine=True)

class HalfCheetahEnvReward(RewardMachineEnv): # Original+Reward
    def __init__(self):
        env = HalfCheetahEnvRM()
        rm_files = ["./env/reward_machines/transferred_halfcheetah.txt"]
        super().__init__(env, rm_files)

class HalfCheetahEnvOriginal(RewardMachineEnv): # Original
    def __init__(self):
        env = HalfCheetahEnvRM()
        rm_files = ["./env/reward_machines/halfcheetah.txt"]
        super().__init__(env, rm_files)

class HalfCheetahEnvEvaluate(RewardMachineEnv): # Original Evaluate
    def __init__(self):
        env = HalfCheetahEnvRM()
        rm_files = ["./env/reward_machines/halfcheetah.txt"]
        super().__init__(env, rm_files, is_evaluate_env=True)

class HumanoidRMEnv(RewardMachineEnv):  # RM+reward
    def __init__(self):
        env = HumanoidEnvRM()
        rm_files = ["./env/reward_machines/transferred_halfcheetah.txt"]
        super().__init__(env, rm_files, use_reward_machine=True)

class HumanoidRMEnvEmptyRM(RewardMachineEnv):  # RM
    def __init__(self):
        env = HumanoidEnvRM()
        rm_files = ["./env/reward_machines/halfcheetah.txt"]
        super().__init__(env, rm_files, use_reward_machine=True)

class HumanoidRMEnvEvaluate(RewardMachineEnv):  # RM Evaluate
    def __init__(self):
        env = HumanoidEnvRM()
        rm_files = ["./env/reward_machines/halfcheetah.txt"]
        super().__init__(env, rm_files, is_evaluate_env=True, use_reward_machine=True)

class HumanoidEnvReward(RewardMachineEnv):  # Original+Reward
    def __init__(self):
        env = HumanoidEnvRM()
        rm_files = ["./env/reward_machines/transferred_halfcheetah.txt"]
        super().__init__(env, rm_files)

class HopperRMEnv(RewardMachineEnv):  # RM+reward
    def __init__(self):
        env = HopperEnvRM()
        rm_files = ["./env/reward_machines/transferred_halfcheetah.txt"]
        super().__init__(env, rm_files, use_reward_machine=True)

class HopperRMEnvEmptyRM(RewardMachineEnv):  # RM
    def __init__(self):
        env = HopperEnvRM()
        rm_files = ["./env/reward_machines/halfcheetah.txt"]
        super().__init__(env, rm_files, use_reward_machine=True)

class HopperRMEnvEvaluate(RewardMachineEnv):  # RM Evaluate
    def __init__(self):
        env = HopperEnvRM()
        rm_files = ["./env/reward_machines/halfcheetah.txt"]
        super().__init__(env, rm_files, is_evaluate_env=True, use_reward_machine=True)

class HopperEnvReward(RewardMachineEnv):  # Original+Reward
    def __init__(self):
        env = HopperEnvRM()
        rm_files = ["./env/reward_machines/transferred_halfcheetah.txt"]
        super().__init__(env, rm_files)

if __name__ == "__main__":
    test_env = HalfCheetahRMEnv()
    obs = test_env.reset()
    for i in range(50):
        act = test_env.action_space.sample()
        next_obs, reward, done, info = test_env.step(act)
        print(i, reward, info)
        obs = next_obs