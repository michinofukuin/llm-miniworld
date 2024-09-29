import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
import numpy as np
from reward_machines.reward_machine import RewardMachine

def flatten_observation_space(obs_space):
    if isinstance(obs_space, Dict):
        # 使用 flatten_space 展平整个 Dict 空间
        flat_space = gym.spaces.flatten_space(obs_space)
        return flat_space
    else:
        return obs_space

class RewardMachineEnv(gym.Wrapper):
    def __init__(self, env, rm_files, is_evaluate_env=False, use_reward_machine=False, origianl_task=False, add_rs=False):
        """
        RM environment
        --------------------
        It adds a set of RMs to the environment:
            - Every episode, the agent has to solve a different RM task
            - This code keeps track of the current state on the current RM task
            - The id of the RM state is appended to the observations
            - The reward given to the agent comes from the RM

        Parameters
        --------------------
            - env: original environment. It must implement the following function:
                - get_events(...): Returns the propositions that currently hold on the environment.
            - rm_files: list of strings with paths to the RM files.
        """
        super().__init__(env)

        # Loading the reward machines
        self.rm_files = rm_files
        self.reward_machines = []
        self.num_rm_states = 0
        for rm_file in rm_files:
            rm = RewardMachine(rm_file)
            self.num_rm_states += len(rm.get_states())
            self.reward_machines.append(rm)
        self.num_rms = len(self.reward_machines)
        self._use_reward_machine = use_reward_machine

        # The observation space is a dictionary including the env features and a one-hot representation of the state in the reward machine
        # self.observation_dict = spaces.Dict({'features': env.observation_space, 'rm-state': spaces.Box(low=0, high=1, shape=(self.num_rm_states,), dtype=np.uint8)})
        self.observation_dict = gym.spaces.Dict({'obs': env.observation_space['obs'], 'goal': env.observation_space['goal'], 'rm-state': Box(low=0, high=1, shape=(self.num_rm_states,), dtype=np.uint8)})
        if self._use_reward_machine:
            # 创建一个新的 Box 作为最终的 observation_space
            self.observation_space = self.observation_dict
        else:
            self.observation_space = env.observation_space
        # Computing one-hot encodings for the non-terminal RM states
        self.rm_state_features = {}
        for rm_id, rm in enumerate(self.reward_machines):
            for u_id in rm.get_states():
                u_features = np.zeros(self.num_rm_states)
                u_features[len(self.rm_state_features)] = 1
                self.rm_state_features[(rm_id,u_id)] = u_features
        self.rm_done_feat = np.zeros(self.num_rm_states) # for terminal RM states, we give as features an array of zeros
        # Selecting the current RM task
        self.current_rm_id = -1
        self.current_rm = None

        # Is an evaluated environment: To compare with original setting without reward machine
        self._is_evaluate_env = is_evaluate_env
        self._is_original_task = origianl_task
        self._add_rs = add_rs
        if self._add_rs:
            for reward_machine in self.reward_machines:
                reward_machine.add_reward_shaping(gamma=0.99, rs_gamma=0.9)
        # print("test")

    def reset(self):
        # Reseting the environment and selecting the next RM tasks
        self.obs = self.env.reset()
        self.current_rm_id = (self.current_rm_id+1)%self.num_rms
        self.current_rm    = self.reward_machines[self.current_rm_id]
        self.current_u_id  = self.current_rm.reset()
        info={}

        # Adding the RM state to the observation
        return self.get_observation(self.obs, self.current_rm_id, self.current_u_id, False), info

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, t, info = self.env.step(action)

        # getting the output of the detectors and saving information for generating counterfactual experiences
        true_props = self.env.get_events()
        self.crm_params = self.obs, action, next_obs, env_done, true_props, info
        self.obs = next_obs
        last_u_id = self.current_u_id
        # update the RM state
        self.current_u_id, rm_rew, rm_done = self.current_rm.step(self.current_u_id, true_props, info, add_rs=self._add_rs)
        info['rm_state'] = self.current_u_id
        info['events'] = true_props
        # if self._is_evaluate_env:
        #     if self.current_u_id != last_u_id:
        #         print("RM transfered from: " + str(last_u_id) + " to " + str(self.current_u_id))
        # returning the result of this action
        done = rm_done or env_done or t
        rm_obs = self.get_observation(next_obs, self.current_rm_id, self.current_u_id, done)
        if not self._is_evaluate_env:
            return_reward = rm_rew
        else:
            return_reward = original_reward
        if self._is_original_task:
            return_reward = rm_rew + original_reward * 1000 - 1
            # print(rm_rew)
            if self._is_evaluate_env:
                return_reward = original_reward
        return rm_obs, return_reward, done, t, info

    def get_observation(self, next_obs, rm_id, u_id, done):
        # Print next_obs to check its structure

        # Check if next_obs is a dictionary
        if isinstance(next_obs, dict):
            rm_feat = self.rm_done_feat if done else self.rm_state_features[(rm_id, u_id)]
            rm_obs = {
                'obs': next_obs['obs'].astype(np.uint8),  # Access the observation matrix
                'goal': next_obs['goal'].astype(np.uint8),  # Access the goal state
                'rm-state': rm_feat.astype(np.uint8)  # Add the reward machine state
            }
            npc = {
                'obs': next_obs['obs'].astype(np.uint8),  # Access the observation matrix
                'goal': next_obs['goal'].astype(np.uint8),  # Access the goal state
            }
        # Check if next_obs is a tuple
        elif isinstance(next_obs, tuple):
            next_obs_dict = next_obs[0]  # Access the first dictionary in the tuple
            rm_feat = self.rm_done_feat if done else self.rm_state_features[(rm_id, u_id)]
            rm_obs = {
                'obs': next_obs_dict['obs'].astype(np.uint8),  # Access 'obs' from the first dict in the tuple
                'goal': next_obs_dict['goal'].astype(np.uint8),  # Access 'goal' from the first dict in the tuple
                'rm-state': rm_feat.astype(np.uint8)  # Add the reward machine state
            }
            npc = {
                'obs': next_obs_dict['obs'].astype(np.uint8),  # Access 'obs' from the first dict in the tuple
                'goal': next_obs_dict['goal'].astype(np.uint8),  # Access 'goal' from the first dict in the tuple
            }
        else:
            raise ValueError(f"Unexpected next_obs type: {type(next_obs)}")

        if self._use_reward_machine:
            return rm_obs
        else:
            return npc



class RewardMachineWrapper(gym.Wrapper):
    def __init__(self, env, add_crm, add_rs, gamma, rs_gamma):
        """
        RM wrapper
        --------------------
        It adds crm (counterfactual experience) and/or reward shaping to *info* in the step function

        Parameters
        --------------------
            - env(RewardMachineEnv): It must be an RM environment
            - add_crm(bool):   if True, it will add a set of counterfactual experiences to info
            - add_rs(bool):    if True, it will add reward shaping to info
            - gamma(float):    Discount factor for the environment
            - rs_gamma(float): Discount factor for shaping the rewards in the RM
        """
        super().__init__(env)
        self.add_crm = add_crm
        self.add_rs  = add_rs
        if add_rs:
            for rm in env.reward_machines:
                rm.add_reward_shaping(gamma, rs_gamma)

    def get_num_rm_states(self):
        return self.env.num_rm_states

    def reset(self):
        self.valid_states = None # We use this set to compute RM states that are reachable by the last experience (None means that all of them are reachable!)
        return self.env.reset()

    def step(self, action):
        # RM and RM state before executing the action
        rm_id = self.env.current_rm_id
        rm    = self.env.current_rm
        u_id  = self.env.current_u_id

        # executing the action in the environment
        rm_obs, rm_rew, done, info = self.env.step(action)

        # adding crm if needed
        if self.add_crm:
            crm_experience = self._get_crm_experience(*self.crm_params)
            info["crm-experience"] = crm_experience
        elif self.add_rs:
            # Computing reward using reward shaping
            _, _, _, rs_env_done, rs_true_props, rs_info = self.crm_params
            _, rs_rm_rew, _ = rm.step(u_id, rs_true_props, rs_info, self.add_rs, rs_env_done)
            info["rs-reward"] = rs_rm_rew

        return rm_obs, rm_rew, done, info

    def _get_rm_experience(self, rm_id, rm, u_id, obs, action, next_obs, env_done, true_props, info):
        rm_obs = self.env.get_observation(obs, rm_id, u_id, False)
        next_u_id, rm_rew, rm_done = rm.step(u_id, true_props, info, self.add_rs, env_done)
        done = rm_done or env_done
        rm_next_obs = self.env.get_observation(next_obs, rm_id, next_u_id, done)
        return (rm_obs,action,rm_rew,rm_next_obs,done), next_u_id

    def _get_crm_experience(self, obs, action, next_obs, env_done, true_props, info):
        """
        Returns a list of counterfactual experiences generated per each RM state.
        Format: [..., (obs, action, r, new_obs, done), ...]
        """
        reachable_states = set()
        experiences = []
        for rm_id, rm in enumerate(self.reward_machines):
            for u_id in rm.get_states():
                exp, next_u = self._get_rm_experience(rm_id, rm, u_id, obs, action, next_obs, env_done, true_props, info)
                reachable_states.add((rm_id,next_u))
                if self.valid_states is None or (rm_id,u_id) in self.valid_states:
                    # We only add experience that are possible (i.e., it is possible to reach state u_id given the previous experience)
                    experiences.append(exp)

        self.valid_states = reachable_states
        return experiences


class AtariRewardMachineEnv(gym.Wrapper):
    def __init__(self, env, rm_files, is_evaluate_env=False, use_reward_machine=False, origianl_task=False,
                 add_rs=False):
        super().__init__(env)

        # Loading the reward machines
        self.rm_files = rm_files
        self.reward_machines = []
        self.num_rm_states = 0
        for rm_file in rm_files:
            rm = RewardMachine(rm_file)
            self.num_rm_states += len(rm.get_states())
            self.reward_machines.append(rm)
        self.num_rms = len(self.reward_machines)
        self._use_reward_machine = use_reward_machine

        # Update observation_dict: 'rm-state' now includes both RM id and the u_id
        self.observation_dict = spaces.Dict({'features': env.observation_space,
                                             'rm-state': spaces.Tuple((spaces.Discrete(len(rm_files)),
                                                                       spaces.Discrete(self.num_rm_states)))})

        if self._use_reward_machine:
            self.observation_space = self.observation_dict
        else:
            self.observation_space = env.observation_space

        # Selecting the current RM task
        self.current_rm_id = -1
        self.current_rm = None

        self._is_evaluate_env = is_evaluate_env
        self._is_original_task = origianl_task
        self._add_rs = add_rs
        if self._add_rs:
            for reward_machine in self.reward_machines:
                reward_machine.add_reward_shaping(gamma=0.99, rs_gamma=0.9)

    def reset(self):
        self.obs = self.env.reset()
        self.current_rm_id = (self.current_rm_id + 1) % self.num_rms
        self.current_rm = self.reward_machines[self.current_rm_id]
        self.current_u_id = self.current_rm.reset()

        return self.get_observation(self.obs, self.current_rm_id, self.current_u_id, False)

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)
        true_props = self.env.get_events()
        self.crm_params = self.obs, action, next_obs, env_done, true_props, info

        self.obs = next_obs
        last_u_id = self.current_u_id

        # Update the RM state
        self.current_u_id, rm_rew, rm_done = self.current_rm.step(
            self.current_u_id, true_props, info, add_rs=self._add_rs
        )
        info['rm_state'] = self.current_u_id
        info['events'] = true_props

        done = rm_done or env_done
        rm_obs = self.get_observation(next_obs, self.current_rm_id, self.current_u_id, done)
        rm_rew /= 1000
        if not self._is_evaluate_env:
            return_reward = rm_rew
        else:
            return_reward = original_reward

        if self._is_original_task:
            return_reward = rm_rew + original_reward * 1000 - 1
            if self._is_evaluate_env:
                return_reward = original_reward
        return rm_obs, return_reward, done, info

    def get_observation(self, next_obs, rm_id, u_id, done_):
        return {'features': next_obs, 'u-id': u_id} if not done_ else {'features': next_obs, 'u-id': -1}
