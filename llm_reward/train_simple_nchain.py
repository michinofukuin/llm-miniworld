import gym
from gym import spaces
import numpy as np
import torch


# 创建自定义环境
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # 动作空间: 左(0), 右(1), 上(2), 下(3)
        self.action_space = spaces.Discrete(4)

        # 最大宽度和高度
        self.max_w = 7
        self.max_h = 2

        # 观察空间（Agent位置）
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.max_h - 1, self.max_w - 1]),
                                            dtype=np.int32)

        # 初始状态
        self.start_pos = (0, 5)

        # 最大步数
        self.max_steps = 100

    def reset(self):
        self.agent_pos = np.array(self.start_pos)
        self.current_step = 0
        return self.agent_pos.copy()

    def step(self, action):
        old_position = self.agent_pos.copy()

        if action == 0:  # 左
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 1:  # 右
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.max_w - 1)
        elif action == 2:  # 上
            if self.agent_pos[1] == 3 and self.agent_pos[0] == 0:
                self.agent_pos[0] = min(self.agent_pos[0] + 1, 1)
            else:
                self.agent_pos = old_position
        elif action == 3:  # 下
            if self.agent_pos[1] == 3 and self.agent_pos[0] == 1:
                self.agent_pos[0] -= 1
            else:
                self.agent_pos = old_position

        self.current_step += 1

        done = (self.agent_pos == [0, 0]).all() or (self.current_step >= self.max_steps)

        reward = (((self.max_steps - self.current_step) / self.max_steps) * 1000) if done and (
                    self.agent_pos == [0, 0]).all() else 0

        info = {}

        return self.agent_pos.copy(), reward, done, info

    def set_obs(self, x, y):
        self.agent_pos[0] = x
        self.agent_pos[1] = y


# 定义环境实例
env = CustomEnv()

# 导入DQN模型
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

# 创建并训练DQN模型
model = DQN("MlpPolicy", env, verbose=1)
# eval_callback = EvalCallback(env, best_model_save_path='./models/', log_path='./logs/', eval_freq=5000)
model.learn(total_timesteps=int(300000))
# 输出每个格子转移的return
for x in range(env.max_h - 1):
    for y in range(env.max_w):
        source_state = np.array([x, y], dtype=np.int32)
        for action in range(4):
            env.reset()
            env.set_obs(x, y)
            next_state, _, _, _ = env.step(action)
            tensor_input = torch.tensor(source_state, dtype=torch.float32).unsqueeze(dim=0).to(model.device)
            value = model.policy.q_net(tensor_input).to("cpu").detach().max(1)[0].item()
            tensor_input_next = torch.tensor(next_state, dtype=torch.float32).unsqueeze(dim=0).to(model.device)
            value_next = model.policy.q_net(tensor_input_next).to("cpu").detach().max(1)[0].item()
            value_diff = 100 * (value_next - value)
            print(f"<起始状态: {source_state}, 达到状态: {next_state}, 动作: {action}, return: {value_diff}>")
