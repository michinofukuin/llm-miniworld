import numpy as np
import gym
from gym import spaces
import pygame
import os
import matplotlib.pyplot as plt
# Define actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Define entities
AGENT = 'agent'
OBJECTS = ['biscuit', 'candy']

NUM_SHAPES = 3  # square, circle, triangle
NUM_TYPES = 2   # biscuit, candy

# Shape encoding: square=0, circle=1, triangle=2
SHAPE_MAPPING = {
    0: "square",
    1: "circle",
    2: "triangle"
}
SHAPE_MAPPING_REVERSE = {v: k for k, v in SHAPE_MAPPING.items()}
# Color encoding: biscuit=blue, candy=red, agent=black
COLOR_MAPPING = {
    'biscuit': 1,
    'candy': 2,
    'agent': 3
}

COLOR_DICT = {
    1: 'blue',
    2: 'red',
    3: 'black'
}

# Shape and color mapping to matplotlib markers
SHAPE_MARKER_MAPPING = {
    ('square', 'biscuit'): ('s', 'blue'),
    ('circle', 'biscuit'): ('o', 'blue'),
    ('triangle', 'biscuit'): ('^', 'blue'),
    ('square', 'candy'): ('s', 'red'),
    ('circle', 'candy'): ('o', 'red'),
    ('triangle', 'candy'): ('^', 'red'),
    ('agent', 'agent'): ('*', 'black')  # Star shape for the agent
}
SHAPE_TYPE_MAPPING = {
    (0, 0): 2,  # square_biscuit
    (0, 1): 3,  # square_candy
    (1, 0): 4,  # circle_biscuit
    (1, 1): 5,  # circle_candy
    (2, 0): 6,  # triangle_biscuit
    (2, 1): 7   # triangle_candy
}
class SignEnvironment(gym.Env):  # Inherit from gym.Env
    metadata = {'render.modes': ['human']}
    
    def __init__(self, size=[10, 10], num_tasks=3, num_distractors=2, seed=None):
        super(SignEnvironment, self).__init__()
        if seed is not None:
            self.set_seed(seed)
        self.nrow, self.ncol = size
        self._her = False
        self.nS = self.nrow * self.ncol
        self.num_tasks = num_tasks
        self.num_distractors = num_distractors
        self.action_space = spaces.Discrete(4)  # Four discrete actions: LEFT, DOWN, RIGHT, UP
        obs_space = spaces.Box(low=0, high=8, shape=(self.nrow, self.ncol), dtype=np.float32)

        # Goal space: one-hot encoded shape (8 classes)
        goal_space = spaces.Discrete(8)

        # Object type space: one-hot encoded type (2 classes)
        # obj_type_space = spaces.MultiBinary(NUM_TYPES)

        # Combine these into a Dict observation space
        if self._her is True:
            self.observation_space = spaces.Dict({
            'observation': obs_space,
            'achieved_goal': goal_space,
            'desired_goal': goal_space,
        })
        else:
            self.observation_space = spaces.Dict({
            'obs': obs_space,
            'goal': goal_space,
        })

        # self.observation_space = spaces.Box(low=0, high=3, shape=(self.nrow, self.ncol), dtype=np.int32)  # Grid observation

        self.screen = None  # To hold the pygame screen
        self.cell_size = 50  # Size of each cell in pixels
        self.max_step = 1000
        self.objects = []
        self.state = {}
        self.current_stage = 0
        self.agent_path = []
        self.agent_initial_position = (0, 0)
        self.sign_position = (0, self.ncol - 1)
        self.goals = self.generate_goals()  # Generate shape goals
        self.current_goal = self.goals[0]  # Initial goal is the first shape
        self.target_object_type = np.random.choice(OBJECTS)  # Randomly select target type

    def generate_goals(self):
        """Generate task goals based on shapes."""
        return np.random.choice([0, 1, 2], self.num_tasks, replace=False).tolist()
    
    def set_seed(self, seed: int):
        """设置环境的随机种子。"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(seed)
        if hasattr(self.observation_space, 'seed'):
            self.observation_space.seed(seed)

    def reset(self):
        """重置环境，并重新设置任务序列和当前任务阶段。"""
        self.total_steps = 0  # 初始化总步数
        self.current_stage = 0
        self.agent_path = []
        self.state['agent'] = self.agent_initial_position  # 固定代理初始位置
        self.goals = self.generate_goals()
        self.current_goal = self.goals[self.current_stage]  # 初始目标设为当前阶段的形状
        self.target_object_type = np.random.choice(OBJECTS)  # 在重置时选择目标类型
        self.initialize_objects()  # 初始化物体和干扰项
        return self.get_obs()

    def initialize_objects(self):
        """Initialize object positions, including tasks and distractors."""
        self.state['object_positions'] = {}
        self.objects = []

        # Generate task objects
        for shape in self.goals:
            shape_name = SHAPE_MAPPING[shape]
            obj = f"{shape_name}_{self.target_object_type}"
            self.objects.append(obj)

        # Record all task combinations
        task_objects = set(self.objects)

        # Add distractors, ensuring they differ from all tasks
        for _ in range(self.num_distractors):
            while True:
                distractor_shape = np.random.choice(list(SHAPE_MAPPING.values()))
                distractor_type = np.random.choice(OBJECTS)
                distractor = f"{distractor_shape}_{distractor_type}"

                if distractor not in task_objects:
                    self.objects.append(distractor)
                    break

        # Assign random positions to all objects
        positions = np.random.permutation(self.nS)[:len(self.objects)]
        for i, ob in enumerate(self.objects):
            pos = self.from_s(positions[i])
            while pos == self.sign_position or pos == self.agent_initial_position:
                pos = self.from_s(np.random.randint(0, self.nS))
            self._add_obj(ob, pos)

    def _add_obj(self, objtype, pos):
        """Add object to specified position."""
        if objtype not in self.state['object_positions']:
            self.state['object_positions'][objtype] = []
        self.state['object_positions'][objtype].append(pos)

    def from_s(self, s):
        row = int(s / self.ncol)
        return (row, s - row * self.ncol)

    def move_agent(self, action):
        act = ACTIONS[action]
        pos = self.state['agent']
        row, col = pos[0] + act[0], pos[1] + act[1]
        if 0 <= row < self.nrow and 0 <= col < self.ncol:
            self.state['agent'] = (row, col)
            self.agent_path.append((row, col))

    def step(self, action):
        self.move_agent(action)
        self.total_steps += 1  # 记录总步数
        reward = self.compute_reward()
        obs = self.get_obs()
        done = self.current_stage > self.num_tasks or self.total_steps >= self.max_step  # 完成所有任务或达到最大步数时结束
        info = {'stage': self.current_stage}
        return obs, reward, done, info


    def compute_reward(self):
        agent_pos = self.state['agent']

        # 如果当前阶段为 0，并且到达标志位置，则进入下一个阶段
        if self.current_stage == 0 and agent_pos == self.sign_position:
            self.current_stage += 1
            self.current_goal = self.goals[0]  # 刚碰到标志时，目标为第一个任务的形状
            return 0

        # 阶段 1-num_tasks: 按顺序执行任务
        if 1 <= self.current_stage <= self.num_tasks:
            current_shape = SHAPE_MAPPING[self.goals[self.current_stage - 1]]  # 更新为当前阶段对应的形状
            for obj, positions in self.state['object_positions'].items():
                shape = obj.split('_')[0]
                obj_type = obj.split('_')[1]
                for pos in positions:
                    if pos == agent_pos:
                        if shape == current_shape and obj_type == self.target_object_type:
                            self.current_stage += 1
                            if self.current_stage > self.num_tasks:
                                # 完成所有任务，计算奖励
                                return 1000 * (1 - self.total_steps / self.max_step)
                            else:
                                self.current_goal = self.goals[self.current_stage-1]  # 更新为下一个目标形状
                                return 0  # 完成正确任务但未完成所有任务时奖励为0
                        else:
                            return 0  # 碰到错误的物体时奖励为0
        return 0  # 没有碰到任何物体时奖励为0

    def get_obs(self):
        """获取当前环境观察。"""
        # 初始化grid, 每个grid位置将包含有关形状和类型的编码
        grid = np.zeros((self.nrow, self.ncol), dtype=int)

        # 绘制agent位置
        mapping = {
        'square_biscuit': 2,
        'circle_biscuit': 3,
        'triangle_biscuit': 4,
        'square_candy': 5,
        'circle_candy': 6,
        'triangle_candy': 7,
        }
    
        # 定义特殊值
        agent_value = 8
        sign_value = 1

        # 绘制objects位置，使用组合映射后的值
        for obj, positions in self.state['object_positions'].items():
            if '_' in obj:
                shape_type = obj 
                value = mapping.get(shape_type, 0)  
                for pos in positions:
                    grid[pos] = value

        # 绘制agent位置，优先级最高
        grid[self.sign_position] = sign_value
        agent_pos = self.state['agent']
        achieved_goal = grid[agent_pos[0], agent_pos[1]]
        grid[agent_pos] = agent_value 
        # 定义goal
        goal_s = mapping.get(f"{SHAPE_MAPPING[self.current_goal]}_{self.target_object_type}", 0)
        if self.current_stage == 0:
            goal = 1
        else:
            goal = goal_s
        desired_goal = goal
        if self._her is True:
             obs = {
            'observation': grid,  # 包含物体的形状和类型信息的grid
            'achieved_goal': achieved_goal,  # 当前要寻找的目标形状
            'desired_goal': desired_goal,
        }
        else:
            obs = {
            'obs': grid,  # 包含物体的形状和类型信息的grid
            'goal': goal,  # 当前要寻找的目标形状
        }
        return obs


    def get_obs_2(self):
        """Get current environment observation."""
        grid = np.zeros((self.nrow, self.ncol), dtype=int)

        # Draw agent
        agent_pos = self.state['agent']
        grid[agent_pos] = COLOR_MAPPING['agent']

        # Draw objects
        for obj_type, positions in self.state['object_positions'].items():
            for pos in positions:
                # Safely handle object parsing
                if '_' in obj_type:
                    _, obj_type = obj_type.split('_')
                grid[pos] = COLOR_MAPPING.get(obj_type, 0)
        
        goal_encoding = np.zeros(NUM_SHAPES, dtype=np.float32)
        goal_encoding[self.current_goal] = 1
        goal = goal_encoding
        
        # Set object type
        type_encoding = np.zeros(NUM_TYPES, dtype=np.float32)
        type_encoding[OBJECTS.index(self.target_object_type)] = 1
        obj_type = type_encoding

        obs = {
        'obs': grid,
        }

        return obs

    def render(self, mode='human', show_on_screen=False):
        """Render the current environment state, optionally showing it on the screen."""
        # Initialize pygame if not already initialized
        if self.screen is None:
            pygame.init()
            window_size = (self.ncol * self.cell_size, self.nrow * self.cell_size)
            if show_on_screen:
                self.screen = pygame.display.set_mode(window_size)  # Create a window if showing on screen
                pygame.display.set_caption("Sign Environment")
            else:
                self.screen = pygame.Surface(window_size)  # Create an off-screen surface

        # Fill the screen with white background
        self.screen.fill((255, 255, 255))

        # Get the current observation to use for rendering
        obs = self.get_obs()
        grid = obs['obs'] if not self._her else obs['observation']  # Adjust based on HER or not

        # Render the grid and objects
        for i in range(self.nrow):
            for j in range(self.ncol):
                pygame.draw.rect(self.screen, (200, 200, 200),
                                pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                                1)  # Draw grid lines

                # Determine the content of the cell and draw accordingly
                cell_value = grid[i, j]
                if cell_value == 1:  # Sign
                    self.draw_shape('sign', 'yellow', i, j)
                elif cell_value == 8:  # Agent
                    self.draw_shape('agent', 'black', i, j)
                else:
                    shape_type = {v: k for k, v in self.mapping.items()}.get(cell_value, None)
                    if shape_type:
                        shape, obj_type = shape_type.split('_')
                        color = COLOR_DICT[COLOR_MAPPING[obj_type]]
                        self.draw_shape(shape, color, i, j)

        if show_on_screen:
            pygame.display.flip()  # Update the display if showing on screen

        return self.screen

    def draw_shape(self, shape, color, i, j):
        """Draw the specified shape with the given color at grid location (i, j)."""
        center_x = j * self.cell_size + self.cell_size // 2
        center_y = i * self.cell_size + self.cell_size // 2
        color_rgb = pygame.Color(color)

        if shape == 'square':
            pygame.draw.rect(self.screen, color_rgb,
                            (center_x - self.cell_size // 4, center_y - self.cell_size // 4,
                            self.cell_size // 2, self.cell_size // 2))
        elif shape == 'circle':
            pygame.draw.circle(self.screen, color_rgb, (center_x, center_y), self.cell_size // 4)
        elif shape == 'triangle':
            pygame.draw.polygon(self.screen, color_rgb,
                                [(center_x, center_y - self.cell_size // 4),
                                (center_x - self.cell_size // 4, center_y + self.cell_size // 4),
                                (center_x + self.cell_size // 4, center_y + self.cell_size // 4)])
        elif shape == 'sign':
            pygame.draw.circle(self.screen, color_rgb, (center_x, center_y), self.cell_size // 2, 2)
        elif shape == 'agent':
            # Draw a star for the agent
            pygame.draw.circle(self.screen, color_rgb, (center_x, center_y), self.cell_size // 4)
            pygame.draw.polygon(self.screen, color_rgb,
                                [(center_x, center_y - self.cell_size // 4),
                                (center_x - self.cell_size // 8, center_y + self.cell_size // 8),
                                (center_x + self.cell_size // 8, center_y + self.cell_size // 8)])
            pygame.draw.polygon(self.screen, color_rgb,
                                [(center_x - self.cell_size // 8, center_y),
                                (center_x + self.cell_size // 8, center_y - self.cell_size // 4),
                                (center_x - self.cell_size // 8, center_y - self.cell_size // 4)])
            pygame.draw.polygon(self.screen, color_rgb,
                                [(center_x + self.cell_size // 8, center_y),
                                (center_x + self.cell_size // 8, center_y + self.cell_size // 4),
                                (center_x - self.cell_size // 8, center_y + self.cell_size // 4)])


    import pygame
    import os

    def plot_agent_path(self, res_path="res/agent_path.png", show_on_screen=False):
        """Plot and save the agent's path along with environment objects using Pygame."""
        
        # Initialize pygame if not already initialized
        if self.screen is None:
            pygame.init()
            window_size = (self.ncol * self.cell_size, self.nrow * self.cell_size)
            if show_on_screen:
                self.screen = pygame.display.set_mode(window_size)  # Create a window if showing on screen
                pygame.display.set_caption("Sign Environment")
            else:
                self.screen = pygame.Surface(window_size)  # Create an off-screen surface

        # Fill the screen with white background
        self.screen.fill((255, 255, 255))

        # Render the grid, objects, and sign
        for i in range(self.nrow):
            for j in range(self.ncol):
                pygame.draw.rect(self.screen, (200, 200, 200),
                                pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                                1)  # Draw grid lines

                # Draw sign
                cell_content = self.get_obs()['obs'][i, j]
                if cell_content == 1:  # Sign
                    pygame.draw.circle(self.screen, (255, 255, 0),
                                    (j * self.cell_size + self.cell_size // 2, i * self.cell_size + self.cell_size // 2),
                                    self.cell_size // 4, 2)
                elif 2 <= cell_content <= 7:  # Objects
                    color = (0, 0, 255) if cell_content in [2, 3, 4] else (255, 0, 0)
                    pygame.draw.circle(self.screen, color,
                                    (j * self.cell_size + self.cell_size // 2, i * self.cell_size + self.cell_size // 2),
                                    self.cell_size // 4)
                elif cell_content == 8:  # Agent
                    pygame.draw.polygon(self.screen, (0, 0, 0),
                                        [(j * self.cell_size + self.cell_size // 2, i * self.cell_size + self.cell_size // 4),
                                        (j * self.cell_size + self.cell_size // 4, i * self.cell_size + 3 * self.cell_size // 4),
                                        (j * self.cell_size + 3 * self.cell_size // 4, i * self.cell_size + 3 * self.cell_size // 4)])

        # Draw the agent's path with arrows, ensuring it does not overlap with objects
        path_x, path_y = zip(*[(pos[1], pos[0]) for pos in self.agent_path])
        for k in range(len(path_x) - 1):
            start_pos = (path_x[k] * self.cell_size + self.cell_size // 2, path_y[k] * self.cell_size + self.cell_size // 2)
            end_pos = (path_x[k+1] * self.cell_size + self.cell_size // 2, path_y[k+1] * self.cell_size + self.cell_size // 2)
            pygame.draw.line(self.screen, (0, 0, 255), start_pos, end_pos, 2)
            # Draw arrowhead
            arrow_size = self.cell_size // 8
            angle = pygame.math.Vector2(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]).angle_to((1, 0))
            pygame.draw.polygon(self.screen, (0, 0, 255),
                                [(end_pos[0] + arrow_size * pygame.math.cos(angle + 135),
                                end_pos[1] + arrow_size * pygame.math.sin(angle + 135)),
                                (end_pos[0] + arrow_size * pygame.math.cos(angle - 135),
                                end_pos[1] + arrow_size * pygame.math.sin(angle - 135)),
                                end_pos])

        # Draw the agent's initial position as a star
        initial_pos = self.agent_initial_position
        pygame.draw.polygon(self.screen, (0, 0, 0),
                            [(initial_pos[1] * self.cell_size + self.cell_size // 2, initial_pos[0] * self.cell_size + self.cell_size // 4),
                            (initial_pos[1] * self.cell_size + self.cell_size // 4, initial_pos[0] * self.cell_size + 3 * self.cell_size // 4),
                            (initial_pos[1] * self.cell_size + 3 * self.cell_size // 4, initial_pos[0] * self.cell_size + 3 * self.cell_size // 4)])

        # Save the surface as an image
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        pygame.image.save(self.screen, res_path)

        # Optionally show the plot on the screen
        if show_on_screen:
            pygame.display.flip()

        # Optionally, quit pygame to clean up resources
        if not show_on_screen:
            pygame.quit()
            self.screen = None


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


import gym
from stable_baselines3 import PPO, DQN
import torch
import random
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn
# Initialize your custom environment
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
env = SignEnvironment(size=[4, 4], num_tasks=3, num_distractors=0,seed=42)
test_env = SignEnvironment(size=[4, 4], num_tasks=3, num_distractors=0,seed=0)
log_dir = "logs_llm/test/1.5e-5-0.2_55/"
model_dir = "logs_llm/eval_model/1.5e-5-0.2_55/"
# Create the PPO model
eval_callback = EvalCallback(eval_env=test_env, best_model_save_path=model_dir, log_path=log_dir+"evaluate/",
                                 eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False)
model = PPO("MultiInputPolicy",
            env,
            device="cuda:0",
            policy_kwargs=dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU, net_arch=[dict(pi=[256,256], vf=[256, 256])]),
            n_steps=512,
            batch_size=128,
            n_epochs=20,
            learning_rate=4e-5,
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
model.learn(total_timesteps=600000, callback=eval_callback)

# Save the model
model.save("logs_llm/4_4_3/4e-5-0.2")

# # Load the model
# model = PPO.load("/mnt/data/chenhaosheng/logs_llm/4e-5-0.2.zip")

# # Evaluate the model
# obs = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)


# obs = env.reset()
# env.plot_agent_path()  # 显示agent的运动路径
# env.close()

