import numpy as np
import gym
from gym import spaces
import pygame

# Define actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Define entities
AGENT = 'agent'
OBJECTS = ['biscuit', 'candy']

# Shape encoding: square=0, circle=1, triangle=2
SHAPE_MAPPING = {
    0: "square",
    1: "circle",
    2: "triangle"
}

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

class SignEnvironment(gym.Env):  # Inherit from gym.Env
    metadata = {'render.modes': ['human']}
    
    def __init__(self, size=[10, 10], num_tasks=3, num_distractors=2):
        super(SignEnvironment, self).__init__()
        
        self.nrow, self.ncol = size
        self.nS = self.nrow * self.ncol
        self.num_tasks = num_tasks
        self.num_distractors = num_distractors
        self.action_space = spaces.Discrete(4)  # Four discrete actions: LEFT, DOWN, RIGHT, UP
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.nrow, self.ncol), dtype=np.int32)  # Grid observation

        self.screen = None  # To hold the pygame screen
        self.cell_size = 50  # Size of each cell in pixels

        self.objects = []
        self.state = {}
        self.current_stage = 0

        self.agent_initial_position = (self.nrow - 1, 0)
        self.sign_position = (0, self.ncol - 1)
        self.goals = self.generate_goals()  # Generate shape goals
        self.current_goal = self.goals[0]  # Initial goal is the first shape
        self.target_object_type = np.random.choice(OBJECTS)  # Randomly select target type

    def generate_goals(self):
        """Generate task goals based on shapes."""
        return np.random.choice([0, 1, 2], self.num_tasks).tolist()

    def reset(self):
        """Reset environment and set task sequence and current stage."""
        self.current_stage = 0
        self.state['agent'] = self.agent_initial_position  # Fixed agent initial position
        self.goals = self.generate_goals()
        self.current_goal = self.goals[self.current_stage]  # Initial goal is the first shape
        self.target_object_type = np.random.choice(OBJECTS)  # Choose target type at reset
        self.initialize_objects()  # Initialize objects and distractors
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

    def step(self, action):
        self.move_agent(action)
        obs = self.get_obs()
        reward = self.compute_reward()
        done = self.current_stage > self.num_tasks
        info = {'stage': self.current_stage}
        return obs, reward, done, info

    def compute_reward(self):
        agent_pos = self.state['agent']

        # Stage 0: Reach the sign position
        if self.current_stage == 0 and agent_pos == self.sign_position:
            self.current_stage += 1
            self.current_goal = self.goals[0]  # Target is the first task's shape when reaching the sign
            return 1

        # Stage 1-num_tasks: Interact with objects
        if 1 <= self.current_stage <= self.num_tasks:
            current_shape = SHAPE_MAPPING[self.goals[self.current_stage - 1]]  # Update to current stage shape
            for obj, positions in self.state['object_positions'].items():
                shape = obj.split('_')[0]
                obj_type = obj.split('_')[1]
                for pos in positions:
                    if pos == agent_pos:
                        if shape == current_shape and obj_type == self.target_object_type:
                            self.current_stage += 1
                            if self.current_stage < self.num_tasks:
                                self.current_goal = self.goals[self.current_stage]  # Update to next target shape
                            return 1  # Correct task completed
                        else:
                            return -1  # Incorrect object touched
        return 0  # No interaction

    def get_obs(self):
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

        return grid

    def render(self, mode='human'):
        """Render the current environment state using pygame."""
        # Initialize pygame if not already initialized
        if self.screen is None:
            pygame.init()
            window_size = (self.ncol * self.cell_size, self.nrow * self.cell_size)
            self.screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("Sign Environment")

        # Fill the screen with white background
        self.screen.fill((255, 255, 255))

        # Render the grid
        for i in range(self.nrow):
            for j in range(self.ncol):
                pygame.draw.rect(self.screen, (200, 200, 200),
                                 pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                                 1)  # Draw grid lines

                # Determine the color and shape
                cell_content = self.get_obs()[i, j]
                if cell_content == COLOR_MAPPING['biscuit']:
                    shape = self.goals[self.current_stage - 1] if self.current_stage > 0 else self.goals[0]
                    shape_name = SHAPE_MAPPING[shape]
                    color = COLOR_DICT[COLOR_MAPPING['biscuit']]
                    self.draw_shape(shape_name, color, i, j)
                elif cell_content == COLOR_MAPPING['candy']:
                    shape = self.goals[self.current_stage - 1] if self.current_stage > 0 else self.goals[0]
                    shape_name = SHAPE_MAPPING[shape]
                    color = COLOR_DICT[COLOR_MAPPING['candy']]
                    self.draw_shape(shape_name, color, i, j)
                elif (i, j) == self.state['agent']:
                    # Draw the agent as a star
                    self.draw_shape('agent', 'black', i, j)

        pygame.display.flip()

        # Pause for observation
        pygame.time.wait(500)  # Wait for 500 milliseconds (0.5 seconds)

    def draw_shape(self, shape, color, i, j):
        """Draw the specified shape with the given color at grid location (i, j)."""
        center_x = j * self.cell_size + self.cell_size // 2
        center_y = i * self.cell_size + self.cell_size // 2
        if shape == 'square':
            pygame.draw.rect(self.screen, color,
                             (center_x - self.cell_size // 4, center_y - self.cell_size // 4,
                              self.cell_size // 2, self.cell_size // 2))
        elif shape == 'circle':
            pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 4)
        elif shape == 'triangle':
            pygame.draw.polygon(self.screen, color,
                                [(center_x, center_y - self.cell_size // 4),
                                 (center_x - self.cell_size // 4, center_y + self.cell_size // 4),
                                 (center_x + self.cell_size // 4, center_y + self.cell_size // 4)])
        elif shape == 'agent':
            # Draw a star for the agent
            pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 4)
            pygame.draw.polygon(self.screen, color,
                                [(center_x, center_y - self.cell_size // 4),
                                 (center_x - self.cell_size // 8, center_y + self.cell_size // 8),
                                 (center_x + self.cell_size // 8, center_y + self.cell_size // 8)])
            pygame.draw.polygon(self.screen, color,
                                [(center_x - self.cell_size // 8, center_y),
                                 (center_x + self.cell_size // 8, center_y - self.cell_size // 4),
                                 (center_x - self.cell_size // 8, center_y - self.cell_size // 4)])
            pygame.draw.polygon(self.screen, color,
                                [(center_x + self.cell_size // 8, center_y),
                                 (center_x + self.cell_size // 8, center_y + self.cell_size // 4),
                                 (center_x - self.cell_size // 8, center_y + self.cell_size // 4)])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


import gym
from stable_baselines3 import PPO

# Initialize your custom environment
env = SignEnvironment(size=[10, 10], num_tasks=3, num_distractors=2)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=50000)

# Save the model
model.save("ppo_sign_environment")

# Load the model
model = PPO.load("ppo_sign_environment")

# Evaluate the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

