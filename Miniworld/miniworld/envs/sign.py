import math
import random
from typing import Optional, Tuple
import gymnasium as gym
from gymnasium import utils
from gymnasium.core import ObsType
import numpy as np
from miniworld.entity import COLOR_NAMES, Box, Key, MeshEnt, TextFrame
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

object_mapping = {
    'blue_box': 1,
    'red_box': 2,
    'green_box': 3,
    'blue_key': 4,
    'red_key': 5,
    'green_key': 6
}
colors = ['blue', 'red', 'green']
entities = ['box', 'key']
color_entity_pos_mapping = {
    'blue_box': (1, 0, 1),
    'red_box': (9, 0, 1),
    'green_box': (9, 0, 5),
    'blue_key': (5, 0, 1),
    'red_key': (1, 0, 5),
    'green_key': (1, 0, 9)
}
color_to_index = {'blue': 0, 'red': 1, 'green': 2}

# 初始化颜色序列时，使用索引而不是名称
class BigKey(Key):
    """A key with a bigger size for better visibility."""

    def __init__(self, color, size=0.6):
        assert color in COLOR_NAMES
        self.color = color
        MeshEnt.__init__(self, mesh_name=f"key_{color}", height=size, static=False)


class Sign(MiniWorldEnv, utils.EzPickle):
    metadata = {  # type: ignore
        "render_modes": [
            "human",
            "rgb_array",
        ],
    }
    def __init__(self, size=10, max_episode_steps=1000, num_tasks=3, **kwargs):
        self.num_tasks = num_tasks
        self.rng = np.random.default_rng()
        params = DEFAULT_PARAMS.no_random()
        params.set("forward_step", 0.7)  # larger steps
        params.set("turn_step", 45)  # 45 degree rotation
        self.tot_step = 0
        self.max_step = max_episode_steps
        obj = random.choice([0, 1])
        self.obj = obj
        self.goal = [size, 0, size]
        self.current_stage = 0
        self.color_sequence = random.sample(['blue', 'red', 'green'], num_tasks)  # 随机化颜色顺序
        self._size = size
        if obj not in [0, 1]:
            raise ValueError("Goals must be 0 (box) or 1 (key).")

        MiniWorldEnv.__init__(
            self,
            params=DEFAULT_PARAMS,
            max_episode_steps=max_episode_steps,
            domain_rand=False,
            **kwargs,
        )
        utils.EzPickle.__init__(self, size, max_episode_steps, num_tasks, **kwargs)
        goal_space = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([self._size + 1, self._size + 1, self._size + 1]), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({
            'obs': self.observation_space,  # 原来的 observation_space
            'goal': goal_space  # 新的 goal space
        })
        self.action_space = gym.spaces.Discrete(self.actions.move_forward + 2)

    def seed(self, seed=None):
        """设置随机种子"""
        # 设置 numpy 和 python 的随机数种子
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def generate_task_letters(self, tasks):
        """生成任务颜色的首字母字符串"""
        color_initials = {
            "BLUE": "B",
            "GREEN": "G",
            "RED": "R"
        }
        return ''.join(color_initials[task.upper()] for task in tasks)

    def _gen_world(self):
        gap_size = 0.25
        top_room = self.add_rect_room(
            min_x=0, max_x=self._size, min_z=0, max_z=self._size * 0.65
        )
        left_room = self.add_rect_room(
            min_x=0,
            max_x=self._size * 3 / 5,
            min_z=self._size * 0.65 + gap_size,
            max_z=self._size * 1.3,
        )
        right_room = self.add_rect_room(
            min_x=self._size * 3 / 5,
            max_x=self._size,
            min_z=self._size * 0.65 + gap_size,
            max_z=self._size * 1.3,
        )
        self.connect_rooms(top_room, left_room, min_x=0, max_x=self._size * 3 / 5)
        self.connect_rooms(
            left_room,
            right_room,
            min_z=self._size * 0.65 + gap_size,
            max_z=self._size * 1.3,
        )

        self._objects = [
            # Boxes
            (
                self.place_entity(Box(color="blue"), pos=(1, 0, 1)),
                self.place_entity(Box(color="red"), pos=(9, 0, 1)),
                self.place_entity(Box(color="green"), pos=(9, 0, 5)),
            ),
            # Keys
            (
                self.place_entity(BigKey(color="blue"), pos=(5, 0, 1)),
                self.place_entity(BigKey(color="red"), pos=(1, 0, 5)),
                self.place_entity(BigKey(color="green"), pos=(1, 0, 9)),
            ),
        ]

        # 显示所有任务的颜色
        text = self.generate_task_letters(self.color_sequence)
        # text = 'blue'
        sign = TextFrame(
            pos=[self._size, 0, self._size],
            dir=math.pi,
            str=text,
            height=1,
        )
        self.entities.append(sign)

        # 确保代理和标志位置不同
        self.place_agent(min_x=4, max_x=5, min_z=4, max_z=6)

    def _is_close(self, x1, z1, x2, z2, threshold=1.0):
        """判断两个点是否接近"""
        return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2) < threshold

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        # 自定义结束动作
        # if action == self.actions.move_forward + 1:
        #     termination = True
        # if action == self.actions.move_forward + 1:  # custom end episode action
        #     termination = True
        # 定义颜色和物体类型的映射
        color_entity_pos_mapping = {
            'blue_box': (1, 0, 1),
            'red_box': (9, 0, 1),
            'green_box': (9, 0, 5),
            'blue_key': (5, 0, 1),
            'red_key': (1, 0, 5),
            'green_key': (1, 0, 9)
        }
        info = {}
        info['stage'] = self.current_stage
        # 阶段0：代理发现标志
        if self.current_stage == 0:
            for entity in self.entities:
                if isinstance(entity, TextFrame) and self.near(entity):  # 确保检测到标志
                    self.current_stage += 1
                    info['stage'] = self.current_stage
                    reward = 0  # 在找到标志时给予轻微奖励
                    
                    # 设置当前任务的goal为下一阶段目标物体的坐标
                    next_color = self.color_sequence[self.current_stage - 1]
                    obj_type = 'box' if self.obj == 0 else 'key'
                    goal_key = f"{next_color}_{obj_type}"
                    self.goal = color_entity_pos_mapping[goal_key]  # 获取目标物体的坐标
                    self.tot_step += 1
                    return {"obs": obs.astype(np.uint8), "goal": np.array(self.goal, dtype=np.uint8)}, reward, termination, truncation, info

        # 阶段1至num_tasks：完成任务
        if 1 <= self.current_stage <= self.num_tasks:
            for obj_index, object_pair in enumerate(self._objects):
                for color_index, ob in enumerate(object_pair):
                    if self.near(ob):
                        # 检查当前阶段的颜色和目标是否匹配
                        if color_index == color_to_index[self.color_sequence[self.current_stage - 1]] and obj_index == self.obj:
                            self.current_stage += 1  # 进入下一个阶段
                            info['stage'] = self.current_stage
                            
                            if self.current_stage > self.num_tasks:
                                reward = 1000 * (1 - self.tot_step / self.max_step)
                                termination = True  # 所有任务完成
                                self.tot_step += 1
                                state = {"obs": obs.astype(np.uint8), "goal": np.array(self.goal, dtype=np.uint8)}
                                return state, reward, termination, truncation, info
                            else:
                                # 更新目标物体的坐标
                                next_color = self.color_sequence[self.current_stage - 1]
                                obj_type = 'box' if self.obj == 0 else 'key'
                                goal_key = f"{next_color}_{obj_type}"
                                self.goal = color_entity_pos_mapping[goal_key]
                            break

        state = {"obs": obs.astype(np.uint8), "goal": np.array(self.goal, dtype=np.uint8)}
        self.tot_step += 1
        if self.tot_step >= self.max_step:
            truncation = True
        return state, reward, termination, truncation, info


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        # 重置环境状态
        self.current_stage = 0 
        self.tot_step = 0
        obj = random.choice([0, 1])
        self.obj = obj
        self.goal = np.array([self._size, 0, self._size], dtype=np.uint8)  # 定义目标
        self.color_sequence = random.sample(['blue', 'red', 'green'], self.num_tasks)  # 随机化颜色顺序
        # 调用父类的 reset 方法
        obs, info = super().reset(seed=seed, options=options)
        
        # 返回观测值和目标
        oo = {
            'obs': obs.astype(np.uint8),  # 原来的 observation_space
            'goal': self.goal  # 新的 goal 值
        }
        return oo, info # 返回实际的观测值字典和 info 字典
