import math
import random
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import utils
from gymnasium.core import ObsType
from gymnasium.spaces import Dict, Discrete

from miniworld.entity import COLOR_NAMES, Box, Key, MeshEnt, TextFrame
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS


class BigKey(Key):
    """A key with a bigger size for better visibility."""

    def __init__(self, color, size=0.6):
        assert color in COLOR_NAMES
        MeshEnt.__init__(self, mesh_name=f"key_{color}", height=size, static=False)


class Sign(MiniWorldEnv, utils.EzPickle):
    def __init__(self, size=10, max_episode_steps=20, num_tasks=3, goal=0, **kwargs):
        self.num_tasks = num_tasks
        self.goal = goal  # 目标在整个任务中保持一致
        self.current_stage = 0
        self.color_sequence = random.sample(['blue', 'red', 'green'], num_tasks)  # 随机化颜色顺序
        self._size = size
        if goal not in [0, 1]:
            raise ValueError("Goals must be 0 (box) or 1 (key).")

        MiniWorldEnv.__init__(
            self,
            params=DEFAULT_PARAMS,
            max_episode_steps=max_episode_steps,
            domain_rand=False,
            **kwargs,
        )
        utils.EzPickle.__init__(self, size, max_episode_steps, num_tasks, goal, **kwargs)

        self.observation_space = Dict(obs=self.observation_space, goal=Discrete(2))
        self.action_space = gym.spaces.Discrete(self.actions.move_forward + 2)

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
        sign = TextFrame(
            pos=[self._size, 1.35, self._size + gap_size],
            dir=math.pi,
            str=text,
            height=1,
        )
        self.entities.append(sign)

        # 确保代理和标志位置不同
        agent_x, agent_z = random.uniform(0, self._size), random.uniform(0, self._size)
        while self._is_close(agent_x, agent_z, self._size, self._size + gap_size):
            agent_x, agent_z = random.uniform(0, self._size), random.uniform(0, self._size)
        self.place_agent(min_x=agent_x, max_x=agent_x+1, min_z=agent_z, max_z=agent_z+1)

    def _is_close(self, x1, z1, x2, z2, threshold=1.0):
        """判断两个点是否接近"""
        return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2) < threshold

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        # 自定义结束动作
        if action == self.actions.move_forward + 1:
            termination = True

        # 当前阶段信息
        info['stage'] = self.current_stage

        # 阶段0：代理发现标志
        if self.current_stage == 0:
            for entity in self.entities:
                if isinstance(entity, TextFrame) and self.near(entity):  # 确保检测到标志
                    self.current_stage += 1
                    info['stage'] = self.current_stage
                    reward = 0.5  # 在找到标志时给予轻微奖励
                    return {"obs": obs, "goal": self.goal}, reward, termination, truncation, info

        # 阶段1至num_tasks：完成任务
        if 1 <= self.current_stage <= self.num_tasks:
            for obj_index, object_pair in enumerate(self._objects):
                for color_index, obj in enumerate(object_pair):
                    if self.near(obj):
                        # 检查当前阶段的颜色和目标是否匹配
                        if self.color_sequence[self.current_stage - 1] == obj.color and obj_index == self.goal:
                            reward = 1.0  # 正确完成任务
                            self.current_stage += 1  # 进入下一个阶段
                            info['stage'] = self.current_stage
                            if self.current_stage > self.num_tasks:
                                termination = True  # 所有任务完成
                        else:
                            reward = -1.0  # 错误任务
                        termination = True  # 触碰到物体后，终止本次尝试

        state = {"obs": obs, "goal": self.goal}
        return state, reward, termination, truncation, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self.current_stage = 0  # 重置当前阶段
        self.color_sequence = random.sample(['blue', 'red', 'green'], self.num_tasks)  # 重置颜色顺序
        obs, info = super().reset(seed=seed, options=options)
        return {"obs": obs, "goal": self.goal}, info
