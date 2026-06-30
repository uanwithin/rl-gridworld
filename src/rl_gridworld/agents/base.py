"""算法基类。

每个 RL 算法都实现 :meth:`solve`，吃进一个环境，吐出一个 :class:`Policy`。
统一这个契约后，训练脚本和可视化层就能以同样的方式驱动任意算法。
"""

from abc import ABC, abstractmethod

from rl_gridworld.core.types import Policy
from rl_gridworld.env.grid_world import GridWorld


class Agent(ABC):
    """所有算法的抽象基类。"""

    @abstractmethod
    def solve(self, env: GridWorld) -> Policy:
        """求解环境，返回学到的策略。"""
        raise NotImplementedError
