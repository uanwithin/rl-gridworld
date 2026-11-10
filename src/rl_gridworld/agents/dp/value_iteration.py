"""价值迭代 (Value Iteration)。

有模型算法：直接利用环境的 P / R，反复套用 Bellman 最优算子
    v(s) <- max_a [ R(s,a) + γ Σ_s' P(s'|s,a) v(s') ]
直到 v 收敛，再对最终的 Q 取贪婪得到最优策略。
"""

import warnings

import numpy as np

from rl_gridworld.agents.base import Agent
from rl_gridworld.core.types import ActionValueFunction, Solution, ValueFunction
from rl_gridworld.env.grid_world import GridWorld


class ValueIteration(Agent):
    def __init__(
        self, gamma: float = 0.9, theta: float = 1e-6, max_iters: int = 1000
    ) -> None:
        self.gamma = gamma
        self.theta = theta  # 收敛阈值
        self.max_iters = max_iters

    def _q_from_v(self, env: GridWorld, v: np.ndarray) -> np.ndarray:
        """由当前 v 计算 Q(s, a) = R(s,a) + γ Σ_s' P(s'|s,a) v(s')。"""
        return env.R + self.gamma * env.P @ v

    def solve(self, env: GridWorld) -> Solution:
        v = np.zeros(env.n_states)
        n_iters = 0
        for n_iters in range(1, self.max_iters + 1):
            v_new = self._q_from_v(env, v).max(axis=1)
            delta = np.abs(v_new - v).max()
            v = v_new
            if delta < self.theta:
                break
        else:
            warnings.warn(
                f"Value Iteration 在 {self.max_iters} 轮内未收敛", stacklevel=2
            )

        Q = ActionValueFunction(self._q_from_v(env, v))
        return Solution(
            policy=Q.greedy_policy(), V=ValueFunction(v), Q=Q, n_iters=n_iters
        )
