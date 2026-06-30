"""价值迭代 (Value Iteration)。

有模型算法：直接利用环境的 P / R，反复套用 Bellman 最优算子
    v(s) <- max_a [ R(s,a) + γ Σ_s' P(s'|s,a) v(s') ]
直到 v 收敛，再对最终的 Q 取贪婪得到最优策略。
"""

import numpy as np

from rl_gridworld.agents.base import Agent
from rl_gridworld.core.types import ActionValueFunction, Policy, ValueFunction
from rl_gridworld.env.grid_world import GridWorld


class ValueIteration(Agent):
    def __init__(
        self, gamma: float = 0.9, theta: float = 1e-6, max_iters: int = 1000
    ) -> None:
        self.gamma = gamma
        self.theta = theta  # 收敛阈值
        self.max_iters = max_iters
        # 求解后填充，供可视化使用
        self.V: ValueFunction | None = None
        self.Q: ActionValueFunction | None = None
        self.n_iters: int = 0

    def _q_from_v(self, env: GridWorld, v: np.ndarray) -> np.ndarray:
        """由当前 v 计算 Q(s, a) = R(s,a) + γ Σ_s' P(s'|s,a) v(s')。"""
        return env.R + self.gamma * env.P.dot(v)

    def solve(self, env: GridWorld) -> Policy:
        v = np.zeros(env.n_states)
        for i in range(1, self.max_iters + 1):
            q = self._q_from_v(env, v)
            v_new = q.max(axis=1)
            delta = np.abs(v_new - v).max()
            v = v_new
            self.n_iters = i
            if delta < self.theta:
                break

        q = self._q_from_v(env, v)
        self.V = ValueFunction(v)
        self.Q = ActionValueFunction(q)
        return self.Q.greedy_policy()
