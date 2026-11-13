"""RL 的公共数据类型。

所有算法的输出都对齐到这里定义的 Solution (Policy / ValueFunction /
ActionValueFunction)。可视化层只认这些类型，因此任意算法的结果都能用同一套
代码画出来，方便横向对比。
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Policy:
    """策略 π(a|s)。

    用一张 (n_states, n_actions) 的概率表表示。确定性策略就是每行一个 1。
    """

    # (n_states, n_actions) -> float，每行和为 1
    probs: np.ndarray

    @property
    def n_states(self) -> int:
        return self.probs.shape[0]

    @property
    def n_actions(self) -> int:
        return self.probs.shape[1]

    @classmethod
    def greedy_from_actions(cls, actions: np.ndarray, n_actions: int) -> "Policy":
        """由每个状态的贪婪动作 (n_states,) 构造确定性策略。"""
        n_states = len(actions)
        probs = np.zeros((n_states, n_actions))
        probs[np.arange(n_states), actions] = 1.0
        return cls(probs)

    def greedy_actions(self) -> np.ndarray:
        """每个状态概率最大的动作，(n_states,)。"""
        return self.probs.argmax(axis=1)


@dataclass
class ValueFunction:
    """状态价值 V(s)，(n_states,)。"""

    # 这里自己有个误解的点，自己以为状态值是这一个状态的，实际并不是
    # 表示的是：从 s 出发、按策略 π 行动，所能得到的折扣回报的期望
    values: np.ndarray


@dataclass
class ActionValueFunction:
    """动作价值 Q(s, a)，(n_states, n_actions)。"""

    values: np.ndarray

    # 动作值也一样
    # 表示的是：第一步做 a，之后按策略 π 走
    def greedy_policy(self) -> Policy:
        """对 Q 取贪婪得到确定性策略。"""
        return Policy.greedy_from_actions(
            self.values.argmax(axis=1), self.values.shape[1]
        )


@dataclass
class Solution:
    """一次求解的结果。不同算法填充不同字段，可视化按需取用。"""

    policy: Policy
    V: ValueFunction | None = None
    Q: ActionValueFunction | None = None
    # DP 算法是迭代轮数，model-free 算法是 episode 数
    n_iters: int = 0
    # 每个 episode 的累计回报，用来画学习曲线
    returns: list[float] = field(default_factory=list)
