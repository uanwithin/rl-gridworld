"""RL 的公共数据类型。

所有算法的输入/输出都对齐到这里定义的 Policy / ValueFunction /
ActionValueFunction。可视化层只认这些类型，因此任意算法的结果都能用同一套
代码画出来，方便横向对比。
"""

from dataclasses import dataclass

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

    values: np.ndarray


@dataclass
class ActionValueFunction:
    """动作价值 Q(s, a)，(n_states, n_actions)。"""

    values: np.ndarray

    def greedy_policy(self) -> Policy:
        """对 Q 取贪婪得到确定性策略。"""
        return Policy.greedy_from_actions(
            self.values.argmax(axis=1), self.values.shape[1]
        )
