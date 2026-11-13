"""策略评估 (Policy Evaluation)。

给定策略 π，求它的状态值 v_π，也就是求解贝尔曼方程
    v_π = r_π + γ P_π v_π
有两种解法：

- 解析解: v_π = (I - γ P_π)^{-1} r_π
- 迭代解: v_{j+1} = r_π + γ P_π v_j，从任意 v_0 出发收敛到 v_π

迭代解有两个旋钮：初值 v_0 和步数。跑到收敛是策略迭代的评估步，
从上一轮的值热启动、只跑有限步是截断策略迭代的评估步。
"""

import numpy as np

from rl_gridworld.core.types import Policy, ValueFunction
from rl_gridworld.env.grid_world import GridWorld


def q_from_v(env: GridWorld, v: np.ndarray, gamma: float) -> np.ndarray:
    """向前看一步: q(s,a) = r(s,a) + γ Σ_s' p(s'|s,a) v(s')，(n_states, n_actions)。

    传入 v_π 得到 q_π；传入值迭代的中间向量 v_k，得到的只是一个近似。
    """
    return env.R + gamma * env.P @ v


def _policy_reward(env: GridWorld, policy: Policy) -> np.ndarray:
    """按策略加权的即时奖励 r_π(s) = Σ_a π(a|s) r(s,a)，(n_states,)。"""
    raise NotImplementedError


def _policy_transition(env: GridWorld, policy: Policy) -> np.ndarray:
    """按策略加权的状态转移 P_π[s, s'] = Σ_a π(a|s) p(s'|s,a)，(n_states, n_states)。

    每行是关于 s' 的概率分布，和为 1。
    """
    raise NotImplementedError


def evaluate_policy_exact(
    env: GridWorld, policy: Policy, gamma: float
) -> ValueFunction:
    """解析解: 解线性方程组 (I - γ P_π) v = r_π。"""
    raise NotImplementedError


def evaluate_policy_iterative(
    env: GridWorld,
    policy: Policy,
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 1000,
    v0: ValueFunction | None = None,
) -> ValueFunction:
    """迭代解: 从 v0 (默认全 0) 出发反复执行 v <- r_π + γ P_π v，
    直到 max|v_new - v| < theta 或跑满 max_iters 轮。

    跑满 max_iters 时直接返回当前估计，不报警告：截断策略迭代就是靠很小的
    max_iters 只做有限步评估。
    """
    raise NotImplementedError
