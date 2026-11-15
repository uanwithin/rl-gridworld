"""策略迭代与截断策略迭代 (Policy Iteration)。

利用环境模型 P / R，从均匀随机策略出发，交替执行策略评估和贪婪改进。

1. 策略评估：固定 π，估计其状态价值。
   - eval_steps=None：解 (I - γ P_π) v_π = r_π，得到准确评估值。
   - eval_steps=m：从上一轮的 v 出发，执行 m 次 v <- r_π + γ P_π v。
     这是截断评估，v 不一定已经收敛到当前策略的 v_π。

2. 策略改进：计算 q(s,a;v) = r(s,a) + γ Σ_s' p(s'|s,a) v(s')，
   再令 π_new(s) = argmax_a q(s,a;v)。当 v=v_π 时，q 才是当前策略的 q_π。
   多个动作价值相同时，argmax 选择索引最小的动作。

普通模式在策略不变时停止；对有限折扣 MDP，准确评估下的稳定策略是最优的。
截断模式不能仅凭策略暂时不变停止，而检查贝尔曼最优残差：
    max_s |max_a q(s,a;v) - v(s)| < theta
残差不是价值误差本身；0 <= γ < 1 时，价值最大误差不超过残差 / (1 - γ)。

max_iters 和返回的 n_iters 表示外层“评估 + 改进”的轮数，而非环境交互步数。
普通模式返回策略及其评估值；跑满上限时会额外准确评估最后改进的策略。
截断模式返回最新的贪婪策略和当前近似 v，不额外评估，V 不保证等于该策略的 v_π。
达到上限不保证收敛。max_iters=0 时返回初始随机策略，以及准确评估值（普通模式）
或全零初值（截断模式），不进行策略改进。
"""

import numpy as np

from rl_gridworld.agents.base import Agent
from rl_gridworld.agents.dp.policy_evaluation import (
    evaluate_policy_exact,
    evaluate_policy_iterative,
    q_from_v,
)
from rl_gridworld.core.types import Policy, Solution, ValueFunction
from rl_gridworld.env.grid_world import GridWorld


class PolicyIteration(Agent):
    def __init__(
        self,
        gamma: float = 0.9,
        max_iters: int = 1000,
        eval_steps: int | None = None,
        theta: float = 1e-6,
    ) -> None:
        if not 0 <= gamma < 1:
            raise ValueError("gamma 必须满足 0 <= gamma < 1")
        if (
            isinstance(max_iters, bool)
            or not isinstance(max_iters, (int, np.integer))
            or max_iters < 0
        ):
            raise ValueError("max_iters 必须是非负整数")
        if eval_steps is not None and (
            isinstance(eval_steps, bool)
            or not isinstance(eval_steps, (int, np.integer))
            or eval_steps <= 0
        ):
            raise ValueError("eval_steps 必须是正整数或 None")
        if not np.isfinite(theta) or theta <= 0:
            raise ValueError("theta 必须是有限正数")
        self.gamma = gamma
        self.max_iters = max_iters
        self.eval_steps = eval_steps
        self.theta = theta

    def solve(self, env: GridWorld) -> Solution:
        policy = Policy(probs=np.full((env.n_states, env.n_actions), 1 / env.n_actions))
        value = ValueFunction(values=np.zeros(env.n_states))

        # 外部截断策略评估：是否接近最优价值 v
        #       更新方式：在每个状态，选择动作值最大的动作
        for i in range(self.max_iters):
            if self.eval_steps is None:
                value = evaluate_policy_exact(env, policy, self.gamma)
            else:
                # 内部策略评估：判断的是 v 是否接近于当前策略的 v
                #    更新方式：按照固定策略(\pi)加权平均。
                # 一个很差的策略也能被评估得非常准确，因此“当前策略评估收敛”不代表“已经找到最优策略”
                value = evaluate_policy_iterative(
                    env,
                    policy,
                    self.gamma,
                    theta=0.0,  # 禁用内部提前停止，严格执行 eval_steps 次更新。
                    max_iters=self.eval_steps,
                    v0=value,
                )
            # 先执行当前动作 a，后续回报使用当前的 v 来估计！！！
            q = q_from_v(env, value.values, self.gamma)
            # 选出每个状态的最大动作估计值，这就是新的策略
            actions = np.argmax(q, axis=1)
            # 这个结果，就叫做：贝尔曼最优算子，它就是对当前 v 做一次价值迭代更新
            new_policy = Policy.greedy_from_actions(actions, env.n_actions)

            if self.eval_steps is None:
                if np.array_equal(new_policy.probs, policy.probs):
                    return Solution(policy=policy, V=value, n_iters=i + 1)
            else:
                # 检查价值是否近似满足贝尔曼最优方程，而不是策略是否暂时不变。
                # 这里检查的是做了一次最优价值更新之后，每个状态的价值最多还会变化多少，得到的结果是：贝尔曼最优残差
                residual = np.max(np.abs(q.max(axis=1) - value.values))
                converged = residual < self.theta
                if converged:
                    # 残差针对最优动作计算，因此返回对当前价值贪婪的新策略。
                    return Solution(policy=new_policy, V=value, n_iters=i + 1)

            policy = new_policy

        # 普通模式：准确评估最后一次改进后的策略。
        # 截断模式：保留当前价值估计，不额外评估。
        if self.eval_steps is None:
            value = evaluate_policy_exact(env, policy, self.gamma)

        return Solution(policy=policy, V=value, n_iters=self.max_iters)
