"""按需执行一轮 DP 更新，不预先计算后续结果。"""

import numpy as np

from rl_gridworld.agents.dp.policy_evaluation import (
    evaluate_policy_exact,
    evaluate_policy_iterative,
    q_from_v,
)
from rl_gridworld.core.types import ActionValueFunction, Policy, Solution, ValueFunction
from rl_gridworld.core.updates import UpdateCallback, notify
from rl_gridworld.env.grid_world import GridWorld


class DPStepper:
    """交互用状态机；策略迭代的一轮包含一次准确评估和一次改进。"""

    def __init__(
        self,
        env: GridWorld,
        algorithm: str,
        gamma: float = 0.9,
        theta: float = 1e-6,
        max_iters: int = 1000,
        on_update: UpdateCallback | None = None,
    ) -> None:
        if algorithm not in {
            "value_iteration",
            "policy_evaluation",
            "policy_iteration",
        }:
            raise ValueError(f"不支持的单步算法：{algorithm}")
        if not 0 <= gamma < 1:
            raise ValueError("gamma 必须满足 0 <= gamma < 1")
        if not np.isfinite(theta) or theta <= 0 or max_iters < 1:
            raise ValueError("theta 必须为有限正数，max_iters 必须为正整数")
        self.env, self.algorithm, self.gamma = env, algorithm, gamma
        self.theta, self.max_iters, self.on_update = theta, max_iters, on_update
        self.reset()

    def reset(self) -> None:
        self.solution = Solution(
            policy=Policy(
                np.full((self.env.n_states, self.env.n_actions), 1 / self.env.n_actions)
            ),
            V=ValueFunction(np.zeros(self.env.n_states)),
        )
        self.delta = np.zeros(self.env.n_states)
        self.policy_changed = np.zeros(self.env.n_states, dtype=bool)
        self.done = False
        self.converged = False

    def step(self) -> Solution:
        if self.done:
            return self.solution
        env, previous = self.env, self.solution
        previous_value = previous.V
        if previous_value is None:
            raise RuntimeError("单步算法缺少上一轮状态价值")
        old = previous_value.values
        iteration = previous.n_iters + 1
        policy = previous.policy
        if self.algorithm == "value_iteration":
            value = ValueFunction(q_from_v(env, old, self.gamma).max(axis=1))
            phase = "价值更新"
        elif self.algorithm == "policy_evaluation":
            value = evaluate_policy_iterative(
                env, policy, self.gamma, max_iters=1, v0=previous_value
            )
            phase = "策略评估"
        else:
            value = evaluate_policy_exact(env, policy, self.gamma)
            phase = "策略评估（解析解）"
        notify(self.on_update, iteration, phase, old, value.values)
        q = None
        if self.algorithm != "policy_evaluation":
            q = ActionValueFunction(q_from_v(env, value.values, self.gamma))
            policy = q.greedy_policy()
        self.policy_changed = np.any(policy.probs != previous.policy.probs, axis=1)
        if self.algorithm == "policy_iteration":
            notify(
                self.on_update,
                iteration,
                "策略改进",
                previous.policy.probs,
                policy.probs,
            )
            self.converged = not self.policy_changed.any()
        else:
            self.converged = np.max(np.abs(value.values - old)) < self.theta
        self.done = self.converged or iteration >= self.max_iters
        # 与批量策略迭代一致：达到上限时准确评估最后的策略。
        if self.algorithm == "policy_iteration" and self.done and not self.converged:
            final_value = evaluate_policy_exact(env, policy, self.gamma)
            notify(
                self.on_update,
                iteration,
                "最终策略评估（解析解）",
                value.values,
                final_value.values,
            )
            value = final_value
            q = ActionValueFunction(q_from_v(env, value.values, self.gamma))
        self.delta = value.values - old
        self.solution = Solution(policy=policy, V=value, Q=q, n_iters=iteration)
        return self.solution
