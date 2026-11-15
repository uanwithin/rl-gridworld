"""命令行入口。

把「环境 + 算法 + 可视化」粘起来。新增算法时在 ALGORITHMS 里注册即可。
"""

import argparse
from collections.abc import Callable, Sequence

import numpy as np

from rl_gridworld.agents.dp.policy_evaluation import evaluate_policy_iterative
from rl_gridworld.agents.dp.policy_iteration import PolicyIteration
from rl_gridworld.agents.dp.value_iteration import ValueIteration
from rl_gridworld.core.types import Policy, Solution
from rl_gridworld.env.grid_world import GridWorld, build_gridworld
from rl_gridworld.gui.ascii import (
    render_grid,
    render_policy,
    render_update,
    render_values,
)


def _evaluate_uniform_policy(env: GridWorld, args: argparse.Namespace) -> Solution:
    policy = Policy(probs=np.full((env.n_states, env.n_actions), 1 / env.n_actions))
    value = evaluate_policy_iterative(
        env, policy, gamma=args.gamma, on_update=args.on_update
    )
    return Solution(policy=policy, V=value)


# 算法注册表: name -> 根据环境和命令行参数返回 Solution 的函数
ALGORITHMS: dict[str, Callable[[GridWorld, argparse.Namespace], Solution]] = {
    "value_iteration": lambda env, args: ValueIteration(gamma=args.gamma).solve(
        env, on_update=args.on_update
    ),
    "policy_evaluation": _evaluate_uniform_policy,
    "policy_iteration": lambda env, args: PolicyIteration(gamma=args.gamma).solve(
        env, on_update=args.on_update
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL GridWorld")
    parser.add_argument(
        "algo",
        nargs="?",
        default="value_iteration",
        choices=list(ALGORITHMS),
        help="要运行的算法",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="折扣因子")
    parser.add_argument("--no-plot", action="store_true", help="只在终端输出，不弹窗")
    parser.add_argument(
        "--quiet", action="store_true", help="仅显示最终结果，隐藏逐轮更新"
    )
    parser.add_argument(
        "--step", action="store_true", help="按键执行一轮迭代并刷新绘图"
    )
    args = parser.parse_args(argv)
    if args.step and args.no_plot:
        parser.error("--step 需要绘图窗口，不能与 --no-plot 同时使用")
    if args.step and not 0 <= args.gamma < 1:
        parser.error("单步模式要求 0 <= gamma < 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    env = build_gridworld()
    print("环境布局:")
    print(render_grid(env))

    args.on_update = (
        None if args.quiet else lambda update: print(render_update(env, update))
    )
    if not args.quiet:
        print("\n更新明细（坐标从 0 开始；价值初值为 0）：")
    if args.step:
        from rl_gridworld.agents.dp.stepper import DPStepper
        from rl_gridworld.gui import plots

        runner = DPStepper(env, args.algo, gamma=args.gamma, on_update=args.on_update)
        controller = plots.plot_steps(runner)
        print(
            "点击图窗后，按空格/回车/右方向键执行一轮，按 R 重置；也可点击 Next 按钮。"
        )
        plots.show()
        del controller
        return

    solution = ALGORITHMS[args.algo](env, args)
    if args.algo == "policy_evaluation":
        print("\n[policy_evaluation] 迭代法评估均匀随机策略")
    else:
        print(f"\n[{args.algo}] 迭代 {solution.n_iters} 次")

    value = solution.V
    if value is not None:
        print("状态价值 V(s):")
        print(render_values(env, value))

    if args.algo == "policy_evaluation":
        print(f"策略：每个状态下，各动作概率均为 {1 / env.n_actions:.0%}")
    else:
        print("策略 (每格的贪婪动作):")
        print(render_policy(env, solution.policy))

    if not args.no_plot:
        from rl_gridworld.gui import plots

        plots.plot_solution(env, solution)
        plots.show()


if __name__ == "__main__":
    main()
