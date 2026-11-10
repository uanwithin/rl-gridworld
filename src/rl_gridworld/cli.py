"""命令行入口。

把「环境 + 算法 + 可视化」粘起来。新增算法时在 ALGORITHMS 里注册即可。
"""

import argparse
from collections.abc import Callable, Sequence

from rl_gridworld.agents.base import Agent
from rl_gridworld.agents.dp.value_iteration import ValueIteration
from rl_gridworld.env.grid_world import build_gridworld
from rl_gridworld.gui.ascii import render_grid, render_policy, render_values

# 算法注册表: name -> 由命令行参数构造 Agent 的函数
ALGORITHMS: dict[str, Callable[[argparse.Namespace], Agent]] = {
    "value_iteration": lambda args: ValueIteration(gamma=args.gamma),
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    env = build_gridworld()
    print("环境布局:")
    print(render_grid(env))

    agent = ALGORITHMS[args.algo](args)
    solution = agent.solve(env)
    print(f"\n[{args.algo}] 迭代 {solution.n_iters} 次")
    if solution.V is not None:
        print("状态价值 V(s):")
        print(render_values(env, solution.V))
    print("策略 (每格的贪婪动作):")
    print(render_policy(env, solution.policy))

    if not args.no_plot:
        from rl_gridworld.gui import plots

        plots.plot_solution(env, solution)
        plots.show()


if __name__ == "__main__":
    main()
