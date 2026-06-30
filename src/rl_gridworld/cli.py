"""命令行入口。

把「环境 + 算法 + 可视化」粘起来。新增算法时在 ALGORITHMS 里注册即可。
"""

import argparse

from rl_gridworld.agents.base import Agent
from rl_gridworld.agents.dp.value_iteration import ValueIteration
from rl_gridworld.env.grid_world import build_gridworld
from rl_gridworld.gui import plots
from rl_gridworld.gui.ascii import render_ascii

# 算法注册表: name -> 构造函数
ALGORITHMS: dict[str, type[Agent]] = {
    "value_iteration": ValueIteration,
}


def main() -> None:
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
    args = parser.parse_args()

    env = build_gridworld()
    print("环境布局:")
    render_ascii()

    agent = ALGORITHMS[args.algo](gamma=args.gamma)  # type: ignore
    policy = agent.solve(env)
    print(f"[{args.algo}] 收敛于 {getattr(agent, 'n_iters', '?')} 次迭代")

    if not args.no_plot:
        _, axes = plots.plt.subplots(1, 2, figsize=(11, 5))
        if agent.V is not None:  # type: ignore
            plots.plot_value(agent.V, ax=axes[0])  # type: ignore
        plots.plot_policy(policy, ax=axes[1])
        plots.show()


if __name__ == "__main__":
    main()
