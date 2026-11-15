"""matplotlib 可视化。

只依赖 core.types 和环境的地图布局，因此任意算法产出的 Solution 都能直接画。
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from rl_gridworld.agents.dp.stepper import DPStepper
from rl_gridworld.core.types import Policy, Solution, ValueFunction
from rl_gridworld.env.grid_world import ACTION_DELTAS, Action, GridWorld


def _draw_grid(env: GridWorld, ax: Axes) -> None:
    """画网格线并标出目标和禁区。坐标系: x 是列，y 是行，(0,0) 在左上角。"""
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(range(env.cols + 1))
    ax.set_yticks(range(env.rows + 1))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    tr, tc = env.target
    ax.add_patch(Rectangle((tc, tr), 1, 1, color="gold", alpha=0.5))
    for fr, fc in env.forbidden:
        ax.add_patch(Rectangle((fc, fr), 1, 1, color="dimgray", alpha=0.6))


def plot_value(env: GridWorld, value: ValueFunction, ax: Axes | None = None) -> Axes:
    """把状态价值 V(s) 画成热力图。"""
    if ax is None:
        _, ax = plt.subplots()
    grid = value.values.reshape(env.rows, env.cols)
    im = ax.imshow(grid, cmap="viridis", origin="upper")
    for r in range(env.rows):
        for c in range(env.cols):
            ax.text(c, r, f"{grid[r, c]:.1f}", ha="center", va="center", color="w")
    ax.set_title("State Value V(s)")
    ax.figure.colorbar(im, ax=ax)
    return ax


def plot_policy(env: GridWorld, policy: Policy, ax: Axes | None = None) -> Axes:
    """把策略画成箭头图。

    每个动作的箭头长度与 π(a|s) 成正比，原地不动用圆点表示，
    所以确定性策略和 ε-greedy 之类的随机策略都能画。
    """
    if ax is None:
        _, ax = plt.subplots()
    _draw_grid(env, ax)

    for s in range(env.n_states):
        r, c = env.state_to_cell(int(s))
        x, y = c + 0.5, r + 0.5
        for a in Action:
            p = policy.probs[s, a]
            if p < 1e-3:
                continue
            if a == Action.STAY:
                ax.plot(x, y, "o", color="tab:blue", markersize=3 + 5 * p)
                continue
            # y 轴已翻转，行增量可以直接当作 dy
            dr, dc = ACTION_DELTAS[a]
            ax.arrow(
                x, y, dc * 0.35 * p, dr * 0.35 * p,
                head_width=0.12, head_length=0.12, fc="tab:blue", ec="tab:blue",
                length_includes_head=True,
            )  # fmt: skip
    ax.set_title("Policy")
    return ax


def plot_returns(returns: list[float], ax: Axes | None = None) -> Axes:
    """把每个 episode 的累计回报画成学习曲线 (给 model-free 算法用)。"""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.asarray(returns))
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("Learning Curve")
    return ax


def plot_solution(env: GridWorld, solution: Solution) -> None:
    """把 Solution 里已有的内容并排画在一张图上。"""
    panels: list[Callable[[Axes], Axes]] = []
    if (V := solution.V) is not None:
        panels.append(lambda ax: plot_value(env, V, ax))
    panels.append(lambda ax: plot_policy(env, solution.policy, ax))
    if solution.returns:
        panels.append(lambda ax: plot_returns(solution.returns, ax))

    _, axes = plt.subplots(
        1, len(panels), figsize=(5.5 * len(panels), 5), squeeze=False
    )
    for draw, ax in zip(panels, axes[0]):
        draw(ax)


def show() -> None:
    plt.show()


class StepPlot:
    """保留图和按钮引用，通过 GUI 事件驱动一次实际迭代。"""

    def __init__(self, stepper: DPStepper) -> None:
        from matplotlib.widgets import Button

        self.stepper = stepper
        self.figure, self.axes = plt.subplots(1, 3, figsize=(16, 6))
        self.figure.subplots_adjust(bottom=0.23, top=0.80, wspace=0.35)
        self.colorbar = None
        self.next_button = Button(
            self.figure.add_axes((0.36, 0.05, 0.14, 0.07)), "Next [Space]"
        )
        self.reset_button = Button(
            self.figure.add_axes((0.53, 0.05, 0.12, 0.07)), "Reset [R]"
        )
        self.next_button.on_clicked(self.advance)
        self.reset_button.on_clicked(self.reset)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def on_key(self, event) -> None:
        if event.key in (" ", "enter", "right", "n"):
            self.advance()
        elif event.key == "r":
            self.reset()

    def advance(self, event=None) -> None:
        if not self.stepper.done:
            self.stepper.step()
            self.draw()

    def reset(self, event=None) -> None:
        self.stepper.reset()
        self.draw()

    def draw(self) -> None:
        runner = self.stepper
        env, solution = runner.env, runner.solution
        value = solution.V
        if value is None:
            raise RuntimeError("单步算法未提供状态价值")
        if self.colorbar is not None:
            self.colorbar.remove()
        for ax in self.axes:
            ax.clear()
        values, changes, policy = self.axes
        grid = value.values.reshape(env.rows, env.cols)
        values.imshow(grid, cmap="viridis", origin="upper")
        for s, state_value in enumerate(value.values):
            r, c = env.state_to_cell(s)
            values.text(
                c, r, f"{state_value:.4g}", ha="center", va="center", color="white"
            )
        values.set_title("Current V(s)")
        limit = max(float(np.max(np.abs(runner.delta))), 1e-12)
        im = changes.imshow(
            runner.delta.reshape(env.rows, env.cols),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        for s, delta in enumerate(runner.delta):
            r, c = env.state_to_cell(s)
            changes.text(
                c,
                r,
                f"{delta:+.3g}",
                ha="center",
                va="center",
                color="white" if abs(delta) > 0.6 * limit else "black",
            )
        changes.set_title("Change in V(s)")
        self.colorbar = self.figure.colorbar(im, ax=changes)
        plot_policy(env, solution.policy, policy)
        for s in np.flatnonzero(runner.policy_changed):
            r, c = env.state_to_cell(int(s))
            policy.add_patch(
                Rectangle((c, r), 1, 1, fill=False, edgecolor="crimson", linewidth=3)
            )
        policy.set_title("Policy (red border = changed)")
        status = (
            "Converged"
            if runner.converged
            else "Iteration limit reached"
            if runner.done
            else "Ready"
        )
        self.figure.suptitle(
            f"{runner.algorithm} | Iteration {solution.n_iters} | {status}\n"
            f"max |dV| = {np.max(np.abs(runner.delta)):.6g} | "
            "Space / Enter / Right: next round; R: reset"
        )
        note = ""
        if runner.algorithm == "policy_iteration":
            note = "One round = policy evaluation + improvement. V evaluates the previous policy until convergence."
        self.figure.supxlabel(note, y=0.16, fontsize=10)
        self.next_button.set_active(not runner.done)
        self.next_button.label.set_text("Finished" if runner.done else "Next [Space]")
        self.figure.canvas.draw_idle()


def plot_steps(stepper: DPStepper) -> StepPlot:
    return StepPlot(stepper)
