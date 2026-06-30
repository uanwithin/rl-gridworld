"""matplotlib 可视化。

只依赖 core.types，因此任意算法产出的 Policy / ValueFunction 都能直接画。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from rl_gridworld.core.types import Policy, ValueFunction
from rl_gridworld.env.grid_world import (
    FORBIDDEN,
    GRID_COLS,
    GRID_ROWS,
    TARGET,
)

# 每个动作画成箭头的 (dx, dy)，原地动作用一个点表示
_ARROWS = [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)]


def _draw_grid(ax: Axes) -> None:
    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(0, GRID_ROWS)
    ax.set_xticks(range(GRID_COLS + 1))
    ax.set_yticks(range(GRID_ROWS + 1))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # 让 (0,0) 在左上角，符合 grid 习惯


def _cell_center(r: int, c: int) -> tuple[float, float]:
    return c + 0.5, r + 0.5


def plot_value(value: ValueFunction, ax: Axes | None = None) -> Axes:
    """把状态价值 V(s) 画成热力图。"""
    if ax is None:
        _, ax = plt.subplots()
    grid = value.values.reshape(GRID_ROWS, GRID_COLS)
    im = ax.imshow(grid, cmap="viridis", origin="upper")
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            ax.text(c, r, f"{grid[r, c]:.1f}", ha="center", va="center", color="w")
    ax.set_title("State Value V(s)")
    ax.figure.colorbar(im, ax=ax)
    return ax


def plot_policy(policy: Policy, ax: Axes | None = None) -> Axes:
    """把策略画成箭头图 (每个状态画其贪婪动作)。"""
    if ax is None:
        _, ax = plt.subplots()
    _draw_grid(ax)

    # 标注目标和禁区
    tr, tc = TARGET
    ax.add_patch(plt.Rectangle((tc, tr), 1, 1, color="gold", alpha=0.5))
    for fr, fc in FORBIDDEN:
        ax.add_patch(plt.Rectangle((fc, fr), 1, 1, color="dimgray", alpha=0.6))

    actions = policy.greedy_actions()
    for s, a in enumerate(actions):
        r, c = divmod(s, GRID_COLS)
        x, y = _cell_center(r, c)
        dx, dy = _ARROWS[a]
        if dx == 0 and dy == 0:
            ax.plot(x, y, "o", color="tab:blue")
        else:
            ax.arrow(
                x, y, dx * 0.3, dy * 0.3,
                head_width=0.12, head_length=0.12, fc="tab:blue", ec="tab:blue",
            )
    ax.set_title("Greedy Policy")
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


def show() -> None:
    plt.show()
