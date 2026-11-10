"""GridWorld 环境。

一个确定性的有限 MDP：agent 在 rows x cols 的网格里移动。

- 撞墙: 留在原地，得到 ``boundary`` 奖励
- 进入禁区: 允许进入，得到 ``forbidden`` 奖励
- 进入目标: 得到 ``target`` 奖励
- 其他: 得到 ``other`` 奖励

目标格不是终止状态：在目标格原地不动会持续拿到目标奖励。
状态按行优先编号: ``s = r * cols + c``，(0, 0) 是左上角。
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

Cell = tuple[int, int]


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


# 每个动作对应的 (d_row, d_col)，按 Action 的值索引
ACTION_DELTAS: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))


@dataclass(frozen=True)
class Rewards:
    boundary: float = -1.0
    forbidden: float = -1.0
    target: float = 1.0
    other: float = 0.0


# 默认地图
GRID_ROWS = 5
GRID_COLS = 5
TARGET: Cell = (3, 2)
FORBIDDEN: tuple[Cell, ...] = ((1, 1), (1, 2), (2, 2), (3, 1), (4, 1))

N_STATES = GRID_ROWS * GRID_COLS
N_ACTIONS = len(Action)

R_BOUNDARY = Rewards.boundary
R_FORBIDDEN = Rewards.forbidden
R_TARGET = Rewards.target
R_OTHER = Rewards.other


class GridWorld:
    """由地图布局推导出完整模型的 GridWorld。

    构造后提供三张表，供有模型算法 (DP) 直接使用：

    - ``P[s, a, s2]``: 转移概率
    - ``R[s, a]``: 即时奖励
    - ``next_states[s, a]``: 确定性下一状态
    """

    def __init__(
        self,
        rows: int = GRID_ROWS,
        cols: int = GRID_COLS,
        target: Cell = TARGET,
        forbidden: tuple[Cell, ...] = FORBIDDEN,
        rewards: Rewards = Rewards(),
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.target = target
        self.forbidden = tuple(forbidden)
        self.rewards = rewards
        self._validate()

        self.next_states = np.zeros((self.n_states, self.n_actions), dtype=int)
        self.R = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in Action:
                next_cell, reward = self._move(self.state_to_cell(s), a)
                self.next_states[s, a] = self.cell_to_state(next_cell)
                self.R[s, a] = reward

        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        s_idx, a_idx = np.indices(self.next_states.shape)
        self.P[s_idx, a_idx, self.next_states] = 1.0

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return len(Action)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def state_to_cell(self, s: int) -> Cell:
        r, c = divmod(s, self.cols)
        return r, c

    def cell_to_state(self, cell: Cell) -> int:
        r, c = cell
        return r * self.cols + c

    def step(self, s: int, a: int) -> tuple[int, float]:
        """在状态 s 执行动作 a，返回 (下一状态, 奖励)。"""
        return int(self.next_states[s, a]), float(self.R[s, a])

    def _move(self, cell: Cell, a: Action) -> tuple[Cell, float]:
        dr, dc = ACTION_DELTAS[a]
        nxt = (cell[0] + dr, cell[1] + dc)
        if not self.in_bounds(nxt):
            return cell, self.rewards.boundary
        if nxt == self.target:
            return nxt, self.rewards.target
        if nxt in self.forbidden:
            return nxt, self.rewards.forbidden
        return nxt, self.rewards.other

    def _validate(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(f"网格尺寸必须为正: {self.rows}x{self.cols}")
        for cell in (self.target, *self.forbidden):
            if not self.in_bounds(cell):
                raise ValueError(f"格子 {cell} 超出 {self.rows}x{self.cols} 网格")
        if self.target in self.forbidden:
            raise ValueError(f"目标 {self.target} 不能同时是禁区")


def build_gridworld() -> GridWorld:
    """构造默认 5x5 地图。"""
    return GridWorld()
