"""终端纯文本可视化。所有函数都返回字符串，由调用方决定是否打印。"""

from rl_gridworld.core.types import Policy, ValueFunction
from rl_gridworld.env.grid_world import Action, GridWorld

# emoji 在终端里占两列，所以补一个空格凑成三列
C_AGENT = "🤖 "
C_FORBIDDEN = "███"
C_TARGET = "🏁 "
C_OTHER = "   "

ACTION_SYMBOLS = {
    Action.UP: "↑",
    Action.DOWN: "↓",
    Action.LEFT: "←",
    Action.RIGHT: "→",
    Action.STAY: "o",
}


def _table(env: GridWorld, cells: list[str], width: int) -> str:
    """把按状态编号排列的格子文本拼成带边框的表格。"""
    sep = "+" + ("-" * width + "+") * env.cols
    lines = [sep]
    for r in range(env.rows):
        row = cells[r * env.cols : (r + 1) * env.cols]
        lines.append("|" + "|".join(row) + "|")
        lines.append(sep)
    return "\n".join(lines)


def render_grid(env: GridWorld, agent_state: int | None = None) -> str:
    """渲染地图布局。

    :param agent_state: 当前 agent 的一维状态，None 表示不画 agent。
    """
    cells = []
    for s in range(env.n_states):
        cell = env.state_to_cell(s)
        if s == agent_state:
            cells.append(C_AGENT)
        elif cell == env.target:
            cells.append(C_TARGET)
        elif cell in env.forbidden:
            cells.append(C_FORBIDDEN)
        else:
            cells.append(C_OTHER)
    return _table(env, cells, width=3)


def render_values(env: GridWorld, value: ValueFunction) -> str:
    """把 V(s) 按网格排成数字表。"""
    return _table(env, [f"{v:^7.2f}" for v in value.values], width=7)


def render_policy(env: GridWorld, policy: Policy) -> str:
    """把每个状态的贪婪动作画成箭头。"""
    cells = [f" {ACTION_SYMBOLS[Action(a)]} " for a in policy.greedy_actions()]
    return _table(env, cells, width=3)
