"""终端纯文本可视化。"""

from rl_gridworld.env.grid_world import (
    FORBIDDEN,
    GRID_COLS,
    GRID_ROWS,
    TARGET,
)

C_AGENT = "🤖"
C_FORBIDDEN = "███"
C_TARGET = "🏁"
C_OTHER = "   "


def render_ascii(agent_state: int | None = None) -> None:
    """用纯文本字符在终端渲染 GridWorld。

    :param agent_state: 当前 agent 的一维状态 (0 ~ N_STATES-1)，None 表示不画 agent。
    """
    print("+" + "---+" * GRID_COLS)
    for r in range(GRID_ROWS):
        row_str = "|"
        for c in range(GRID_COLS):
            cell = (r, c)
            cell_state = r * GRID_COLS + c

            if agent_state is not None and cell_state == agent_state:
                char = C_AGENT
            elif cell == TARGET:
                char = C_TARGET
            elif cell in FORBIDDEN:
                char = C_FORBIDDEN
            else:
                char = C_OTHER

            row_str += f"{char:<2}|"

        print(row_str)
        print("+" + "---+" * GRID_COLS)
    print()
