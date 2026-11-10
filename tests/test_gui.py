import matplotlib

matplotlib.use("Agg")

from rl_gridworld.agents.dp.value_iteration import ValueIteration  # noqa: E402
from rl_gridworld.cli import main  # noqa: E402
from rl_gridworld.env.grid_world import build_gridworld  # noqa: E402
from rl_gridworld.gui import plots  # noqa: E402
from rl_gridworld.gui.ascii import (  # noqa: E402
    C_FORBIDDEN,
    C_TARGET,
    render_grid,
    render_policy,
    render_values,
)


def test_render_grid_marks_layout():
    env = build_gridworld()
    text = render_grid(env)
    assert text.count(C_TARGET) == 1
    assert text.count(C_FORBIDDEN) == len(env.forbidden)
    # 每行格子一行文字 + 上下边框
    assert len(text.splitlines()) == 2 * env.rows + 1


def test_render_solution():
    env = build_gridworld()
    solution = ValueIteration().solve(env)
    assert solution.V is not None
    assert "10.00" in render_values(env, solution.V)
    assert "↑" in render_policy(env, solution.policy)


def test_plot_solution_smoke():
    env = build_gridworld()
    plots.plot_solution(env, ValueIteration().solve(env))
    plots.plt.close("all")


def test_cli_no_plot(capsys):
    main(["--no-plot"])
    out = capsys.readouterr().out
    assert "value_iteration" in out
    assert "V(s)" in out
