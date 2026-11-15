import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent

from rl_gridworld.agents.dp.stepper import DPStepper
from rl_gridworld.agents.dp.value_iteration import ValueIteration
from rl_gridworld.agents.dp.policy_iteration import PolicyIteration
from rl_gridworld.agents.dp.policy_evaluation import evaluate_policy_iterative
from rl_gridworld.cli import parse_args
from rl_gridworld.env.grid_world import build_gridworld
from rl_gridworld.gui import plots


@pytest.mark.parametrize(
    "algorithm", ["value_iteration", "policy_evaluation", "policy_iteration"]
)
def test_stepper_matches_batch(algorithm):
    env = build_gridworld()
    runner = DPStepper(env, algorithm)
    initial_policy = runner.solution.policy
    assert runner.solution.n_iters == 0
    runner.step()
    assert runner.solution.n_iters == 1
    while not runner.done:
        runner.step()
    result = runner.solution
    result_value = result.V
    assert result_value is not None
    if algorithm == "policy_evaluation":
        expected = evaluate_policy_iterative(env, initial_policy, 0.9)
        np.testing.assert_allclose(result_value.values, expected.values)
    else:
        expected = (
            ValueIteration() if algorithm == "value_iteration" else PolicyIteration()
        ).solve(env)
        expected_value = expected.V
        assert expected_value is not None
        np.testing.assert_allclose(result_value.values, expected_value.values)
        np.testing.assert_array_equal(result.policy.probs, expected.policy.probs)
        assert result.n_iters == expected.n_iters
    assert runner.converged
    assert runner.step() is result
    runner.reset()
    assert runner.solution.n_iters == 0
    assert not runner.done
    reset_value = runner.solution.V
    assert reset_value is not None
    assert not reset_value.values.any()


def test_policy_limit_matches_batch():
    env = build_gridworld()
    runner = DPStepper(env, "policy_iteration", max_iters=1)
    actual = runner.step()
    expected = PolicyIteration(max_iters=1).solve(env)
    actual_value = actual.V
    expected_value = expected.V
    assert actual_value is not None
    assert expected_value is not None
    np.testing.assert_allclose(actual_value.values, expected_value.values)
    assert runner.done and not runner.converged


def test_plot_keyboard_buttons_and_axes():
    runner = DPStepper(build_gridworld(), "value_iteration", max_iters=3)
    controller = plots.plot_steps(runner)
    canvas = controller.figure.canvas
    count = len(controller.figure.axes)

    def key(value):
        canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", canvas, key=value)
        )

    key("x")
    assert runner.solution.n_iters == 0
    for value in [" ", "enter", "right"]:
        key(value)
    assert runner.solution.n_iters == 3
    assert len(controller.figure.axes) == count
    key(" ")
    assert runner.solution.n_iters == 3
    key("r")
    assert runner.solution.n_iters == 0
    controller.advance()
    assert runner.solution.n_iters == 1
    assert len(controller.figure.axes) == count
    plots.plt.close(controller.figure)


def test_step_flag_conflict():
    with pytest.raises(SystemExit):
        parse_args(["--step", "--no-plot"])
