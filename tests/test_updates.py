import numpy as np

from rl_gridworld.agents.dp.policy_evaluation import evaluate_policy_iterative
from rl_gridworld.agents.dp.policy_iteration import PolicyIteration
from rl_gridworld.agents.dp.value_iteration import ValueIteration
from rl_gridworld.cli import main
from rl_gridworld.core.types import Policy
from rl_gridworld.env.grid_world import build_gridworld


def test_value_updates_match_bellman_and_final_result():
    env = build_gridworld()
    updates = []
    result = ValueIteration().solve(env, on_update=updates.append)
    value = result.V
    assert value is not None
    assert len(updates) == result.n_iters
    for index, update in enumerate(updates):
        assert update.iteration == index + 1
        np.testing.assert_allclose(
            update.after, (env.R + 0.9 * env.P @ update.before).max(axis=1)
        )
        if index:
            np.testing.assert_array_equal(update.before, updates[index - 1].after)
    np.testing.assert_array_equal(updates[-1].after, value.values)
    updates[-1].after[:] = 123
    assert not np.all(value.values == 123)


def test_policy_evaluation_updates():
    env = build_gridworld()
    policy = Policy(np.full((env.n_states, env.n_actions), 1 / env.n_actions))
    updates = []
    result = evaluate_policy_iterative(
        env, policy, 0.9, max_iters=3, on_update=updates.append
    )
    assert len(updates) == 3
    for update in updates:
        np.testing.assert_allclose(
            update.after,
            ((env.R + 0.9 * env.P @ update.before) * policy.probs).sum(axis=1),
        )
    np.testing.assert_array_equal(updates[-1].after, result.values)


def test_policy_iteration_reports_probabilities_and_final_evaluation():
    env = build_gridworld()
    updates = []
    result = PolicyIteration(max_iters=1).solve(env, on_update=updates.append)
    value = result.V
    assert value is not None
    assert len(updates) == 3
    np.testing.assert_allclose(updates[1].before, 1 / env.n_actions)
    np.testing.assert_array_equal(updates[1].after, result.policy.probs)
    np.testing.assert_array_equal(updates[-1].after, value.values)


def test_truncated_policy_iteration_reports_each_inner_step():
    updates = []
    PolicyIteration(max_iters=1, eval_steps=3).solve(
        build_gridworld(), on_update=updates.append
    )
    assert len(updates) == 4
    assert "内部第 3 步" in updates[2].phase


def test_cli_updates_and_quiet(capsys):
    main(["--no-plot"])
    output = capsys.readouterr().out
    assert "第 1 轮" in output
    assert "ΔV=" in output
    main(["--no-plot", "--quiet"])
    output = capsys.readouterr().out
    assert "第 1 轮" not in output
    assert "V(s)" in output
