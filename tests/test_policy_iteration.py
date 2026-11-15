import numpy as np
import pytest

from rl_gridworld.agents.dp.policy_evaluation import evaluate_policy_exact, q_from_v
from rl_gridworld.agents.dp.policy_iteration import PolicyIteration
from rl_gridworld.agents.dp.value_iteration import ValueIteration
from rl_gridworld.env.grid_world import GridWorld


@pytest.mark.parametrize("eval_steps", [None, 1, 5])
def test_matches_optimal_value(eval_steps):
    env = GridWorld()
    result = PolicyIteration(eval_steps=eval_steps, theta=1e-9).solve(env)
    reference = ValueIteration(theta=1e-11).solve(env)
    value = result.V
    reference_value = reference.V
    assert value is not None
    assert reference_value is not None
    np.testing.assert_allclose(value.values, reference_value.values, atol=1e-8)
    # Returned policy should achieve the optimum as well as the value estimate.
    actual = evaluate_policy_exact(env, result.policy, 0.9)
    np.testing.assert_allclose(actual.values, reference_value.values, atol=1e-8)
    residual = np.max(
        np.abs(q_from_v(env, value.values, 0.9).max(axis=1) - value.values)
    )
    assert residual < 1e-9


def test_truncated_early_stop_returns_improved_policy():
    env = GridWorld()
    result = PolicyIteration(eval_steps=1, theta=100).solve(env)
    value = result.V
    assert value is not None
    assert result.n_iters == 1
    assert np.all((result.policy.probs == 0) | (result.policy.probs == 1))
    np.testing.assert_array_equal(
        result.policy.greedy_actions(),
        q_from_v(env, value.values, 0.9).argmax(axis=1),
    )


def test_truncated_limit_does_not_add_evaluation():
    env = GridWorld()
    result = PolicyIteration(eval_steps=1, max_iters=1).solve(env)
    value = result.V
    assert value is not None
    # One update from zero under the initial uniform policy is mean immediate reward.
    np.testing.assert_allclose(value.values, env.R.mean(axis=1))
    assert result.n_iters == 1


@pytest.mark.parametrize("eval_steps", [None, 3])
def test_zero_outer_iterations(eval_steps):
    env = GridWorld()
    result = PolicyIteration(eval_steps=eval_steps, max_iters=0).solve(env)
    value = result.V
    assert value is not None
    assert result.n_iters == 0
    np.testing.assert_allclose(result.policy.probs, 1 / env.n_actions)
    expected = (
        evaluate_policy_exact(env, result.policy, 0.9).values
        if eval_steps is None
        else np.zeros(env.n_states)
    )
    np.testing.assert_allclose(value.values, expected)


def test_exact_limit_keeps_policy_and_value_consistent():
    env = GridWorld()
    result = PolicyIteration(max_iters=1).solve(env)
    value = result.V
    assert value is not None
    np.testing.assert_allclose(
        value.values, evaluate_policy_exact(env, result.policy, 0.9).values
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eval_steps": 0},
        {"eval_steps": -1},
        {"eval_steps": 1.5},
        {"eval_steps": True},
        {"max_iters": -1},
        {"max_iters": 1.5},
        {"gamma": 1},
        {"gamma": -0.1},
        {"theta": 0},
        {"theta": float("nan")},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        PolicyIteration(**kwargs)
