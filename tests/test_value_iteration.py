import numpy as np

from rl_gridworld.agents.dp.value_iteration import ValueIteration
from rl_gridworld.env.grid_world import TARGET, Action, build_gridworld


def test_value_iteration_converges_and_solves():
    env = build_gridworld()
    solution = ValueIteration(gamma=0.9).solve(env)

    assert solution.V is not None
    assert solution.Q is not None
    assert 0 < solution.n_iters < 1000
    assert solution.policy.probs.shape == (env.n_states, env.n_actions)
    assert np.allclose(solution.policy.probs.sum(axis=1), 1.0)

    # 目标状态的价值应达到全局最大 (可能与邻近状态并列)
    target_state = env.cell_to_state(TARGET)
    assert np.isclose(solution.V.values[target_state], solution.V.values.max())


def test_value_iteration_matches_known_optimum():
    env = build_gridworld()
    gamma = 0.9
    solution = ValueIteration(gamma=gamma).solve(env)
    assert solution.V is not None

    # 在目标格一直原地不动: v = 1 + γ + γ² + ... = 1 / (1 - γ)
    target_state = env.cell_to_state(TARGET)
    assert np.isclose(solution.V.values[target_state], 1 / (1 - gamma), atol=1e-4)

    actions = solution.policy.greedy_actions()
    assert actions[target_state] == Action.STAY
    below_target = env.cell_to_state((TARGET[0] + 1, TARGET[1]))
    assert actions[below_target] == Action.UP
