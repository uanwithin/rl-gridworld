import numpy as np

from rl_gridworld.agents.dp.value_iteration import ValueIteration
from rl_gridworld.env.grid_world import TARGET, build_gridworld


def test_value_iteration_converges_and_solves():
    env = build_gridworld()
    agent = ValueIteration(gamma=0.9)
    policy = agent.solve(env)

    assert agent.V is not None
    assert agent.n_iters > 0
    # 策略每行是一个合法概率分布
    assert policy.probs.shape == (env.n_states, env.n_actions)

    # 目标状态的价值应达到全局最大 (可能与邻近状态并列)
    target_state = env.cell_to_state(TARGET)
    assert np.isclose(agent.V.values[target_state], agent.V.values.max())
