import numpy as np

from rl_gridworld.env.grid_world import (
    N_ACTIONS,
    N_STATES,
    R_BOUNDARY,
    R_TARGET,
    TARGET,
    build_gridworld,
)


def test_shapes():
    env = build_gridworld()
    assert env.P.shape == (N_STATES, N_ACTIONS, N_STATES)
    assert env.R.shape == (N_STATES, N_ACTIONS)
    assert env.n_states == N_STATES
    assert env.n_actions == N_ACTIONS


def test_transition_is_a_distribution():
    env = build_gridworld()
    # 每个 (s, a) 的转移概率之和都应为 1
    assert np.allclose(env.P.sum(axis=2), 1.0)


def test_boundary_keeps_state_and_penalizes():
    env = build_gridworld()
    # 状态 0 = 左上角，向上 (a=0) 会撞墙
    s2, r = env.step(0, 0)
    assert s2 == 0
    assert r == R_BOUNDARY


def test_state_cell_roundtrip():
    env = build_gridworld()
    for s in range(N_STATES):
        assert env.cell_to_state(env.state_to_cell(s)) == s


def test_reward_entering_target():
    env = build_gridworld()
    target_state = build_gridworld().cell_to_state(TARGET)
    # 找一个能一步走进 target 的 (s, a)
    found = False
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            s2, r = env.step(s, a)
            if s2 == target_state and s != target_state:
                assert r == R_TARGET
                found = True
    assert found
