import numpy as np
import pytest

from rl_gridworld.env.grid_world import (
    FORBIDDEN,
    N_ACTIONS,
    N_STATES,
    R_BOUNDARY,
    R_FORBIDDEN,
    R_TARGET,
    TARGET,
    Action,
    GridWorld,
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
    # 状态 0 = 左上角，向上会撞墙
    s2, r = env.step(0, Action.UP)
    assert s2 == 0
    assert r == R_BOUNDARY


def test_actions_move_in_expected_direction():
    env = build_gridworld()
    s = env.cell_to_state((2, 3))
    expected = {
        Action.UP: (1, 3),
        Action.DOWN: (3, 3),
        Action.LEFT: (2, 2),
        Action.RIGHT: (2, 4),
        Action.STAY: (2, 3),
    }
    for a, cell in expected.items():
        assert env.state_to_cell(env.step(s, a)[0]) == cell


def test_state_cell_roundtrip():
    env = build_gridworld()
    for s in range(N_STATES):
        assert env.cell_to_state(env.state_to_cell(s)) == s


def test_reward_entering_target():
    env = build_gridworld()
    target_state = env.cell_to_state(TARGET)
    # 找一个能一步走进 target 的 (s, a)
    found = False
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            s2, r = env.step(s, a)
            if s2 == target_state and s != target_state:
                assert r == R_TARGET
                found = True
    assert found


def test_target_is_not_terminal():
    env = build_gridworld()
    target_state = env.cell_to_state(TARGET)
    assert env.step(target_state, Action.STAY) == (target_state, R_TARGET)


def test_entering_forbidden_is_allowed_but_penalized():
    env = build_gridworld()
    forbidden = FORBIDDEN[0]
    s = env.cell_to_state((forbidden[0] - 1, forbidden[1]))
    s2, r = env.step(s, Action.DOWN)
    assert env.state_to_cell(s2) == forbidden
    assert r == R_FORBIDDEN


def test_custom_layout():
    env = GridWorld(rows=3, cols=4, target=(2, 3), forbidden=((1, 1),))
    assert env.n_states == 12
    assert env.P.shape == (12, N_ACTIONS, 12)
    assert np.allclose(env.P.sum(axis=2), 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": (5, 0)},
        {"forbidden": ((0, -1),)},
        {"target": (1, 1), "forbidden": ((1, 1),)},
    ],
)
def test_invalid_layout_raises(kwargs):
    with pytest.raises(ValueError):
        GridWorld(**kwargs)
