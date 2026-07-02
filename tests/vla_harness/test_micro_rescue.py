from __future__ import annotations

import numpy as np

from lerobot.vla_harness.config import HarnessConfig
from lerobot.vla_harness.rescue import MicroRescuePlanner, build_rescue_index


def test_micro_rescue_returns_expected_neighbor():
    states = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]], dtype=np.float64)
    actions = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1]], dtype=np.float64)
    episodes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    index = build_rescue_index(
        states,
        actions,
        episodes,
        state_scales=np.array([1.0]),
        mode_ids=["m0"] * len(states),
        horizon_steps=2,
    )
    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.micro_rescue.shadow_mode = False
    planner = MicroRescuePlanner(index, cfg)

    decision = planner.query(np.array([0.05]), now_s=10.0)
    assert decision.accepted
    assert decision.snippet is not None
    assert decision.metadata["selected_neighbor"]["episode_index"] == 0


def test_micro_rescue_refuses_ood_state():
    states = np.array([[0.0], [0.1], [0.2]], dtype=np.float64)
    actions = np.array([[0.1], [0.1], [0.1]], dtype=np.float64)
    episodes = np.array([0, 0, 0], dtype=np.int64)
    index = build_rescue_index(
        states,
        actions,
        episodes,
        state_scales=np.array([1.0]),
        mode_ids=["m0"] * len(states),
        horizon_steps=2,
    )
    cfg = HarnessConfig(enable=True)
    cfg.micro_rescue.max_state_distance = 0.05
    planner = MicroRescuePlanner(index, cfg)
    decision = planner.query(np.array([10.0]), now_s=10.0)
    assert not decision.accepted
    assert decision.reason == "no_rescue_neighbor_passed_filters"


def test_micro_rescue_prepends_joint_space_ramp_in():
    states = np.array([[0.0], [0.1], [0.2]], dtype=np.float64)
    actions = np.array([[10.0], [20.0], [30.0]], dtype=np.float64)
    episodes = np.array([0, 0, 0], dtype=np.int64)
    index = build_rescue_index(
        states,
        actions,
        episodes,
        state_scales=np.array([1.0]),
        mode_ids=["m0"] * len(states),
        horizon_steps=2,
    )
    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.micro_rescue.shadow_mode = False
    cfg.micro_rescue.ramp_in_steps = 5
    planner = MicroRescuePlanner(index, cfg, fps=15)

    decision = planner.query(np.array([0.0]), now_s=10.0)

    assert decision.accepted
    assert decision.snippet is not None
    assert np.allclose(decision.snippet[:5, 0], np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
    assert np.allclose(decision.snippet[5, 0], 20.0)
    assert decision.metadata["ramp_in_steps"] == 5
    assert decision.metadata["pre_ramp_first_step_l2"] == 10.0
    assert decision.metadata["post_ramp_first_step_l2"] == 2.0


def test_micro_rescue_ramp_in_respects_max_joint_delta():
    states = np.array([[0.0], [0.1]], dtype=np.float64)
    actions = np.array([[12.0], [13.0]], dtype=np.float64)
    episodes = np.array([0, 0], dtype=np.int64)
    index = build_rescue_index(
        states,
        actions,
        episodes,
        state_scales=np.array([1.0]),
        mode_ids=["m0"] * len(states),
        horizon_steps=2,
    )
    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.micro_rescue.shadow_mode = False
    cfg.micro_rescue.ramp_in_steps = 1
    cfg.micro_rescue.ramp_in_max_joint_delta = 5.0
    planner = MicroRescuePlanner(index, cfg, fps=15)

    decision = planner.query(np.array([0.0]), now_s=10.0)

    assert decision.accepted
    assert decision.snippet is not None
    assert np.allclose(decision.snippet[:3, 0], np.array([4.0, 8.0, 12.0]))
    assert decision.metadata["ramp_in_steps"] == 3
    assert decision.metadata["post_ramp_first_step_l2"] == 4.0


def test_rescue_snippets_do_not_cross_episode_boundary():
    states = np.array([[0.0], [0.1], [10.0], [10.1]], dtype=np.float64)
    actions = np.array([[1.0], [2.0], [10.0], [11.0]], dtype=np.float64)
    episodes = np.array([0, 0, 1, 1], dtype=np.int64)
    index = build_rescue_index(
        states,
        actions,
        episodes,
        state_scales=np.array([1.0]),
        mode_ids=["m0"] * len(states),
        horizon_steps=3,
    )

    assert index.frame_indices.tolist() == [0, 1, 0, 1]
    assert index.snippet_ends[1] == 2
    assert np.allclose(index.action_snippets[1], np.array([[2.0], [2.0], [2.0]]))
