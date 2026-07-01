from __future__ import annotations

import numpy as np

from lerobot.vla_harness.config import HarnessConfig
from lerobot.vla_harness.invariants import InvariantGuard, InvariantMiner
from lerobot.vla_harness.mode import ModeEstimate, build_mode_profile


def test_invariant_guard_flags_high_confidence_violation():
    states = np.array(
        [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]],
        dtype=np.float64,
    )
    actions = np.array(
        [[0.0], [0.0], [0.1], [1.0], [1.0], [1.0]],
        dtype=np.float64,
    )
    episode_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    profile = build_mode_profile(states, actions, episode_ids)
    invariants = InvariantMiner(min_support=0.0, max_train_violation_rate=1.0).mine(
        states, actions, episode_ids, profile
    )

    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.invariant_guard.min_support = 0.0
    cfg.invariant_guard.max_train_violation_rate = 1.0
    guard = InvariantGuard(invariants, cfg)

    mode_id = profile.modes[0].mode_id
    violations = guard.evaluate_chunk(
        current_state=np.array([0.0]),
        action_chunk=np.array([[10.0]]),
        mode_estimate=ModeEstimate(mode_id=mode_id, confidence=1.0, distances={mode_id: 0.0}),
        mode_history=[mode_id, mode_id],
    )

    assert violations
    assert any(v.reason == "action_outside_mode_envelope" for v in violations)


def test_invariant_guard_stays_shadow_when_mode_confidence_low():
    states = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    actions = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    episode_ids = np.array([0, 0, 0], dtype=np.int64)
    profile = build_mode_profile(states, actions, episode_ids)
    invariants = InvariantMiner(min_support=0.0, max_train_violation_rate=1.0).mine(
        states, actions, episode_ids, profile
    )
    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.invariant_guard.min_support = 0.0
    cfg.invariant_guard.max_train_violation_rate = 1.0
    cfg.invariant_guard.min_mode_confidence = 0.9
    guard = InvariantGuard(invariants, cfg)

    mode_id = profile.modes[0].mode_id
    violations = guard.evaluate_chunk(
        current_state=np.array([0.0]),
        action_chunk=np.array([[1.0]]),
        mode_estimate=ModeEstimate(mode_id=mode_id, confidence=0.1, distances={mode_id: 0.0}),
    )

    assert violations
    assert all(v.severity == "shadow" for v in violations)


def test_invariant_miner_creates_data_derived_actuator_hold_guard():
    states = np.array(
        [
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        dtype=np.float64,
    )
    actions = states.copy()
    episode_ids = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    profile = build_mode_profile(states, actions, episode_ids)
    invariants = InvariantMiner(min_support=1.0, max_train_violation_rate=0.0).mine(
        states, actions, episode_ids, profile
    )

    actuator_invariants = [
        item for item in invariants if item.category == "catastrophic_actuator_release"
    ]
    assert actuator_invariants

    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.invariant_guard.shadow_mode = False
    guard = InvariantGuard(actuator_invariants, cfg)
    violations = guard.evaluate_chunk(
        current_state=np.array([0.0]),
        action_chunk=np.array([[1.0]]),
        mode_estimate=ModeEstimate(mode_id=profile.modes[0].mode_id, confidence=1.0, distances={}),
    )
    assert violations
    assert violations[0].severity == "hard"
