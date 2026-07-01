from __future__ import annotations

import numpy as np

from lerobot.vla_harness.config import HarnessConfig
from lerobot.vla_harness.envelopes import ActionEnvelopeGuard, build_speed_envelope_profile


def test_speed_envelope_allows_in_distribution_chunk():
    actions = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]], dtype=np.float64)
    profile = build_speed_envelope_profile(actions, ["m0"] * len(actions), 0.0, 1.0)
    guard = ActionEnvelopeGuard(profile, HarnessConfig(enable=True))
    adjusted, violations = guard.evaluate(np.array([0.2]), np.array([[0.2], [0.3]]), "m0")
    assert not violations
    assert np.allclose(adjusted, np.array([[0.2], [0.3]]))


def test_speed_envelope_flags_spike_and_escalates_repeated_clamp():
    actions = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]], dtype=np.float64)
    profile = build_speed_envelope_profile(actions, ["m0"] * len(actions), 0.0, 0.8)
    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.speed_envelope.shadow_mode = False
    cfg.speed_envelope.max_consecutive_clamps = 2
    guard = ActionEnvelopeGuard(profile, cfg)

    _, first = guard.evaluate(np.array([0.0]), np.array([[5.0]]), "m0")
    _, second = guard.evaluate(np.array([0.0]), np.array([[5.0]]), "m0")

    assert any(v.reason == "value_outside_envelope" for v in first)
    assert any(v.reason == "repeated_speed_clamp_requires_flush" for v in second)


def test_speed_envelope_uses_action_state_delta_when_states_are_available():
    states = np.array([[0.0], [10.0]], dtype=np.float64)
    actions = np.array([[0.1], [10.1]], dtype=np.float64)
    profile = build_speed_envelope_profile(
        actions,
        ["m0", "m0"],
        0.0,
        1.0,
        states=states,
        episode_ids=np.array([0, 1], dtype=np.int64),
    )

    assert np.allclose(profile.delta.low, [0.1])
    assert np.allclose(profile.delta.high, [0.1])
