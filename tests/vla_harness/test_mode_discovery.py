from __future__ import annotations

import numpy as np

from lerobot.vla_harness.mode import build_mode_profile, discover_mode_ids


def test_mode_discovery_uses_episode_boundaries_for_velocity_features():
    states = np.array([[0.0], [0.1], [10.0], [10.1]], dtype=np.float64)
    actions = np.zeros_like(states)
    episode_ids = np.array([0, 0, 1, 1], dtype=np.int64)

    _, features, feature_keys = discover_mode_ids(states, actions, episode_ids)

    velocity_col = feature_keys.index("state_velocity_norm")
    assert features[2, velocity_col] == 0.0


def test_mode_profile_is_data_clustered_not_fixed_tertiles():
    states = np.array(
        [[0.0], [0.0], [0.0], [0.1], [0.2], [0.3], [2.0], [2.4], [2.8]],
        dtype=np.float64,
    )
    actions = np.array(
        [[0.0], [0.0], [0.0], [0.2], [0.2], [0.2], [1.0], [1.0], [1.0]],
        dtype=np.float64,
    )
    episode_ids = np.zeros(len(states), dtype=np.int64)

    profile = build_mode_profile(states, actions, episode_ids)

    assert profile.stable
    assert {mode.mode_id for mode in profile.modes} >= {"mode_plateau", "mode_excursion"}
