from __future__ import annotations

import json

import numpy as np

from lerobot.vla_harness.profile import (
    HarnessProfileMiner,
    HarnessProfileMinerConfig,
    load_harness_profile,
)


def test_profile_miner_builds_reusable_profile(tmp_path, monkeypatch):
    states = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.2, 1.0],
        ],
        dtype=np.float64,
    )
    actions = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [0.1, 0.0],
            [0.1, 0.0],
        ],
        dtype=np.float64,
    )
    episode_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    miner = HarnessProfileMiner(
        HarnessProfileMinerConfig(
            dataset_repo_id="synthetic/test",
            output_dir=str(tmp_path),
            fps=15,
        )
    )
    monkeypatch.setattr(
        miner,
        "_extract_arrays_from_dataset",
        lambda: (states, actions, episode_ids, ["s0", "s1"], ["a0", "a1"], 15),
    )

    bundle = miner.build()
    assert bundle.profile.diagnostics.num_frames == 6
    assert bundle.profile.diagnostics.num_episodes == 2
    assert bundle.profile.rescue_index is not None
    assert bundle.profile.invariants

    miner.export()
    loaded = load_harness_profile(tmp_path / "harness_profile.json")
    assert loaded.profile.dataset_repo_id == "synthetic/test"
    assert loaded.rescue_index is not None
    assert loaded.rescue_index.action_snippets.shape[0] == 6

    payload = json.loads((tmp_path / "harness_profile.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["diagnostics"]["num_frames"] == 6
