# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute the safe start-pose region of a LeRobot dataset.

A policy overfit on consistent demonstrations only behaves in-distribution if the
robot is reset to a pose inside the training start-pose manifold. This module
downloads a dataset's per-episode first-frame ``observation.state`` and summarises
the per-joint distribution (median + IQR + range), so the RTC GUI can auto-fill a
safe reset target and detect when a newer dataset version warrants re-analysis.

Pure (no Tk); safe to import and unit-test on its own.
"""

from __future__ import annotations

import glob
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

# Canonical joint -> substring tokens used to locate its column in the dataset feature names.
# Tokens are specific enough to disambiguate shoulder_pan/shoulder_lift and wrist_flex/wrist_roll.
_JOINT_TOKENS: dict[str, tuple[str, ...]] = {
    "shoulder_pan": ("shoulder_pan",),
    "shoulder_lift": ("shoulder_lift",),
    "elbow_flex": ("elbow",),
    "wrist_flex": ("wrist_flex",),
    "wrist_roll": ("wrist_roll",),
    "gripper": ("gripper", "jaw"),
}

# Per-dataset analysis cache so the GUI can detect "is this still the version I analysed?".
CACHE_PATH = Path.home() / ".cache" / "lerobot" / "rtc_gui_start_pose.json"


@dataclass
class StartPoseStats:
    """Per-joint start-pose distribution for one dataset revision."""

    repo_id: str
    revision: str
    n_episodes: int
    joints: list[str]
    # joint -> {"min","q1","median","q3","max","iqr"}
    stats: dict[str, dict[str, float]]

    def medians(self) -> dict[str, float]:
        return {j: self.stats[j]["median"] for j in self.joints}


def resolve_revision(repo_id: str, token: str | None = None) -> str:
    """Latest commit SHA of the dataset on the Hub (used to detect new versions)."""
    return HfApi().dataset_info(repo_id, token=token).sha


def _match_columns(names: list[str]) -> dict[str, int]:
    """Map canonical joints to their column index in the dataset's state feature names."""
    out: dict[str, int] = {}
    for joint, tokens in _JOINT_TOKENS.items():
        for i, name in enumerate(names):
            lname = name.lower()
            if any(tok in lname for tok in tokens):
                out[joint] = i
                break
    return out


def analyze_start_pose(repo_id: str, revision: str | None = None, token: str | None = None) -> StartPoseStats:
    """Download the dataset and summarise its per-episode start-pose distribution.

    Only ``meta/info.json`` and the data parquet files are fetched (no videos).
    """
    sha = revision or resolve_revision(repo_id, token)
    root = snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=sha,
        token=token,
        allow_patterns=["meta/info.json", "data/**/*.parquet"],
    )
    info = json.loads((Path(root) / "meta" / "info.json").read_text(encoding="utf-8"))
    names = info["features"]["observation.state"]["names"]

    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    if not files:
        raise RuntimeError(f"No data parquet files found for {repo_id} @ {sha[:8]}")
    frames = [
        pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)

    starts = df[df.frame_index == 0].sort_values("episode_index")
    if len(starts) == 0:
        raise RuntimeError(f"No frame_index==0 rows for {repo_id}")
    matrix = np.stack(starts["observation.state"].to_numpy())  # (n_episodes, dof)

    col_for = _match_columns(names)
    if not col_for:
        raise RuntimeError(f"Could not map any joint from state feature names: {names}")

    stats: dict[str, dict[str, float]] = {}
    for joint, idx in col_for.items():
        col = matrix[:, idx].astype(np.float64)
        q1, median, q3 = np.percentile(col, [25, 50, 75])
        stats[joint] = {
            "min": float(col.min()),
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "max": float(col.max()),
            "iqr": float(q3 - q1),
        }

    return StartPoseStats(
        repo_id=repo_id,
        revision=sha,
        n_episodes=int(len(matrix)),
        joints=list(stats.keys()),
        stats=stats,
    )


def load_cached_stats(repo_id: str) -> StartPoseStats | None:
    """Return the last analysed stats for ``repo_id``, or None if never analysed."""
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    entry = data.get(repo_id)
    return StartPoseStats(**entry) if entry else None


def save_stats(stats: StartPoseStats) -> None:
    """Persist ``stats`` (keyed by repo_id) so future sessions can detect new versions."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if CACHE_PATH.exists():
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data[stats.repo_id] = asdict(stats)
    CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
