# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class StepRecord:
    """
    Record of XAI metrics for a single inference step.

    Attributes:
        step_idx: Index of the step in the episode.
        timestamp: Unix timestamp when this step was recorded.
        attn_entropy: Attention entropy from P0-V (0 if not computed).
        attn_compressed: Compressed attention heatmap [7, 7] from P0-V.
        convergence_speed: List of convergence deltas from P1-A.
        boundary_sim: Cosine similarity at chunk boundary from P3-RTC.
        status: Status string from P3-RTC ('ok', 'warning_jerk', 'critical_jerk').
        flagged: Whether this step was flagged for offline analysis.
        flag_reason: Reason for flagging if any.
    """

    step_idx: int
    timestamp: float = 0.0
    attn_entropy: float = 0.0
    attn_compressed: torch.Tensor | None = None
    convergence_speed: list[float] = field(default_factory=list)
    boundary_sim: float | None = None
    status: str | None = None
    flagged: bool = False
    flag_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = {
            "step_idx": self.step_idx,
            "timestamp": self.timestamp,
            "attn_entropy": self.attn_entropy,
            "convergence_speed": self.convergence_speed,
            "boundary_sim": self.boundary_sim,
            "status": self.status,
            "flagged": self.flagged,
            "flag_reason": self.flag_reason,
        }
        if self.attn_compressed is not None:
            d["attn_compressed"] = self.attn_compressed.tolist()
        return d


@dataclass
class EpisodeXAIBuffer:
    """
    Buffer containing all XAI records for a single episode.

    This buffer collects step-level data during inference and provides
    methods for episode-level analysis and persistence.

    Attributes:
        episode_id: Unique identifier for this episode.
        episode_index: Index of episode in the evaluation run.
        timestamp_start: Unix timestamp when episode started.
        timestamp_end: Unix timestamp when episode ended.
        step_records: List of StepRecord for each step.
        flagged_steps: List of step indices that were flagged.
        episode_quality: Quality score from P3-RTC (0-1).
        mean_entropy: Mean attention entropy from P0-V.
        attention_stability: Std of attention centroid positions.
    """

    episode_id: str
    episode_index: int = 0
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    step_records: list[StepRecord] = field(default_factory=list)
    flagged_steps: list[int] = field(default_factory=list)
    episode_quality: float | None = None
    mean_entropy: float | None = None
    attention_stability: float | None = None

    # Offline results (set after episode ends)
    gmar_heatmaps: dict[int, torch.Tensor] | None = None
    action_bundle: dict | None = None
    integrated_gradients: dict | None = None
    correlation_matrix: torch.Tensor | None = None

    should_include_in_training: bool = True

    def add_step(self, record: StepRecord) -> None:
        """Add a step record to the buffer."""
        self.step_records.append(record)
        if record.flagged and record.step_idx not in self.flagged_steps:
            self.flagged_steps.append(record.step_idx)

    def compute_summary(self) -> dict[str, Any]:
        """Compute episode-level summary statistics."""
        n_flagged = sum(1 for r in self.step_records if r.flagged)
        if not self.step_records:
            return {
                "episode_id": self.episode_id,
                "episode_index": self.episode_index,
                "n_steps": 0,
                "n_flagged": 0,
                "episode_quality": None,
                "mean_entropy": None,
                "should_include_in_training": True,
            }

        entropies = [r.attn_entropy for r in self.step_records]
        boundary_sims = [r.boundary_sim for r in self.step_records if r.boundary_sim is not None]

        return {
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "n_steps": len(self.step_records),
            "n_flagged": n_flagged,
            "flagged_ratio": n_flagged / len(self.step_records),
            "episode_quality": self.episode_quality,
            "mean_entropy": sum(entropies) / len(entropies) if entropies else None,
            "max_entropy": max(entropies) if entropies else None,
            "mean_boundary_sim": sum(boundary_sims) / len(boundary_sims) if boundary_sims else None,
            "should_include_in_training": self.should_include_in_training,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "duration_s": self.timestamp_end - self.timestamp_start if self.timestamp_end > 0 else None,
        }

    def should_run_offline_xai(self, threshold: float = 0.1) -> bool:
        """Return True if flagged step ratio exceeds threshold."""
        if not self.step_records:
            return False
        return len(self.flagged_steps) / len(self.step_records) > threshold

    def save(self, output_dir: Path) -> None:
        """
        Save episode buffer to disk.

        Creates:
            output_dir/
                episode_{episode_id}/
                    summary.json
                    step_records.json
                    heatmaps/
                        step_{idx}_attn.png (if save_heatmaps)
        """
        ep_dir = output_dir / f"episode_{self.episode_id}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary = self.compute_summary()
        with open(ep_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save step records
        step_data = [r.to_dict() for r in self.step_records]
        with open(ep_dir / "step_records.json", "w") as f:
            json.dump(step_data, f, indent=2)

        # Save heatmaps if present
        if self.has_heatmaps():
            heatmap_dir = ep_dir / "heatmaps"
            heatmap_dir.mkdir(exist_ok=True)
            self._save_heatmaps(heatmap_dir)

    def has_heatmaps(self) -> bool:
        """Return True if any step has attention heatmap."""
        return any(r.attn_compressed is not None for r in self.step_records)

    def _save_heatmaps(self, heatmap_dir: Path) -> None:
        """Save attention heatmaps as PNG files."""
        import numpy as np
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        for record in self.step_records:
            if record.attn_compressed is not None:
                heatmap = record.attn_compressed.numpy()
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(heatmap, cmap="hot", interpolation="nearest")
                ax.axis("off")
                ax.set_title(f"Step {record.step_idx}")
                fig.savefig(heatmap_dir / f"step_{record.step_idx:04d}_attn.png", bbox_inches="tight")
                plt.close(fig)
