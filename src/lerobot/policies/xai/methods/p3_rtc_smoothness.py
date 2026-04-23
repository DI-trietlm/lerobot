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

"""
P3-RTC: Chunk Boundary Smoothness Monitor.

Real-time method that computes cosine similarity between chunk boundary transitions.
Low compute cost (~0), runs during inference.

This detects "jerk" - sudden changes in action at chunk boundaries which can
be dangerous for robot hardware and indicate model uncertainty.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lerobot.policies.pretrained import PreTrainedPolicy

from ..config import XAIConfig
from .base import XAIMethod


class P3RTCSmoothnessMonitor(XAIMethod):
    """
    Computes cosine similarity at action chunk boundaries.

    Compares the tail of the previous chunk with the head of the new chunk.
    High similarity (> 0.75) = smooth transition.
    Low similarity (< 0.5) = critical jerk.

    This is a real-time method with ~0 compute cost.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy) -> None:
        super().__init__(config, policy)
        self.overlap_steps = config.p3_rtc_overlap_steps
        self.low_threshold = config.boundary_low_threshold
        self.critical_threshold = config.boundary_critical_threshold
        self._prev_chunk: torch.Tensor | None = None
        self._history: list[float] = []

    def name(self) -> str:
        return "p3_rtc_smoothness"

    def is_realtime(self) -> bool:
        return True

    def reset(self) -> None:
        """Reset state between episodes."""
        self._prev_chunk = None
        self._history = []

    def update(self, new_chunk: torch.Tensor) -> float | None:
        """
        Update with new action chunk and compute boundary similarity.

        Args:
            new_chunk: Action chunk [B, chunk_size, dim_action]

        Returns:
            Cosine similarity score, or None if insufficient history.
        """
        if self._prev_chunk is None:
            self._prev_chunk = new_chunk.detach().clone()
            return None

        prev_tail = self._prev_chunk[:, -self.overlap_steps:, :]
        new_head = new_chunk[:, :self.overlap_steps, :]

        sim = F.cosine_similarity(
            prev_tail.flatten(1),
            new_head.flatten(1),
            dim=1
        ).mean().item()

        self._history.append(sim)
        self._prev_chunk = new_chunk.detach().clone()

        return sim

    def get_status(self, sim: float | None) -> str:
        """
        Get status string based on similarity score.

        Args:
            sim: Cosine similarity score from update()

        Returns:
            Status string: 'ok', 'warning_jerk', or 'critical_jerk'
        """
        if sim is None:
            return "ok"
        if sim < self.critical_threshold:
            return "critical_jerk"
        if sim < self.low_threshold:
            return "warning_jerk"
        return "ok"

    def episode_quality(self) -> float:
        """
        Compute episode quality score based on boundary smoothness.

        Returns:
            Ratio of smooth transitions (>= low_threshold) to total transitions.
        """
        if not self._history:
            return 1.0
        good = sum(s >= self.low_threshold for s in self._history)
        return good / len(self._history)

    def get_history(self) -> list[float]:
        """Return the history of similarity scores."""
        return self._history.copy()
