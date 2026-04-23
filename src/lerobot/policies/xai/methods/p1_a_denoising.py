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
P1-A: Denoising Trajectory Tracker.

Real-time method that tracks denoising trajectory during action generation.
Lightweight logging (~0 compute), only memory storage.

This monitors how quickly the model "commits" to an action - slow convergence
indicates uncertainty or ambiguous observations.
"""

from __future__ import annotations

import torch

from lerobot.policies.pretrained import PreTrainedPolicy

from ..config import XAIConfig
from .base import XAIMethod


class P1ADenoisingTracker(XAIMethod):
    """
    Tracks denoising trajectory convergence during action generation.

    Monitors the delta between consecutive x_t states during the flow matching
    denoising process. Fast convergence = confident action selection.

    This is a real-time method with ~0 compute (just memory storage).
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy) -> None:
        super().__init__(config, policy)
        self._trajectory: list[torch.Tensor] = []
        self._enabled = False

    def name(self) -> str:
        return "p1_a_denoising"

    def is_realtime(self) -> bool:
        return True

    def enable(self) -> None:
        """Enable tracking of denoising steps."""
        self._enabled = True

    def disable(self) -> None:
        """Disable tracking."""
        self._enabled = False

    def track(self, x_t: torch.Tensor) -> None:
        """
        Track a denoising step.

        Call this from within the generate_actions() loop to capture
        intermediate x_t states.

        Args:
            x_t: Current denoised action tensor.
        """
        if self._enabled:
            self._trajectory.append(x_t.detach().cpu().clone())

    def get_convergence_speed(self) -> list[float]:
        """
        Compute convergence deltas between consecutive x_t.

        Returns:
            List of delta magnitudes (L2 norm of difference).
            Empty list if less than 2 steps tracked.
        """
        if len(self._trajectory) < 2:
            return []

        deltas = []
        for i in range(1, len(self._trajectory)):
            delta = (self._trajectory[i] - self._trajectory[i - 1]).pow(2).mean().sqrt().item()
            deltas.append(delta)
        return deltas

    def get_final_std(self) -> float | None:
        """
        Compute std of final x_t across chunk timesteps.

        Returns:
            Mean std across batch, or None if no trajectory.
        """
        if not self._trajectory:
            return None
        final = self._trajectory[-1]
        return final.std(dim=1).mean().item()

    def get_mean_delta(self) -> float | None:
        """
        Compute mean delta across all steps.

        Returns:
            Mean delta, or None if insufficient data.
        """
        deltas = self.get_convergence_speed()
        if not deltas:
            return None
        return sum(deltas) / len(deltas)

    def is_converged(self, threshold: float = 0.01) -> bool:
        """
        Check if trajectory has converged based on final delta.

        Args:
            threshold: Delta below which is considered converged.

        Returns:
            True if last delta is below threshold.
        """
        deltas = self.get_convergence_speed()
        if not deltas:
            return False
        return deltas[-1] < threshold

    def clear(self) -> None:
        """Clear trajectory history."""
        self._trajectory.clear()

    def reset(self) -> None:
        """Reset state."""
        self.clear()
        self._enabled = False

    def get_trajectory_length(self) -> int:
        """Return number of tracked steps."""
        return len(self._trajectory)
