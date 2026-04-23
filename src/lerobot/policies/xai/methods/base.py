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
Base class for XAI methods.

Each XAI method (P0-V, P1-V, P1-A, P2-A, P2-X, P3-A, P3-RTC) inherits from XAIMethod
and implements the abstract interface.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

    from .config import XAIConfig


class XAIMethod(abc.ABC):
    """
    Abstract base class for all XAI methods.

    XAI methods are categorized as:
    - Real-time: Run during inference (P0-V, P1-A, P3-RTC)
    - Offline: Run after episode ends (P1-V, P2-A, P2-X, P3-A)

    Subclasses must implement:
        - name(): Return method identifier string
        - is_realtime(): Return True for real-time methods

    Optional overrides:
        - start_episode(episode_id): Called at episode start
        - on_step(batch, action_chunk, step_idx): Called after each inference step
        - end_episode(): Called at episode end, returns dict of results

    Args:
        config: XAIConfig with method-specific settings
        policy: The policy to analyze
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy) -> None:
        self.config = config
        self.policy = policy

    @abc.abstractmethod
    def name(self) -> str:
        """Return the method name.

        Returns:
            Method identifier, e.g. 'p0_v_attention', 'p3_rtc_smoothness'.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_realtime(self) -> bool:
        """Return True if this is a real-time method.

        Real-time methods are called during inference (per step).
        Offline methods are called after episode ends.

        Returns:
            True for real-time methods, False for offline.
        """
        raise NotImplementedError

    def start_episode(self, episode_id: str, episode_index: int = 0) -> None:
        """
        Called at the start of each episode.

        Args:
            episode_id: Unique episode identifier.
            episode_index: Index of episode in the evaluation run.
        """
        pass

    def on_step(
        self,
        batch: dict,  # noqa: ARG002
        action_chunk: dict,  # noqa: ARG002
        step_idx: int,  # noqa: ARG002
    ) -> None:
        """
        Called after each inference step (real-time methods only).

        Args:
            batch: The observation batch passed to the policy.
            action_chunk: The action chunk returned by the policy.
            step_idx: Index of the current step in the episode.
        """
        pass

    def end_episode(self) -> dict | None:
        """
        Called at the end of each episode.

        For real-time methods, this can compute episode-level summaries.
        For offline methods, this triggers the analysis.

        Returns:
            Dict containing method-specific results, or None.
        """
        return None

    def reset(self) -> None:
        """Reset internal state. Called between episodes."""
        pass
