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
XAIPipeline: Orchestration of all XAI methods.

This module provides the XAIPipeline class that coordinates all XAI methods
(real-time and offline) during policy evaluation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from .config import XAIConfig
from .buffer import EpisodeXAIBuffer, StepRecord

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


class XAIPipeline:
    """
    Orchestrates XAI methods during policy evaluation.

    This pipeline coordinates real-time XAI methods (P0-V, P1-A, P3-RTC)
    and provides hooks for offline methods (P1-V, P2-A, P2-X, P3-A).

    Usage:
        xai_pipeline = XAIPipeline(policy, xai_config)
        xai_pipeline.wrap_policy(policy)

        # During evaluation, episodes are tracked automatically
        for episode in episodes:
            xai_pipeline.start_episode(episode_id)
            for step in episode.steps():
                action = policy.select_action(observation)
                xai_pipeline.on_after_action(observation, action, step_idx)
            buffer = xai_pipeline.end_episode()
    """

    def __init__(self, policy: PreTrainedPolicy, config: XAIConfig) -> None:
        self.config = config
        self.policy = policy
        self.current_episode: EpisodeXAIBuffer | None = None

        # Initialize real-time method handlers
        self._p0_v = None
        self._p1_a = None
        self._p3_rtc = None

        # Track step count
        self._step_count = 0

    def _verify_policy_support(self) -> None:
        """Verify policy has required architecture for XAI methods."""
        if not hasattr(self.policy, "model"):
            raise ValueError(f"Policy {type(self.policy)} does not have 'model' attribute")

        if not hasattr(self.policy.model, "vlm"):
            raise ValueError(
                f"Policy model does not have VLM (Florence-2) for attention-based XAI. "
                f"Available attributes: {list(vars(self.policy.model))}"
            )

    def _init_real_time_methods(self) -> None:
        """Initialize enabled real-time XAI methods."""
        from lerobot.policies.xai.methods.p0_v_attention_map import P0VAttentionMap
        from lerobot.policies.xai.methods.p1_a_denoising import P1ADenoisingTracker
        from lerobot.policies.xai.methods.p3_rtc_smoothness import P3RTCSmoothnessMonitor

        if self.config.use_p0_v_attention:
            self._p0_v = P0VAttentionMap(self.config, self.policy)

        if self.config.use_p1_a_denoising:
            self._p1_a = P1ADenoisingTracker(self.config, self.policy)

        if self.config.use_p3_rtc_smoothness:
            self._p3_rtc = P3RTCSmoothnessMonitor(self.config, self.policy)

    def wrap_policy(self, policy: PreTrainedPolicy) -> PreTrainedPolicy:
        """
        Wrap a policy to automatically track episodes.

        This wraps the policy's select_action method to automatically
        call on_after_action after each step.

        Args:
            policy: The policy to wrap.

        Returns:
            The wrapped policy.
        """
        self._verify_policy_support()
        self._init_real_time_methods()

        original_select_action = policy.select_action
        step_counter = {"count": 0}

        def wrapped_select_action(batch, **kwargs):
            result = original_select_action(batch, **kwargs)
            self.on_after_action(batch, result, step_counter["count"])
            step_counter["count"] += 1
            return result

        policy.select_action = wrapped_select_action
        return policy

    def start_episode(self, episode_id: str, episode_index: int = 0) -> None:
        """
        Start tracking a new episode.

        Args:
            episode_id: Unique identifier for this episode.
            episode_index: Index of this episode in the evaluation run.
        """
        self.current_episode = EpisodeXAIBuffer(
            episode_id=episode_id,
            episode_index=episode_index,
            timestamp_start=time.time(),
        )
        self._step_count = 0

        # Initialize real-time methods
        if self._p0_v is not None:
            self._p0_v.register()

        if self._p1_a is not None:
            self._p1_a.enable()

        if self._p3_rtc is not None:
            self._p3_rtc.reset()

    def on_after_action(
        self,
        batch: dict,
        action_chunk: torch.Tensor,
        step_idx: int | None = None,
    ) -> None:
        """
        Called after each policy.select_action() call.

        Records XAI metrics for this step.

        Args:
            batch: The observation batch passed to the policy.
            action_chunk: The action chunk returned by the policy.
            step_idx: Step index (auto-incremented if not provided).
        """
        if self.current_episode is None:
            return

        if step_idx is None:
            step_idx = self._step_count

        record = StepRecord(step_idx=step_idx, timestamp=time.time())

        # P0-V: Attention entropy
        if self._p0_v is not None:
            try:
                entropy, compressed = self._p0_v.compute_entropy()
                record.attn_entropy = entropy
                record.attn_compressed = compressed
            except Exception as e:
                logger.warning(f"P0-V attention computation failed: {e}")
            finally:
                self._p0_v.clear()

        # P1-A: Denoising convergence (if tracked externally)
        if self._p1_a is not None:
            try:
                record.convergence_speed = self._p1_a.get_convergence_speed()
            except Exception as e:
                logger.warning(f"P1-A convergence tracking failed: {e}")
            finally:
                self._p1_a.clear()

        # P3-RTC: Boundary smoothness
        if self._p3_rtc is not None:
            try:
                record.boundary_sim = self._p3_rtc.update(action_chunk)
                record.status = self._p3_rtc.get_status(record.boundary_sim)
            except Exception as e:
                logger.warning(f"P3-RTC smoothness computation failed: {e}")

        # Flagging logic
        if record.attn_entropy > self.config.entropy_threshold:
            record.flagged = True
            record.flag_reason = "high_entropy"
        elif getattr(record, "status", None) == "critical_jerk":
            record.flagged = True
            record.flag_reason = "critical_jerk"
        elif getattr(record, "status", None) == "warning_jerk":
            record.flagged = True
            record.flag_reason = "warning_jerk"

        self.current_episode.add_step(record)
        self._step_count += 1

    def end_episode(self) -> EpisodeXAIBuffer | None:
        """
        End the current episode and return the buffer.

        Performs cleanup and computes episode-level statistics.

        Returns:
            EpisodeXAIBuffer with all recorded data, or None if no episode was active.
        """
        if self.current_episode is None:
            return None

        self.current_episode.timestamp_end = time.time()

        # Cleanup P0-V hooks
        if self._p0_v is not None:
            self._p0_v.remove()

        # Compute episode quality
        if self._p3_rtc is not None:
            self.current_episode.episode_quality = self._p3_rtc.episode_quality()

        # Compute mean entropy
        if self.current_episode.step_records:
            entropies = [r.attn_entropy for r in self.current_episode.step_records]
            self.current_episode.mean_entropy = sum(entropies) / len(entropies)

        # Determine training inclusion
        quality_threshold = self.config.quality_threshold
        self.current_episode.should_include_in_training = (
            self.current_episode.episode_quality >= quality_threshold
            if self.current_episode.episode_quality is not None
            else True
        )

        episode = self.current_episode
        self.current_episode = None
        return episode

    def on_denoising_step(self, x_t: torch.Tensor) -> None:
        """
        Track a denoising step (call from within generate_actions loop).

        This should be called by the policy's generate_actions method
        to enable P1-A denoising trajectory tracking.

        Args:
            x_t: The current denoised action tensor.
        """
        if self._p1_a is not None and self._p1_a._enabled:
            self._p1_a.track(x_t)

    def get_current_buffer(self) -> EpisodeXAIBuffer | None:
        """Return the current episode buffer if an episode is active."""
        return self.current_episode

    def has_real_time_methods(self) -> bool:
        """Return True if any real-time methods are enabled."""
        return self._p0_v is not None or self._p1_a is not None or self._p3_rtc is not None
