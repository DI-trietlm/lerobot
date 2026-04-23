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

"""Tests for XAIPipeline orchestration."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import torch

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.buffer import EpisodeXAIBuffer, StepRecord
from lerobot.policies.xai.pipeline import XAIPipeline


class TestXAIPipelineInit:
    """Tests for XAIPipeline initialization."""

    def test_init_with_no_xai_methods(self):
        """Test pipeline initializes with no XAI methods."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)

        assert pipeline.config == config
        assert pipeline.policy == mock_policy
        assert pipeline.current_episode is None
        assert pipeline._p0_v is None
        assert pipeline._p1_a is None
        assert pipeline._p3_rtc is None

    def test_init_stores_policy(self):
        """Test pipeline stores policy reference."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)

        assert pipeline.policy is mock_policy


class TestVerifyPolicySupport:
    """Tests for _verify_policy_support method."""

    def test_raises_when_no_model_attribute(self):
        """Test error when policy has no model attribute."""
        config = XAIConfig()
        mock_policy = MagicMock(spec=[])

        pipeline = XAIPipeline(mock_policy, config)

        with pytest.raises(ValueError, match="does not have 'model' attribute"):
            pipeline._verify_policy_support()

    def test_raises_when_no_vlm_attribute(self):
        """Test error when policy model has no VLM."""
        config = XAIConfig()
        mock_policy = MagicMock()
        mock_policy.model = MagicMock(spec=[])

        pipeline = XAIPipeline(mock_policy, config)

        with pytest.raises(ValueError, match="does not have VLM"):
            pipeline._verify_policy_support()

    def test_passes_with_valid_policy(self):
        """Test passes when policy has model with vlm."""
        config = XAIConfig()
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)

        pipeline._verify_policy_support()


class TestInitRealTimeMethods:
    """Tests for _init_real_time_methods."""

    @patch("lerobot.policies.xai.methods.p0_v_attention_map.P0VAttentionMap")
    def test_inits_p0_v_when_enabled(self, MockP0V):
        """Test P0-V is initialized when enabled."""
        config = XAIConfig(use_p0_v_attention=True)
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._init_real_time_methods()

        assert pipeline._p0_v is not None
        MockP0V.assert_called_once_with(config, mock_policy)

    @patch("lerobot.policies.xai.methods.p1_a_denoising.P1ADenoisingTracker")
    def test_inits_p1_a_when_enabled(self, MockP1A):
        """Test P1-A is initialized when enabled."""
        config = XAIConfig(use_p1_a_denoising=True)
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._init_real_time_methods()

        assert pipeline._p1_a is not None
        MockP1A.assert_called_once_with(config, mock_policy)

    @patch("lerobot.policies.xai.methods.p3_rtc_smoothness.P3RTCSmoothnessMonitor")
    def test_inits_p3_rtc_when_enabled(self, MockP3RTC):
        """Test P3-RTC is initialized when enabled."""
        config = XAIConfig(use_p3_rtc_smoothness=True)
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._init_real_time_methods()

        assert pipeline._p3_rtc is not None
        MockP3RTC.assert_called_once_with(config, mock_policy)

    @patch("lerobot.policies.xai.methods.p0_v_attention_map.P0VAttentionMap")
    @patch("lerobot.policies.xai.methods.p1_a_denoising.P1ADenoisingTracker")
    @patch("lerobot.policies.xai.methods.p3_rtc_smoothness.P3RTCSmoothnessMonitor")
    def test_inits_multiple_methods(self, MockP3RTC, MockP1A, MockP0V):
        """Test multiple methods can be initialized together."""
        config = XAIConfig(
            use_p0_v_attention=True,
            use_p1_a_denoising=True,
            use_p3_rtc_smoothness=True,
        )
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._init_real_time_methods()

        assert pipeline._p0_v is not None
        assert pipeline._p1_a is not None
        assert pipeline._p3_rtc is not None

    def test_inits_no_methods_when_disabled(self):
        """Test no methods initialized when all disabled."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._init_real_time_methods()

        assert pipeline._p0_v is None
        assert pipeline._p1_a is None
        assert pipeline._p3_rtc is None


class TestWrapPolicy:
    """Tests for wrap_policy method."""

    def test_wraps_select_action(self):
        """Test select_action is wrapped."""
        config = XAIConfig()
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()
        original_select_action = MagicMock(return_value=torch.zeros(1, 7))
        mock_policy.select_action = original_select_action

        pipeline = XAIPipeline(mock_policy, config)
        pipeline.wrap_policy(mock_policy)

        assert mock_policy.select_action is not original_select_action

    def test_returns_wrapped_policy(self):
        """Test wrap_policy returns the policy."""
        config = XAIConfig()
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()
        mock_policy.select_action = MagicMock(return_value=torch.zeros(1, 7))

        pipeline = XAIPipeline(mock_policy, config)
        result = pipeline.wrap_policy(mock_policy)

        assert result is mock_policy


class TestStartEpisode:
    """Tests for start_episode method."""

    def test_creates_new_buffer(self):
        """Test start_episode creates new EpisodeXAIBuffer."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline.start_episode("test_episode_1", episode_index=0)

        assert pipeline.current_episode is not None
        assert isinstance(pipeline.current_episode, EpisodeXAIBuffer)
        assert pipeline.current_episode.episode_id == "test_episode_1"
        assert pipeline.current_episode.episode_index == 0

    def test_resets_step_count(self):
        """Test step count is reset on new episode."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._step_count = 999
        pipeline.start_episode("test_episode")

        assert pipeline._step_count == 0

    @patch("lerobot.policies.xai.methods.p0_v_attention_map.P0VAttentionMap")
    def test_registers_p0_v_on_new_episode(self, MockP0V):
        """Test P0-V register is called on new episode."""
        config = XAIConfig(use_p0_v_attention=True)
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()

        mock_p0_v = MagicMock()
        MockP0V.return_value = mock_p0_v

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._init_real_time_methods()
        pipeline.start_episode("test_episode")

        mock_p0_v.register.assert_called_once()

    @patch("lerobot.policies.xai.methods.p1_a_denoising.P1ADenoisingTracker")
    def test_enables_p1_a_on_new_episode(self, MockP1A):
        """Test P1-A enable is called on new episode."""
        config = XAIConfig(use_p1_a_denoising=True)
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()

        mock_p1_a = MagicMock()
        MockP1A.return_value = mock_p1_a

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._init_real_time_methods()
        pipeline.start_episode("test_episode")

        mock_p1_a.enable.assert_called_once()

    @patch("lerobot.policies.xai.methods.p3_rtc_smoothness.P3RTCSmoothnessMonitor")
    def test_resets_p3_rtc_on_new_episode(self, MockP3RTC):
        """Test P3-RTC reset is called on new episode."""
        config = XAIConfig(use_p3_rtc_smoothness=True)
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()

        mock_p3_rtc = MagicMock()
        MockP3RTC.return_value = mock_p3_rtc

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._init_real_time_methods()
        pipeline.start_episode("test_episode")

        mock_p3_rtc.reset.assert_called_once()


class TestOnAfterAction:
    """Tests for on_after_action method."""

    def test_returns_early_when_no_active_episode(self):
        """Test returns early if no episode is active."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline.on_after_action({}, torch.zeros(1, 7), 0)

        assert pipeline._step_count == 0

    def test_increments_step_count(self):
        """Test step count is incremented."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline.start_episode("test_episode")
        initial_count = pipeline._step_count

        pipeline.on_after_action({}, torch.zeros(1, 7), 0)

        assert pipeline._step_count == initial_count + 1

    def test_adds_step_record(self):
        """Test a StepRecord is added to the buffer."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline.start_episode("test_episode")

        action = torch.zeros(1, 7)
        pipeline.on_after_action({}, action, 0)

        assert len(pipeline.current_episode.step_records) == 1
        assert pipeline.current_episode.step_records[0].step_idx == 0

    def test_flags_high_entropy(self):
        """Test step is flagged when entropy exceeds threshold."""
        config = XAIConfig(entropy_threshold=0.5)
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._p0_v = MagicMock()
        pipeline._p0_v.compute_entropy.return_value = (0.8, None)
        pipeline._p0_v.clear.return_value = None

        pipeline.start_episode("test_episode")
        pipeline.on_after_action({}, torch.zeros(1, 7), 0)

        record = pipeline.current_episode.step_records[0]
        assert record.flagged is True
        assert record.flag_reason == "high_entropy"

    def test_flags_critical_jerk(self):
        """Test step is flagged for critical jerk."""
        config = XAIConfig(use_p3_rtc_smoothness=True)
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._p3_rtc = MagicMock()
        pipeline._p3_rtc.update.return_value = 0.1
        pipeline._p3_rtc.get_status.return_value = "critical_jerk"

        pipeline.start_episode("test_episode")
        pipeline.on_after_action({}, torch.zeros(1, 7), 0)

        record = pipeline.current_episode.step_records[0]
        assert record.flagged is True
        assert record.flag_reason == "critical_jerk"


class TestEndEpisode:
    """Tests for end_episode method."""

    def test_returns_none_when_no_active_episode(self):
        """Test end_episode returns None when no episode is active."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        result = pipeline.end_episode()

        assert result is None

    def test_clears_current_episode(self):
        """Test current_episode is cleared after end."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline.start_episode("test_episode")
        pipeline.end_episode()

        assert pipeline.current_episode is None

    @patch("lerobot.policies.xai.methods.p0_v_attention_map.P0VAttentionMap")
    def test_removes_p0_v_hooks(self, MockP0V):
        """Test P0-V hooks are removed."""
        config = XAIConfig(use_p0_v_attention=True)
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()

        mock_p0_v = MagicMock()
        MockP0V.return_value = mock_p0_v

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._init_real_time_methods()
        pipeline.start_episode("test_episode")
        pipeline.end_episode()

        mock_p0_v.remove.assert_called_once()

    @patch("lerobot.policies.xai.methods.p3_rtc_smoothness.P3RTCSmoothnessMonitor")
    def test_computes_episode_quality(self, MockP3RTC):
        """Test episode quality is computed from P3-RTC."""
        config = XAIConfig(use_p3_rtc_smoothness=True, quality_threshold=0.5)
        mock_policy = MagicMock()
        mock_policy.model = MagicMock()
        mock_policy.model.vlm = MagicMock()

        mock_p3_rtc = MagicMock()
        mock_p3_rtc.episode_quality.return_value = 0.8
        MockP3RTC.return_value = mock_p3_rtc

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._init_real_time_methods()
        pipeline.start_episode("test_episode")
        pipeline.end_episode()

        mock_p3_rtc.episode_quality.assert_called_once()


class TestOnDenoisingStep:
    """Tests for on_denoising_step method."""

    def test_tracks_when_p1_a_enabled(self):
        """Test denoising step is tracked."""
        config = XAIConfig(use_p1_a_denoising=True)
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._p1_a = MagicMock()
        pipeline._p1_a._enabled = True

        x_t = torch.zeros(1, 7)
        pipeline.on_denoising_step(x_t)

        pipeline._p1_a.track.assert_called_once_with(x_t)

    def test_does_not_track_when_disabled(self):
        """Test denoising step is not tracked when disabled."""
        config = XAIConfig(use_p1_a_denoising=True)
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._p1_a = MagicMock()
        pipeline._p1_a._enabled = False

        x_t = torch.zeros(1, 7)
        pipeline.on_denoising_step(x_t)

        pipeline._p1_a.track.assert_not_called()


class TestGetCurrentBuffer:
    """Tests for get_current_buffer method."""

    def test_returns_none_when_no_episode(self):
        """Test returns None when no episode is active."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)

        assert pipeline.get_current_buffer() is None

    def test_returns_buffer_when_episode_active(self):
        """Test returns buffer when episode is active."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline.start_episode("test_episode")

        assert pipeline.get_current_buffer() is pipeline.current_episode


class TestHasRealTimeMethods:
    """Tests for has_real_time_methods method."""

    def test_returns_false_when_no_methods(self):
        """Test returns False when no real-time methods."""
        config = XAIConfig()
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)

        assert pipeline.has_real_time_methods() is False

    def test_returns_true_when_p0_v_enabled(self):
        """Test returns True when P0-V is enabled."""
        config = XAIConfig(use_p0_v_attention=True)
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._p0_v = MagicMock()

        assert pipeline.has_real_time_methods() is True

    def test_returns_true_when_p1_a_enabled(self):
        """Test returns True when P1-A is enabled."""
        config = XAIConfig(use_p1_a_denoising=True)
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._p1_a = MagicMock()

        assert pipeline.has_real_time_methods() is True

    def test_returns_true_when_p3_rtc_enabled(self):
        """Test returns True when P3-RTC is enabled."""
        config = XAIConfig(use_p3_rtc_smoothness=True)
        mock_policy = MagicMock()

        pipeline = XAIPipeline(mock_policy, config)
        pipeline._p3_rtc = MagicMock()

        assert pipeline.has_real_time_methods() is True