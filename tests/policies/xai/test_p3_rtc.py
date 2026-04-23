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

import pytest
import torch
from unittest.mock import MagicMock

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.p3_rtc_smoothness import P3RTCSmoothnessMonitor


@pytest.fixture
def config():
    return XAIConfig(
        p3_rtc_overlap_steps=3,
        boundary_low_threshold=0.75,
        boundary_critical_threshold=0.5,
    )


@pytest.fixture
def mock_policy():
    return MagicMock()


@pytest.fixture
def monitor(config, mock_policy):
    return P3RTCSmoothnessMonitor(config, mock_policy)


class TestP3RTCSmoothnessMonitor:
    """Tests for P3-RTC Smoothness Monitor."""

    def test_name(self, monitor):
        assert monitor.name() == "p3_rtc_smoothness"

    def test_is_realtime(self, monitor):
        assert monitor.is_realtime() is True

    def test_initialization(self, config, mock_policy):
        monitor = P3RTCSmoothnessMonitor(config, mock_policy)
        assert monitor.overlap_steps == 3
        assert monitor.low_threshold == 0.75
        assert monitor.critical_threshold == 0.5
        assert monitor._prev_chunk is None
        assert monitor._history == []

    def test_update_first_chunk_returns_none(self, monitor):
        chunk = torch.randn(2, 10, 7)
        result = monitor.update(chunk)
        assert result is None
        assert monitor._prev_chunk is not None

    def test_update_similar_chunks(self, monitor):
        # Create identical chunks for maximum similarity
        chunk = torch.ones(2, 10, 7)
        monitor.update(chunk)
        sim = monitor.update(chunk)  # Same chunk
        assert sim is not None
        assert sim > 0.99

    def test_update_opposite_chunks(self, monitor):
        # Nearly opposite chunks should give low similarity
        chunk1 = torch.ones(2, 10, 7)
        chunk2 = -torch.ones(2, 10, 7)  # Directly opposite
        monitor.update(chunk1)
        sim = monitor.update(chunk2)
        assert sim is not None
        assert sim < 0  # Cosine similarity of -1 for exact opposite

    def test_update_history_accumulates(self, monitor):
        chunk = torch.randn(2, 10, 7)
        monitor.update(chunk)
        monitor.update(chunk + 0.1)
        monitor.update(chunk + 0.2)
        assert len(monitor._history) == 2  # 2 transitions
        assert monitor.get_history() == monitor._history

    def test_reset_clears_state(self, monitor):
        chunk = torch.randn(2, 10, 7)
        monitor.update(chunk)
        monitor.update(chunk + 0.1)
        assert len(monitor._history) == 1
        monitor.reset()
        assert monitor._prev_chunk is None
        assert monitor._history == []

    def test_get_status_ok(self, monitor):
        assert monitor.get_status(None) == "ok"
        assert monitor.get_status(0.9) == "ok"
        assert monitor.get_status(0.8) == "ok"
        assert monitor.get_status(0.75) == "ok"

    def test_get_status_warning_jerk(self, monitor):
        assert monitor.get_status(0.74) == "warning_jerk"
        assert monitor.get_status(0.6) == "warning_jerk"
        assert monitor.get_status(0.51) == "warning_jerk"

    def test_get_status_critical_jerk(self, monitor):
        # Strictly below 0.5 is critical
        assert monitor.get_status(0.49) == "critical_jerk"
        assert monitor.get_status(0.3) == "critical_jerk"
        assert monitor.get_status(-0.1) == "critical_jerk"

    def test_episode_quality_no_history(self, monitor):
        assert monitor.episode_quality() == 1.0

    def test_episode_quality_all_smooth(self, monitor):
        monitor._history = [0.9, 0.95, 0.85, 0.8]
        assert monitor.episode_quality() == 1.0

    def test_episode_quality_mixed(self, monitor):
        # 3 smooth (>0.75), 2 jerky (<0.75)
        monitor._history = [0.9, 0.8, 0.7, 0.6, 0.85]
        assert monitor.episode_quality() == 0.6  # 3/5

    def test_episode_quality_all_jerky(self, monitor):
        monitor._history = [0.3, 0.4, 0.2, 0.1]
        assert monitor.episode_quality() == 0.0

    def test_episode_quality_exactly_low_threshold(self, monitor):
        # Exactly at threshold counts as smooth
        monitor._history = [0.75, 0.75, 0.75]
        assert monitor.episode_quality() == 1.0

    def test_episode_quality_just_below_threshold(self, monitor):
        monitor._history = [0.74, 0.74, 0.74]
        assert monitor.episode_quality() == 0.0


class TestP3RTCBoundaryComputation:
    """Tests for boundary similarity computation."""

    def test_boundary_extraction(self, config, mock_policy):
        monitor = P3RTCSmoothnessMonitor(config, mock_policy)
        chunk1 = torch.randn(1, 10, 4)
        chunk2 = torch.randn(1, 10, 4)
        monitor.update(chunk1)
        monitor.update(chunk2)
        # History should have one entry
        assert len(monitor._history) == 1

    def test_multidimensional_actions(self, config, mock_policy):
        monitor = P3RTCSmoothnessMonitor(config, mock_policy)
        # Different action dimensions
        chunk1 = torch.randn(2, 5, 12)
        chunk2 = torch.randn(2, 5, 12)
        monitor.update(chunk1)
        sim = monitor.update(chunk2)
        assert sim is not None
        assert -1.0 <= sim <= 1.0

    def test_batch_processing(self, config, mock_policy):
        monitor = P3RTCSmoothnessMonitor(config, mock_policy)
        batch_size = 4
        chunk1 = torch.randn(batch_size, 10, 7)
        chunk2 = torch.randn(batch_size, 10, 7)
        monitor.update(chunk1)
        sim = monitor.update(chunk2)
        # Should return mean across batch
        assert sim is not None
        assert -1.0 <= sim <= 1.0
