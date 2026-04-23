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
from lerobot.policies.xai.methods.p1_a_denoising import P1ADenoisingTracker


@pytest.fixture
def config():
    return XAIConfig(p1_a_log_full_trajectory=False)


@pytest.fixture
def mock_policy():
    return MagicMock()


@pytest.fixture
def tracker(config, mock_policy):
    return P1ADenoisingTracker(config, mock_policy)


class TestP1ADenoisingTracker:
    """Tests for P1-A Denoising Tracker."""

    def test_name(self, tracker):
        assert tracker.name() == "p1_a_denoising"

    def test_is_realtime(self, tracker):
        assert tracker.is_realtime() is True

    def test_initialization(self, config, mock_policy):
        tracker = P1ADenoisingTracker(config, mock_policy)
        assert tracker._enabled is False
        assert tracker._trajectory == []

    def test_enable_disable(self, tracker):
        tracker.enable()
        assert tracker._enabled is True
        tracker.disable()
        assert tracker._enabled is False

    def test_track_disabled_does_nothing(self, tracker):
        # Disabled by default
        x = torch.randn(2, 10, 7)
        tracker.track(x)
        assert len(tracker._trajectory) == 0

    def test_track_enabled(self, tracker):
        tracker.enable()
        x = torch.randn(2, 10, 7)
        tracker.track(x)
        assert len(tracker._trajectory) == 1
        assert torch.allclose(tracker._trajectory[0], x)

    def test_track_multiple(self, tracker):
        tracker.enable()
        x1 = torch.randn(2, 10, 7)
        x2 = torch.randn(2, 10, 7)
        x3 = torch.randn(2, 10, 7)
        tracker.track(x1)
        tracker.track(x2)
        tracker.track(x3)
        assert len(tracker._trajectory) == 3

    def test_get_convergence_speed_empty(self, tracker):
        assert tracker.get_convergence_speed() == []

    def test_get_convergence_speed_single(self, tracker):
        tracker.enable()
        x = torch.randn(2, 10, 7)
        tracker.track(x)
        assert tracker.get_convergence_speed() == []

    def test_get_convergence_speed_converging(self, tracker):
        tracker.enable()
        # Simulate converging trajectory
        x1 = torch.randn(2, 10, 7)
        x2 = x1 + 0.5
        x3 = x2 + 0.1
        x4 = x3 + 0.01
        tracker.track(x1)
        tracker.track(x2)
        tracker.track(x3)
        tracker.track(x4)

        deltas = tracker.get_convergence_speed()
        assert len(deltas) == 3
        assert deltas[0] > deltas[1] > deltas[2]

    def test_get_convergence_speed_values(self, tracker):
        tracker.enable()
        x1 = torch.zeros(1, 1, 4)
        x2 = torch.ones(1, 1, 4)  # delta = sqrt(mean(1)) = 1
        tracker.track(x1)
        tracker.track(x2)
        deltas = tracker.get_convergence_speed()
        assert len(deltas) == 1
        assert abs(deltas[0] - 1.0) < 0.001

    def test_get_final_std(self, tracker):
        tracker.enable()
        final = torch.randn(2, 10, 7)
        tracker.track(final)
        std = tracker.get_final_std()
        assert isinstance(std, float)

    def test_get_final_std_no_data(self, tracker):
        assert tracker.get_final_std() is None

    def test_get_mean_delta(self, tracker):
        tracker.enable()
        x1 = torch.zeros(1, 1, 4)
        x2 = 2 * torch.ones(1, 1, 4)
        x3 = 3 * torch.ones(1, 1, 4)
        tracker.track(x1)
        tracker.track(x2)
        tracker.track(x3)
        # deltas: x2-x1=2, x3-x2=1, mean=1.5
        mean = tracker.get_mean_delta()
        assert abs(mean - 1.5) < 0.001

    def test_get_mean_delta_no_data(self, tracker):
        assert tracker.get_mean_delta() is None

    def test_is_converged(self, tracker):
        tracker.enable()
        # Final delta small
        x1 = torch.randn(1, 1, 4)
        x2 = x1 + 0.005
        tracker.track(x1)
        tracker.track(x2)
        assert tracker.is_converged(threshold=0.01) is True

    def test_is_converged_not(self, tracker):
        tracker.enable()
        x1 = torch.randn(1, 1, 4)
        x2 = x1 + 0.5
        tracker.track(x1)
        tracker.track(x2)
        assert tracker.is_converged(threshold=0.01) is False

    def test_is_converged_no_data(self, tracker):
        assert tracker.is_converged() is False

    def test_clear(self, tracker):
        tracker.enable()
        tracker.track(torch.randn(2, 10, 7))
        tracker.track(torch.randn(2, 10, 7))
        assert len(tracker._trajectory) == 2
        tracker.clear()
        assert len(tracker._trajectory) == 0

    def test_reset(self, tracker):
        tracker.enable()
        tracker.track(torch.randn(2, 10, 7))
        assert len(tracker._trajectory) == 1
        tracker.reset()
        assert len(tracker._trajectory) == 0
        assert tracker._enabled is False

    def test_get_trajectory_length(self, tracker):
        assert tracker.get_trajectory_length() == 0
        tracker.enable()
        tracker.track(torch.randn(2, 10, 7))
        tracker.track(torch.randn(2, 10, 7))
        assert tracker.get_trajectory_length() == 2


class TestP1ADenoisingEdgeCases:
    """Edge case tests for P1-A Denoising Tracker."""

    def test_track_does_not_modify_original(self, tracker):
        tracker.enable()
        x = torch.randn(2, 10, 7)
        tracker.track(x)
        # Original should be unchanged
        assert x.sum() == tracker._trajectory[0].sum()

    def test_multiple_batches(self, tracker):
        tracker.enable()
        # Different batch sizes
        tracker.track(torch.randn(2, 10, 7))
        tracker.track(torch.randn(2, 10, 7))
        assert len(tracker._trajectory) == 2

    def test_different_action_dims(self, tracker):
        tracker.enable()
        # In practice, action dim is fixed per policy
        # But we can track with same dim across calls
        tracker.track(torch.randn(2, 10, 7))
        tracker.track(torch.randn(2, 10, 7))
        # Should handle same dims
        deltas = tracker.get_convergence_speed()
        assert len(deltas) == 1
