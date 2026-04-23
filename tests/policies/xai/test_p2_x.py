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

"""Tests for P2-X Integrated Gradients method."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import torch

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.p2_x_integrated_gradients import P2XIntegratedGradients


class TestP2XIntegratedGradients:
    """Tests for P2XIntegratedGradients class."""

    def test_name(self):
        """Test method name."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        assert method.name() == "p2_x_integrated_gradients"

    def test_is_realtime(self):
        """Test is_realtime returns False."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        assert method.is_realtime() is False

    def test_initialization(self):
        """Test initialization stores config and policy."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        assert method.config == config
        assert method.policy == mock_policy

    def test_initialization_ig_tensors_none(self):
        """Test ig_vlm and ig_proprio are None initially."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        assert method.ig_vlm is None
        assert method.ig_proprio is None

    def test_end_episode_returns_none_when_no_ig(self):
        """Test end_episode returns None when no IG computed."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        result = method.end_episode()

        assert result is None

    def test_end_episode_returns_ig_when_computed(self):
        """Test end_episode returns IG when available."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)
        method.ig_vlm = torch.zeros(1, 100, 512)
        method.ig_proprio = torch.zeros(1, 14)

        result = method.end_episode()

        assert result is not None
        assert "ig_vlm" in result
        assert "ig_proprio" in result


class TestP2XIntegratedGradientsAttribution:
    """Tests for attribution computation."""

    def test_check_attribution_health_all_ok(self):
        """Test check_attribution_health with healthy attribution."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        ig_result = {
            "vision_pct": 50.0,
            "language_pct": 30.0,
            "proprio_pct": 20.0,
        }

        issues = method.check_attribution_health(ig_result)

        assert len(issues) == 0

    def test_check_attribution_health_high_vision(self):
        """Test check_attribution_health detects over-reliance on vision."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        ig_result = {
            "vision_pct": 80.0,
            "language_pct": 5.0,
            "proprio_pct": 15.0,
        }

        issues = method.check_attribution_health(ig_result)

        assert len(issues) == 2
        assert any("vision" in i.lower() for i in issues)
        assert any("language" in i.lower() for i in issues)

    def test_check_attribution_health_low_language(self):
        """Test check_attribution_health detects low language attribution."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        ig_result = {
            "vision_pct": 70.0,
            "language_pct": 5.0,
            "proprio_pct": 25.0,
        }

        issues = method.check_attribution_health(ig_result)

        assert any("language" in i.lower() for i in issues)

    def test_check_attribution_health_high_proprio(self):
        """Test check_attribution_health detects over-reliance on proprio."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        ig_result = {
            "vision_pct": 40.0,
            "language_pct": 10.0,
            "proprio_pct": 50.0,
        }

        issues = method.check_attribution_health(ig_result)

        assert any("proprio" in i.lower() for i in issues)


class TestP2XIntegratedGradientsEdgeCases:
    """Edge case tests for P2XIntegratedGradients."""

    def test_check_attribution_health_edge_cases(self):
        """Test check_attribution_health handles boundary values."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        ig_result = {
            "vision_pct": 75.0,
            "language_pct": 10.0,
            "proprio_pct": 15.0,
        }

        issues = method.check_attribution_health(ig_result)

        assert len(issues) == 0

    def test_check_attribution_health_exactly_10_language(self):
        """Test check_attribution_health with exactly 10% language."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        ig_result = {
            "vision_pct": 60.0,
            "language_pct": 10.0,
            "proprio_pct": 30.0,
        }

        issues = method.check_attribution_health(ig_result)

        assert len(issues) == 0

    def test_check_attribution_health_exactly_5_proprio(self):
        """Test check_attribution_health with exactly 5% proprio."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)

        ig_result = {
            "vision_pct": 70.0,
            "language_pct": 25.0,
            "proprio_pct": 5.0,
        }

        issues = method.check_attribution_health(ig_result)

        assert len(issues) == 0

    def test_multiple_end_episode_calls(self):
        """Test multiple end_episode calls work correctly."""
        config = XAIConfig(use_p2_x_integrated_gradients=True)
        mock_policy = MagicMock()

        method = P2XIntegratedGradients(config, mock_policy)
        method.ig_vlm = torch.zeros(1, 100, 512)
        method.ig_proprio = torch.zeros(1, 14)

        result1 = method.end_episode()
        result2 = method.end_episode()

        assert result1 is not None
        assert result2 is not None