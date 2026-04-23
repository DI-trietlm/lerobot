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

"""Tests for P2-A Action Sample Bundle method."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import torch

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.p2_a_bundle import P2ABundle


class TestP2ABundle:
    """Tests for P2ABundle class."""

    def test_name(self):
        """Test method name."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)

        assert method.name() == "p2_a_bundle"

    def test_is_realtime(self):
        """Test is_realtime returns False."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)

        assert method.is_realtime() is False

    def test_initialization(self):
        """Test initialization stores config and policy."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)

        assert method.config == config
        assert method.policy == mock_policy

    def test_clear_samples(self):
        """Test clear_samples empties the list."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)
        method.samples = [torch.zeros(1, 7, 2), torch.zeros(1, 7, 2)]

        method.clear_samples()

        assert len(method.samples) == 0

    def test_add_sample(self):
        """Test add_sample appends to list."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)
        action = torch.zeros(1, 7, 2)

        method.add_sample(action)

        assert len(method.samples) == 1

    def test_get_bundle_stats_empty(self):
        """Test get_bundle_stats returns None values when no samples."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)

        stats = method.get_bundle_stats()

        assert stats["mean"] is None
        assert stats["std"] is None
        assert stats["cv"] is None
        assert stats["is_multimodal"] is False
        assert stats["n_samples"] == 0

    def test_get_bundle_stats_with_samples(self):
        """Test get_bundle_stats computes correct statistics."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)
        method.samples = [
            torch.ones(1, 7, 2) * 0.5,
            torch.ones(1, 7, 2) * 1.5,
        ]

        stats = method.get_bundle_stats()

        assert stats["n_samples"] == 2
        assert stats["mean"] is not None
        assert stats["std"] is not None
        assert "cv" in stats

    def test_get_bundle_stats_cv_calculation(self):
        """Test coefficient of variation is computed."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)
        method.samples = [
            torch.ones(1, 7, 2),
            torch.ones(1, 7, 2) * 2,
        ]

        stats = method.get_bundle_stats()

        assert stats["cv"] is not None

    def test_end_episode_returns_stats(self):
        """Test end_episode returns bundle statistics."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)
        method.samples = [torch.zeros(1, 7, 2)]

        result = method.end_episode()

        assert result is not None
        assert "n_samples" in result


class TestP2ABundleEdgeCases:
    """Edge case tests for P2ABundle."""

    def test_add_sample_detaches_tensor(self):
        """Test add_sample detaches the tensor from computation graph."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)
        action = torch.zeros(1, 7, 2, requires_grad=True)

        method.add_sample(action)

        stored = method.samples[0]
        assert not stored.requires_grad

    def test_sample_bundle_uses_config_n_samples(self):
        """Test sample_bundle uses n_samples from config."""
        config = XAIConfig(use_p2_a_bundle=True, p2_a_n_samples=10)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)

        mock_policy._build_model_inputs = MagicMock(return_value={
            "input_ids": torch.zeros(1, 10),
            "image_input": torch.zeros(1, 3, 224, 224),
            "image_mask": torch.ones(1, 10),
            "domain_id": torch.zeros(1),
            "proprio": torch.zeros(1, 14),
        })
        mock_policy.model.forward_vlm = MagicMock(return_value={
            "vlm_features": torch.zeros(1, 100, 512),
            "aux_visual_inputs": torch.zeros(1, 196, 512),
        })
        mock_policy.model.generate_actions = MagicMock(return_value=torch.zeros(1, 7, 2))

        method.sample_bundle({})

        assert mock_policy.model.generate_actions.call_count == 10

    def test_detect_multimodal_import_error(self):
        """Test detect_multimodal handles sklearn ImportError."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)

        with patch.dict("sys.modules", {"sklearn.mixture": None}):
            result = method._detect_multimodal(torch.randn(10, 1, 1, 7))

        assert result is False

    def test_detect_multimodal_insufficient_samples(self):
        """Test detect_multimodal returns False with too few samples."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)

        result = method._detect_multimodal(torch.randn(1, 1, 1, 7), n_clusters=2)

        assert result is False

    def test_multiple_end_episode_calls(self):
        """Test multiple end_episode calls don't accumulate samples."""
        config = XAIConfig(use_p2_a_bundle=True)
        mock_policy = MagicMock()

        method = P2ABundle(config, mock_policy)
        method.samples = [torch.zeros(1, 7, 2)]

        result1 = method.end_episode()
        result2 = method.end_episode()

        assert result1["n_samples"] == result2["n_samples"]