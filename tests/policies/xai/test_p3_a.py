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

"""Tests for P3-A Action Dimension Correlation method."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import torch

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.p3_a_correlation import P3ACorrelation


class TestP3ACorrelation:
    """Tests for P3ACorrelation class."""

    def test_name(self):
        """Test method name."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        assert method.name() == "p3_a_correlation"

    def test_is_realtime(self):
        """Test is_realtime returns False."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        assert method.is_realtime() is False

    def test_initialization(self):
        """Test initialization stores config and policy."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        assert method.config == config
        assert method.policy == mock_policy

    def test_clear_samples(self):
        """Test clear_samples empties the list."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)
        method.action_samples = [torch.zeros(1, 7, 2), torch.zeros(1, 7, 2)]

        method.clear_samples()

        assert len(method.action_samples) == 0

    def test_add_action(self):
        """Test add_action appends to list."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)
        action = torch.zeros(1, 7, 2)

        method.add_action(action)

        assert len(method.action_samples) == 1

    def test_compute_correlation_empty(self):
        """Test compute_correlation returns None values when no samples."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        result = method.compute_correlation()

        assert result["correlation"] is None
        assert result["covariance"] is None
        assert result["std_per_dim"] is None
        assert result["n_samples"] == 0

    def test_compute_correlation_with_samples(self):
        """Test compute_correlation computes matrix from samples."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)
        method.action_samples = [torch.randn(10, 7, 2) for _ in range(5)]

        result = method.compute_correlation()

        assert result["correlation"] is not None
        assert result["covariance"] is not None
        assert result["std_per_dim"] is not None
        assert result["n_samples"] == 50

    def test_compute_correlation_shape(self):
        """Test compute_correlation returns correct matrix shape."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)
        method.action_samples = [torch.randn(10, 7, 2) for _ in range(3)]

        result = method.compute_correlation()

        corr = result["correlation"]
        assert corr.shape == (14, 14)

    def test_end_episode_returns_none_when_empty(self):
        """Test end_episode returns None when no samples."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        result = method.end_episode()

        assert result is None

    def test_end_episode_returns_result_when_samples(self):
        """Test end_episode returns correlation result."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)
        method.action_samples = [torch.randn(10, 7, 2)]

        result = method.end_episode()

        assert result is not None
        assert "correlation" in result


class TestP3ACorrelationSpurious:
    """Tests for spurious correlation detection."""

    def test_detect_spurious_correlations_none_found(self):
        """Test detect_spurious_correlations finds none when expected pairs present."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        corr_matrix = torch.eye(4)

        expected_pairs = [(0, 1), (2, 3)]

        spurious = method.detect_spurious_correlations(corr_matrix, expected_pairs, threshold=0.5)

        assert len(spurious) == 0

    def test_detect_spurious_correlations_finds_unexpected(self):
        """Test detect_spurious_correlations finds unexpected high correlations."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        corr_matrix = torch.zeros(4, 4)
        corr_matrix[0, 1] = 0.8
        corr_matrix[1, 0] = 0.8

        corr_matrix[2, 3] = 0.8
        corr_matrix[3, 2] = 0.8

        expected_pairs = [(2, 3)]

        spurious = method.detect_spurious_correlations(corr_matrix, expected_pairs, threshold=0.7)

        assert len(spurious) == 1
        assert spurious[0][0] == 0
        assert spurious[0][1] == 1

    def test_detect_spurious_correlations_with_no_expected(self):
        """Test detect_spurious_correlations works with no expected pairs."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        corr_matrix = torch.zeros(3, 3)
        corr_matrix[0, 1] = 0.9

        spurious = method.detect_spurious_correlations(corr_matrix, None, threshold=0.7)

        assert len(spurious) == 1


class TestP3ACorrelationCompare:
    """Tests for correlation comparison."""

    def test_compare_with_reference(self):
        """Test compare_with_reference computes Frobenius norm."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        ref = torch.eye(3)
        test = torch.eye(3) * 0.9

        diff_norm = method.compare_with_reference(ref, test)

        assert diff_norm is not None
        assert diff_norm > 0

    def test_compare_with_reference_mismatch_shape(self):
        """Test compare_with_reference returns inf on shape mismatch."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        ref = torch.eye(3)
        test = torch.eye(5)

        diff_norm = method.compare_with_reference(ref, test)

        assert diff_norm == float("inf")

    def test_compare_with_reference_identical(self):
        """Test compare_with_reference returns 0 for identical matrices."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        mat = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

        diff_norm = method.compare_with_reference(mat, mat.clone())

        assert diff_norm < 1e-6


class TestP3ACorrelationEdgeCases:
    """Edge case tests for P3ACorrelation."""

    def test_detect_spurious_correlations_diagonal_ignored(self):
        """Test detect_spurious_correlations ignores diagonal (self-correlation)."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        corr_matrix = torch.eye(3)
        corr_matrix[0, 0] = 1.0

        spurious = method.detect_spurious_correlations(corr_matrix, [], threshold=0.5)

        assert len(spurious) == 0

    def test_compute_correlation_normalizes_properly(self):
        """Test correlation matrix is properly clamped to [-1, 1]."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)

        method.action_samples = [torch.randn(20, 3, 2) for _ in range(3)]

        result = method.compute_correlation()

        corr = result["correlation"]
        assert corr.min() >= -1
        assert corr.max() <= 1

    def test_add_action_detaches_tensor(self):
        """Test add_action detaches the tensor."""
        config = XAIConfig(use_p3_a_correlation=True)
        mock_policy = MagicMock()

        method = P3ACorrelation(config, mock_policy)
        action = torch.zeros(1, 7, 2, requires_grad=True)

        method.add_action(action)

        stored = method.action_samples[0]
        assert not stored.requires_grad