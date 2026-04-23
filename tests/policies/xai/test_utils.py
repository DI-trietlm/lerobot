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

from lerobot.policies.xai.utils import attention_entropy, compress_heatmap, normalize_heatmap


class TestAttentionEntropy:
    """Test attention_entropy function."""

    def test_uniform_distribution_high_entropy(self):
        """Uniform attention = maximum entropy."""
        uniform = torch.ones(196) / 196
        entropy = attention_entropy(uniform)
        # Max entropy for 196 tokens = -sum(1/n * log(1/n)) = log(n) ≈ 5.28
        assert entropy > 5.0
        assert entropy < 5.5

    def test_focused_attention_low_entropy(self):
        """Single focused spot = minimum entropy."""
        focused = torch.zeros(196)
        focused[0] = 10.0  # Large value to ensure one-hot after softmax
        entropy = attention_entropy(focused)
        # With 10.0 at index 0, softmax concentrates ~99% on that token
        # entropy ≈ -0.99*log(0.99) - 195*~0.0005*log(~0.0005) ≈ 0.1
        assert entropy < 0.2

    def test_two_spot_attention(self):
        """Two equal spots = low entropy (concentrated on 2 tokens)."""
        two_spot = torch.zeros(196)
        two_spot[0] = 5.0
        two_spot[1] = 5.0  # Equal values -> equal weight after normalization
        entropy = attention_entropy(two_spot)
        # With 2 equal values (50% each after normalization):
        # H = -2 * 0.5 * log(0.5) = log(2) ≈ 0.693
        assert 0.5 < entropy < 1.0

    def test_gradually_focused(self):
        """Test entropy decreases as attention becomes more focused."""
        # Uniform
        uniform = torch.ones(100) / 100
        e_uniform = attention_entropy(uniform)

        # 50% on first token
        half = torch.zeros(100)
        half[:50] = 1.0 / 50
        e_half = attention_entropy(half)

        # 10% on first token
        tenth = torch.zeros(100)
        tenth[:10] = 1.0 / 10
        e_tenth = attention_entropy(tenth)

        assert e_uniform > e_half > e_tenth


class TestCompressHeatmap:
    """Test compress_heatmap function."""

    def test_compress_2d(self):
        """Test compressing a 2D heatmap."""
        heatmap = torch.randn(14, 14)
        compressed = compress_heatmap(heatmap, target_size=(7, 7))
        assert compressed.shape == (7, 7)

    def test_compress_3d(self):
        """Test compressing a 3D heatmap [1, 1, H, W]."""
        heatmap = torch.randn(1, 1, 14, 14)
        compressed = compress_heatmap(heatmap, target_size=(7, 7))
        assert compressed.shape == (7, 7)

    def test_compress_no_change_needed(self):
        """Test when heatmap is already target size."""
        heatmap = torch.randn(7, 7)
        compressed = compress_heatmap(heatmap, target_size=(7, 7))
        assert compressed.shape == (7, 7)

    def test_compress_retains_relative_values(self):
        """Test that compression preserves relative hot/cold spots."""
        heatmap = torch.zeros(14, 14)
        heatmap[0:7, 0:7] = 1.0  # Top-left quadrant is hot
        heatmap[7:, 7:] = 0.0  # Bottom-right is cold

        compressed = compress_heatmap(heatmap, target_size=(7, 7))

        # Top-left block should be hotter than bottom-right
        assert compressed[0:3, 0:3].mean() > compressed[4:, 4:].mean()


class TestNormalizeHeatmap:
    """Test normalize_heatmap function."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        heatmap = torch.tensor([1.0, 2.0, 3.0, 4.0])
        normalized = normalize_heatmap(heatmap)
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert torch.allclose(normalized, torch.tensor([0.0, 1.0/3, 2.0/3, 1.0]))

    def test_normalize_negative_values(self):
        """Test normalization with negative values."""
        heatmap = torch.tensor([-2.0, 0.0, 2.0, 4.0])
        normalized = normalize_heatmap(heatmap)
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_normalize_constant(self):
        """Test normalization of constant tensor."""
        constant = torch.ones(10) * 5.0
        normalized = normalize_heatmap(constant)
        assert torch.all(normalized == 0.0)

    def test_normalize_zeros(self):
        """Test normalization of all zeros."""
        zeros = torch.zeros(10)
        normalized = normalize_heatmap(zeros)
        assert torch.all(normalized == 0.0)

    def test_normalize_inplace_false(self):
        """Test that normalization doesn't modify original."""
        original = torch.tensor([1.0, 5.0, 10.0])
        normalized = normalize_heatmap(original)
        assert original[0] == 1.0
        assert normalized[0] == 0.0
