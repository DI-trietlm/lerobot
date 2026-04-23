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

"""Tests for P1-V GMAR method."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import torch

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.p1_v_gmar import P1VGMAR


class TestP1VGMAR:
    """Tests for P1VGMAR class."""

    def test_name(self):
        """Test method name."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        method = P1VGMAR(config, mock_policy)

        assert method.name() == "p1_v_gmar"

    def test_is_realtime(self):
        """Test is_realtime returns False."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        method = P1VGMAR(config, mock_policy)

        assert method.is_realtime() is False

    def test_initialization(self):
        """Test initialization stores config and policy."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        method = P1VGMAR(config, mock_policy)

        assert method.config == config
        assert method.policy == mock_policy

    def test_find_encoder_layers_empty(self):
        """Test _find_encoder_layers returns empty when no VLM."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock(spec=[])

        method = P1VGMAR(config, mock_policy)
        layers = method._find_encoder_layers()

        assert layers == []

    def test_find_encoder_layers_with_vlm(self):
        """Test _find_encoder_layers finds layers in VLM."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        mock_encoder = MagicMock()
        mock_encoder.layers = [MagicMock(), MagicMock(), MagicMock()]
        mock_vlm = MagicMock()
        mock_vlm.language_model.model.encoder = mock_encoder
        mock_policy.model.vlm = mock_vlm

        method = P1VGMAR(config, mock_policy)
        layers = method._find_encoder_layers()

        assert len(layers) == 3

    def test_register_hooks_idempotent(self):
        """Test hooks can be registered and removed idempotently."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        mock_encoder = MagicMock()
        mock_layer = MagicMock()
        mock_layer.self_attn = MagicMock()
        mock_layer.self_attn.register_forward_hook = MagicMock(return_value=MagicMock())
        mock_encoder.layers = [mock_layer]
        mock_vlm = MagicMock()
        mock_vlm.language_model.model.encoder = mock_encoder
        mock_policy.model.vlm = mock_vlm

        method = P1VGMAR(config, mock_policy)

        method._register_hooks(layer_indices=[0])
        method._register_hooks(layer_indices=[0])

        assert len(method.hooks) == 2

        method._remove_hooks()

        assert len(method.hooks) == 0

    def test_remove_hooks_empty(self):
        """Test _remove_hooks works when no hooks registered."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        method = P1VGMAR(config, mock_policy)
        method._remove_hooks()

        assert len(method.hooks) == 0

    def test_compute_gmar_returns_dict(self):
        """Test compute_gmar returns expected dict structure."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        method = P1VGMAR(config, mock_policy)

        with patch.object(method, "_find_encoder_layers", return_value=[]):
            result = method.compute_gmar({})

        assert "heatmap" in result
        assert "attribution_scores" in result

    def test_end_episode_returns_none(self):
        """Test end_episode returns None (no per-episode computation)."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        method = P1VGMAR(config, mock_policy)

        result = method.end_episode()

        assert result is None

    def test_attention_maps_cleared_on_compute(self):
        """Test attention_maps is cleared after compute_gmar."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        mock_encoder = MagicMock()
        mock_layer = MagicMock()
        mock_layer.self_attn = MagicMock()
        mock_layer.self_attn.register_forward_hook = MagicMock(return_value=MagicMock())
        mock_encoder.layers = [mock_layer]
        mock_vlm = MagicMock()
        mock_vlm.language_model.model.encoder = mock_encoder
        mock_policy.model.vlm = mock_vlm

        method = P1VGMAR(config, mock_policy)
        method._register_hooks()

        method.attention_maps.append(torch.zeros(1, 2, 10, 10))

        with patch.object(method, "_find_encoder_layers", return_value=[]):
            method.compute_gmar({})

        assert len(method.attention_maps) == 0


class TestP1VGMAREdgeCases:
    """Edge case tests for P1VGMAR."""

    def test_compute_gmar_with_no_encoder(self):
        """Test compute_gmar handles missing encoder gracefully."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock(spec=[])

        method = P1VGMAR(config, mock_policy)

        result = method.compute_gmar({})

        assert result["heatmap"] is None
        assert result["attribution_scores"] == {}

    def test_register_hooks_with_invalid_index(self):
        """Test register_hooks handles invalid index."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        mock_encoder = MagicMock()
        mock_layer = MagicMock()
        mock_layer.self_attn = MagicMock()
        mock_layer.self_attn.register_forward_hook = MagicMock(return_value=MagicMock())
        mock_encoder.layers = [mock_layer]
        mock_vlm = MagicMock()
        mock_vlm.language_model.model.encoder = mock_encoder
        mock_policy.model.vlm = mock_vlm

        method = P1VGMAR(config, mock_policy)

        method._register_hooks(layer_indices=[999])

        assert len(method.hooks) == 0

    def test_multiple_compute_calls(self):
        """Test multiple compute_gmar calls don't accumulate state."""
        config = XAIConfig(use_p1_v_gmar=True)
        mock_policy = MagicMock()

        method = P1VGMAR(config, mock_policy)

        with patch.object(method, "_find_encoder_layers", return_value=[]):
            result1 = method.compute_gmar({})
            result2 = method.compute_gmar({})

        assert result1["heatmap"] is None
        assert result2["heatmap"] is None