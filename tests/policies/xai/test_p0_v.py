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
from unittest.mock import MagicMock, patch

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.p0_v_attention_map import P0VAttentionMap


@pytest.fixture
def config():
    return XAIConfig(
        p0_v_layer_indices=[-1, -3],
        p0_v_patch_grid=(14, 14),
        entropy_threshold=3.5,
    )


@pytest.fixture
def mock_policy():
    """Create a mock policy with fake encoder structure."""
    policy = MagicMock()

    # Create fake encoder layers with self_attn
    class FakeSelfAttn:
        def __init__(self):
            self.hooks = []

        def register_forward_hook(self, hook_fn):
            class FakeHandle:
                def remove(self):
                    pass
            return FakeHandle()

    class FakeEncoderLayer:
        def __init__(self):
            self.self_attn = FakeSelfAttn()

    # Create fake encoder with layers
    encoder_layers = [FakeEncoderLayer() for _ in range(12)]
    policy.model.vlm.language_model.model.encoder.layers = encoder_layers

    return policy


@pytest.fixture
def monitor(config, mock_policy):
    return P0VAttentionMap(config, mock_policy)


class TestP0VAttentionMap:
    """Tests for P0-V Attention Map."""

    def test_name(self, monitor):
        assert monitor.name() == "p0_v_attention"

    def test_is_realtime(self, monitor):
        assert monitor.is_realtime() is True

    def test_initialization(self, config, mock_policy):
        monitor = P0VAttentionMap(config, mock_policy)
        assert monitor.layer_indices == [-1, -3]
        assert monitor.patch_grid == (14, 14)
        assert monitor.num_img_tokens == 196
        assert len(monitor._hooks) == 0
        assert len(monitor._attention_maps) == 0

    def test_find_encoder_layers(self, monitor, mock_policy):
        layers = monitor._find_encoder_layers()
        assert len(layers) == 12

    def test_register_hooks(self, monitor):
        monitor.register()
        assert len(monitor._hooks) == 2  # 2 layer indices

    def test_register_hooks_idempotent(self, monitor):
        monitor.register()
        monitor.register()  # Should not add more hooks
        assert len(monitor._hooks) == 2

    def test_remove_hooks(self, monitor):
        monitor.register()
        assert len(monitor._hooks) == 2
        monitor.remove()
        assert len(monitor._hooks) == 0

    def test_clear_attention_maps(self, monitor):
        # Simulate having some maps
        monitor._attention_maps.append(torch.randn(2, 8, 210, 210))
        assert len(monitor._attention_maps) == 1
        monitor.clear()
        assert len(monitor._attention_maps) == 0

    def test_reset(self, monitor):
        monitor.register()
        monitor._attention_maps.append(torch.randn(2, 8, 210, 210))
        monitor.reset()
        assert len(monitor._hooks) == 0
        assert len(monitor._attention_maps) == 0

    def test_get_num_layers(self, monitor):
        assert monitor.get_num_layers() == 2

    def test_get_num_attention_maps_empty(self, monitor):
        assert monitor.get_num_attention_maps() == 0


class TestP0VComputeEntropy:
    """Tests for entropy and heatmap computation."""

    def test_compute_entropy_no_maps(self, monitor):
        entropy, heatmap = monitor.compute_entropy()
        assert entropy == 0.0
        assert heatmap.shape == (7, 7)
        assert torch.all(heatmap == 0)

    def test_compute_entropy_with_fake_attention(self, monitor):
        # Simulate attention map: [B, heads, seq, seq]
        # seq = 196 (img) + 14 (lang) = 210
        batch_size = 2
        num_heads = 8
        num_img = 196
        num_lang = 14
        seq_len = num_img + num_lang

        # Uniform attention over image tokens
        attn = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        # Language tokens (rows 196-209) attend uniformly to image tokens (cols 0-195)
        attn[:, :, 196:, :196] = 1.0 / num_img

        monitor._attention_maps.append(attn)

        entropy, heatmap = monitor.compute_entropy()

        # Uniform distribution over 196 tokens should give high entropy
        assert entropy > 4.0
        assert heatmap.shape == (7, 7)

    def test_compute_entropy_focused_attention(self, monitor):
        # Focused attention: all language tokens attend to ONE image patch
        batch_size = 1
        num_heads = 8
        num_img = 196
        num_lang = 14
        seq_len = num_img + num_lang

        # Create attention where rows sum to 1 (like real softmax attention)
        # Image tokens: indices 0-195
        # Language tokens: indices 196-209
        attn = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        # ALL language tokens (rows 196-209, all heads) attend ONLY to image token 0 (col 0)
        attn[:, :, 196:, 0] = 1.0

        monitor._attention_maps.append(attn)

        entropy, heatmap = monitor.compute_entropy()

        # All attention on 1 token should give very low entropy
        assert entropy < 0.5

    def test_compute_entropy_batch_size_handling(self, monitor):
        # Use batch size > 1, should use first element
        batch_size = 3
        num_heads = 8
        num_img = 196
        num_lang = 14
        seq_len = num_img + num_lang

        attn = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        # First batch element: ALL attend to token 0 (very focused)
        attn[0, :, 196:, 0] = 1.0
        # Second batch element: uniform over all 196 (diffuse)
        attn[1, :, 196:, :] = 1.0 / num_img
        # Third: also uniform
        attn[2, :, 196:, :] = 1.0 / num_img

        monitor._attention_maps.append(attn)

        entropy, heatmap = monitor.compute_entropy()

        # Should use first batch element's very focused attention
        assert entropy < 0.5

    def test_heatmap_shape(self, monitor):
        # 14x14 = 196 patches, pooled 2x2 -> 7x7
        batch_size = 1
        num_heads = 8
        num_img = 196
        num_lang = 14
        seq_len = num_img + num_lang

        attn = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        # Language tokens attend uniformly to all tokens (including other lang tokens)
        attn[:, :, 196:, :] = 1.0 / seq_len

        monitor._attention_maps.append(attn)

        _, heatmap = monitor.compute_entropy()

        assert heatmap.shape == (7, 7)


class TestP0VIntegration:
    """Integration tests for P0-V with mock policy."""

    def test_full_lifecycle(self, mock_policy):
        config = XAIConfig(
            use_p0_v_attention=True,
            p0_v_layer_indices=[-1],
            p0_v_patch_grid=(14, 14),
        )
        monitor = P0VAttentionMap(config, mock_policy)

        # Register hooks
        monitor.register()
        assert len(monitor._hooks) == 1

        # Simulate attention capture
        batch_size = 1
        num_heads = 8
        seq_len = 210
        attn = torch.randn(batch_size, num_heads, seq_len, seq_len)
        monitor._attention_maps.append(attn)

        # Compute
        entropy, heatmap = monitor.compute_entropy()
        assert isinstance(entropy, float)
        assert heatmap.shape == (7, 7)

        # Cleanup
        monitor.reset()
        assert len(monitor._hooks) == 0
        assert len(monitor._attention_maps) == 0
