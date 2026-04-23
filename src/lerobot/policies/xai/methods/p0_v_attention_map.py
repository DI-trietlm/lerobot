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
P0-V: Raw Attention Map.

Real-time method that captures attention weights from Florence-2 encoder.
Very low compute cost (~0), only memory bandwidth for capturing.

This provides immediate visibility into where the model is attending:
- Language tokens attending to image patches
- Attention entropy as a measure of focus vs confusion
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lerobot.policies.pretrained import PreTrainedPolicy

from ..config import XAIConfig
from ..utils import attention_entropy
from .base import XAIMethod


class P0VAttentionMap(XAIMethod):
    """
    Captures raw attention weights from Florence-2 encoder.

    Registers forward hooks on specified encoder layers to capture
    attention weights without modifying model behavior.

    This is a real-time method with ~0 compute cost.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy) -> None:
        super().__init__(config, policy)
        self.layer_indices = list(config.p0_v_layer_indices)
        self.patch_grid = config.p0_v_patch_grid
        self.num_img_tokens = config.p0_v_patch_grid[0] * config.p0_v_patch_grid[1]
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._attention_maps: list[torch.Tensor] = []

    def name(self) -> str:
        return "p0_v_attention"

    def is_realtime(self) -> bool:
        return True

    def _find_encoder_layers(self) -> list:
        """
        Find Florence-2 encoder layers.

        Returns:
            List of encoder layer modules.

        Raises:
            AttributeError: If policy doesn't have expected Florence-2 structure.
        """
        vlm = self.policy.model.vlm
        return vlm.language_model.model.encoder.layers

    def register(self) -> None:
        """Register forward hooks on encoder layers."""
        if self._hooks:
            return  # Already registered

        encoder_layers = self._find_encoder_layers()
        for idx in self.layer_indices:
            layer = encoder_layers[idx]
            hook = layer.self_attn.register_forward_hook(self._hook_fn)
            self._hooks.append(hook)

    def _hook_fn(self, module, input, output) -> None:
        """
        Hook function to capture attention weights.

        Args:
            module: The self-attention module.
            input: Input to the module (ignored).
            output: Output from the module. If tuple, second element is attention weights.
        """
        if isinstance(output, tuple) and len(output) > 1:
            self._attention_maps.append(output[1].detach().cpu())

    def remove(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear(self) -> None:
        """Clear captured attention maps."""
        self._attention_maps.clear()

    def reset(self) -> None:
        """Reset state - removes hooks and clears maps."""
        self.remove()
        self._attention_maps = []

    def compute_entropy(self) -> tuple[float, torch.Tensor]:
        """
        Compute attention entropy and compressed heatmap.

        Returns:
            Tuple of (entropy, compressed_heatmap).
            - entropy: Scalar attention entropy (high = diffuse, low = focused).
            - compressed_heatmap: [7, 7] tensor of attention mass.
        """
        if not self._attention_maps:
            return 0.0, torch.zeros(7, 7)

        # Use last layer's attention
        attn = self._attention_maps[-1]  # [B, num_heads, seq_len, seq_len]

        if attn.nelement() == 0:
            return 0.0, torch.zeros(7, 7)

        # Get batch size and determine image vs language tokens
        batch_size = attn.shape[0]
        seq_len = attn.shape[-1]

        # Language tokens are at the END of the sequence
        # Image tokens are at the BEGINNING (first num_img_tokens)
        num_lang_tokens = seq_len - self.num_img_tokens

        if num_lang_tokens <= 0:
            return 0.0, torch.zeros(7, 7)

        # Language-to-image attention: [B, heads, lang_tokens, img_tokens]
        # Language tokens (rows num_img_tokens to end) attend to image tokens (cols 0 to num_img_tokens-1)
        try:
            lang_to_img = attn[:, :, self.num_img_tokens:, :self.num_img_tokens]
        except IndexError:
            return 0.0, torch.zeros(7, 7)

        # Average over heads and language tokens -> [B, num_patches]
        img_attn = lang_to_img.mean(dim=(1, 2))

        if img_attn.numel() == 0:
            return 0.0, torch.zeros(7, 7)

        # Use first batch element
        img_attn_1d = img_attn[0]

        # Compute entropy
        entropy = attention_entropy(img_attn_1d)

        # Compress to heatmap
        heatmap = img_attn_1d.reshape(self.patch_grid)
        compressed = F.avg_pool2d(
            heatmap.unsqueeze(0).unsqueeze(0),
            kernel_size=2
        )[0, 0]

        return entropy, compressed

    def get_num_layers(self) -> int:
        """Return number of layers being monitored."""
        return len(self.layer_indices)

    def get_num_attention_maps(self) -> int:
        """Return number of captured attention maps."""
        return len(self._attention_maps)
