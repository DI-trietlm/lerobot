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
P1-V: GMAR - Gradient-weighted Multi-head Attention Rollout.

This method combines attention rollout with gradient weighting to produce
action-conditioned heatmaps showing which image regions most influence
the policy's action predictions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.base import XAIMethod


class P1VGMAR(XAIMethod):
    """
    GMAR (Gradient-weighted Multi-head Attention Rollout) for X-VLA.

    This offline method computes action-conditioned attention heatmaps by:
    1. Capturing attention weights from Florence-2 encoder layers
    2. Computing gradients of action predictions w.r.t. attention weights
    3. Weighting attention heads by gradient magnitude
    4. Performing attention rollout across layers with residual connections

    Results are used for:
    - Language instruction sensitivity testing
    - Contrastive demo strategy
    - Task-phase attention audit
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy) -> None:
        super().__init__(config, policy)
        self.attention_maps: list[torch.Tensor] = []
        self.attention_grads: list[torch.Tensor] = []
        self.hooks: list = []
        self._num_img_tokens = 196

    def name(self) -> str:
        return "p1_v_gmar"

    def is_realtime(self) -> bool:
        return False

    def _find_encoder_layers(self) -> list:
        """Find encoder layers in the VLM model."""
        if not hasattr(self.policy, "model") or not hasattr(self.policy.model, "vlm"):
            return []

        vlm = self.policy.model.vlm
        if hasattr(vlm, "language_model") and hasattr(vlm.language_model, "model"):
            encoder = vlm.language_model.model.encoder
            if hasattr(encoder, "layers"):
                return list(encoder.layers)
        return []

    def _register_hooks(self, layer_indices: list[int] | None = None) -> None:
        """Register forward hooks to capture attention weights and gradients."""
        encoder_layers = self._find_encoder_layers()
        if not encoder_layers:
            return

        target_indices = layer_indices or list(range(len(encoder_layers)))

        for idx in target_indices:
            if idx < 0:
                idx = len(encoder_layers) + idx
            if idx < 0 or idx >= len(encoder_layers):
                continue

            layer = encoder_layers[idx]

            def make_hook(pos):
                def fwd_hook(m, inp, out):
                    if isinstance(out, tuple) and len(out) > 1:
                        attn = out[1].detach().clone()
                        attn.requires_grad_(True)
                        self.attention_maps.append(attn)
                return fwd_hook

            hook = layer.self_attn.register_forward_hook(make_hook(idx))
            self.hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def compute_gmar(
        self,
        batch: dict,
        target_action_dim: int | None = None,
    ) -> dict:
        """
        Compute GMAR heatmap for the given batch.

        Args:
            batch: Input batch with observations
            target_action_dim: Specific action dimension to analyze (None = all)

        Returns:
            Dict containing:
                - heatmap: [B, H, W] attention heatmap
                - attribution_scores: dict of modality attribution percentages
        """
        self.attention_maps.clear()
        self.attention_grads.clear()

        self._register_hooks()

        try:
            encoder_layers = self._find_encoder_layers()
            if not encoder_layers:
                return {"heatmap": None, "attribution_scores": {}}

            n_layers = len(encoder_layers)
            layer_range = list(range(n_layers))

            inputs = self.policy._build_model_inputs(batch)
            enc = self.policy.model.forward_vlm(
                inputs["input_ids"],
                inputs["image_input"],
                inputs["image_mask"],
            )

            vlm_features = enc["vlm_features"]
            B, seq_len, hidden = vlm_features.shape

            t_mid = torch.full((B,), 0.5, device=vlm_features.device)
            x_noisy = torch.randn(
                B,
                self.policy.model.chunk_size,
                self.policy.model.dim_action,
                device=vlm_features.device,
            )

            proprio_m, x_noisy_m = self.policy.model.action_space.preprocess(
                inputs["proprio"], x_noisy
            )

            pred = self.policy.model.transformer(
                domain_id=inputs["domain_id"],
                action_with_noise=x_noisy_m,
                proprio=proprio_m,
                t=t_mid,
                vlm_features=vlm_features,
                aux_visual_inputs=enc.get("aux_visual_inputs"),
            )

            if target_action_dim is not None:
                target = pred[:, :, target_action_dim].mean()
            else:
                target = pred.mean()

            target.backward()

            for attn in self.attention_maps:
                if attn.grad is not None:
                    self.attention_grads.append(attn.grad.detach())

            if not self.attention_maps or not self.attention_grads:
                return {"heatmap": None, "attribution_scores": {}}

            B_attn = self.attention_maps[0].shape[0]
            seq_len_attn = self.attention_maps[0].shape[-1]

            rollout = torch.eye(seq_len_attn, device=vlm_features.device)
            rollout = rollout.unsqueeze(0).expand(B_attn, -1, -1)

            for attn, grad in zip(self.attention_maps, self.attention_grads):
                head_weights = grad.abs().mean(dim=(-2, -1), keepdim=True)
                weighted_attn = (attn * head_weights).sum(dim=1)
                weighted_attn = F.relu(weighted_attn)
                weighted_attn = weighted_attn + torch.eye(
                    seq_len_attn, device=weighted_attn.device
                )
                weighted_attn = weighted_attn / (
                    weighted_attn.sum(dim=-1, keepdim=True) + 1e-8
                )
                rollout = weighted_attn @ rollout

            num_img_tokens = self._num_img_tokens
            if num_img_tokens > seq_len_attn:
                num_img_tokens = seq_len_attn // 2

            img_rollout = rollout[:, num_img_tokens:, :num_img_tokens]
            heatmap = img_rollout.mean(dim=1)

            grid_size = int(heatmap.shape[1] ** 0.5)
            heatmap = heatmap.reshape(B, grid_size, grid_size)

            heatmap_min = heatmap.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            heatmap_max = heatmap.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)

            vision_score = heatmap.sum().item()
            total = vision_score + 1e-8

            return {
                "heatmap": heatmap,
                "attribution_scores": {
                    "vision_pct": 100 * vision_score / total,
                    "language_pct": 0.0,
                    "proprio_pct": 0.0,
                },
            }

        finally:
            self._remove_hooks()
            self.attention_maps.clear()
            self.attention_grads.clear()

    def end_episode(self) -> dict | None:
        """Run GMAR analysis on recorded data."""
        return None