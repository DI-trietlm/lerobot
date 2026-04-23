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
P2-X: Integrated Gradients - Cross-modal Attribution.

This method computes Integrated Gradients to attribute action predictions
to different input modalities (vision, language, proprioception).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.base import XAIMethod


class P2XIntegratedGradients(XAIMethod):
    """
    Integrated Gradients for cross-modal attribution.

    Computes IG attribution to understand how much each input modality
    (vision, language, proprioception) contributes to action predictions.

    This is an offline method - very expensive (50x forward pass).
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy) -> None:
        super().__init__(config, policy)
        self.ig_vlm: torch.Tensor | None = None
        self.ig_proprio: torch.Tensor | None = None

    def name(self) -> str:
        return "p2_x_integrated_gradients"

    def is_realtime(self) -> bool:
        return False

    def compute_ig(
        self,
        batch: dict,
        n_steps: int | None = None,
        target_joints: list[int] | None = None,
    ) -> dict:
        """
        Compute Integrated Gradients attribution.

        Args:
            batch: Input batch with observations
            n_steps: Number of interpolation steps (default from config)
            target_joints: Specific joints to analyze (None = all)

        Returns:
            Dict containing attribution percentages and IG tensors
        """
        n_steps = n_steps or self.config.p2_x_n_steps
        target_joints = target_joints or self.config.p2_x_target_joints

        inputs = self.policy._build_model_inputs(batch)
        B = inputs["input_ids"].shape[0]

        baseline_inputs = {
            "input_ids": inputs["input_ids"],
            "image_input": torch.zeros_like(inputs["image_input"]),
            "image_mask": inputs["image_mask"],
            "domain_id": inputs["domain_id"],
            "proprio": torch.zeros_like(inputs["proprio"]),
        }

        baseline_enc = self.policy.model.forward_vlm(
            baseline_inputs["input_ids"],
            baseline_inputs["image_input"],
            baseline_inputs["image_mask"],
        )

        actual_enc = self.policy.model.forward_vlm(
            inputs["input_ids"],
            inputs["image_input"],
            inputs["image_mask"],
        )

        grads_vlm = torch.zeros_like(actual_enc["vlm_features"])
        grads_proprio = torch.zeros_like(inputs["proprio"])

        for alpha in torch.linspace(0, 1, n_steps):
            alpha_val = alpha.item()

            interp_vlm = (
                baseline_enc["vlm_features"]
                + alpha_val * (actual_enc["vlm_features"] - baseline_enc["vlm_features"])
            ).requires_grad_(True)

            interp_proprio = (
                baseline_inputs["proprio"]
                + alpha_val * (inputs["proprio"] - baseline_inputs["proprio"])
            ).requires_grad_(True)

            t_mid = torch.full((B,), 0.5, device=interp_vlm.device)

            x_noisy = torch.randn(
                B,
                self.policy.model.chunk_size,
                self.policy.model.dim_action,
                device=interp_vlm.device,
            )

            proprio_m, x_noisy_m = self.policy.model.action_space.preprocess(
                interp_proprio, x_noisy
            )

            pred = self.policy.model.transformer(
                domain_id=inputs["domain_id"],
                action_with_noise=x_noisy_m,
                proprio=proprio_m,
                t=t_mid,
                vlm_features=interp_vlm,
                aux_visual_inputs=actual_enc.get("aux_visual_inputs"),
            )

            if target_joints:
                target = pred[:, :, target_joints].mean()
            else:
                target = pred.mean()

            target.backward()

            grads_vlm += interp_vlm.grad.detach()
            grads_proprio += interp_proprio.grad.detach()

            interp_vlm.grad = None
            interp_proprio.grad = None

        grads_vlm /= n_steps
        grads_proprio /= n_steps

        ig_vlm = (
            actual_enc["vlm_features"] - baseline_enc["vlm_features"]
        ) * grads_vlm
        ig_proprio = (inputs["proprio"] - baseline_inputs["proprio"]) * grads_proprio

        vision_score = ig_vlm[:, :196, :].abs().sum().item()
        language_score = ig_vlm[:, 196:, :].abs().sum().item()
        proprio_score = ig_proprio.abs().sum().item()

        total = vision_score + language_score + proprio_score + 1e-8

        self.ig_vlm = ig_vlm
        self.ig_proprio = ig_proprio

        return {
            "vision_pct": 100 * vision_score / total,
            "language_pct": 100 * language_score / total,
            "proprio_pct": 100 * proprio_score / total,
            "ig_vlm": ig_vlm,
            "ig_proprio": ig_proprio,
        }

    def check_attribution_health(self, ig_result: dict) -> list[str]:
        """
        Check if attribution values are within expected ranges.

        Args:
            ig_result: Result from compute_ig

        Returns:
            List of warning messages for unhealthy attribution
        """
        issues = []

        v = ig_result["vision_pct"]
        l = ig_result["language_pct"]
        p = ig_result["proprio_pct"]

        if v > 75:
            issues.append("Model over-relying on vision - add more language-diverse demos")
        if l < 10:
            issues.append("Model ignoring language - add contrastive language demos")
        if p < 5:
            issues.append("Model ignoring proprio - check proprio normalization")
        if p > 45:
            issues.append("Model over-relying on proprio - check visual diversity")

        return issues

    def end_episode(self) -> dict | None:
        """Return stored IG results."""
        if self.ig_vlm is not None:
            return {
                "ig_vlm": self.ig_vlm,
                "ig_proprio": self.ig_proprio,
            }
        return None