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
P2-A: Action Sample Bundle - Uncertainty and Multimodality Analysis.

This method samples multiple action trajectories from the same observation
to analyze the distribution of actions predicted by the policy.
"""

from __future__ import annotations

import torch

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.base import XAIMethod


class P2ABundle(XAIMethod):
    """
    Action Sample Bundle for uncertainty analysis.

    Samples multiple action trajectories from the same observation to:
    - Estimate action mean and variance (uncertainty)
    - Detect multimodal distributions
    - Support active data collection based on uncertainty

    This is an offline method - too expensive for real-time use.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy) -> None:
        super().__init__(config, policy)
        self.samples: list[torch.Tensor] = []

    def name(self) -> str:
        return "p2_a_bundle"

    def is_realtime(self) -> bool:
        return False

    def clear_samples(self) -> None:
        """Clear stored samples."""
        self.samples.clear()

    def add_sample(self, action: torch.Tensor) -> None:
        """Add an action sample to the bundle."""
        self.samples.append(action.detach().cpu().clone())

    def get_bundle_stats(self) -> dict:
        """
        Compute statistics from the collected samples.

        Returns:
            Dict with mean, std, cv (coefficient of variation), and multimodality
        """
        if not self.samples:
            return {
                "mean": None,
                "std": None,
                "cv": None,
                "is_multimodal": False,
                "n_samples": 0,
            }

        samples = torch.stack(self.samples)

        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        cv = std / (mean.abs() + 1e-8)

        is_multimodal = self._detect_multimodal(samples)

        return {
            "mean": mean,
            "std": std,
            "cv": cv,
            "is_multimodal": is_multimodal,
            "n_samples": len(self.samples),
        }

    def _detect_multimodal(self, samples: torch.Tensor, n_clusters: int = 2) -> bool:
        """
        Detect if distribution is multimodal using simple clustering.

        Uses 1D GMM on action magnitude at first timestep.
        """
        try:
            from sklearn.mixture import GaussianMixture

            N = samples.shape[0]
            if N < n_clusters:
                return False

            feats = samples[:, 0, 0, :].numpy()

            gm = GaussianMixture(n_components=n_clusters, random_state=42).fit(feats)

            return all(w > 0.2 for w in gm.weights_)

        except ImportError:
            return False

    def sample_bundle(
        self,
        batch: dict,
        n_samples: int | None = None,
    ) -> dict:
        """
        Generate action sample bundle for the given batch.

        Args:
            batch: Input batch with observations
            n_samples: Number of samples to generate (default from config)

        Returns:
            Dict containing samples, mean, std, cv, and multimodality flag
        """
        n_samples = n_samples or self.config.p2_a_n_samples

        self.clear_samples()

        inputs = self.policy._build_model_inputs(batch)

        enc = self.policy.model.forward_vlm(
            inputs["input_ids"],
            inputs["image_input"],
            inputs["image_mask"],
        )

        with torch.no_grad():
            for _ in range(n_samples):
                action = self.policy.model.generate_actions(
                    input_ids=inputs["input_ids"],
                    image_input=inputs["image_input"],
                    image_mask=inputs["image_mask"],
                    domain_id=inputs["domain_id"],
                    proprio=inputs["proprio"],
                )
                self.add_sample(action)

        return self.get_bundle_stats()

    def end_episode(self) -> dict | None:
        """Return bundle statistics for the episode."""
        return self.get_bundle_stats()