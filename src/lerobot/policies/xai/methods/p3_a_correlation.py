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
P3-A: Action Dimension Correlation Heatmap.

This method analyzes the correlation structure between action dimensions
to detect spurious correlations learned from training data.
"""

from __future__ import annotations

import torch

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.base import XAIMethod


class P3ACorrelation(XAIMethod):
    """
    Action Dimension Correlation Analysis.

    Computes correlation matrix between action dimensions to:
    - Detect spurious correlations from demo bias
    - Verify coupling structure matches robot kinematics
    - Support demo diversity metric

    This is an offline method.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy) -> None:
        super().__init__(config, policy)
        self.action_samples: list[torch.Tensor] = []

    def name(self) -> str:
        return "p3_a_correlation"

    def is_realtime(self) -> bool:
        return False

    def clear_samples(self) -> None:
        """Clear stored action samples."""
        self.action_samples.clear()

    def add_action(self, action: torch.Tensor) -> None:
        """Add an action sample to the collection."""
        self.action_samples.append(action.detach().cpu().clone())

    def compute_correlation(
        self,
        observations: list[dict] | None = None,
        n_samples_per_obs: int = 100,
    ) -> dict:
        """
        Compute correlation matrix from action samples.

        Args:
            observations: List of observation batches (uses stored samples if None)
            n_samples_per_obs: Number of samples per observation

        Returns:
            Dict with correlation matrix, covariance, std per dimension
        """
        if observations is None and not self.action_samples:
            return {
                "correlation": None,
                "covariance": None,
                "std_per_dim": None,
                "n_samples": 0,
            }

        all_samples = []

        if observations is not None:
            self.clear_samples()

            for obs in observations:
                bundle = self._generate_bundle(obs, n_samples_per_obs)
                first_step = bundle[:, 0, 0, :]
                all_samples.append(first_step)

            all_samples = torch.cat(all_samples, dim=0)
        else:
            if len(self.action_samples) == 0:
                return {
                    "correlation": None,
                    "covariance": None,
                    "std_per_dim": None,
                    "n_samples": 0,
                }
            all_samples = torch.cat(self.action_samples, dim=0)

        original_ndim = all_samples.ndim
        if original_ndim == 3:
            all_samples = all_samples.reshape(all_samples.shape[0], -1)

        cov = torch.cov(all_samples.T)
        std = all_samples.std(dim=0)
        std_outer = std.unsqueeze(1) @ std.unsqueeze(0)
        corr = cov / (std_outer + 1e-8)
        corr = corr.clamp(-1, 1)

        return {
            "correlation": corr,
            "covariance": cov,
            "std_per_dim": std,
            "n_samples": all_samples.shape[0],
        }

    def _generate_bundle(
        self,
        batch: dict,
        n_samples: int,
    ) -> torch.Tensor:
        """Generate action bundle for a single observation."""
        inputs = self.policy._build_model_inputs(batch)

        enc = self.policy.model.forward_vlm(
            inputs["input_ids"],
            inputs["image_input"],
            inputs["image_mask"],
        )

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                action = self.policy.model.generate_actions(
                    input_ids=inputs["input_ids"],
                    image_input=inputs["image_input"],
                    image_mask=inputs["image_mask"],
                    domain_id=inputs["domain_id"],
                    proprio=inputs["proprio"],
                )
                samples.append(action)

        return torch.stack(samples)

    def detect_spurious_correlations(
        self,
        corr_matrix: torch.Tensor,
        expected_pairs: list[tuple[int, int]] | None = None,
        threshold: float = 0.7,
    ) -> list[tuple[int, int, float]]:
        """
        Detect unexpected high correlations.

        Args:
            corr_matrix: [dim, dim] correlation matrix
            expected_pairs: List of (i, j) pairs expected to be correlated (kinematic)
            threshold: Correlation threshold for flagging

        Returns:
            List of (i, j, correlation) tuples for unexpected high correlations
        """
        if expected_pairs is None:
            expected_pairs = []

        dim = corr_matrix.shape[0]
        expected_set = set(frozenset(p) for p in expected_pairs)

        unexpected = []
        for i in range(dim):
            for j in range(i + 1, dim):
                corr_val = corr_matrix[i, j].item()
                if abs(corr_val) > threshold:
                    if frozenset({i, j}) not in expected_set:
                        unexpected.append((i, j, corr_val))

        return unexpected

    def compare_with_reference(
        self,
        reference_corr: torch.Tensor,
        test_corr: torch.Tensor,
    ) -> float:
        """
        Compare correlation matrices to detect overfitting.

        Args:
            reference_corr: Reference (dataset) correlation matrix
            test_corr: Model prediction correlation matrix

        Returns:
            Frobenius norm of the difference
        """
        if reference_corr.shape != test_corr.shape:
            return float("inf")

        diff = reference_corr - test_corr
        return torch.norm(diff, p="fro").item()

    def end_episode(self) -> dict | None:
        """Return correlation analysis for the episode."""
        if not self.action_samples:
            return None

        result = self.compute_correlation()
        return result