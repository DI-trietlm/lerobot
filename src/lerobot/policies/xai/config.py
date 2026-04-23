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

from dataclasses import dataclass, field


@dataclass
class XAIConfig:
    """
    Configuration for XAI (Explainable AI) methods.

    All XAI methods are disabled by default. Enable them by setting
    the corresponding flags to True.

    Real-time methods (low overhead, can run during inference):
        - use_p0_v_attention: Raw Attention Map
        - use_p1_a_denoising: Denoising Trajectory
        - use_p3_rtc_smoothness: Chunk Boundary Smoothness

    Offline methods (higher overhead, run after episode):
        - use_p1_v_gmar: GMAR (Gradient-weighted Multi-head Attention Rollout)
        - use_p2_a_bundle: Action Sample Bundle
        - use_p2_x_integrated_gradients: Integrated Gradients
        - use_p3_a_correlation: Action Dimension Correlation
    """

    # =========================================================
    # Real-time methods (default: all off)
    # =========================================================
    use_p0_v_attention: bool = False
    use_p1_a_denoising: bool = False
    use_p3_rtc_smoothness: bool = False

    # =========================================================
    # Offline methods (default: all off)
    # =========================================================
    use_p1_v_gmar: bool = False
    use_p2_a_bundle: bool = False
    use_p2_x_integrated_gradients: bool = False
    use_p3_a_correlation: bool = False

    # =========================================================
    # Common settings
    # =========================================================
    output_dir: str = "xai_outputs"
    save_heatmaps: bool = True
    save_episode_summary: bool = True

    # =========================================================
    # P0-V: Attention Map settings
    # =========================================================
    p0_v_layer_indices: list = field(default_factory=lambda: [-1, -3, -6])
    p0_v_patch_grid: tuple = (14, 14)
    entropy_threshold: float = 3.5

    # =========================================================
    # P1-V: GMAR settings
    # =========================================================
    p1_v_target_action_dim: int | None = None

    # =========================================================
    # P1-A: Denoising Trajectory settings
    # =========================================================
    p1_a_log_full_trajectory: bool = False

    # =========================================================
    # P2-A: Action Sample Bundle settings
    # =========================================================
    p2_a_n_samples: int = 50
    p2_a_trigger_on_flagged: bool = True

    # =========================================================
    # P2-X: Integrated Gradients settings
    # =========================================================
    p2_x_n_steps: int = 50
    p2_x_target_joints: list | None = None

    # =========================================================
    # P3-RTC: Chunk Boundary Smoothness settings
    # =========================================================
    p3_rtc_overlap_steps: int = 3
    boundary_low_threshold: float = 0.75
    boundary_critical_threshold: float = 0.5

    # =========================================================
    # Episode quality settings
    # =========================================================
    quality_threshold: float = 0.80
    flagged_episode_threshold: float = 0.1

    def has_any_xai_enabled(self) -> bool:
        """Return True if any XAI method is enabled."""
        return any([
            self.use_p0_v_attention,
            self.use_p1_a_denoising,
            self.use_p3_rtc_smoothness,
            self.use_p1_v_gmar,
            self.use_p2_a_bundle,
            self.use_p2_x_integrated_gradients,
            self.use_p3_a_correlation,
        ])

    def has_realtime_methods(self) -> bool:
        """Return True if any real-time XAI method is enabled."""
        return any([
            self.use_p0_v_attention,
            self.use_p1_a_denoising,
            self.use_p3_rtc_smoothness,
        ])

    def has_offline_methods(self) -> bool:
        """Return True if any offline XAI method is enabled."""
        return any([
            self.use_p1_v_gmar,
            self.use_p2_a_bundle,
            self.use_p2_x_integrated_gradients,
            self.use_p3_a_correlation,
        ])
