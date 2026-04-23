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

from lerobot.policies.xai.config import XAIConfig


class TestXAIConfigDefaults:
    """Test XAIConfig default values."""

    def test_all_flags_default_false(self):
        cfg = XAIConfig()
        assert cfg.use_p0_v_attention is False
        assert cfg.use_p1_v_gmar is False
        assert cfg.use_p1_a_denoising is False
        assert cfg.use_p2_a_bundle is False
        assert cfg.use_p2_x_integrated_gradients is False
        assert cfg.use_p3_a_correlation is False
        assert cfg.use_p3_rtc_smoothness is False

    def test_common_defaults(self):
        cfg = XAIConfig()
        assert cfg.output_dir == "xai_outputs"
        assert cfg.save_heatmaps is True
        assert cfg.save_episode_summary is True

    def test_threshold_defaults(self):
        cfg = XAIConfig()
        assert cfg.entropy_threshold == 3.5
        assert cfg.quality_threshold == 0.80
        assert cfg.boundary_low_threshold == 0.75
        assert cfg.boundary_critical_threshold == 0.5
        assert cfg.flagged_episode_threshold == 0.1

    def test_p0_v_defaults(self):
        cfg = XAIConfig()
        assert cfg.p0_v_layer_indices == [-1, -3, -6]
        assert cfg.p0_v_patch_grid == (14, 14)

    def test_p2_a_defaults(self):
        cfg = XAIConfig()
        assert cfg.p2_a_n_samples == 50
        assert cfg.p2_a_trigger_on_flagged is True

    def test_p2_x_defaults(self):
        cfg = XAIConfig()
        assert cfg.p2_x_n_steps == 50
        assert cfg.p2_x_target_joints is None

    def test_p3_rtc_defaults(self):
        cfg = XAIConfig()
        assert cfg.p3_rtc_overlap_steps == 3


class TestXAIConfigCustom:
    """Test XAIConfig with custom values."""

    def test_custom_flags(self):
        cfg = XAIConfig(
            use_p0_v_attention=True,
            use_p1_v_gmar=True,
            use_p2_a_bundle=True,
        )
        assert cfg.use_p0_v_attention is True
        assert cfg.use_p1_v_gmar is True
        assert cfg.use_p2_a_bundle is True
        assert cfg.use_p3_rtc_smoothness is False

    def test_custom_thresholds(self):
        cfg = XAIConfig(
            entropy_threshold=4.0,
            quality_threshold=0.9,
            boundary_low_threshold=0.8,
        )
        assert cfg.entropy_threshold == 4.0
        assert cfg.quality_threshold == 0.9
        assert cfg.boundary_low_threshold == 0.8


class TestXAIConfigHelpers:
    """Test XAIConfig helper methods."""

    def test_has_any_xai_enabled_all_false(self):
        cfg = XAIConfig()
        assert cfg.has_any_xai_enabled() is False

    def test_has_any_xai_enabled_one_true(self):
        cfg = XAIConfig(use_p0_v_attention=True)
        assert cfg.has_any_xai_enabled() is True

    def test_has_any_xai_enabled_multiple_true(self):
        cfg = XAIConfig(use_p0_v_attention=True, use_p1_v_gmar=True, use_p2_a_bundle=True)
        assert cfg.has_any_xai_enabled() is True

    def test_has_realtime_methods_all_false(self):
        cfg = XAIConfig()
        assert cfg.has_realtime_methods() is False

    def test_has_realtime_methods_one_true(self):
        cfg = XAIConfig(use_p0_v_attention=True)
        assert cfg.has_realtime_methods() is True
        cfg = XAIConfig(use_p3_rtc_smoothness=True)
        assert cfg.has_realtime_methods() is True

    def test_has_offline_methods_all_false(self):
        cfg = XAIConfig()
        assert cfg.has_offline_methods() is False

    def test_has_offline_methods_one_true(self):
        cfg = XAIConfig(use_p1_v_gmar=True)
        assert cfg.has_offline_methods() is True
        assert cfg.has_realtime_methods() is False

    def test_has_realtime_and_offline(self):
        cfg = XAIConfig(use_p0_v_attention=True, use_p1_v_gmar=True)
        assert cfg.has_realtime_methods() is True
        assert cfg.has_offline_methods() is True
