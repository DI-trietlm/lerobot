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

from lerobot.policies.xai.buffer import EpisodeXAIBuffer, StepRecord


class TestStepRecord:
    """Test StepRecord dataclass."""

    def test_creation_minimal(self):
        record = StepRecord(step_idx=0)
        assert record.step_idx == 0
        assert record.timestamp == 0.0
        assert record.attn_entropy == 0.0
        assert record.attn_compressed is None
        assert record.convergence_speed == []
        assert record.boundary_sim is None
        assert record.status is None
        assert record.flagged is False
        assert record.flag_reason is None

    def test_creation_full(self):
        record = StepRecord(
            step_idx=5,
            timestamp=1234567890.0,
            attn_entropy=2.5,
            attn_compressed=torch.randn(7, 7),
            convergence_speed=[0.5, 0.3, 0.1],
            boundary_sim=0.85,
            status="ok",
            flagged=True,
            flag_reason="high_entropy",
        )
        assert record.step_idx == 5
        assert record.timestamp == 1234567890.0
        assert record.attn_entropy == 2.5
        assert record.attn_compressed.shape == (7, 7)
        assert record.convergence_speed == [0.5, 0.3, 0.1]
        assert record.boundary_sim == 0.85
        assert record.status == "ok"
        assert record.flagged is True
        assert record.flag_reason == "high_entropy"

    def test_to_dict(self):
        record = StepRecord(
            step_idx=0,
            timestamp=1.0,
            attn_entropy=2.0,
            boundary_sim=0.9,
            flagged=False,
        )
        d = record.to_dict()
        assert d["step_idx"] == 0
        assert d["timestamp"] == 1.0
        assert d["attn_entropy"] == 2.0
        assert d["boundary_sim"] == 0.9
        assert d["flagged"] is False
        assert "attn_compressed" not in d  # None values omitted from dict


class TestEpisodeXAIBuffer:
    """Test EpisodeXAIBuffer dataclass."""

    def test_creation_minimal(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_001")
        assert buffer.episode_id == "ep_001"
        assert buffer.episode_index == 0
        assert buffer.step_records == []
        assert buffer.flagged_steps == []
        assert buffer.episode_quality is None

    def test_add_step_no_flag(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_001")
        record = StepRecord(step_idx=0, timestamp=1.0, flagged=False)
        buffer.add_step(record)
        assert len(buffer.step_records) == 1
        assert buffer.flagged_steps == []

    def test_add_step_flagged(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_001")
        record1 = StepRecord(step_idx=0, flagged=False)
        record2 = StepRecord(step_idx=1, flagged=True, flag_reason="high_entropy")
        record3 = StepRecord(step_idx=2, flagged=True, flag_reason="high_entropy")
        record4 = StepRecord(step_idx=3, flagged=False)

        buffer.add_step(record1)
        buffer.add_step(record2)
        buffer.add_step(record3)
        buffer.add_step(record4)

        assert len(buffer.step_records) == 4
        assert buffer.flagged_steps == [1, 2]

    def test_add_step_duplicate_flag_prevented(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_001")
        record = StepRecord(step_idx=1, flagged=True, flag_reason="high_entropy")
        buffer.add_step(record)
        buffer.add_step(record)  # Same step, shouldn't duplicate
        assert len(buffer.flagged_steps) == 1

    def test_compute_summary_empty(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_001")
        summary = buffer.compute_summary()
        assert summary["episode_id"] == "ep_001"
        assert summary["n_steps"] == 0
        assert summary["n_flagged"] == 0
        assert summary["episode_quality"] is None
        assert summary["should_include_in_training"] is True

    def test_compute_summary_with_records(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_002", episode_index=1)
        buffer.step_records = [
            StepRecord(step_idx=0, attn_entropy=2.0, boundary_sim=0.9),
            StepRecord(step_idx=1, attn_entropy=3.0, boundary_sim=0.8),
            StepRecord(step_idx=2, attn_entropy=1.0, boundary_sim=0.95),
        ]
        buffer.episode_quality = 0.85

        summary = buffer.compute_summary()
        assert summary["episode_id"] == "ep_002"
        assert summary["episode_index"] == 1
        assert summary["n_steps"] == 3
        assert summary["n_flagged"] == 0
        assert summary["episode_quality"] == 0.85
        assert summary["mean_entropy"] == 2.0
        assert summary["max_entropy"] == 3.0
        assert abs(summary["mean_boundary_sim"] - 0.883) < 0.01

    def test_compute_summary_with_flagged(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_003")
        buffer.step_records = [
            StepRecord(step_idx=0, flagged=False),
            StepRecord(step_idx=1, flagged=True),
            StepRecord(step_idx=2, flagged=False),
            StepRecord(step_idx=3, flagged=True),
            StepRecord(step_idx=4, flagged=False),
        ]
        summary = buffer.compute_summary()
        assert summary["n_steps"] == 5
        assert summary["n_flagged"] == 2
        assert summary["flagged_ratio"] == 0.4

    def test_should_run_offline_xai_empty(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_001")
        assert buffer.should_run_offline_xai() is False

    def test_should_run_offline_xai_below_threshold(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_001")
        buffer.step_records = [StepRecord(step_idx=i) for i in range(20)]
        buffer.flagged_steps = [1]  # 5% flagged
        assert buffer.should_run_offline_xai(threshold=0.1) is False

    def test_should_run_offline_xai_above_threshold(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_001")
        buffer.step_records = [StepRecord(step_idx=i) for i in range(10)]
        buffer.flagged_steps = [1, 2]  # 20% flagged
        assert buffer.should_run_offline_xai(threshold=0.1) is True

    def test_has_heatmaps_false(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_001")
        buffer.step_records = [
            StepRecord(step_idx=0),
            StepRecord(step_idx=1),
        ]
        assert buffer.has_heatmaps() is False

    def test_has_heatmaps_true(self):
        buffer = EpisodeXAIBuffer(episode_id="ep_001")
        buffer.step_records = [
            StepRecord(step_idx=0),
            StepRecord(step_idx=1, attn_compressed=torch.randn(7, 7)),
        ]
        assert buffer.has_heatmaps() is True
