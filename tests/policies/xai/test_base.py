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
from unittest.mock import MagicMock

from lerobot.policies.xai.config import XAIConfig
from lerobot.policies.xai.methods.base import XAIMethod


class DummyRealtimeMethod(XAIMethod):
    """Dummy implementation for testing."""

    def name(self) -> str:
        return "dummy_realtime"

    def is_realtime(self) -> bool:
        return True


class DummyOfflineMethod(XAIMethod):
    """Dummy implementation for testing."""

    def name(self) -> str:
        return "dummy_offline"

    def is_realtime(self) -> bool:
        return False


class TestXAIMethodBase:
    """Test XAIMethod base class."""

    def test_realtime_method_name(self):
        cfg = XAIConfig()
        mock_policy = MagicMock()
        method = DummyRealtimeMethod(cfg, mock_policy)
        assert method.name() == "dummy_realtime"
        assert method.is_realtime() is True

    def test_offline_method_name(self):
        cfg = XAIConfig()
        mock_policy = MagicMock()
        method = DummyOfflineMethod(cfg, mock_policy)
        assert method.name() == "dummy_offline"
        assert method.is_realtime() is False

    def test_config_stored(self):
        cfg = XAIConfig(use_p0_v_attention=True)
        mock_policy = MagicMock()
        method = DummyRealtimeMethod(cfg, mock_policy)
        assert method.config == cfg
        assert method.config.use_p0_v_attention is True

    def test_policy_stored(self):
        cfg = XAIConfig()
        mock_policy = MagicMock()
        method = DummyRealtimeMethod(cfg, mock_policy)
        assert method.policy == mock_policy

    def test_start_episode_default_noop(self):
        cfg = XAIConfig()
        mock_policy = MagicMock()
        method = DummyRealtimeMethod(cfg, mock_policy)
        # Should not raise
        method.start_episode("ep_001", 0)
        method.start_episode("ep_002", 5)

    def test_on_step_default_noop(self):
        cfg = XAIConfig()
        mock_policy = MagicMock()
        method = DummyRealtimeMethod(cfg, mock_policy)
        # Should not raise
        method.on_step(batch={}, action_chunk={}, step_idx=0)

    def test_end_episode_default_returns_none(self):
        cfg = XAIConfig()
        mock_policy = MagicMock()
        method = DummyRealtimeMethod(cfg, mock_policy)
        assert method.end_episode() is None

    def test_reset_default_noop(self):
        cfg = XAIConfig()
        mock_policy = MagicMock()
        method = DummyRealtimeMethod(cfg, mock_policy)
        # Should not raise
        method.reset()

    def test_abstract_method_enforcement(self):
        """Test that subclasses must implement name() and is_realtime()."""
        cfg = XAIConfig()

        class IncompleteMethod(XAIMethod):
            pass

        with pytest.raises(TypeError):
            IncompleteMethod(cfg, MagicMock())

    def test_realtime_method_not_abstract_offline(self):
        """Test that is_realtime can be overridden in subclass."""

        class CustomRealtime(XAIMethod):
            def name(self) -> str:
                return "custom"

            def is_realtime(self) -> bool:
                return True

        cfg = XAIConfig()
        method = CustomRealtime(cfg, MagicMock())
        assert method.is_realtime() is True
