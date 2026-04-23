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
XAI (Explainable AI) module for LeRobot.

This module provides 7 XAI methods for analyzing policy behavior:
- P0-V: Raw Attention Map
- P1-V: GMAR (Gradient-weighted Multi-head Attention Rollout)
- P1-A: Denoising Trajectory
- P2-A: Action Sample Bundle
- P2-X: Integrated Gradients
- P3-A: Action Dimension Correlation
- P3-RTC: Chunk Boundary Smoothness
"""

from .buffer import EpisodeXAIBuffer, StepRecord
from .config import XAIConfig
from .methods.base import XAIMethod
from .pipeline import XAIPipeline

__all__ = ["XAIConfig", "StepRecord", "EpisodeXAIBuffer", "XAIMethod", "XAIPipeline"]
