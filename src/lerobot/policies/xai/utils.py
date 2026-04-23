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
Shared utilities for XAI methods.
"""

import torch
import torch.nn.functional as F


def attention_entropy(attn_map_1d: torch.Tensor) -> float:
    """
    Compute entropy of attention distribution.

    Higher entropy = more uniform/diffuse attention (model may be confused).
    Lower entropy = more focused attention.

    Args:
        attn_map_1d: Attention weights [num_tokens]. Does not need to sum to 1.
            Will be normalized internally.

    Returns:
        Entropy value. Max entropy when uniform distribution.
    """
    # Normalize to get proper probability distribution
    p = attn_map_1d / (attn_map_1d.sum() + 1e-8)
    entropy = -(p * (p + 1e-8).log()).sum().item()
    return entropy


def compress_heatmap(heatmap: torch.Tensor, target_size: tuple[int, int] = (7, 7)) -> torch.Tensor:
    """
    Compress a heatmap to target size using average pooling.

    Args:
        heatmap: Heatmap tensor [H, W] or [1, 1, H, W].
        target_size: Target (height, width).

    Returns:
        Compressed heatmap [H', W'].
    """
    if heatmap.ndim == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif heatmap.ndim == 3:
        heatmap = heatmap.unsqueeze(0)

    h, w = target_size
    return F.avg_pool2d(heatmap, kernel_size=(heatmap.shape[2] // h, heatmap.shape[3] // w))[0, 0]


def normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Normalize heatmap to [0, 1] range.

    Args:
        heatmap: Heatmap tensor.

    Returns:
        Normalized heatmap with values in [0, 1].
    """
    min_val = heatmap.min()
    max_val = heatmap.max()
    if max_val - min_val < 1e-8:
        return torch.zeros_like(heatmap)
    return (heatmap - min_val) / (max_val - min_val)
