"""Thread-local context manager for capturing attention weights during X-VLA inference.

Zero overhead when disabled: is_capturing() is a single thread-local attribute lookup.
"""

import contextlib
import threading

import numpy as np

_tls = threading.local()


def is_capturing() -> bool:
    return getattr(_tls, "enabled", False)


@contextlib.contextmanager
def capture_attention():
    """Context manager that activates attention weight collection.

    Yields the thread-local state object with:
        .store  — dict[stage_name, dict[layer_idx, Tensor[B,H,T,T]]]
        .meta   — dict with token boundary info:
                    T_vis, T_text           (Florence-2 stage)
                    num_actions, T_vlm, T_aux, len_soft_prompts  (SoftPromptedTransformer stage)
    """
    _tls.enabled = True
    _tls.store = {}
    _tls.meta = {}
    try:
        yield _tls
    finally:
        _tls.enabled = False


def store_attn(stage: str, layer_idx: int, weights) -> None:
    """Store attention weights for a given stage and layer (no-op when not capturing)."""
    if is_capturing():
        _tls.store.setdefault(stage, {})[layer_idx] = weights.detach().cpu()


def store_meta(**kwargs) -> None:
    """Store token boundary metadata (no-op when not capturing)."""
    if is_capturing():
        _tls.meta.update(kwargs)


# ---------------------------------------------------------------------------
# Heatmap utilities (used by visualize_attention.py)
# ---------------------------------------------------------------------------

def attn_to_spatial_heatmap(
    attn: np.ndarray,
    query_start: int,
    query_end: int,
    key_start: int,
    key_end: int,
    spatial_h: int,
    spatial_w: int,
) -> np.ndarray:
    """Convert attention weights to a spatial heatmap.

    Parameters
    ----------
    attn : ndarray [B, H, T_tgt, T_src]
    query_start/end : row slice selecting which query tokens to average
    key_start/end   : col slice selecting which key tokens to reshape spatially
    spatial_h, spatial_w : spatial dimensions to reshape key tokens into

    Returns
    -------
    ndarray [B, spatial_h, spatial_w] float32, values in [0, 1]
    """
    # Slice: [B, H, Q, K]
    sliced = attn[:, :, query_start:query_end, key_start:key_end]
    # Average over heads and query tokens → [B, K]
    heatmap = sliced.mean(axis=(1, 2))
    B, K = heatmap.shape
    expected = spatial_h * spatial_w
    if K != expected:
        raise ValueError(
            f"Key tokens ({K}) != spatial_h*spatial_w ({spatial_h}*{spatial_w}={expected})"
        )
    heatmap = heatmap.reshape(B, spatial_h, spatial_w)
    # Normalize per-sample
    mn = heatmap.min(axis=(1, 2), keepdims=True)
    mx = heatmap.max(axis=(1, 2), keepdims=True)
    denom = np.where(mx - mn > 1e-8, mx - mn, 1.0)
    return ((heatmap - mn) / denom).astype(np.float32)


def overlay_heatmap_on_image(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a [H, W] heatmap (0-1) onto an RGB image using JET colormap.

    Returns an RGB uint8 image of the same spatial size as img_rgb.
    """
    import cv2

    h, w = img_rgb.shape[:2]
    heat_u8 = (heatmap * 255).astype(np.uint8)
    heat_resized = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = (alpha * colored_rgb + (1 - alpha) * img_rgb).astype(np.uint8)
    return blended
