#!/usr/bin/env python3
"""
XAI — SmolVLA Cross / Self Attention Map for recorded observation frames.

Turns a `recorded_obs-*` directory (per-camera PNG frame sequences) into one
annotated MP4 *per camera*, where every frame reproduces the style of the
reference figure from "Towards Understanding Cross and Self-Attention in Stable
Diffusion for Text-Guided Image Editing":

    row 1 (cross-attention):  [source] [word_0] [word_1] ... [word_W]
    row 2 (self-attention):   [source] [Top-0 ] [Top-1 ] ... [Top-K ]

How it works
------------
SmolVLA's prefix sequence is  [img_cam0 | img_cam1 | language | state].  During
the `fill_kv_cache` prefix pass *every* VLM layer runs full self-attention over
that sequence (see `SmolVLMWithExpertModel.forward`), so a single forward gives
us both:

  * cross-attention  = language-token rows  ->  image-patch columns
  * self-attention   = image-patch rows     ->  image-patch columns (per camera)

We monkey-patch the model's `eager_attention_forward` to capture the softmax
probabilities, average them over heads and over all 16 layers, then slice the
[P, P] prefix attention matrix into the cross / self blocks.

The recorded data has no robot state vector, so the single state token is fed as
zeros — it does not affect the language<->image attention we visualise.

This script does NOT need a GPU-less machine to be useful: like the other xai/
scripts it is meant to run on the GPU server where the model weights live.

Usage
-----
    # smoke test: load model, map cameras, show token boundaries, then exit
    python3 xai_smolvla_attention_video.py --obs-dir ../recorded_obs-05-06-03 --dry-run

    # full run (two videos written to xai/outputs/)
    python3 xai_smolvla_attention_video.py \
        --obs-dir ../recorded_obs-05-06-03 \
        --model di-techinnova/smolvla-pick-cup-0.2 \
        --instruction "Put orange cup into white box." \
        --fps 5 --topk 6 --max-words 6
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Make `src/` importable when run from anywhere inside the repo.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (_REPO / "src", _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

OUTPUT_DIR = _HERE / "outputs"


# --------------------------------------------------------------------------- #
# Attention capture
# --------------------------------------------------------------------------- #
def _capturing_eager_attention_forward(
    self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
):
    """Drop-in replacement for `eager_attention_forward` that also records the
    head-averaged attention probabilities into `self._captured_attn`.

    The math is identical to the original implementation in
    `smolvlm_with_expert.py`; we only add the `probs` capture.
    """
    num_att_heads = self.num_attention_heads
    num_key_value_heads = self.num_key_value_heads
    num_key_value_groups = num_att_heads // num_key_value_heads

    sequence_length = key_states.shape[1]

    key_states = key_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    key_states = key_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )

    value_states = value_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    value_states = value_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )

    query_states = query_states.to(dtype=torch.float32)
    key_states = key_states.to(dtype=torch.float32)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    att_weights *= head_dim**-0.5

    att_weights = att_weights.to(dtype=torch.float32)
    big_neg = torch.finfo(att_weights.dtype).min
    masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
    probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)

    # --- capture: mean over heads, first batch element -> [Q, K] ----------- #
    if getattr(self, "_capture_enabled", False):
        self._captured_attn.append(probs[0].mean(dim=0).detach().to("cpu", torch.float32).numpy())

    probs = probs.to(dtype=value_states.dtype)
    att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
    att_output = att_output.permute(0, 2, 1, 3)
    att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
    return att_output


def install_attention_capture(policy: SmolVLAPolicy):
    """Patch the VLM-with-expert so its attention interface records probs."""
    vlm = policy.model.vlm_with_expert
    vlm._captured_attn = []
    vlm._capture_enabled = False
    vlm.eager_attention_forward = types.MethodType(_capturing_eager_attention_forward, vlm)
    return vlm


# --------------------------------------------------------------------------- #
# Batch construction
# --------------------------------------------------------------------------- #
def resolve_camera_keys(policy: SmolVLAPolicy):
    """Return the policy image-feature keys in a stable order."""
    return list(policy.config.image_features.keys())


def load_frame_tensor(path: Path, device, dtype) -> torch.Tensor:
    """Load a PNG as a (1, 3, H, W) float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H, W, 3
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1, 3, H, W
    return t.to(device=device, dtype=dtype)


def tokenize_instruction(policy: SmolVLAPolicy, instruction: str, device):
    """Tokenize the task string the same way the SmolVLA processor would
    (NewLineTaskProcessorStep appends a newline)."""
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    text = instruction.rstrip("\n") + "\n"
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
    )
    input_ids = enc["input_ids"].to(device)
    # SmolVLA's attention path uses the language mask as a boolean pad mask.
    attn_mask = enc["attention_mask"].to(device).bool()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return input_ids, attn_mask, tokens


def clean_token_label(tok: str) -> str:
    """Strip the SentencePiece / GPT2 markers for display and force an
    ASCII-safe label (Windows consoles use cp1252 and choke on Ġ/Ċ/▁)."""
    label = (
        tok.replace("Ġ", "")
        .replace("▁", "")
        .replace("Ċ", "")  # GPT2-style newline byte token (U+010A)
        .replace("\n", "")
        .strip()
    )
    return label.encode("ascii", "ignore").decode("ascii")


def select_word_columns(tokens, lang_mask_row, max_words: int):
    """Pick which language-token positions to show as cross-attention columns.

    Keeps real (masked-in) tokens, drops obvious special / empty tokens, and
    caps the count at `max_words`.
    """
    specials = {"<s>", "</s>", "<|endoftext|>", "<pad>", "<unk>", "<end_of_utterance>"}
    chosen = []
    for j, tok in enumerate(tokens):
        if j >= len(lang_mask_row) or lang_mask_row[j] == 0:
            continue
        label = clean_token_label(tok)
        if not label or tok in specials:
            continue
        chosen.append((j, label))
    if len(chosen) > max_words:
        chosen = chosen[:max_words]
    return chosen


# --------------------------------------------------------------------------- #
# Heatmap helpers
# --------------------------------------------------------------------------- #
def normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def overlay_heatmap(disp_rgb: np.ndarray, grid: np.ndarray, alpha: float, cmap: str) -> np.ndarray:
    """Blend a small (g, g) heatmap grid over an RGB uint8 image."""
    h, w = disp_rgb.shape[:2]
    heat = cv2.resize(normalize01(grid), (w, h), interpolation=cv2.INTER_CUBIC)
    colored = (matplotlib.colormaps[cmap](heat)[..., :3] * 255.0).astype(np.float32)
    blended = alpha * colored + (1.0 - alpha) * disp_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def model_input_image(frame_t: torch.Tensor, policy: SmolVLAPolicy) -> np.ndarray:
    """Reproduce the exact image the model sees (resize-with-pad to e.g. 512x512)
    so the attention grid aligns spatially. Returns an RGB uint8 array."""
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

    img = frame_t
    if policy.config.resize_imgs_with_padding is not None:
        img = resize_with_pad(img, *policy.config.resize_imgs_with_padding, pad_value=0)
    arr = img[0].permute(1, 2, 0).detach().to("cpu", torch.float32).numpy()  # H, W, 3 in [0,1]
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def cross_attention_maps(attn, lang_start, word_cols, cam_start, cam_end, grid_side, mode):
    """Per-word language->image maps.

    mode="raw":      attention as-is (dominated by attention-sink saliency, so
                     every word looks alike).
    mode="contrast": divide each patch column by its overall saliency to remove
                     the sink bias, then subtract the mean-over-words baseline and
                     keep the positive part — i.e. where THIS word attends *more*
                     than the average word. Reveals word-specific structure.
    """
    rows = np.stack([attn[lang_start + j, cam_start:cam_end] for (j, _l) in word_cols])  # [W, N]
    if mode == "contrast" and len(rows) > 1:
        saliency = attn[:, cam_start:cam_end].mean(axis=0)  # avg attention each patch receives
        rows = rows / (saliency[None, :] + 1e-8)
        rows = rows - rows.mean(axis=0, keepdims=True)
        rows = np.clip(rows, 0.0, None)
    return [normalize01(rows[i].reshape(grid_side, grid_side)) for i in range(len(rows))]


def self_attention_components(block: np.ndarray, grid_side: int, topk: int, center: bool = True):
    """SVD of the image->image attention block; return top-K right singular
    vectors reshaped to (grid, grid) and normalized to [0, 1].

    center=True subtracts the per-column mean first, which removes the dominant
    saliency mode (otherwise Top-0 is just the attention-sink pattern that also
    shows up in cross-attention), exposing the structural / segmentation modes.
    """
    b = block.astype(np.float64)
    if center:
        b = b - b.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(b, full_matrices=False)
    except np.linalg.LinAlgError:
        return [np.zeros((grid_side, grid_side), np.float32) for _ in range(topk)]
    comps = []
    for i in range(min(topk, vt.shape[0])):
        comp = np.abs(vt[i]).reshape(grid_side, grid_side)
        comps.append(normalize01(comp))
    while len(comps) < topk:
        comps.append(np.zeros((grid_side, grid_side), np.float32))
    return comps


# --------------------------------------------------------------------------- #
# Per-frame attention
# --------------------------------------------------------------------------- #
def reduce_layers(attn_stack: np.ndarray, layer) -> np.ndarray:
    """Collapse the per-layer attention stack [L, P, P] into one [P, P] matrix."""
    if layer == "mean":
        return attn_stack.mean(axis=0)
    if layer == "last":
        return attn_stack[-1]
    return attn_stack[int(layer)]


def compute_prefix_attention(policy, batch, vlm):
    """Run only the SmolVLA prefix self-attention pass and return the
    per-layer (head-averaged) attention stack [L, P, P]."""
    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    vlm._captured_attn = []
    vlm._capture_enabled = True
    model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=policy.config.use_cache,
        fill_kv_cache=True,
    )
    vlm._capture_enabled = False

    attn_stack = np.stack(vlm._captured_attn, axis=0)  # [num_layers, P, P]
    return attn_stack, len(images)  # [L, P, P], num_cameras


def derive_token_layout(prefix_len, num_cameras, lang_len):
    """Recover per-camera image block size N and language start index."""
    # prefix_len = num_cameras * N + lang_len + 1(state)
    rem = prefix_len - lang_len - 1
    assert rem % num_cameras == 0, f"Cannot split {rem} image tokens across {num_cameras} cameras"
    n = rem // num_cameras
    grid_side = int(round(n**0.5))
    assert grid_side * grid_side == n, f"Image token count {n} is not a perfect square"
    lang_start = num_cameras * n
    return n, grid_side, lang_start


# --------------------------------------------------------------------------- #
# Figure rendering
# --------------------------------------------------------------------------- #
def render_figure(
    disp_rgb,
    word_cols,
    word_heatmaps,
    self_components,
    instruction,
    frame_name,
    camera_name,
    model_name,
    alpha,
    cmap,
    cross_label="Cross-attention\n(text -> image)",
    self_label="Self-attention\n(SVD top-k)",
    fig_dpi=100,
) -> np.ndarray:
    """Build the 2-row reference-style figure and return it as a BGR uint8 array."""
    n_words = len(word_cols)
    topk = len(self_components)
    ncols = 1 + max(n_words, topk)

    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 2.2, 5.0), dpi=fig_dpi)
    if axes.ndim == 1:
        axes = axes[None, :]

    for ax in axes.ravel():
        ax.axis("off")

    # Row 0 — cross-attention
    axes[0, 0].imshow(disp_rgb)
    axes[0, 0].set_title("Source", fontsize=10)
    axes[0, 0].set_ylabel(cross_label, fontsize=9)
    axes[0, 0].axis("on")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    for c, ((_, label), heat) in enumerate(zip(word_cols, word_heatmaps), start=1):
        axes[0, c].imshow(overlay_heatmap(disp_rgb, heat, alpha, cmap))
        axes[0, c].set_title(label, fontsize=10)

    # Row 1 — self-attention SVD components
    axes[1, 0].imshow(disp_rgb)
    axes[1, 0].set_title("Source", fontsize=10)
    axes[1, 0].set_ylabel(self_label, fontsize=9)
    axes[1, 0].axis("on")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    for c, comp in enumerate(self_components, start=1):
        axes[1, c].imshow(overlay_heatmap(disp_rgb, comp, alpha, cmap))
        axes[1, c].set_title(f"Top-{c - 1}", fontsize=10)

    fig.suptitle(
        f'{model_name}   |   "{instruction}"\ncamera: {camera_name}   frame: {frame_name}',
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    rgb = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def list_frames(obs_dir: Path, camera: str):
    cam_dir = obs_dir / "images" / camera
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"Camera directory not found: {cam_dir}")
    return sorted(cam_dir.glob("*.png"))


def main(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    obs_dir = Path(args.obs_dir).resolve()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 64)
    print("SmolVLA Cross / Self Attention Video")
    print("=" * 64)
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Obs dir: {obs_dir}")

    print("\n[1] Loading SmolVLA policy ...")
    policy = SmolVLAPolicy.from_pretrained(args.model)
    policy.to(device)
    # The VLM loads in bfloat16 but the action-expert projections are float32.
    # Unify everything to float32 — consistent dtype across the concatenated
    # prefix embeddings, and the right choice for CPU inference.
    policy.float()
    policy.eval()
    vlm = install_attention_capture(policy)
    model_dtype = torch.float32

    cam_keys = resolve_camera_keys(policy)
    print(f"    image_features ({len(cam_keys)}): {cam_keys}")

    # Recorded camera folders we will render (one video each).
    recorded_cams = args.cameras
    n_render = min(len(recorded_cams), len(cam_keys))
    if len(recorded_cams) > len(cam_keys):
        print(f"    [warn] {len(recorded_cams)} recorded cameras but model has only "
              f"{len(cam_keys)} image inputs. Rendering the first {n_render}.")
    if len(cam_keys) > n_render:
        print(f"    [info] model expects {len(cam_keys)} image inputs; the extra "
              f"{len(cam_keys) - n_render} will be fed black frames (not rendered).")

    print("\n[2] Tokenizing instruction ...")
    input_ids, lang_mask, tokens = tokenize_instruction(policy, args.instruction, device)
    print(f"    tokens: {[clean_token_label(t) for t in tokens]}")

    frame_lists = {cam: list_frames(obs_dir, cam) for cam in recorded_cams[:n_render]}
    n_frames = min(len(v) for v in frame_lists.values())
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    print(f"\n[3] {n_frames} frames per camera")

    state_zeros = torch.zeros(1, policy.config.max_state_dim, device=device, dtype=model_dtype)

    def build_batch(f_idx):
        """Full batch covering every model image input (real frames in the first
        `n_render` blocks, black frames after). Returns (batch, per-block display tensors)."""
        batch = {OBS_LANGUAGE_TOKENS: input_ids, OBS_LANGUAGE_ATTENTION_MASK: lang_mask}
        block_frames = []
        for b_idx, key in enumerate(cam_keys):
            if b_idx < n_render:
                t = load_frame_tensor(frame_lists[recorded_cams[b_idx]][f_idx], device, model_dtype)
            else:
                ref = block_frames[0]
                t = torch.zeros_like(ref)
            batch[key] = t
            block_frames.append(t)
        batch[OBS_STATE] = state_zeros
        return batch, block_frames

    if args.dry_run:
        batch, _ = build_batch(0)
        with torch.no_grad():
            attn_stack, num_cams = compute_prefix_attention(policy, batch, vlm)
        attn = reduce_layers(attn_stack, args.layer)
        n, grid_side, lang_start = derive_token_layout(attn.shape[0], num_cams, input_ids.shape[1])
        print(f"    layers={attn_stack.shape[0]}  prefix_len={attn.shape[0]}  cameras={num_cams}  "
              f"tokens/cam={n} ({grid_side}x{grid_side})  lang_start={lang_start}")
        print(f"    layer={args.layer}  cross_mode={args.cross_mode}  self_mode={args.self_mode}")
        print("\n[Dry run] OK - exiting before video render.")
        return

    word_cols = select_word_columns(tokens, lang_mask[0].tolist(), args.max_words)
    print(f"    cross-attention columns: {[w[1] for w in word_cols]}")
    cross_label = (
        "Cross-attention\n(text -> image,\nsaliency-debiased)" if args.cross_mode == "contrast"
        else "Cross-attention\n(text -> image, raw)"
    )
    self_label = (
        "Self-attention\n(SVD, saliency-\nremoved)" if args.self_mode == "centered"
        else "Self-attention\n(SVD top-k, raw)"
    )

    writers = {}
    try:
        for f_idx in range(n_frames):
            # One forward per frame; the prefix attention matrix already covers
            # every camera block, so we render all recorded cameras from it.
            batch, block_frames = build_batch(f_idx)
            with torch.no_grad():
                attn_stack, num_cams = compute_prefix_attention(policy, batch, vlm)
            attn = reduce_layers(attn_stack, args.layer)
            n, grid_side, lang_start = derive_token_layout(attn.shape[0], num_cams, input_ids.shape[1])

            for block_idx in range(n_render):
                cam = recorded_cams[block_idx]
                frame_name = frame_lists[cam][f_idx].name
                cam_start, cam_end = block_idx * n, (block_idx + 1) * n

                word_heatmaps = cross_attention_maps(
                    attn, lang_start, word_cols, cam_start, cam_end, grid_side, args.cross_mode
                )
                block = attn[cam_start:cam_end, cam_start:cam_end]
                self_components = self_attention_components(
                    block, grid_side, args.topk, center=(args.self_mode == "centered")
                )

                disp_rgb = model_input_image(block_frames[block_idx], policy)
                fig_bgr = render_figure(
                    disp_rgb, word_cols, word_heatmaps, self_components,
                    args.instruction, frame_name, cam, args.model, args.alpha, args.cmap,
                    cross_label=cross_label, self_label=self_label,
                )

                if cam not in writers:
                    h, w = fig_bgr.shape[:2]
                    out_path = OUTPUT_DIR / f"smolvla_attn_{obs_dir.name}_{cam}.mp4"
                    writers[cam] = (
                        cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                        args.fps, (w, h)),
                        out_path,
                    )
                writers[cam][0].write(fig_bgr)

            print(f"\r    rendered {f_idx + 1}/{n_frames} frames", end="")
    finally:
        print()
        for cam, (writer, out_path) in writers.items():
            writer.release()
            print(f"    saved {out_path}")

    print("\nDone.")


def parse_args():
    p = argparse.ArgumentParser(
        description="SmolVLA cross/self attention video over recorded observation frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--obs-dir", default=str(_REPO / "recorded_obs-05-06-03"),
                   help="recorded_obs-* directory containing images/<camera>/*.png")
    p.add_argument("--model", default="di-techinnova/smolvla-pick-cup-0.2",
                   help="SmolVLA model id or local path")
    p.add_argument("--instruction", default="Put orange cup into white box.",
                   help="Task instruction text")
    p.add_argument("--cameras", nargs="+", default=["camera1", "camera2"],
                   help="Recorded camera folder names, mapped in order to model image inputs")
    p.add_argument("--fps", type=float, default=5.0, help="Output video FPS")
    p.add_argument("--topk", type=int, default=6, help="Number of self-attention SVD components")
    p.add_argument("--max-words", type=int, default=6, help="Max cross-attention word columns")
    p.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay opacity [0-1]")
    p.add_argument("--cmap", default="turbo", help="Matplotlib colormap")
    p.add_argument("--layer", default="mean",
                   help="Which attention layer: 'mean', 'last', or an int index")
    p.add_argument("--cross-mode", choices=["contrast", "raw"], default="raw",
                   help="raw: honest attention (saliency-dominated, words look alike); "
                        "contrast: saliency-debiased per-word maps (WARNING: when raw words "
                        "are near-identical this mostly amplifies noise and can mislead)")
    p.add_argument("--self-mode", choices=["centered", "raw"], default="raw",
                   help="raw: plain SVD (Top-0 is the saliency mode); "
                        "centered: mean-center before SVD to expose structural modes")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--max-frames", type=int, default=None, help="Limit number of frames")
    p.add_argument("--dry-run", action="store_true",
                   help="Load model, probe one frame's token layout, then exit")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
