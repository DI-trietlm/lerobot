"""Visualize attention heatmaps captured from X-VLA inference.

Usage
-----
# View Stage 1 (Florence-2): text tokens attending to visual patches
python scripts/visualize_attention.py \\
    --input_dir attention_captures \\
    --stage florence2 --mode text_to_visual --layer 6 \\
    --output_dir heatmap_out

# View Stage 3 (SoftPromptedTransformer): action tokens attending to VLM tokens
python scripts/visualize_attention.py \\
    --input_dir attention_captures \\
    --stage soft_transformer --mode action_to_vlm --layer 12 \\
    --output_dir heatmap_out --make_video

Modes
-----
florence2 stage:
  text_to_visual   — rows=text tokens, cols=visual patches (what does text "look at"?)
  visual_self      — rows=visual patches, cols=visual patches (visual self-attention)

soft_transformer stage:
  action_to_vlm         — rows=action tokens, cols=vlm tokens
  action_to_softprompt  — rows=action tokens, cols=soft prompt tokens
  action_to_aux         — rows=action tokens, cols=aux visual tokens
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import json
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_timestep(ts_dir: Path):
    meta_path = ts_dir / "meta.json"
    if not meta_path.exists():
        return None, None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta, ts_dir


def list_timestep_dirs(input_dir: Path):
    dirs = sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit())
    return dirs


def load_attn(ts_dir: Path, stage: str, layer: int) -> np.ndarray | None:
    fname = ts_dir / f"attn_{stage}_layer{layer:02d}.npy"
    if not fname.exists():
        return None
    return np.load(str(fname)).astype(np.float32)


def load_image_for_mode(ts_dir: Path, meta: dict, stage: str, mode: str) -> np.ndarray | None:
    """Load the correct camera image for the given stage/mode.

    florence2 stages and action_to_vlm → primary camera (cam1, merged with text in Florence-2).
    action_to_aux → first aux camera (cam2, fed directly to SoftPromptedTransformer).
    Falls back to first image found if explicit names are not in meta.
    """
    use_aux = stage == "soft_transformer" and mode == "action_to_aux"

    if use_aux:
        aux_list = meta.get("aux_images", [])
        candidates = [ts_dir / name for name in aux_list if name]
    else:
        primary = meta.get("primary_image", "")
        candidates = [ts_dir / primary] if primary else []

    for p in candidates:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Fallback: any PNG that looks like a camera image (old format or any naming)
    for p in sorted(ts_dir.glob("*.png")):
        if any(tok in p.name for tok in ("observation", "camera", "image", "cam_")):
            img = cv2.imread(str(p))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def attn_to_heatmap(attn: np.ndarray, query_start: int, query_end: int,
                    key_start: int, key_end: int,
                    spatial_h: int, spatial_w: int) -> np.ndarray:
    """Average attention over heads and query tokens, reshape to spatial grid."""
    sliced = attn[:, :, query_start:query_end, key_start:key_end]  # [B, H, Q, K]
    heatmap = sliced.mean(axis=(0, 1, 2))                           # [K]
    expected = spatial_h * spatial_w
    if heatmap.shape[0] != expected:
        # Fallback: take first expected tokens if mismatch
        heatmap = heatmap[:expected]
    heatmap = heatmap.reshape(spatial_h, spatial_w)
    mn, mx = heatmap.min(), heatmap.max()
    if mx - mn > 1e-8:
        heatmap = (heatmap - mn) / (mx - mn)
    return heatmap.astype(np.float32)


def overlay(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    heat_u8 = (heatmap * 255).astype(np.uint8)
    heat_resized = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (alpha * colored_rgb + (1 - alpha) * img_rgb).astype(np.uint8)


def spatial_dims_from_meta(meta: dict, key_start: int, key_end: int):
    """Return (spatial_h, spatial_w, skip_global) using meta stored at capture time.

    T_vis includes 1 leading global_avg_pool token that is NOT spatial.
    Spatial tokens are at positions [T_vis_global : T_vis] within each view block.
    """
    spatial_h = meta.get("spatial_h", 0)
    spatial_w = meta.get("spatial_w", 0)
    if spatial_h > 0 and spatial_w > 0:
        return spatial_h, spatial_w
    # Fallback: guess from key range, assume square, no global token
    n = key_end - key_start
    sq = math.isqrt(n)
    if sq * sq == n:
        return sq, sq
    for h in range(sq, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_heatmap_for_timestep(ts_dir: Path, meta: dict, stage: str, mode: str,
                                layer: int) -> np.ndarray | None:
    attn = load_attn(ts_dir, stage, layer)
    if attn is None:
        print(f"  [warn] No attention file for stage={stage} layer={layer} in {ts_dir.name}")
        return None

    # Token boundaries
    T_vis = meta.get("T_vis", 0)
    T_text = meta.get("T_text", 0)
    num_actions = meta.get("num_actions", 0)
    T_vlm = meta.get("T_vlm", 0)
    T_aux = meta.get("T_aux", 0)
    len_soft_prompts = meta.get("len_soft_prompts", 0)

    # Global token info (spatial_avg_pool = 1 non-spatial leading token per view)
    T_vis_global = meta.get("T_vis_global", 0)  # typically 1 for X-VLA
    spatial_h = meta.get("spatial_h", 0)
    spatial_w = meta.get("spatial_w", 0)

    if stage == "florence2":
        # Florence-2 sequence: [global_token | spatial_patches(T_vis-1) | text_tokens(T_text)]
        # global_token = spatial_avg_pool (1 token, skip for spatial heatmap)
        # spatial_patches span [T_vis_global : T_vis]
        vis_spatial_start = T_vis_global          # skip global token
        vis_spatial_end   = T_vis                 # = 1 + spatial_h*spatial_w
        text_start        = T_vis
        text_end          = T_vis + T_text

        if mode == "text_to_visual":
            # text tokens attending to spatial visual patches
            if vis_spatial_end <= vis_spatial_start:
                print("  [warn] No spatial visual tokens found")
                return None
            sh, sw = spatial_dims_from_meta(meta, vis_spatial_start, vis_spatial_end)
            return attn_to_heatmap(attn, text_start, text_end,
                                   vis_spatial_start, vis_spatial_end, sh, sw)

        elif mode == "visual_self":
            # spatial patches attending to each other
            if vis_spatial_end <= vis_spatial_start:
                return None
            sh, sw = spatial_dims_from_meta(meta, vis_spatial_start, vis_spatial_end)
            return attn_to_heatmap(attn, vis_spatial_start, vis_spatial_end,
                                   vis_spatial_start, vis_spatial_end, sh, sw)

        else:
            print(f"  [warn] Unknown mode '{mode}' for stage 'florence2'. "
                  "Use: text_to_visual, visual_self")
            return None

    elif stage == "soft_transformer":
        # Sequence: [action(A) | vlm(V) | aux(X) | soft_prompt(S)]
        # Within vlm block: positions [A : A+T_vis_global] = global token,
        #                              [A+T_vis_global : A+T_vis] = spatial patches of cam1
        # Within aux block: same structure per extra camera view
        A = num_actions
        V = T_vlm
        X = T_aux
        S = len_soft_prompts
        seq_total = A + V + X + S

        if mode == "action_to_vlm":
            # action tokens → spatial patches of cam1 (skip global token inside vlm block)
            vis_key_start = A + T_vis_global
            vis_key_end   = A + T_vis    # cam1 spatial patches only
            if vis_key_end <= vis_key_start:
                return None
            sh, sw = spatial_dims_from_meta(meta, vis_key_start, vis_key_end)
            return attn_to_heatmap(attn, 0, A, vis_key_start, vis_key_end, sh, sw)

        elif mode == "action_to_aux":
            # action tokens → spatial patches of cam2 (skip global token inside aux block)
            if X == 0:
                return None
            aux_key_start = A + V + T_vis_global
            aux_key_end   = A + V + T_vis    # first extra camera's spatial patches
            sh, sw = spatial_dims_from_meta(meta, aux_key_start, aux_key_end)
            return attn_to_heatmap(attn, 0, A, aux_key_start, aux_key_end, sh, sw)

        elif mode == "action_to_softprompt":
            if S == 0:
                return None
            sh, sw = max(1, int(S**0.5)), max(1, math.ceil(S / max(1, int(S**0.5))))
            return attn_to_heatmap(attn, 0, A, seq_total - S, seq_total, sh, sw)

        else:
            print(f"  [warn] Unknown mode '{mode}' for stage 'soft_transformer'. "
                  "Use: action_to_vlm, action_to_aux, action_to_softprompt")
            return None

    else:
        print(f"  [warn] Unknown stage '{stage}'. Use: florence2, soft_transformer")
        return None


def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_dirs = list_timestep_dirs(input_dir)
    if not ts_dirs:
        print(f"No timestep directories found in {input_dir}")
        sys.exit(1)

    layers = list(range(12 if args.stage == "florence2" else 24)) if args.layer == "all" \
        else [int(args.layer)]

    frames = []

    for ts_dir in ts_dirs:
        meta, _ = load_timestep(ts_dir)
        if meta is None:
            continue
        img = load_image_for_mode(ts_dir, meta, args.stage, args.mode)
        if img is None:
            print(f"  [warn] No image found in {ts_dir.name}")
            continue

        for layer_idx in layers:
            heatmap = build_heatmap_for_timestep(ts_dir, meta, args.stage, args.mode, layer_idx)
            if heatmap is None:
                continue

            result = overlay(img, heatmap, alpha=0.5)

            # Label overlay
            label = f"t={ts_dir.name} | {args.stage} | L{layer_idx:02d} | {args.mode}"
            cv2.putText(
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

            out_path = output_dir / f"{ts_dir.name}_layer{layer_idx:02d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            frames.append(result)
            print(f"  Saved {out_path.name}")

    if args.make_video and frames:
        h, w = frames[0].shape[:2]
        video_path = output_dir / "attention_heatmap.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"\nVideo saved → {video_path}")

    print(f"\nDone. {len(frames)} frame(s) written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize X-VLA attention heatmaps")
    parser.add_argument("--input_dir", required=True, help="Directory with captured attention data")
    parser.add_argument("--output_dir", default="heatmap_out", help="Output directory for heatmap images")
    parser.add_argument(
        "--stage", required=True,
        choices=["florence2", "soft_transformer"],
        help="Which attention stage to visualize"
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["text_to_visual", "visual_self", "action_to_vlm", "action_to_aux", "action_to_softprompt"],
        help="Which attention slice to extract"
    )
    parser.add_argument(
        "--layer", default="all",
        help="Layer index (int) or 'all'. Florence-2 has 12 layers, SoftTransformer has 24."
    )
    parser.add_argument("--make_video", action="store_true", help="Render frames as mp4 video")
    parser.add_argument("--fps", type=float, default=5.0, help="Video FPS (default 5)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
