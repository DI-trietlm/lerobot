#!/usr/bin/env python3
"""
XAI — SmolVLA Occlusion Sensitivity (causal grounding of the predicted action).

Unlike raw attention (which on this SmolVLA checkpoint is saliency-dominated and
NOT grounded per word), occlusion sensitivity is a *causal* test: we hide one
region of the camera image at a time and measure how much the model's predicted
action changes. Regions whose occlusion changes the action the most are the
regions the model actually relies on to decide what to do.

Key detail: flow-matching sampling is stochastic, so the SAME fixed noise tensor
is reused for the baseline and every occluded run — otherwise the action
difference would be dominated by sampling noise, not by the occlusion.

Because each occluded cell needs a full action sample (~65 forwards per image),
this produces **static figures on a few key frames**, not a full video.

Usage
-----
    # quick timing / sanity check (1 frame, 1 camera, coarse grid)
    python3 xai_smolvla_occlusion.py --device cpu --frames 0 --cameras camera1 --grid 6

    # the real thing: a few representative frames, both cameras
    python3 xai_smolvla_occlusion.py --device cpu \
        --obs-dir ../recorded_obs-05-06-03 \
        --instruction "Put orange cup into white box." \
        --frames 0 16 32 --cameras camera1 camera2 --grid 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (_REPO / "src", _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse the loaders/overlay helpers from the attention script.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("_attn", _HERE / "xai_smolvla_attention_video.py")
_attn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_attn)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

OUTPUT_DIR = _HERE / "outputs"

DEFAULT_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def base_input_image(frame_t: torch.Tensor, size) -> torch.Tensor:
    """resize-with-pad a [1,3,H,W] frame in [0,1] to the model input size, kept in [0,1]."""
    return resize_with_pad(frame_t, *size, pad_value=0).clamp(0, 1)


def predict_action(policy, base_imgs, cam_keys, lang_tokens, lang_masks, noise, device, state=None):
    """Sample the action chunk for a set of (already model-sized, [0,1]) camera
    images, using a FIXED noise tensor. Returns a numpy array [chunk, action_dim]."""
    batch = {OBS_LANGUAGE_TOKENS: lang_tokens, OBS_LANGUAGE_ATTENTION_MASK: lang_masks}
    for k in cam_keys:
        batch[k] = base_imgs[k]
    if state is None:
        batch[OBS_STATE] = torch.zeros(1, policy.config.max_state_dim, device=device, dtype=torch.float32)
    else:
        batch[OBS_STATE] = state.to(device=device, dtype=torch.float32)

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state, noise=noise
    )
    adim = policy.config.action_feature.shape[0]
    return actions[:, :, :adim][0].detach().to("cpu", torch.float32).numpy()


def _iter_cells(s, grid):
    cell = s // grid
    for r in range(grid):
        for c in range(grid):
            y0, x0 = r * cell, c * cell
            y1 = s if r == grid - 1 else (r + 1) * cell
            x1 = s if c == grid - 1 else (c + 1) * cell
            yield r, c, y0, y1, x0, x1


def occlusion_sensitivity(policy, base_imgs, cam_keys, target_key, lang_tokens, lang_masks,
                          noise, grid, device, base_action, state=None):
    """Slide a gray patch over the target camera image; return a (grid, grid) map of
    L2 action change vs the unoccluded baseline."""
    img = base_imgs[target_key]  # [1,3,S,S] in [0,1]
    s = img.shape[-1]
    fill = img.mean().item()  # neutral gray = image mean

    sens = np.zeros((grid, grid), np.float32)
    for r, c, y0, y1, x0, x1 in _iter_cells(s, grid):
        occ = img.clone()
        occ[:, :, y0:y1, x0:x1] = fill
        occ_imgs = dict(base_imgs)
        occ_imgs[target_key] = occ
        act = predict_action(policy, occ_imgs, cam_keys, lang_tokens, lang_masks, noise, device, state=state)
        sens[r, c] = float(np.linalg.norm(act - base_action))
    return sens


def occlusion_word_effect(policy, base_imgs, cam_keys, target_key,
                          full_tokens, full_masks, abl_tokens, abl_masks,
                          noise, grid, device, e_base, state=None):
    """Isolate where a specific instruction word is grounded.

    The word's effect on the action is E(image) = action(full) - action(ablated).
    For each occluded cell we recompute E and measure how much it moved from the
    unoccluded baseline e_base. A large move means the word's effect depends on
    that region — i.e. that is where the word is grounded.
    """
    img = base_imgs[target_key]
    s = img.shape[-1]
    fill = img.mean().item()

    eff = np.zeros((grid, grid), np.float32)
    for r, c, y0, y1, x0, x1 in _iter_cells(s, grid):
        occ = img.clone()
        occ[:, :, y0:y1, x0:x1] = fill
        occ_imgs = dict(base_imgs)
        occ_imgs[target_key] = occ
        a_full = predict_action(policy, occ_imgs, cam_keys, full_tokens, full_masks, noise, device, state=state)
        a_abl = predict_action(policy, occ_imgs, cam_keys, abl_tokens, abl_masks, noise, device, state=state)
        eff[r, c] = float(np.linalg.norm((a_full - a_abl) - e_base))
    return eff


def ablate_word(text: str, word: str) -> str:
    """Remove a word (case-insensitive, whole word) from the instruction."""
    import re

    out = re.sub(rf"\b{re.escape(word)}\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def _load_metadata_states(obs_dir: Path, state_keys: list[str]) -> dict[int, torch.Tensor]:
    meta_path = obs_dir / "metadata.jsonl"
    if not meta_path.exists():
        return {}
    states = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            state = row.get("state")
            timestep = row.get("timestep")
            if not isinstance(state, dict) or timestep is None:
                continue
            try:
                values = [float(state[k]) for k in state_keys]
            except KeyError:
                continue
            states[int(timestep)] = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    return states


def _state_for_frame(frame_path: Path, state_by_timestep: dict[int, torch.Tensor]) -> torch.Tensor | None:
    if not state_by_timestep:
        return None
    timestep = int(frame_path.stem)
    if timestep in state_by_timestep:
        return state_by_timestep[timestep]
    nearest = min(state_by_timestep, key=lambda t: abs(t - timestep))
    return state_by_timestep[nearest]


def main(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    obs_dir = Path(args.obs_dir).resolve()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 64)
    print("SmolVLA Occlusion Sensitivity (action grounding)")
    print("=" * 64)
    print(f"Device: {device}  |  Model: {args.model}")

    policy = SmolVLAPolicy.from_pretrained(args.model)
    policy.to(device)
    policy.float()
    policy.eval()

    cam_keys = list(policy.config.image_features.keys())
    n_render = min(len(args.cameras), len(cam_keys))
    print(f"image_features: {cam_keys}")

    lang_tokens, lang_masks, _toks = _attn.tokenize_instruction(policy, args.instruction, device)

    word_mode = args.word is not None
    if word_mode:
        abl_text = args.instruction_ablated or ablate_word(args.instruction, args.word)
        abl_tokens, abl_masks, _ = _attn.tokenize_instruction(policy, abl_text, device)
        print(f'word-contrast mode: "{args.instruction}"  vs  "{abl_text}"  (isolating "{args.word}")')

    size = policy.config.resize_imgs_with_padding or (512, 512)
    # Disable in-policy resize: we feed images already at model size.
    policy.config.resize_imgs_with_padding = None

    # Fixed noise so occlusion is the ONLY thing that changes the sampled action.
    torch.manual_seed(args.seed)
    noise = torch.randn(
        1, policy.config.chunk_size, policy.config.max_action_dim, device=device, dtype=torch.float32
    )

    frame_lists = {
        args.cameras[b]: sorted((obs_dir / "images" / args.cameras[b]).glob("*.png"))
        for b in range(n_render)
    }
    state_keys = [item.strip() for item in args.state_keys.split(",") if item.strip()]
    state_by_timestep = _load_metadata_states(obs_dir, state_keys) if args.use_metadata_state else {}
    if args.use_metadata_state:
        print(f"metadata states loaded: {len(state_by_timestep)}")

    for f_idx in args.frames:
        t0 = time.time()
        # Build base (model-sized) images for every model camera input.
        base_imgs = {}
        raw_for_disp = {}
        for b, key in enumerate(cam_keys):
            if b < n_render:
                cam = args.cameras[b]
                ft = _attn.load_frame_tensor(frame_lists[cam][f_idx], device, torch.float32)
                base_imgs[key] = base_input_image(ft, size)
                raw_for_disp[b] = base_imgs[key]
            else:
                base_imgs[key] = torch.zeros(1, 3, *size, device=device, dtype=torch.float32)

        state = _state_for_frame(frame_lists[args.cameras[0]][f_idx], state_by_timestep)
        base_action = predict_action(
            policy, base_imgs, cam_keys, lang_tokens, lang_masks, noise, device, state=state
        )
        if word_mode:
            a_abl_base = predict_action(
                policy, base_imgs, cam_keys, abl_tokens, abl_masks, noise, device, state=state
            )
            e_base = base_action - a_abl_base  # the word's effect on the action (unoccluded)

        # One figure per frame: source + occlusion overlay for each analysed camera.
        fig, axes = plt.subplots(2, n_render, figsize=(n_render * 4.2, 8.2), squeeze=False)
        for b in range(n_render):
            cam = args.cameras[b]
            key = cam_keys[b]
            if word_mode:
                smap = occlusion_word_effect(
                    policy, base_imgs, cam_keys, key, lang_tokens, lang_masks,
                    abl_tokens, abl_masks, noise, args.grid, device, e_base, state=state,
                )
                row_title = f'{cam}  "{args.word}"-effect grounding (|Δ word-effect|)'
            else:
                smap = occlusion_sensitivity(
                    policy, base_imgs, cam_keys, key, lang_tokens, lang_masks,
                    noise, args.grid, device, base_action, state=state,
                )
                row_title = f"{cam}  occlusion sensitivity (|Δaction|)"
            disp = (raw_for_disp[b][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            overlay = _attn.overlay_heatmap(disp, _attn.normalize01(smap), args.alpha, args.cmap)

            axes[0, b].imshow(disp)
            state_label = "metadata-state" if state is not None else "zero-state"
            axes[0, b].set_title(
                f"{cam}  (frame {frame_lists[cam][f_idx].name}, {state_label})", fontsize=10
            )
            axes[0, b].axis("off")
            axes[1, b].imshow(overlay)
            axes[1, b].set_title(row_title, fontsize=10)
            axes[1, b].axis("off")

        if word_mode:
            suptitle = (
                f'{args.model}   |   "{args.instruction}"\n'
                f'Where the word "{args.word}" is grounded — bright = occluding this region '
                f'most changes the action-difference caused by "{args.word}" '
                f"(grid {args.grid}x{args.grid}, fixed noise)"
            )
        else:
            suptitle = (
                f'{args.model}   |   "{args.instruction}"\n'
                f"Occlusion grounding — bright = occluding this region changes the action most "
                f"(grid {args.grid}x{args.grid}, fixed noise)"
            )
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        tag = f"_word-{args.word}" if word_mode else ""
        out = OUTPUT_DIR / f"smolvla_occlusion{tag}_{obs_dir.name}_f{f_idx:03d}.png"
        fig.savefig(str(out), dpi=110)
        plt.close(fig)
        print(f"  frame {f_idx}: saved {out.name}  ({time.time() - t0:.1f}s)")

    print("Done.")


def parse_args():
    p = argparse.ArgumentParser(
        description="SmolVLA occlusion sensitivity (causal action grounding).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--obs-dir", default=str(_REPO / "recorded_obs-05-06-03"))
    p.add_argument("--model", default="di-techinnova/smolvla-pick-cup-0.2")
    p.add_argument("--instruction", default="Put orange cup into white box.")
    p.add_argument("--cameras", nargs="+", default=["camera1", "camera2"])
    p.add_argument("--frames", nargs="+", type=int, default=[0, 16, 32],
                   help="Frame indices to analyse (static figures)")
    p.add_argument("--grid", type=int, default=8, help="Occlusion grid resolution (cells per side)")
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--seed", type=int, default=0, help="Fixed flow-matching noise seed")
    p.add_argument("--use-metadata-state", action=argparse.BooleanOptionalAction, default=True,
                   help="Use metadata.jsonl state for the analysed frame when available")
    p.add_argument("--state-keys", default=",".join(DEFAULT_STATE_KEYS),
                   help="Comma-separated state key order for metadata.jsonl")
    p.add_argument("--word", default=None,
                   help="Isolate where this instruction word is grounded (word-contrast mode). "
                        "Doubles runtime: runs full vs ablated instruction per occlusion.")
    p.add_argument("--instruction-ablated", default=None,
                   help="Override the ablated instruction (default: --instruction with --word removed)")
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
