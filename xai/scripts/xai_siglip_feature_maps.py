"""
SigLIP feature map visualization for SmolVLA.

Visualizes residual-stream activation maps at selected SigLIP vision encoder
layers by hooking each layer's output, aggregating channels into a 2D map,
and overlaying it on the input image.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOLVLA_MODEL_DIR = os.path.join(PROJECT_DIR, "smolvla-model")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SIGLIP_INPUT_SIZE = (512, 512)
LAYERS_DEFAULT = [0, 3, 6, 9, 11]


AGGREGATION_MODES = {
    "centered_norm": "L2-norm of spatially-centered features (default, removes DC bias)",
    "norm": "Raw L2-norm across channels",
    "mean": "Mean across channels (fast but shows window attention artifacts)",
    "max": "Max across channels (emphasizes strongest activations)",
}


def _aggregate_channels(feat_hwc: torch.Tensor, mode: str) -> np.ndarray:
    """
    Aggregates a (H, W, C) feature tensor into a (H, W) activation scalar map.

    'centered_norm' subtracts the per-channel spatial mean to remove
    positional DC bias from window attention.
    """
    if mode == "centered_norm":
        centered = feat_hwc - feat_hwc.mean(dim=[0, 1], keepdim=True)
        return centered.norm(dim=-1).numpy()
    if mode == "norm":
        return feat_hwc.norm(dim=-1).numpy()
    if mode == "max":
        return feat_hwc.max(dim=-1).values.numpy()
    return feat_hwc.mean(dim=-1).numpy()


def upsample_map(act_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Upscales an activation map to target_hw via bicubic interpolation."""
    target_h, target_w = target_hw
    up = cv2.resize(act_map.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    sigma = target_h * 0.025
    up = cv2.GaussianBlur(up, (0, 0), sigmaX=sigma)
    return np.clip(up, 0.0, 1.0)


def apply_colormap(act_map: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """Converts a [0,1] grayscale map -> RGBA uint8 via a matplotlib colormap."""
    cmap = plt.get_cmap(cmap_name)
    return (cmap(act_map) * 255).astype(np.uint8)


def _act_to_original_size(act_map: np.ndarray, original_pil: Image.Image) -> np.ndarray:
    """
    Upsample act_map to padded 512x512, crop the padded region (top + left strips
    added by resize_with_pad), then resize the content back to the original image
    dimensions. This removes the black-bar artefact when displaying overlays.
    """
    pad_h, pad_w = SIGLIP_INPUT_SIZE
    orig_w, orig_h = original_pil.size  # PIL .size is (W, H)
    ratio = max(orig_w / pad_w, orig_h / pad_h)
    content_w = int(orig_w / ratio)
    content_h = int(orig_h / ratio)
    # resize_with_pad pads on LEFT and TOP; mirror those offsets here.
    offset_w = pad_w - content_w
    offset_h = pad_h - content_h

    act_padded = upsample_map(act_map, (pad_h, pad_w))
    act_content = act_padded[offset_h:, offset_w:]
    act_orig = cv2.resize(
        act_content.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC,
    )
    return np.clip(act_orig, 0.0, 1.0)


def overlay_heatmap(
    original_pil: Image.Image,
    act_map: np.ndarray,
    alpha: float = 0.55,
    cmap_name: str = "turbo",
) -> np.ndarray:
    """De-letterbox the heatmap and overlay on the ORIGINAL (non-padded) image."""
    act_orig = _act_to_original_size(act_map, original_pil)
    img_np = np.array(original_pil).astype(np.float32)
    heat_rgb = apply_colormap(act_orig, cmap_name).astype(np.float32)[..., :3]
    blended = (1 - alpha) * img_np + alpha * heat_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


TITLE_FONT = {"fontsize": 9, "fontweight": "bold", "color": "#e8e8e8"}
LABEL_FONT = {"fontsize": 7.5, "color": "#b0b0b0"}


def _style_ax(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_title(title, **TITLE_FONT, pad=4)
    if subtitle:
        ax.text(0.5, -0.06, subtitle, transform=ax.transAxes,
                ha="center", **LABEL_FONT)
    ax.axis("off")


def resize_with_pad(
    tensor: torch.Tensor,
    width: int,
    height: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    (B, C, H, W) -> (B, C, height, width). Maintain aspect ratio,
    pad left/top with pad_value to match SmolVLA preprocessing.
    """
    if tensor.ndim != 4:
        raise ValueError(f"(B,C,H,W) expected, got {tensor.shape}")
    cur_h, cur_w = tensor.shape[2:]

    ratio = max(cur_w / width, cur_h / height)
    new_w = int(cur_w / ratio)
    new_h = int(cur_h / ratio)

    resized = F.interpolate(
        tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    pad_h = max(0, int(height - new_h))
    pad_w = max(0, int(width - new_w))

    return F.pad(resized, (pad_w, 0, pad_h, 0), value=pad_value)


def parse_layers_arg(layers_str: str, n_layers: int) -> list[int]:
    """Parse "0,6,13" -> [0, 6, 13]; clamp, dedupe, sort."""
    if n_layers <= 0:
        return []
    raw = []
    for part in layers_str.split(","):
        part = part.strip()
        if not part:
            continue
        raw.append(int(part))
    clamped = [min(max(i, 0), n_layers - 1) for i in raw]
    return sorted(set(clamped))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SigLIP residual-stream activation maps for SmolVLA."
    )
    parser.add_argument(
        "--image",
        default=os.path.join(PROJECT_DIR, "test_image", "camera1.png"),
        help="Path to camera1 image (required unless using --all-cams).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Heatmap overlay transparency (default: 0.55).",
    )
    parser.add_argument(
        "--mode",
        default="norm",
        choices=list(AGGREGATION_MODES.keys()),
        help=(
            "Channel aggregation method. "
            + ", ".join(AGGREGATION_MODES.keys())
        ),
    )
    parser.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap name (default: turbo).",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Camera id for single-camera mode (default: 1).",
    )
    parser.add_argument(
        "--all-cams",
        action="store_true",
        help="Process all 3 cameras (requires --image-cam2 and --image-cam3).",
    )
    parser.add_argument(
        "--image-cam2",
        default=os.path.join(PROJECT_DIR, "test_image", "camera2.png"),
        help="Path to camera2 image (required with --all-cams).",
    )
    parser.add_argument(
        "--image-cam3",
        default=os.path.join(PROJECT_DIR, "test_image", "camera3.png"),
        help="Path to camera3 image (required with --all-cams).",
    )
    parser.add_argument(
        "--layers",
        default=",".join(str(x) for x in LAYERS_DEFAULT),
        help="Comma-separated layer indices (default: 0,3,6,9,11).",
    )
    parser.add_argument(
        "--model-dir",
        default=SMOLVLA_MODEL_DIR,
        help="Path to the SmolVLA checkpoint directory (default: ../smolvla-model).",
    )
    parser.add_argument(
        "--debug-dump",
        action="store_true",
        help="Run a minimal debug pass to print layer counts and map shapes.",
    )

    args = parser.parse_args()
    if args.all_cams:
        missing = []
        if not args.image_cam2:
            missing.append("--image-cam2")
        if not args.image_cam3:
            missing.append("--image-cam3")
        if missing:
            parser.error("Missing required arguments for --all-cams: " + ", ".join(missing))
    return args


def preprocess_for_siglip(
    pil_img: Image.Image,
    device: torch.device,
) -> tuple[torch.Tensor, Image.Image]:
    """Resize-with-pad to SigLIP input size and normalize to [-1, 1].

    Returns the model-ready tensor on `device` and the original RGB PIL image
    (used later to draw overlays in the original aspect ratio).
    """
    original_pil = pil_img.convert("RGB")
    img_np = np.array(original_pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    tensor = resize_with_pad(tensor, SIGLIP_INPUT_SIZE[1], SIGLIP_INPUT_SIZE[0])
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device), original_pil


class LayerHook:
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.features: torch.Tensor | None = None
        self._handle = None

    def attach(self, module: torch.nn.Module) -> "LayerHook":
        self._handle = module.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        n = out.shape[1]
        side = int(round(n ** 0.5))
        if side * side != n:
            side_minus = int(round((n - 1) ** 0.5))
            if side_minus * side_minus == n - 1:
                out = out[:, 1:, :]
        self.features = out.detach().cpu().float()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def extract_layer_features(
    vision_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    target_layers: list[int],
    mode: str,
) -> list[np.ndarray]:
    hooks = [LayerHook(i).attach(vision_model.encoder.layers[i]) for i in target_layers]
    try:
        model_dtype = getattr(vision_model, "dtype", None)
        if model_dtype is None:
            try:
                model_dtype = next(vision_model.parameters()).dtype
            except StopIteration:
                model_dtype = pixel_values.dtype
        with torch.no_grad():
            vision_model(
                pixel_values=pixel_values.to(dtype=model_dtype),
                patch_attention_mask=None,
            )
        maps = []
        for h in hooks:
            if h.features is None:
                raise RuntimeError(f"No features captured for layer {h.layer_idx}.")
            feat = h.features[0]
            n = feat.shape[0]
            side = int(round(n ** 0.5))
            if side * side != n:
                raise RuntimeError(
                    f"Non-square patch grid: {n} tokens at layer {h.layer_idx}"
                )
            feat = feat.reshape(side, side, -1).float()
            act = _aggregate_channels(feat, mode)
            lo, hi = np.percentile(act, 2), np.percentile(act, 98)
            act = np.clip((act - lo) / (hi - lo + 1e-8), 0.0, 1.0)
            maps.append(act)
    finally:
        for h in hooks:
            h.remove()
    return maps


def _layer_phase(layer_idx: int, n_layers: int) -> str:
    if n_layers <= 0:
        return "mid"
    if layer_idx < n_layers / 3:
        return "early"
    if layer_idx < 2 * n_layers / 3:
        return "mid"
    return "late"


def save_grid(
    original_pil: Image.Image,
    maps: list[np.ndarray],
    target_layers: list[int],
    n_layers: int,
    out_path: str,
    alpha: float,
    cmap_name: str,
    image_name: str,
) -> None:
    maps_dewindow = [_act_to_original_size(m, original_pil) for m in maps]
    maps_colored = [apply_colormap(m, cmap_name) for m in maps_dewindow]
    maps_overlay = [overlay_heatmap(original_pil, m, alpha, cmap_name) for m in maps]
    orig_np = np.array(original_pil)

    n_cols = 1 + len(target_layers)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(3.0 * n_cols, 7.5),
        facecolor="#1a1a2e",
        gridspec_kw={"wspace": 0.06, "hspace": 0.22},
    )
    fig.patch.set_facecolor("#1a1a2e")

    _style_ax(axes[0, 0], "Original Image", image_name)
    axes[0, 0].imshow(orig_np)

    for col, (layer_idx, m_col) in enumerate(zip(target_layers, maps_colored), start=1):
        phase = _layer_phase(layer_idx, n_layers)
        _style_ax(axes[0, col], f"Layer {layer_idx}", f"{phase} layer")
        axes[0, col].imshow(m_col[..., :3])

    _style_ax(axes[1, 0], "Original Image", "")
    axes[1, 0].imshow(orig_np)

    for col, (layer_idx, m_ov) in enumerate(zip(target_layers, maps_overlay), start=1):
        _style_ax(axes[1, col], f"Layer {layer_idx} (overlay)", "")
        axes[1, col].imshow(m_ov)

    for row_idx, label in enumerate(["Raw activation", "Overlay on image"]):
        axes[row_idx, 0].text(
            -0.08, 0.5, label, transform=axes[row_idx, 0].transAxes,
            fontsize=9, color="#9090b0", rotation=90,
            va="center", ha="center",
        )

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01, shrink=0.7)
    cbar.set_label("Activation intensity", color="#b0b0b0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#b0b0b0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#b0b0b0", fontsize=7)
    cbar.outline.set_edgecolor("#404060")

    fig.suptitle(
        "SigLIP Vision Encoder — Residual Stream Activation Maps",
        fontsize=12, color="#d0d0f0", y=1.01,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_combined(
    original_pil: Image.Image,
    maps: list[np.ndarray],
    out_path: str,
    alpha: float,
    cmap_name: str,
    image_name: str,
) -> None:
    combined = np.mean(np.stack(maps, axis=0), axis=0)
    lo, hi = np.percentile(combined, 2), np.percentile(combined, 98)
    combined = np.clip((combined - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    combined_dewindow = _act_to_original_size(combined, original_pil)
    combined_col = apply_colormap(combined_dewindow, cmap_name)
    overlay = overlay_heatmap(original_pil, combined, alpha, cmap_name)
    orig_np = np.array(original_pil)

    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.5), facecolor="#1a1a2e",
        gridspec_kw={"wspace": 0.06},
    )
    fig.patch.set_facecolor("#1a1a2e")

    _style_ax(axes[0], "Original Image", image_name)
    axes[0].imshow(orig_np)

    _style_ax(axes[1], "Combined Activation", "Mean across selected layers")
    axes[1].imshow(combined_col[..., :3])

    _style_ax(axes[2], "Combined Overlay", "High activation = model focus area")
    axes[2].imshow(overlay)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Activation", color="#b0b0b0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#b0b0b0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#b0b0b0", fontsize=7)
    cbar.outline.set_edgecolor("#404060")

    fig.suptitle(
        "SigLIP — Combined Activation Heatmap",
        fontsize=11, color="#d0d0f0",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_individual_layers(
    original_pil: Image.Image,
    maps: list[np.ndarray],
    target_layers: list[int],
    out_dir: str,
    alpha: float,
    cmap_name: str,
    image_name: str,
    cam_suffix: str,
) -> None:
    orig_np = np.array(original_pil)

    for layer_idx, act in zip(target_layers, maps):
        overlay = overlay_heatmap(original_pil, act, alpha, cmap_name)

        fig, axes = plt.subplots(
            1, 2, figsize=(8, 4), facecolor="#1a1a2e",
            gridspec_kw={"wspace": 0.05},
        )
        fig.patch.set_facecolor("#1a1a2e")

        _style_ax(axes[0], "Original Image", image_name)
        axes[0].imshow(orig_np)

        _style_ax(axes[1], f"Layer {layer_idx} (overlay)", "")
        axes[1].imshow(overlay)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Activation", color="#b0b0b0", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#b0b0b0")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#b0b0b0", fontsize=7)
        cbar.outline.set_edgecolor("#404060")

        stem = os.path.splitext(os.path.basename(image_name))[0]
        out_path = os.path.join(out_dir, f"{stem}_layer{layer_idx}_{cam_suffix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def _debug_dump(model_dir: str) -> None:
    device = torch.device("cuda")
    policy, vision_model = load_smolvla_policy(model_dir, device)
    n_layers = len(vision_model.encoder.layers)
    print(f"[DEBUG] n_layers = {n_layers}")
    print(f"[DEBUG] vision_model class = {type(vision_model).__name__}")
    try:
        print(f"[DEBUG] vision_model.dtype = {vision_model.dtype}")
        model_dtype = vision_model.dtype
    except AttributeError:
        model_dtype = next(vision_model.parameters()).dtype
        print("[DEBUG] vision_model.dtype MISSING — use next(parameters()).dtype")
        print(f"[DEBUG] params dtype = {model_dtype}")
    dummy = torch.zeros(1, 3, 512, 512, dtype=model_dtype, device=device)
    maps = extract_layer_features(vision_model, dummy, [0, 1], "centered_norm")
    print(f"[DEBUG] map shape per layer = {maps[0].shape}, {maps[1].shape}")


def _select_image_for_camera(args: argparse.Namespace, camera_id: int) -> str:
    if camera_id == 1:
        return args.image
    if camera_id == 2:
        return args.image_cam2
    if camera_id == 3:
        return args.image_cam3
    raise ValueError(f"Unsupported camera id: {camera_id}")


def _save_all_cams_composite(
    results: list[tuple[list[np.ndarray], Image.Image, str]],
    target_layers: list[int],
    n_layers: int,
    out_dir: str,
    alpha: float,
    cmap_name: str,
) -> None:
    n_cols = 1 + len(target_layers)
    fig, axes = plt.subplots(
        3, n_cols,
        figsize=(3.0 * n_cols, 10.5),
        facecolor="#1a1a2e",
        gridspec_kw={"wspace": 0.06, "hspace": 0.22},
    )
    fig.patch.set_facecolor("#1a1a2e")

    for row_idx, (maps, original_pil, stem) in enumerate(results):
        maps_colored = [apply_colormap(_act_to_original_size(m, original_pil), cmap_name)
                        for m in maps]
        orig_np = np.array(original_pil)

        _style_ax(axes[row_idx, 0], f"Camera {row_idx + 1}", stem)
        axes[row_idx, 0].imshow(orig_np)

        for col, (layer_idx, m_col) in enumerate(zip(target_layers, maps_colored), start=1):
            phase = _layer_phase(layer_idx, n_layers)
            _style_ax(axes[row_idx, col], f"Layer {layer_idx}", f"{phase} layer")
            axes[row_idx, col].imshow(m_col[..., :3])

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01, shrink=0.7)
    cbar.set_label("Activation intensity", color="#b0b0b0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#b0b0b0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#b0b0b0", fontsize=7)
    cbar.outline.set_edgecolor("#404060")

    fig.suptitle(
        "SigLIP Vision Encoder — All Cameras",
        fontsize=12, color="#d0d0f0", y=1.01,
    )

    out_path = os.path.join(out_dir, f"{results[0][2]}_all_cams.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_single_camera(
    args: argparse.Namespace,
    vision_model: torch.nn.Module,
    camera_id: int,
    target_layers: list[int],
    n_layers: int,
    out_dir: str,
) -> tuple[list[np.ndarray], Image.Image, str]:
    image_path = _select_image_for_camera(args, camera_id)
    pil_img = Image.open(image_path).convert("RGB")
    pixel_values, original_pil = preprocess_for_siglip(pil_img, vision_model.device)
    maps = extract_layer_features(vision_model, pixel_values, target_layers, args.mode)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    cam_suffix = f"cam{camera_id}"

    save_grid(
        original_pil,
        maps,
        target_layers,
        n_layers,
        os.path.join(out_dir, f"{stem}_grid_{cam_suffix}.png"),
        args.alpha,
        args.cmap,
        stem,
    )
    save_combined(
        original_pil,
        maps,
        os.path.join(out_dir, f"{stem}_combined_{cam_suffix}.png"),
        args.alpha,
        args.cmap,
        stem,
    )
    save_individual_layers(
        original_pil,
        maps,
        target_layers,
        out_dir,
        args.alpha,
        args.cmap,
        stem,
        cam_suffix,
    )
    return maps, original_pil, stem


def run_all_cameras(
    args: argparse.Namespace,
    vision_model: torch.nn.Module,
    target_layers: list[int],
    n_layers: int,
    out_dir: str,
) -> None:
    results = []
    for cam_id in [1, 2, 3]:
        results.append(run_single_camera(args, vision_model, cam_id, target_layers, n_layers, out_dir))
    _save_all_cams_composite(results, target_layers, n_layers, out_dir, args.alpha, args.cmap)


def load_smolvla_policy(model_dir: str, device: torch.device):
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(
            f"Missing {safetensors_path}. Run `git lfs pull` in {model_dir}."
        )
    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    if size_mb < 100:
        raise RuntimeError(
            f"{safetensors_path} is {size_mb:.1f}MB — looks like an LFS pointer, not the real weights. "
            f"Run `cd {model_dir} && git lfs pull` first."
        )

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"[INFO] Loading from: {model_dir}")
    policy = SmolVLAPolicy.from_pretrained(model_dir).to(device).eval()
    vision_model = policy.model.vlm_with_expert.vlm.model.vision_model

    cfg = policy.config
    n_trainable = sum(p.requires_grad for p in vision_model.parameters())
    n_total = sum(1 for _ in vision_model.parameters())
    print(
        f"[INFO] freeze_vision_encoder={getattr(cfg, 'freeze_vision_encoder', '?')}  "
        f"train_expert_only={getattr(cfg, 'train_expert_only', '?')}"
    )
    print(
        f"[INFO] SigLIP trainable param-tensors: {n_trainable}/{n_total} "
        f"({'FROZEN' if n_trainable == 0 else 'has grads'})"
    )
    return policy, vision_model


def main() -> int:
    args = parse_args()
    if args.debug_dump:
        if not torch.cuda.is_available():
            print("[FAIL] CUDA required. No CPU fallback.")
            return 1
        _debug_dump(args.model_dir)
        return 0
    if not torch.cuda.is_available():
        print("[FAIL] CUDA required. No CPU fallback.")
        return 1

    device = torch.device("cuda")
    out_dir = os.path.join(OUTPUT_DIR, "siglip_feature_maps")
    os.makedirs(out_dir, exist_ok=True)

    policy, vision_model = load_smolvla_policy(args.model_dir, device)
    n_layers = len(vision_model.encoder.layers)
    print(f"[INFO] SigLIP n_layers = {n_layers}")

    target_layers = parse_layers_arg(args.layers, n_layers)
    if n_layers < 5 and len(target_layers) < n_layers:
        target_layers = list(np.linspace(0, n_layers - 1, n_layers).astype(int))
    print(f"[INFO] Target layers: {target_layers}")

    if args.all_cams:
        run_all_cameras(args, vision_model, target_layers, n_layers, out_dir)
    else:
        run_single_camera(args, vision_model, args.camera_id, target_layers, n_layers, out_dir)

    print("[DONE] outputs at:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
