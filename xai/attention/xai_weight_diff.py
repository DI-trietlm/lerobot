#!/usr/bin/env python3
"""
XAI Phase 4 — Weight diff analysis between two XVLA checkpoints.

Loads raw safetensors state dicts on CPU and reports per-component divergence.
"""

from __future__ import annotations

import argparse
import os
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xai_utils  # noqa: E402

from xai_utils import load_raw_state_dict, load_normalizer_stats, ensure_output_dir


COMPONENT_ORDER = [
    "DaViT (Vision Encoder)",
    "Image Projection",
    "Image Projection Norm",
    "Florence-2 Encoder",
    "Florence-2 Shared Embeddings",
    "Soft Prompts",
    "SoftPromptedTransformer",
    "Unclassified",
]

LAYER_PATTERNS = {
    "DaViT (Vision Encoder)": [
        re.compile(r"model\.vlm\.vision_tower\.blocks\.(\d+)\."),
        re.compile(r"model\.vlm\.vision_tower\.stages\.(\d+)\."),
    ],
    "Florence-2 Encoder": [
        re.compile(r"model\.vlm\.language_model\.model\.encoder\.layers\.(\d+)\.")
    ],
    "SoftPromptedTransformer": [
        re.compile(r"model\.transformer\.blocks\.(\d+)\.")
    ],
}


def classify_key(key: str) -> str:
    if key.startswith("model.vlm.vision_tower."):
        return "DaViT (Vision Encoder)"
    if key.startswith("model.vlm.image_projection"):
        return "Image Projection"
    if key.startswith("model.vlm.image_proj_norm."):
        return "Image Projection Norm"
    if key.startswith("model.vlm.language_model.model.encoder."):
        return "Florence-2 Encoder"
    if key.startswith("model.vlm.language_model.model.shared."):
        return "Florence-2 Shared Embeddings"
    if key.startswith("model.transformer.soft_prompt_hub."):
        return "Soft Prompts"
    if key.startswith("model.transformer."):
        return "SoftPromptedTransformer"
    return "Unclassified"


def format_count(n_params: int) -> str:
    if n_params >= 1_000_000_000:
        return f"{n_params / 1_000_000_000:.2f}B"
    if n_params >= 1_000_000:
        return f"{n_params / 1_000_000:.1f}M"
    if n_params >= 1_000:
        return f"{n_params / 1_000:.1f}K"
    return str(n_params)


def summarize_component(group: dict) -> dict:
    n_params = group["n_params"]
    if n_params == 0:
        return {
            "n_params": 0,
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
            "relative_change": 0.0,
            "cosine_similarity": 1.0,
        }
    mean_abs_diff = group["sum_abs_diff"] / n_params
    mean_abs_w1 = group["sum_abs_w1"] / n_params
    rel_change = mean_abs_diff / (mean_abs_w1 + 1e-12)
    cos_sim = group["dot"] / ((group["norm1"] * group["norm2"]) ** 0.5 + 1e-12)
    return {
        "n_params": n_params,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": group["max_abs_diff"],
        "relative_change": rel_change,
        "cosine_similarity": cos_sim,
    }


def _init_group_stats() -> dict:
    return {
        "n_params": 0,
        "sum_abs_diff": 0.0,
        "sum_abs_w1": 0.0,
        "max_abs_diff": 0.0,
        "dot": 0.0,
        "norm1": 0.0,
        "norm2": 0.0,
        "n_tensors": 0,
    }


def _update_group_stats(group: dict, w1: torch.Tensor, w2: torch.Tensor, chunk_size: int) -> None:
    w1_flat = w1.reshape(-1)
    w2_flat = w2.reshape(-1)
    total = w1_flat.numel()
    group["n_params"] += total

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        w1_chunk = w1_flat[start:end]
        w2_chunk = w2_flat[start:end]
        diff = w1_chunk - w2_chunk
        abs_diff = diff.abs()
        group["sum_abs_diff"] += abs_diff.sum().item()
        group["sum_abs_w1"] += w1_chunk.abs().sum().item()
        group["max_abs_diff"] = max(group["max_abs_diff"], abs_diff.max().item())
        group["dot"] += (w1_chunk * w2_chunk).sum().item()
        group["norm1"] += (w1_chunk * w1_chunk).sum().item()
        group["norm2"] += (w2_chunk * w2_chunk).sum().item()

    group["n_tensors"] += 1


def _extract_layer_index(component: str, key: str) -> int | None:
    patterns = LAYER_PATTERNS.get(component, [])
    for pattern in patterns:
        match = pattern.search(key)
        if match:
            return int(match.group(1))
    return None


def compute_diff_stats(state_a: dict, state_b: dict) -> tuple[dict, dict, dict, dict]:
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    common_keys = keys_a & keys_b

    groups = {name: _init_group_stats() for name in COMPONENT_ORDER}
    layer_stats = {name: {} for name in LAYER_PATTERNS.keys()}
    chunk_size = 5_000_000

    unclassified = 0
    for key in sorted(common_keys):
        group = classify_key(key)
        if group == "Unclassified":
            unclassified += 1
        w1 = state_a[key].float().cpu()
        w2 = state_b[key].float().cpu()

        _update_group_stats(groups[group], w1, w2, chunk_size)

        layer_idx = _extract_layer_index(group, key)
        if layer_idx is not None and group in layer_stats:
            layer_group = layer_stats[group].setdefault(layer_idx, _init_group_stats())
            _update_group_stats(layer_group, w1, w2, chunk_size)

    extras = {
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
        "common": sorted(common_keys),
        "unclassified_count": unclassified,
    }
    return groups, extras, {"keys_a": len(keys_a), "keys_b": len(keys_b)}, layer_stats


def print_report(model_a: str, model_b: str, groups: dict, extras: dict, key_counts: dict) -> None:
    only_a = extras["only_in_a"]
    only_b = extras["only_in_b"]
    common_keys = extras["common"]
    unclassified = extras["unclassified_count"]

    print("=" * 72)
    print("=== XVLA Weight Diff Report ===")
    print(f"Model A: {model_a} ({key_counts['keys_a']} keys)")
    print(f"Model B: {model_b} ({key_counts['keys_b']} keys)")
    pct_a = (len(common_keys) / key_counts["keys_a"] * 100) if key_counts["keys_a"] else 0.0
    pct_b = (len(common_keys) / key_counts["keys_b"] * 100) if key_counts["keys_b"] else 0.0
    print(f"Common keys: {len(common_keys)} ({pct_a:.1f}% of A, {pct_b:.1f}% of B)")

    print(f"Keys only in A: {len(only_a)}")
    if only_a:
        print("  sample:")
        for key in only_a[:5]:
            print(f"    - {key}")
    else:
        print("  sample: (none)")

    print(f"Keys only in B: {len(only_b)}")
    if only_b:
        print("  sample:")
        for key in only_b[:5]:
            print(f"    - {key}")
    else:
        print("  sample: (none)")

    if len(common_keys) > 0:
        unclassified_ratio = unclassified / len(common_keys)
        if unclassified_ratio > 0.05:
            print(
                f"[WARN] Unclassified keys: {unclassified} "
                f"({unclassified_ratio:.1%} of common keys)"
            )

    print("\n--- Per-Component Divergence ---")
    header = (
        f"{'Component':30s} | {'Params':>8s} | {'Mean Diff':>10s} | "
        f"{'Max Diff':>9s} | {'Rel Change':>10s} | {'Cosine Sim':>10s}"
    )
    print(header)
    print("-" * len(header))

    for name in COMPONENT_ORDER:
        stats = summarize_component(groups[name])
        if stats["n_params"] == 0:
            continue
        print(
            f"{name:30s} | "
            f"{format_count(stats['n_params']):>8s} | "
            f"{stats['mean_abs_diff']:<10.2e} | "
            f"{stats['max_abs_diff']:<9.2e} | "
            f"{stats['relative_change'] * 100:>9.2f}% | "
            f"{stats['cosine_similarity']:<10.4f}"
        )

    print("\n--- Verdict ---")
    verdict_targets = [
        "Florence-2 Encoder",
        "DaViT (Vision Encoder)",
        "SoftPromptedTransformer",
    ]
    for name in verdict_targets:
        stats = summarize_component(groups[name])
        if stats["n_params"] == 0:
            print(f"{name}: SKIPPED (no parameters)")
            continue
        mean_diff = stats["mean_abs_diff"]
        if mean_diff == 0.0:
            status = "FROZEN"
        else:
            status = "DIVERGED"
        print(f"{name}: {status} (mean_diff = {mean_diff:.2e})")


def print_normalizer_stats(model_a: str, model_b: str) -> None:
    stats_a = load_normalizer_stats(model_a)
    stats_b = load_normalizer_stats(model_b)

    mean_a = None
    std_a = None
    mean_b = None
    std_b = None

    for key, val in stats_a.items():
        if "mean" in key:
            mean_a = val.squeeze().float().cpu()
        if "std" in key:
            std_a = val.squeeze().float().cpu()
    for key, val in stats_b.items():
        if "mean" in key:
            mean_b = val.squeeze().float().cpu()
        if "std" in key:
            std_b = val.squeeze().float().cpu()

    if mean_a is None or std_a is None or mean_b is None or std_b is None:
        print("\n--- Normalizer Stats ---")
        print("[WARN] Normalizer stats missing mean/std keys")
        return

    if mean_a.ndim == 0:
        mean_a = mean_a.view(1)
    if std_a.ndim == 0:
        std_a = std_a.view(1)
    if mean_b.ndim == 0:
        mean_b = mean_b.view(1)
    if std_b.ndim == 0:
        std_b = std_b.view(1)

    n = min(mean_a.numel(), mean_b.numel(), std_a.numel(), std_b.numel())

    print("\n--- Normalizer Stats ---")
    print(
        f"{'Action':10s} | {'A mean':>10s} | {'A std':>10s} | "
        f"{'B mean':>10s} | {'B std':>10s} | {'Diff':>10s}"
    )
    print("-" * 72)
    for i in range(n):
        diff = (mean_a[i] - mean_b[i]).item()
        print(
            f"action[{i:<2d}] | "
            f"{mean_a[i].item():>10.4f} | {std_a[i].item():>10.4f} | "
            f"{mean_b[i].item():>10.4f} | {std_b[i].item():>10.4f} | "
            f"{diff:>10.4f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare XVLA checkpoint weights on CPU.")
    p.add_argument("--model_a", required=True, help="Path to model A directory")
    p.add_argument("--model_b", required=True, help="Path to model B directory")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Directory for plots (default: xai/artifacts/attention_outputs)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and list key groups without loading weights",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    for label, model_dir in [("A", args.model_a), ("B", args.model_b)]:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model {label} directory not found: {model_dir}")
        model_path = os.path.join(model_dir, "model.safetensors")
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {label} missing model.safetensors: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model {label} missing config.json: {config_path}")

    if args.dry_run:
        print("\n[Dry run] Validating checkpoint headers...")
        from safetensors import safe_open

        def _inspect(model_dir: str) -> dict:
            path = os.path.join(model_dir, "model.safetensors")
            with safe_open(path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
            groups = {name: 0 for name in COMPONENT_ORDER}
            for key in keys:
                groups[classify_key(key)] += 1
            return {"keys": keys, "groups": groups}

        info_a = _inspect(args.model_a)
        info_b = _inspect(args.model_b)
        print(f"Model A keys: {len(info_a['keys'])}")
        print(f"Model B keys: {len(info_b['keys'])}")

        print("\nKey prefix groups (Model A):")
        for name in COMPONENT_ORDER:
            if info_a["groups"][name] > 0:
                print(f"  {name}: {info_a['groups'][name]}")

        print("\nKey prefix groups (Model B):")
        for name in COMPONENT_ORDER:
            if info_b["groups"][name] > 0:
                print(f"  {name}: {info_b['groups'][name]}")

        print("\n[Dry run] Done.")
        return 0

    print("\nLoading model A state dict (CPU) ...")
    state_a = load_raw_state_dict(args.model_a)
    print(f"  Loaded {len(state_a)} tensors from {args.model_a}")

    print("Loading model B state dict (CPU) ...")
    state_b = load_raw_state_dict(args.model_b)
    print(f"  Loaded {len(state_b)} tensors from {args.model_b}")

    groups, extras, key_counts, layer_stats = compute_diff_stats(state_a, state_b)
    print_report(args.model_a, args.model_b, groups, extras, key_counts)

    print_normalizer_stats(args.model_a, args.model_b)

    output_dir = args.output_dir or ensure_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    print("\nSaving visualizations...")
    summary_path = save_summary_plot(groups, output_dir)
    heatmap_path = save_heatmap(layer_stats, output_dir)
    hist_path = save_histograms(state_a, state_b, groups, output_dir)
    print(f"  Saved: {summary_path}")
    if heatmap_path:
        print(f"  Saved: {heatmap_path}")
    if hist_path:
        print(f"  Saved: {hist_path}")

    print("\nDone.")
    return 0


def save_summary_plot(groups: dict, output_dir: str) -> str:
    rows = []
    for name in COMPONENT_ORDER:
        stats = summarize_component(groups[name])
        if stats["n_params"] == 0:
            continue
        rows.append((name, stats["mean_abs_diff"], stats["n_params"]))

    names = [r[0] for r in rows]
    mean_diffs = [r[1] for r in rows]
    colors = ["#55d38e" if md <= 1e-12 else "#e45858" for md in mean_diffs]

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    y = np.arange(len(names))
    ax.barh(y, np.maximum(mean_diffs, 1e-12), color=colors, edgecolor="#2a2a3a")
    ax.set_yticks(y)
    ax.set_yticklabels(names, color="#e8e8e8", fontsize=9)
    ax.set_xlabel("Mean |diff| (log scale)", color="#b0b0b0")
    ax.set_xscale("log")
    ax.tick_params(axis="x", colors="#b0b0b0")
    ax.tick_params(axis="y", colors="#e8e8e8")
    ax.grid(axis="x", color="#2b2b3a", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_title("Weight Diff Summary by Component", color="#d0d0f0", fontsize=12)

    out_path = os.path.join(output_dir, "weight_diff_summary.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def save_heatmap(layer_stats: dict, output_dir: str) -> str | None:
    rows = []
    for component, layer_map in layer_stats.items():
        for layer_idx, stats in layer_map.items():
            summary = summarize_component(stats)
            rows.append((component, layer_idx, summary))

    if not rows:
        return None

    rows.sort(key=lambda r: (r[0], r[1]))
    labels = [f"{comp} | layer {idx}" for comp, idx, _ in rows]
    data = np.array(
        [
            [r[2]["mean_abs_diff"], r[2]["max_abs_diff"], 1.0 - r[2]["cosine_similarity"]]
            for r in rows
        ]
    )

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.3 * len(labels))), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7, color="#e8e8e8")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["mean_diff", "max_diff", "1 - cosine_sim"], color="#b0b0b0")
    ax.tick_params(axis="x", colors="#b0b0b0")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(color="#b0b0b0", labelcolor="#b0b0b0")
    cbar.outline.set_edgecolor("#404060")
    ax.set_title("Per-Layer Divergence Heatmap", color="#d0d0f0", fontsize=12)

    out_path = os.path.join(output_dir, "weight_diff_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _accumulate_hist(
    state_a: dict,
    state_b: dict,
    component: str,
    bins: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, float, float, int]:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    running_sum = 0.0
    running_sq_sum = 0.0
    total_n = 0
    for key in state_a.keys() & state_b.keys():
        if classify_key(key) != component:
            continue
        w1 = state_a[key].float().cpu().reshape(-1)
        w2 = state_b[key].float().cpu().reshape(-1)
        total = w1.numel()
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            diff = (w1[start:end] - w2[start:end]).numpy()
            hist, _ = np.histogram(diff, bins=bins)
            counts += hist
            running_sum += float(diff.sum())
            running_sq_sum += float((diff * diff).sum())
            total_n += diff.size
    return counts, running_sum, running_sq_sum, total_n


def save_histograms(state_a: dict, state_b: dict, groups: dict, output_dir: str) -> str | None:
    components = ["DaViT (Vision Encoder)", "Florence-2 Encoder", "SoftPromptedTransformer"]
    max_diff = {name: groups[name]["max_abs_diff"] for name in components}
    if all(max_diff[name] <= 0 for name in components):
        return None
    chunk_size = 5_000_000

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for ax, comp in zip(axes, components):
        ax.set_facecolor("#1a1a2e")
        if max_diff[comp] <= 0:
            ax.set_title(f"{comp}\n(no data)", color="#e8e8e8", fontsize=9)
            ax.axis("off")
            continue
        span = max(max_diff[comp] * 1.05, 1e-6)
        bins = np.linspace(-span, span, 121)
        counts, running_sum, running_sq_sum, total_n = _accumulate_hist(
            state_a, state_b, comp, bins, chunk_size
        )
        centers = 0.5 * (bins[:-1] + bins[1:])
        ax.bar(centers, counts, width=(bins[1] - bins[0]), color="#7aa0ff", alpha=0.85)
        ax.set_yscale("log")
        if total_n > 0:
            mean = running_sum / total_n
            var = max(running_sq_sum / total_n - mean * mean, 0.0)
            std = var ** 0.5
        else:
            mean = 0.0
            std = 0.0
        ax.set_title(f"{comp}", color="#e8e8e8", fontsize=9)
        ax.set_xlabel("delta", color="#b0b0b0", fontsize=8)
        ax.set_ylabel("count (log)", color="#b0b0b0", fontsize=8)
        ax.tick_params(axis="both", colors="#b0b0b0", labelsize=7)
        ax.text(
            0.98,
            0.96,
            f"mean={mean:.2e}\nstd={std:.2e}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            color="#d0d0f0",
        )

    fig.suptitle("Weight Delta Distributions", color="#d0d0f0", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "weight_diff_histogram.png")
    plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    sys.exit(main())
