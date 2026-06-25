#!/usr/bin/env python
"""Rebuild a LeRobot dataset after removing known safe-pose contamination ranges."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.io_utils import load_info, load_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset


BOOKKEEPING = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

# Inclusive frame ranges found from union(state near safe-pose, action target near safe-pose).
DEFAULT_CUT_RANGES: dict[int, list[tuple[int, int]]] = {
    0: [(0, 2)],
    49: [(344, 348)],
    56: [(0, 3)],
    58: [(0, 8)],
    59: [(0, 9)],
    60: [(0, 2)],
    89: [(406, 413)],
    90: [(0, 12)],
    119: [(432, 436)],
    120: [(0, 15)],
    144: [(365, 368)],
    145: [(0, 15)],
    150: [(0, 3)],
}


def _parse_ranges(value: str | None) -> dict[int, list[tuple[int, int]]]:
    if value is None:
        return DEFAULT_CUT_RANGES
    ranges: dict[int, list[tuple[int, int]]] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ep_text, span = item.split(":", 1)
        start_text, end_text = span.split("-", 1)
        ranges.setdefault(int(ep_text), []).append((int(start_text), int(end_text)))
    return ranges


def _keep_frame(ep: int, frame: int, cut_ranges: dict[int, list[tuple[int, int]]]) -> bool:
    return all(not (start <= frame <= end) for start, end in cut_ranges.get(ep, []))


def _load_parquet_df(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _to_hwc_uint8(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {arr.shape}")
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _task_by_index(tasks: pd.DataFrame) -> dict[int, str]:
    return {int(row.task_index): str(task) for task, row in tasks.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--cut-ranges",
        default=None,
        help="Optional override, e.g. '0:0-2,49:344-348'. Defaults to diagnosed safe-pose ranges.",
    )
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--camera-encoder-vcodec", default="h264")
    parser.add_argument("--camera-encoder-crf", type=float, default=30)
    parser.add_argument("--camera-encoder-g", type=int, default=2)
    parser.add_argument("--camera-encoder-pix-fmt", default="yuv420p")
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.root)
    out_root = Path(args.output_root)
    if out_root.exists() and not args.dry_run:
        raise FileExistsError(f"Output root already exists: {out_root}")

    cut_ranges = _parse_ranges(args.cut_ranges)
    info = load_info(src_root)
    tasks = load_tasks(src_root)
    task_lookup = _task_by_index(tasks)
    df = _load_parquet_df(src_root).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    rows = []
    for ep, g in df.groupby("episode_index", sort=True):
        kept = [
            _keep_frame(int(ep), int(frame), cut_ranges)
            for frame in g["frame_index"].astype(int).to_numpy()
        ]
        n_cut = int(len(kept) - sum(kept))
        rows.append(
            {
                "episode_index": int(ep),
                "n_frames": int(len(kept)),
                "n_keep": int(sum(kept)),
                "n_cut": n_cut,
                "cut_seconds": round(n_cut / int(info.fps), 3),
                "cut_ranges": ";".join(f"{s}-{e}" for s, e in cut_ranges.get(int(ep), [])),
            }
        )

    manifest = pd.DataFrame(rows)
    print(manifest[manifest["n_cut"] > 0].to_string(index=False))
    total = int(manifest["n_frames"].sum())
    kept = int(manifest["n_keep"].sum())
    print(f"\nframes: {total} -> {kept} (cut {total - kept})")
    print(f"episodes with cuts: {int((manifest['n_cut'] > 0).sum())}/{len(manifest)}")

    manifest_path = Path("xai") / f"safe_pose_cut_manifest_{args.output_repo_id.split('/')[-1]}.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"manifest -> {manifest_path}")
    if args.dry_run:
        return

    features = {key: value for key, value in info.features.items() if key not in BOOKKEEPING}
    image_keys = [key for key, value in features.items() if value["dtype"] == "video"]
    vector_keys = [key for key in features if key not in image_keys]
    camera_encoder = VideoEncoderConfig(
        vcodec=args.camera_encoder_vcodec,
        crf=args.camera_encoder_crf,
        g=args.camera_encoder_g,
        pix_fmt=args.camera_encoder_pix_fmt,
    )

    src = LeRobotDataset(args.repo_id, root=src_root, video_backend=args.video_backend)
    dst = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        fps=int(info.fps),
        features=features,
        root=out_root,
        robot_type=info.robot_type,
        use_videos=True,
        video_backend=args.video_backend,
        camera_encoder=camera_encoder,
        encoder_threads=args.encoder_threads,
    )

    for ep, g in df.groupby("episode_index", sort=True):
        ep = int(ep)
        kept_rows = g[
            [
                _keep_frame(ep, int(frame), cut_ranges)
                for frame in g["frame_index"].astype(int).to_numpy()
            ]
        ]
        if kept_rows.empty:
            raise ValueError(f"Episode {ep} would become empty")
        for _, row in kept_rows.iterrows():
            item = src[int(row["index"])]
            frame = {key: np.asarray(row[key], dtype=np.float32) for key in vector_keys}
            for key in image_keys:
                frame[key] = _to_hwc_uint8(item[key])
            frame["task"] = task_lookup[int(row["task_index"])]
            dst.add_frame(frame)
        dst.save_episode()
        n_cut = int(len(g) - len(kept_rows))
        if n_cut:
            print(f"ep {ep:03d}: kept {len(kept_rows)}/{len(g)} (cut {n_cut})")

    dst.finalize()
    print(f"\nDone: {dst.root}")


if __name__ == "__main__":
    main()
