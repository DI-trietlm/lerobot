# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Cut the trailing "stand still" segment at the end of each episode.

Pouring demos end with the arm being teleoped back to ~home and held still for a
second or two. Those frames teach the policy "at home -> stay", which (because the
cup is also returned to its original spot) collides with the "at home -> reach" label
at the start and produces the deploy standstill. This script removes that trailing
idle so the home pose keeps a single meaning.

It rebuilds a NEW dataset (re-encoding videos); the source is left untouched.

Detection: per frame, step-speed = ||action - state|| (deg). Counting from the LAST
frame, drop the run of consecutive frames with speed < --speed-thresh, but only if
that run is longer than --min-cut-s, and never cut below --keep-min-s of the episode.

Usage:
    # preview only (no writing, no video deps):
    uv run python xai/scripts/cut_trailing_idle.py \
        --repo-id di-techinnova/so-arm-101-pouring-0.2 \
        --root /path/to/local/dataset \
        --dry-run
    # actually write the trimmed dataset:
    uv run python xai/scripts/cut_trailing_idle.py \
        --repo-id di-techinnova/so-arm-101-pouring-0.2 \
        --output-repo-id di-techinnova/so-arm-101-pouring-0.2-trimmed
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.io_utils import load_image_as_numpy, load_info, load_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset

BOOKKEEPING = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _trailing_cut(speed: np.ndarray, fps: int, thresh: float, min_cut_s: float, keep_min_s: float) -> int:
    """Number of trailing frames to drop for one episode (0 if none qualify)."""
    n = len(speed)
    idle = 0
    for v in speed[::-1]:
        if v < thresh:
            idle += 1
        else:
            break
    if idle / fps <= min_cut_s:
        return 0
    keep_min = int(round(keep_min_s * fps))
    return max(0, min(idle, n - keep_min))


def plan_cuts(df: pd.DataFrame, fps: int, args) -> dict[int, tuple[int, int]]:
    """episode_index -> (n_frames, n_keep)."""
    S = np.stack(df["observation.state"].to_numpy())
    A = np.stack(df["action"].to_numpy())
    speed = np.linalg.norm(A - S, axis=1)
    plan = {}
    for ep, g in df.groupby("episode_index"):
        order = np.argsort(g["frame_index"].to_numpy())
        sp = speed[g.index.to_numpy()[order]]
        cut = _trailing_cut(sp, fps, args.speed_thresh, args.min_cut_s, args.keep_min_s)
        plan[int(ep)] = (len(sp), len(sp) - cut)
    return plan


def load_source_metadata(args):
    """Load source metadata without requiring encoded videos to exist."""
    root = Path(args.root) if args.root is not None else None
    src = None
    if root is not None and (root / "meta" / "info.json").exists():
        info = load_info(root)
        tasks = load_tasks(root)
    else:
        src = LeRobotDataset(args.repo_id, root=args.root, video_backend=args.video_backend)
        root = Path(src.root)
        info = src.meta.info
        tasks = src.meta.tasks
    return root, info, tasks, src


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="Source dataset repo id.")
    ap.add_argument("--root", default=None, help="Local dir for the source dataset (default: HF cache).")
    ap.add_argument("--output-repo-id", default=None, help="New dataset repo id (default: <repo-id>-trimmed).")
    ap.add_argument("--output-root", default=None, help="Local dir for the new dataset (default: HF cache).")
    ap.add_argument("--speed-thresh", type=float, default=5.0, help="deg; frames below this are 'still'.")
    ap.add_argument("--min-cut-s", type=float, default=0.5, help="only cut trailing idle longer than this.")
    ap.add_argument("--keep-min-s", type=float, default=1.0, help="never trim an episode below this length.")
    ap.add_argument("--video-backend", default="pyav", help="reader backend (pyav avoids torchcodec/ffmpeg issues).")
    ap.add_argument("--camera-encoder-vcodec", default="h264", help="video codec for the output dataset.")
    ap.add_argument("--camera-encoder-crf", type=float, default=30, help="video encoder CRF/quality.")
    ap.add_argument("--camera-encoder-g", type=int, default=2, help="video encoder GOP size.")
    ap.add_argument("--camera-encoder-pix-fmt", default="yuv420p", help="video encoder pixel format.")
    ap.add_argument("--encoder-threads", type=int, default=None, help="number of threads per encoder.")
    ap.add_argument("--max-episodes", type=int, default=None, help="only write the first N episodes (smoke test).")
    ap.add_argument("--csv", default=None, help="per-episode cut table CSV (default: trailing_idle_cuts_<repo>.csv).")
    ap.add_argument("--push-to-hub", action="store_true", help="upload the trimmed dataset to the Hub when done.")
    ap.add_argument("--private", action="store_true", help="make the pushed Hub repo private.")
    ap.add_argument("--dry-run", action="store_true", help="report cuts only; do not write the dataset.")
    args = ap.parse_args()

    print(f"Loading {args.repo_id} ...")
    src_root, src_info, src_tasks, src = load_source_metadata(args)
    fps = src_info.fps

    # Pass 1 — read parquet only (no video decode) to plan the cuts.
    files = sorted(glob.glob(f"{src_root}/data/**/*.parquet", recursive=True))
    df = pd.concat(
        [pd.read_parquet(f, columns=["index", "episode_index", "frame_index", "task_index", "observation.state", "action"])
         for f in files],
        ignore_index=True,
    )
    plan = plan_cuts(df, fps, args)

    total = sum(n for n, _ in plan.values())
    kept = sum(k for _, k in plan.values())
    n_trim = sum(1 for n, k in plan.values() if k < n)
    cut_frames = total - kept
    print(f"\nepisodes: {len(plan)} | frames: {total} -> {kept}  (cut {cut_frames} = {100*cut_frames/total:.1f}%)")
    print(f"episodes trimmed: {n_trim}/{len(plan)}  (trailing idle > {args.min_cut_s}s at speed<{args.speed_thresh})")
    cuts_s = sorted([(n - k) / fps for n, k in plan.values() if k < n], reverse=True)
    if cuts_s:
        print(f"cut per trimmed ep (s): max {cuts_s[0]:.1f} | median {np.median(cuts_s):.1f} | min {cuts_s[-1]:.1f}")
    print("  examples:", {e: f"{n}->{k}" for e, (n, k) in list(plan.items())[:6]})

    # Per-episode cut table -> CSV
    csv_path = args.csv or f"trailing_idle_cuts_{args.repo_id.replace('/', '_')}.csv"
    pd.DataFrame(
        [
            {"episode_index": e, "n_frames": n, "n_keep": k, "n_cut": n - k,
             "cut_seconds": round((n - k) / fps, 3), "trimmed": k < n}
            for e, (n, k) in sorted(plan.items())
        ]
    ).to_csv(csv_path, index=False)
    print(f"per-episode cut table -> {csv_path}")

    if args.dry_run:
        print("\n[dry-run] nothing written. Re-run without --dry-run to create the trimmed dataset.")
        return

    out_id = args.output_repo_id or f"{args.repo_id}-trimmed"
    features = {k: v for k, v in src_info.features.items() if k not in BOOKKEEPING}
    camera_encoder = VideoEncoderConfig(
        vcodec=args.camera_encoder_vcodec,
        crf=args.camera_encoder_crf,
        g=args.camera_encoder_g,
        pix_fmt=args.camera_encoder_pix_fmt,
    )
    print(f"\nCreating trimmed dataset '{out_id}' ...")
    dst = LeRobotDataset.create(
        repo_id=out_id,
        fps=fps,
        features=features,
        root=args.output_root,
        robot_type=src_info.robot_type,
        use_videos=any(ft["dtype"] == "video" for ft in features.values()),
        video_backend=args.video_backend,
        camera_encoder=camera_encoder,
        encoder_threads=args.encoder_threads,
    )

    img_keys = [k for k, ft in src_info.features.items() if ft["dtype"] == "video"]
    vec_keys = [k for k in features if k not in img_keys]  # observation.state, action, ...
    order = df.sort_values(["episode_index", "frame_index"])
    task_by_index = {int(row.task_index): task for task, row in src_tasks.iterrows()}
    source_is_deferred = bool(getattr(src_info, "video_encoding_deferred", False))

    episodes_to_write = sorted(plan)
    if args.max_episodes is not None:
        episodes_to_write = episodes_to_write[:args.max_episodes]
        print(f"Writing only first {len(episodes_to_write)} episode(s) because --max-episodes was set.")

    for ep in episodes_to_write:
        n, keep = plan[ep]
        ep_rows = order[order["episode_index"] == ep].head(keep)
        for _, row in ep_rows.iterrows():
            frame = {k: np.asarray(row[k]) for k in vec_keys}
            for k in img_keys:
                img_path = src_root / "images" / k / f"episode-{int(ep):06d}" / f"frame-{int(row['frame_index']):06d}.png"
                if img_path.exists():
                    frame[k] = load_image_as_numpy(img_path, dtype=np.uint8, channel_first=False)
                else:
                    if source_is_deferred:
                        raise FileNotFoundError(f"Deferred source is missing expected frame image: {img_path}")
                    if src is None:
                        src = LeRobotDataset(args.repo_id, root=args.root, video_backend=args.video_backend)
                    item = src[int(row["index"])]
                    # CHW float[0,1] -> HWC uint8 (matches feature shape)
                    frame[k] = (item[k].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            frame["task"] = task_by_index[int(row["task_index"])]
            dst.add_frame(frame)
        dst.save_episode()
        print(f"  ep {ep:3d}: kept {keep}/{n}")

    dst.finalize()
    print(f"\nDone. Trimmed dataset at: {dst.root}")

    if args.push_to_hub:
        print(f"Pushing '{out_id}' to the Hub (private={args.private}) ...")
        dst.push_to_hub(private=args.private)
        print(f"Pushed: https://huggingface.co/datasets/{out_id}")
    else:
        print("Train on it by pointing dataset.repo_id / dataset.root to this output "
              "(add --push-to-hub to upload).")


if __name__ == "__main__":
    main()
