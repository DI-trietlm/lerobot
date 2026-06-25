#!/usr/bin/env python
"""Append raw LeRobot roots into an encoded dataset after cutting trailing idle.

The source roots are expected to contain parquet metadata plus PNG frames under
``images/``. Only frames referenced by parquet are used, so interrupted extra
PNG directories are ignored.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.io_utils import load_image_as_numpy, load_info, load_tasks, write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset

BOOKKEEPING = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _trailing_cut(speed: np.ndarray, fps: int, thresh: float, min_cut_s: float, keep_min_s: float) -> int:
    idle = 0
    for value in speed[::-1]:
        if value < thresh:
            idle += 1
        else:
            break
    if idle / fps <= min_cut_s:
        return 0
    keep_min = int(round(keep_min_s * fps))
    return max(0, min(idle, len(speed) - keep_min))


def _feature_contract(features: dict) -> dict:
    return {key: {k: v for k, v in ft.items() if k != "info"} for key, ft in features.items() if key not in BOOKKEEPING}


def _load_data(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")
    return pd.concat(
        [
            pd.read_parquet(
                file,
                columns=["index", "episode_index", "frame_index", "task_index", "observation.state", "action"],
            )
            for file in files
        ],
        ignore_index=True,
    )


def _plan_root(root: Path, fps: int, args) -> tuple[pd.DataFrame, dict[int, tuple[int, int]]]:
    df = _load_data(root)
    states = np.stack(df["observation.state"].to_numpy())
    actions = np.stack(df["action"].to_numpy())
    speed = np.linalg.norm(actions - states, axis=1)
    plan = {}
    for ep, group in df.groupby("episode_index"):
        order = np.argsort(group["frame_index"].to_numpy())
        idx = group.index.to_numpy()[order]
        frame_index = group["frame_index"].to_numpy()[order]
        expected = np.arange(len(frame_index))
        if not np.array_equal(frame_index, expected):
            raise ValueError(f"{root.name} episode {int(ep)} has non-contiguous frame_index")
        cut = _trailing_cut(speed[idx], fps, args.speed_thresh, args.min_cut_s, args.keep_min_s)
        plan[int(ep)] = (len(idx), len(idx) - cut)
    return df, plan


def _parse_roots(args) -> list[Path]:
    if args.source_root:
        return [Path(root) for root in args.source_root]
    base = Path(args.source_base)
    roots = []
    for root in base.iterdir():
        if not root.is_dir() or args.name_contains not in root.name:
            continue
        stamp = root.name.rsplit("_", 2)[-2] + "_" + root.name.rsplit("_", 2)[-1]
        if args.start_stamp <= stamp <= args.end_stamp:
            roots.append(root)
    return sorted(roots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-repo-id", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--source-root", action="append", help="Raw root to append. Repeatable.")
    parser.add_argument("--source-base", default=r"F:\llms\hf\models\lerobot\di-techinnova")
    parser.add_argument("--name-contains", default="20260622_")
    parser.add_argument("--start-stamp", default="20260622_104354")
    parser.add_argument("--end-stamp", default="20260622_112314")
    parser.add_argument("--expected-start-episodes", type=int, default=None)
    parser.add_argument("--expected-start-frames", type=int, default=None)
    parser.add_argument("--allow-resume", action="store_true")
    parser.add_argument("--target-video-files-size-mb", type=float, default=None)
    parser.add_argument("--speed-thresh", type=float, default=5.0)
    parser.add_argument("--min-cut-s", type=float, default=0.5)
    parser.add_argument("--keep-min-s", type=float, default=1.0)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--camera-encoder-vcodec", default="h264")
    parser.add_argument("--camera-encoder-crf", type=float, default=30)
    parser.add_argument("--camera-encoder-g", type=int, default=2)
    parser.add_argument("--camera-encoder-pix-fmt", default="yuv420p")
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--manifest", default="xai/append_cut_idle_manifest.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_root = Path(args.target_root)
    roots = _parse_roots(args)
    if not roots:
        raise ValueError("No source roots selected")

    target_info = load_info(target_root)
    if (
        args.expected_start_episodes is not None
        and not args.allow_resume
        and target_info.total_episodes != args.expected_start_episodes
    ):
        raise ValueError(f"Target has {target_info.total_episodes} episodes, expected {args.expected_start_episodes}")
    if (
        args.expected_start_frames is not None
        and not args.allow_resume
        and target_info.total_frames != args.expected_start_frames
    ):
        raise ValueError(f"Target has {target_info.total_frames} frames, expected {args.expected_start_frames}")
    if args.allow_resume and args.expected_start_episodes is not None and target_info.total_episodes < args.expected_start_episodes:
        raise ValueError(f"Target has {target_info.total_episodes} episodes, before expected start {args.expected_start_episodes}")
    if args.allow_resume and args.expected_start_frames is not None and target_info.total_frames < args.expected_start_frames:
        raise ValueError(f"Target has {target_info.total_frames} frames, before expected start {args.expected_start_frames}")

    target_contract = _feature_contract(target_info.features)
    camera_keys = [key for key, ft in target_info.features.items() if ft["dtype"] == "video"]
    vector_keys = [key for key in target_contract if key not in camera_keys]

    print(f"Target: {args.target_repo_id} at {target_root}")
    print(f"Start: episodes={target_info.total_episodes}, frames={target_info.total_frames}, fps={target_info.fps}")
    print("Sources:")
    for root in roots:
        print(f"  {root}")

    planned = []
    data_by_root = {}
    plan_by_root = {}
    next_ep = args.expected_start_episodes if args.allow_resume and args.expected_start_episodes is not None else target_info.total_episodes
    next_index = args.expected_start_frames if args.allow_resume and args.expected_start_frames is not None else target_info.total_frames

    for root in roots:
        info = load_info(root)
        if info.fps != target_info.fps:
            raise ValueError(f"{root.name} fps={info.fps}, target fps={target_info.fps}")
        if _feature_contract(info.features) != target_contract:
            raise ValueError(f"{root.name} feature contract differs from target")
        df, plan = _plan_root(root, info.fps, args)
        tasks = load_tasks(root)
        data_by_root[root] = (df, info, tasks)
        plan_by_root[root] = plan
        for src_ep in sorted(plan):
            n_frames, n_keep = plan[src_ep]
            planned.append(
                {
                    "source_root": root.name,
                    "source_episode": src_ep,
                    "target_episode": next_ep,
                    "source_frames": n_frames,
                    "kept_frames": n_keep,
                    "cut_frames": n_frames - n_keep,
                    "cut_seconds": round((n_frames - n_keep) / info.fps, 3),
                    "target_from_index": next_index,
                    "target_to_index": next_index + n_keep,
                }
            )
            next_ep += 1
            next_index += n_keep

    manifest = pd.DataFrame(planned)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"Manifest -> {manifest_path}")
    print(
        f"Plan: append {len(manifest)} episodes, {int(manifest.kept_frames.sum())} frames "
        f"(cut {int(manifest.cut_frames.sum())} / {100 * manifest.cut_frames.sum() / manifest.source_frames.sum():.1f}%)."
    )
    print(f"Expected final: episodes={next_ep}, frames={next_index}")
    if args.dry_run:
        print("[dry-run] Nothing appended.")
        return

    encoder = VideoEncoderConfig(
        vcodec=args.camera_encoder_vcodec,
        crf=args.camera_encoder_crf,
        g=args.camera_encoder_g,
        pix_fmt=args.camera_encoder_pix_fmt,
    )
    target = LeRobotDataset.resume(
        args.target_repo_id,
        root=target_root,
        video_backend=args.video_backend,
        batch_encoding_size=1,
        camera_encoder=encoder,
        encoder_threads=args.encoder_threads,
    )
    original_video_files_size_mb = target.meta.info.video_files_size_in_mb
    if args.target_video_files_size_mb is not None:
        target.meta.info.video_files_size_in_mb = args.target_video_files_size_mb

    appended = 0
    for root in roots:
        df, info, tasks = data_by_root[root]
        plan = plan_by_root[root]
        ordered = df.sort_values(["episode_index", "frame_index"])
        task_by_index = {int(row.task_index): task for task, row in tasks.iterrows()}
        for src_ep in sorted(plan):
            target_episode = int(
                manifest[
                    (manifest["source_root"] == root.name) & (manifest["source_episode"] == src_ep)
                ]["target_episode"].iloc[0]
            )
            if target_episode < target.meta.total_episodes:
                print(f"skipping already appended target episode {target_episode}: {root.name} ep {src_ep}")
                continue
            if target_episode != target.meta.total_episodes:
                raise ValueError(
                    f"Append order mismatch: next target episode is {target.meta.total_episodes}, "
                    f"but manifest row expects {target_episode}"
                )
            n_frames, n_keep = plan[src_ep]
            rows = ordered[ordered["episode_index"] == src_ep].head(n_keep)
            for _, row in rows.iterrows():
                frame = {key: np.asarray(row[key]) for key in vector_keys}
                for key in camera_keys:
                    img_path = root / "images" / key / f"episode-{int(src_ep):06d}" / f"frame-{int(row['frame_index']):06d}.png"
                    if not img_path.exists():
                        raise FileNotFoundError(img_path)
                    frame[key] = load_image_as_numpy(img_path, dtype=np.uint8, channel_first=False)
                frame["task"] = task_by_index[int(row["task_index"])]
                target.add_frame(frame)
            target.save_episode()
            appended += 1
            print(f"appended {appended:03d}/{len(manifest)}: {root.name} ep {src_ep} -> kept {n_keep}/{n_frames}")

    target.meta.info.video_files_size_in_mb = original_video_files_size_mb
    write_info(target.meta.info, target.root)
    target.finalize()
    print(f"Done. Final target: episodes={target.meta.total_episodes}, frames={target.meta.total_frames}")


if __name__ == "__main__":
    main()
