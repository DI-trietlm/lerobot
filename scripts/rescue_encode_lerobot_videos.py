#!/usr/bin/env python
"""Encode pending LeRobot PNG episode folders into videos and repair metadata.

This is meant for datasets where recording completed with
``streaming_encoding=false`` but delayed/batch video encoding failed before
creating ``videos/`` and the ``videos/...`` columns in ``meta/episodes``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.video_utils import encode_video_frames, get_video_duration_in_s, get_video_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="LeRobot dataset root")
    parser.add_argument("--vcodec", default="h264", help="Video codec to use, matching record config")
    parser.add_argument("--crf", type=float, default=30, help="Encoder CRF/quality value")
    parser.add_argument("--g", type=int, default=2, help="GOP size")
    parser.add_argument("--pix-fmt", default="yuv420p", help="Pixel format")
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rescued videos")
    parser.add_argument("--delete-images", action="store_true", help="Delete image episode folders after success")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak files for metadata")
    return parser.parse_args()


def backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, path.with_suffix(path.suffix + f".bak_rescue_{stamp}"))


def episode_video_path(root: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    return root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    info_path = root / "meta" / "info.json"
    episodes_dir = root / "meta" / "episodes"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    fps = int(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = [key for key, ft in info["features"].items() if ft.get("dtype") == "video"]
    if not video_keys:
        raise RuntimeError("No video features found in meta/info.json")

    episode_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files found under {episodes_dir}")

    camera_encoder = VideoEncoderConfig(
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
        g=args.g,
        crf=args.crf,
    )

    first_video_for_key: dict[str, Path] = {}
    changed_episode_files: list[Path] = []
    encoded_count = 0

    for episode_file in episode_files:
        df = pd.read_parquet(episode_file)
        if "episode_index" not in df.columns or "length" not in df.columns:
            raise RuntimeError(f"{episode_file} is missing episode_index/length columns")

        for row_index, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            expected_frames = int(row["length"])
            video_chunk = ep_idx // chunks_size
            video_file = ep_idx % chunks_size

            for video_key in video_keys:
                img_dir = root / "images" / video_key / f"episode-{ep_idx:06d}"
                if not img_dir.is_dir():
                    raise FileNotFoundError(f"Missing image directory: {img_dir}")

                frame_count = len(list(img_dir.glob("frame-*.png")))
                if frame_count != expected_frames:
                    raise RuntimeError(
                        f"{img_dir} has {frame_count} frames, expected {expected_frames}"
                    )

                out_path = episode_video_path(root, video_key, video_chunk, video_file)
                if args.overwrite or not out_path.exists():
                    print(f"Encoding {video_key} episode {ep_idx:06d} -> {out_path}")
                    encode_video_frames(
                        img_dir,
                        out_path,
                        fps,
                        camera_encoder=camera_encoder,
                        encoder_threads=args.encoder_threads,
                        overwrite=args.overwrite,
                    )
                    encoded_count += 1
                else:
                    print(f"Keeping existing video: {out_path}")

                first_video_for_key.setdefault(video_key, out_path)
                duration = get_video_duration_in_s(out_path)
                prefix = f"videos/{video_key}"
                df.loc[row_index, f"{prefix}/chunk_index"] = video_chunk
                df.loc[row_index, f"{prefix}/file_index"] = video_file
                df.loc[row_index, f"{prefix}/from_timestamp"] = 0.0
                df.loc[row_index, f"{prefix}/to_timestamp"] = duration

        changed_episode_files.append(episode_file)
        if not args.no_backup:
            backup(episode_file)
        df = df.convert_dtypes(dtype_backend="pyarrow")
        df.to_parquet(episode_file)

    for video_key, video_path in first_video_for_key.items():
        info["features"][video_key]["info"] = get_video_info(video_path, camera_encoder=camera_encoder)

    if not args.no_backup:
        backup(info_path)
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    if args.delete_images:
        shutil.rmtree(root / "images")

    print(
        f"Done. Encoded {encoded_count} videos, updated {len(changed_episode_files)} episode parquet file(s)."
    )


if __name__ == "__main__":
    main()
