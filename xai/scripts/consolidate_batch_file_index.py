#!/usr/bin/env python
"""Consolidate a contiguous episode range into one data/meta/video file index."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s


ROOT = Path(r"F:\llms\hf\models\lerobot\di-techinnova\so-arm-101-pouring-0.3-cutted")
CAMERA_KEYS = ["observation.images.camera1", "observation.images.camera2"]


def _to_nested_list(value):
    if isinstance(value, np.ndarray):
        return [_to_nested_list(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_to_nested_list(item) for item in value]
    return value


def _normalize_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in list(df.columns):
        if column.startswith("__index"):
            df = df.drop(columns=[column])
        elif df[column].dtype == "object":
            df[column] = df[column].map(_to_nested_list)
    return df


def _load_parquets(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(root)
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _episode_file_index(ep: int, boundaries: list[int]) -> int:
    for idx, start in enumerate(boundaries):
        next_start = boundaries[idx + 1] if idx + 1 < len(boundaries) else 10**12
        if start <= ep < next_start:
            return idx
    raise ValueError(f"episode {ep} is before first boundary {boundaries[0]}")


def _video_path(camera_key: str, file_index: int) -> Path:
    return ROOT / "videos" / camera_key / "chunk-000" / f"file-{file_index:03d}.mp4"


def _write_split_parquets(df: pd.DataFrame, base_dir: Path, file_indices: pd.Series) -> None:
    for stale in base_dir.glob("chunk-000/file-*.parquet"):
        stale.unlink()
    for file_index in sorted(file_indices.unique()):
        part = df[file_indices == file_index].reset_index(drop=True)
        if part.empty:
            continue
        out = base_dir / "chunk-000" / f"file-{int(file_index):03d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out, index=False)


def main() -> None:
    global ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--start-episode", type=int, required=True)
    parser.add_argument("--end-episode", type=int, required=True)
    parser.add_argument("--file-index", type=int, required=True)
    parser.add_argument(
        "--boundaries",
        default="0,50,120",
        help="Episode starts for canonical file indices, e.g. 0,50,120.",
    )
    parser.add_argument("--backup-dir", default=None)
    args = parser.parse_args()

    ROOT = Path(args.root)
    boundaries = [int(x) for x in args.boundaries.split(",") if x.strip()]
    if args.start_episode not in boundaries or boundaries.index(args.start_episode) != args.file_index:
        raise ValueError(
            f"Expected start episode {args.start_episode} to be boundary for file index {args.file_index}; "
            f"got {boundaries}"
        )

    info = json.loads((ROOT / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    data = _load_parquets(ROOT / "data").sort_values("index").reset_index(drop=True)
    episodes = _load_parquets(ROOT / "meta" / "episodes").sort_values("episode_index").reset_index(
        drop=True
    )

    if info["total_episodes"] != len(episodes):
        raise ValueError(f"info total_episodes={info['total_episodes']} but episodes rows={len(episodes)}")
    if info["total_frames"] != len(data):
        raise ValueError(f"info total_frames={info['total_frames']} but data rows={len(data)}")
    if sorted(episodes["episode_index"].astype(int).tolist()) != list(range(info["total_episodes"])):
        raise ValueError("episodes are not exactly 0..total_episodes-1")
    if not np.array_equal(data["index"].to_numpy(), np.arange(len(data))):
        raise ValueError("data index is not contiguous in parquet order")

    if args.backup_dir:
        backup = Path(args.backup_dir)
        if not backup.exists():
            shutil.copytree(ROOT / "meta" / "episodes", backup / "meta" / "episodes")
            shutil.copytree(ROOT / "data", backup / "data")
            shutil.copy2(ROOT / "meta" / "info.json", backup / "meta" / "info.json")
        else:
            print(f"keeping existing backup: {backup}")

    appended = episodes[
        (episodes["episode_index"] >= args.start_episode)
        & (episodes["episode_index"] < args.end_episode)
    ].sort_values("episode_index")
    if len(appended) != args.end_episode - args.start_episode:
        raise ValueError(f"Expected {args.end_episode - args.start_episode} episodes, got {len(appended)}")

    temp_outputs = {}
    for camera_key in CAMERA_KEYS:
        input_paths = []
        for _, row in appended.iterrows():
            chunk = int(row[f"videos/{camera_key}/chunk_index"])
            file_index = int(row[f"videos/{camera_key}/file_index"])
            path = ROOT / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
            if not path.exists():
                raise FileNotFoundError(path)
            input_paths.append(path)
        output = _video_path(camera_key, args.file_index)
        temp_output = output.with_name(f"file-{args.file_index:03d}.consolidated.tmp.mp4")
        temp_output.unlink(missing_ok=True)
        print(f"concatenating {camera_key}: {len(input_paths)} files -> {temp_output.relative_to(ROOT)}")
        concatenate_video_files(input_paths, temp_output, overwrite=True, compatibility_check=True)
        expected_duration = float(appended["length"].sum() / fps)
        actual_duration = get_video_duration_in_s(temp_output)
        if abs(actual_duration - expected_duration) > 0.2:
            raise ValueError(
                f"{camera_key} duration {actual_duration:.6f}s != expected {expected_duration:.6f}s"
            )
        temp_outputs[camera_key] = temp_output

    # Sync data/meta file indices from canonical boundaries.
    ep_file_indices = episodes["episode_index"].astype(int).map(lambda ep: _episode_file_index(ep, boundaries))
    bounds = data.groupby("episode_index")["index"].agg(["min", "max", "count"]).sort_index()
    for ep, row in bounds.iterrows():
        file_index = _episode_file_index(int(ep), boundaries)
        mask = episodes["episode_index"] == ep
        episodes.loc[mask, "length"] = int(row["count"])
        episodes.loc[mask, "dataset_from_index"] = int(row["min"])
        episodes.loc[mask, "dataset_to_index"] = int(row["max"]) + 1
        episodes.loc[mask, "data/chunk_index"] = 0
        episodes.loc[mask, "data/file_index"] = file_index
        episodes.loc[mask, "meta/episodes/chunk_index"] = 0
        episodes.loc[mask, "meta/episodes/file_index"] = file_index

    cumulative = 0.0
    for _, row in appended.iterrows():
        ep = int(row["episode_index"])
        length = int(row["length"])
        next_cumulative = cumulative + length / fps
        mask = episodes["episode_index"] == ep
        for camera_key in CAMERA_KEYS:
            episodes.loc[mask, f"videos/{camera_key}/chunk_index"] = 0
            episodes.loc[mask, f"videos/{camera_key}/file_index"] = args.file_index
            episodes.loc[mask, f"videos/{camera_key}/from_timestamp"] = cumulative
            episodes.loc[mask, f"videos/{camera_key}/to_timestamp"] = next_cumulative
        cumulative = next_cumulative

    data_file_indices = data["episode_index"].astype(int).map(lambda ep: _episode_file_index(ep, boundaries))
    episodes = _normalize_nested_columns(episodes)
    data = data.drop(columns=[c for c in data.columns if c.startswith("__index")], errors="ignore")
    for col in ["action", "observation.state"]:
        if col in data.columns:
            data[col] = data[col].map(lambda x: np.asarray(x, dtype=np.float32))
    if "timestamp" in data.columns:
        data["timestamp"] = data["timestamp"].astype("float32")

    _write_split_parquets(data, ROOT / "data", data_file_indices)
    _write_split_parquets(episodes, ROOT / "meta" / "episodes", ep_file_indices)

    for camera_key, temp_output in temp_outputs.items():
        output = _video_path(camera_key, args.file_index)
        output.unlink(missing_ok=True)
        shutil.move(str(temp_output), str(output))
        for path in sorted((ROOT / "videos" / camera_key / "chunk-000").glob("file-*.mp4")):
            stem_index = int(path.stem.split("-")[-1])
            if stem_index > args.file_index:
                path.unlink()
        print(f"kept {camera_key}: file-000..file-{args.file_index:03d}.mp4")

    print("done")


if __name__ == "__main__":
    main()
