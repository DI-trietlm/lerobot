#!/usr/bin/env python
"""Consolidate appended episodes into video file-001 for each camera."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s


ROOT = Path(r"F:\llms\hf\models\lerobot\di-techinnova\so-arm-101-pouring-0.3-cutted")
BACKUP = Path("xai/consolidate_appended_videos_backup")
START_EPISODE = 50
END_EPISODE = 120
FPS = 15
CAMERA_KEYS = ["observation.images.camera1", "observation.images.camera2"]


def _to_nested_list(value):
    if isinstance(value, np.ndarray):
        return [_to_nested_list(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_to_nested_list(item) for item in value]
    return value


def _normalize_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if column.startswith("__index"):
            df = df.drop(columns=[column])
        elif df[column].dtype == "object":
            df[column] = df[column].map(_to_nested_list)
    return df


def _load_episodes() -> pd.DataFrame:
    paths = sorted((ROOT / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(ROOT / "meta" / "episodes")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).sort_values(
        "episode_index"
    )


def _backup_metadata() -> None:
    if BACKUP.exists():
        print(f"keeping existing backup: {BACKUP}")
        return
    shutil.copytree(ROOT / "meta" / "episodes", BACKUP / "meta" / "episodes")
    shutil.copy2(ROOT / "meta" / "info.json", BACKUP / "meta" / "info.json")


def _video_path(camera_key: str, file_index: int) -> Path:
    return ROOT / "videos" / camera_key / "chunk-000" / f"file-{file_index:03d}.mp4"


def _ordered_input_paths(episodes: pd.DataFrame, camera_key: str) -> list[Path]:
    paths = []
    appended = episodes[
        (episodes["episode_index"] >= START_EPISODE) & (episodes["episode_index"] < END_EPISODE)
    ].sort_values("episode_index")
    if len(appended) != END_EPISODE - START_EPISODE:
        raise ValueError(f"Expected {END_EPISODE - START_EPISODE} appended episodes, got {len(appended)}")
    for _, row in appended.iterrows():
        chunk = int(row[f"videos/{camera_key}/chunk_index"])
        file_index = int(row[f"videos/{camera_key}/file_index"])
        path = ROOT / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def _rewrite_episode_video_metadata(episodes: pd.DataFrame) -> pd.DataFrame:
    episodes = episodes.copy()
    appended = episodes[
        (episodes["episode_index"] >= START_EPISODE) & (episodes["episode_index"] < END_EPISODE)
    ].sort_values("episode_index")
    cumulative = 0.0
    for _, row in appended.iterrows():
        ep = int(row["episode_index"])
        length = int(row["length"])
        next_cumulative = cumulative + length / FPS
        mask = episodes["episode_index"] == ep
        for camera_key in CAMERA_KEYS:
            episodes.loc[mask, f"videos/{camera_key}/chunk_index"] = 0
            episodes.loc[mask, f"videos/{camera_key}/file_index"] = 1
            episodes.loc[mask, f"videos/{camera_key}/from_timestamp"] = cumulative
            episodes.loc[mask, f"videos/{camera_key}/to_timestamp"] = next_cumulative
        cumulative = next_cumulative
    return episodes


def _write_episodes(episodes: pd.DataFrame) -> None:
    episodes = _normalize_nested_columns(episodes.sort_values("episode_index").reset_index(drop=True))
    episodes[episodes["episode_index"] < START_EPISODE].reset_index(drop=True).to_parquet(
        ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
    )
    episodes[episodes["episode_index"] >= START_EPISODE].reset_index(drop=True).to_parquet(
        ROOT / "meta" / "episodes" / "chunk-000" / "file-001.parquet", index=False
    )


def main() -> None:
    episodes = _load_episodes()
    _backup_metadata()

    temp_outputs = {}
    for camera_key in CAMERA_KEYS:
        inputs = _ordered_input_paths(episodes, camera_key)
        output = _video_path(camera_key, 1)
        temp_output = output.with_name("file-001.consolidated.tmp.mp4")
        temp_output.unlink(missing_ok=True)
        print(f"concatenating {camera_key}: {len(inputs)} files -> {temp_output.relative_to(ROOT)}")
        concatenate_video_files(inputs, temp_output, overwrite=True, compatibility_check=True)
        expected_duration = float(
            episodes[
                (episodes["episode_index"] >= START_EPISODE) & (episodes["episode_index"] < END_EPISODE)
            ]["length"].sum()
            / FPS
        )
        actual_duration = get_video_duration_in_s(temp_output)
        if abs(actual_duration - expected_duration) > 0.2:
            raise ValueError(
                f"{camera_key} consolidated duration {actual_duration:.6f}s differs from "
                f"expected {expected_duration:.6f}s"
            )
        temp_outputs[camera_key] = temp_output

    episodes = _rewrite_episode_video_metadata(episodes)
    _write_episodes(episodes)

    for camera_key, temp_output in temp_outputs.items():
        output = _video_path(camera_key, 1)
        output.unlink(missing_ok=True)
        shutil.move(str(temp_output), str(output))
        for path in sorted((ROOT / "videos" / camera_key / "chunk-000").glob("file-*.mp4")):
            if path.name not in {"file-000.mp4", "file-001.mp4"}:
                path.unlink()
        print(f"kept {camera_key}: file-000.mp4, file-001.mp4")

    print("done")


if __name__ == "__main__":
    main()
