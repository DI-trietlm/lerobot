#!/usr/bin/env python
"""Repair missing appended episodes after an interrupted append/resume run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.io_utils import load_image_as_numpy, load_info
from lerobot.datasets.video_utils import encode_video_frames, get_video_duration_in_s
from lerobot.utils.utils import flatten_dict


ROOT = Path(r"F:\llms\hf\models\lerobot\di-techinnova\so-arm-101-pouring-0.3-cutted")
SOURCE_ROOT = Path(r"F:\llms\hf\models\lerobot\di-techinnova\so-arm-101-pouring-0.3_20260622_104354")
MANIFEST = Path("xai/append_cut_idle_manifest.csv")
BACKUP = Path("xai/repair_missing_append_backup")
TMP = Path("xai/_repair_missing_append_tmp")

EXPECTED_EPISODES = {50, 51, 52}
EXPECTED_TOTAL_EPISODES = 120
EXPECTED_TOTAL_FRAMES = 50829
DEFAULT_VIDEO_FILES_SIZE_MB = 200


def _read_all_parquet(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(root)
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _backup_current() -> None:
    if BACKUP.exists():
        print(f"keeping existing backup: {BACKUP}")
        return
    for rel in ["meta/info.json", "meta/stats.json", "meta/tasks.parquet"]:
        src = ROOT / rel
        dst = BACKUP / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for folder in ["data", "meta/episodes"]:
        shutil.copytree(ROOT / folder, BACKUP / folder)


def _load_source_data() -> pd.DataFrame:
    files = sorted((SOURCE_ROOT / "data").rglob("*.parquet"))
    return pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"],
            )
            for path in files
        ],
        ignore_index=True,
    ).sort_values(["episode_index", "frame_index"])


def _copy_kept_images(camera_key: str, src_ep: int, keep: int, dst_dir: Path) -> list[str]:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_dir = SOURCE_ROOT / "images" / camera_key / f"episode-{src_ep:06d}"
    paths: list[str] = []
    for new_frame in range(keep):
        src = src_dir / f"frame-{new_frame:06d}.png"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = dst_dir / f"frame-{new_frame:06d}.png"
        shutil.copy2(src, dst)
        paths.append(str(dst))
    return paths


def _build_missing_rows(
    source_df: pd.DataFrame,
    manifest: pd.DataFrame,
    features: dict,
    existing_episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    camera_keys = [key for key, feature in features.items() if feature["dtype"] == "video"]
    next_file_index = {
        key: int(existing_episodes[f"videos/{key}/file_index"].max()) + 1 for key in camera_keys
    }
    encoder = VideoEncoderConfig(vcodec="h264", crf=30, g=2, pix_fmt="yuv420p")

    data_rows = []
    episode_rows = []

    for _, mrow in manifest[manifest["target_episode"].isin(EXPECTED_EPISODES)].sort_values("target_episode").iterrows():
        src_ep = int(mrow.source_episode)
        target_ep = int(mrow.target_episode)
        keep = int(mrow.kept_frames)
        target_from = int(mrow.target_from_index)
        target_to = int(mrow.target_to_index)

        rows = source_df[source_df["episode_index"] == src_ep].head(keep).copy()
        if len(rows) != keep:
            raise ValueError(f"source episode {src_ep} has {len(rows)} rows, expected {keep}")

        local_index = np.arange(keep)
        rows["timestamp"] = local_index / 15
        rows["frame_index"] = local_index
        rows["episode_index"] = target_ep
        rows["index"] = np.arange(target_from, target_to)
        rows["task_index"] = 0
        data_rows.append(rows)

        episode_data = {
            "action": np.stack(rows["action"].to_numpy()),
            "observation.state": np.stack(rows["observation.state"].to_numpy()),
            "timestamp": rows["timestamp"].to_numpy(),
            "frame_index": rows["frame_index"].to_numpy(),
            "episode_index": rows["episode_index"].to_numpy(),
            "index": rows["index"].to_numpy(),
            "task_index": rows["task_index"].to_numpy(),
        }

        video_meta = {}
        for key in camera_keys:
            tmp_dir = TMP / key / f"episode-{target_ep:06d}"
            image_paths = _copy_kept_images(key, src_ep, keep, tmp_dir)
            episode_data[key] = image_paths

            file_index = next_file_index[key]
            next_file_index[key] += 1
            out_path = ROOT / "videos" / key / "chunk-000" / f"file-{file_index:03d}.mp4"
            encode_video_frames(tmp_dir, out_path, fps=15, camera_encoder=encoder, overwrite=True)
            duration = get_video_duration_in_s(out_path)
            video_meta.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": file_index,
                    f"videos/{key}/from_timestamp": 0.0,
                    f"videos/{key}/to_timestamp": duration,
                }
            )

        stats = compute_episode_stats(episode_data, features)
        episode_row = {
            "episode_index": target_ep,
            "tasks": ["Pour from orange cup into blue cup."],
            "length": keep,
            "data/chunk_index": 0,
            "data/file_index": 1,
            "dataset_from_index": target_from,
            "dataset_to_index": target_to,
            **video_meta,
            **flatten_dict({"stats": stats}),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 1,
        }
        episode_rows.append(episode_row)

    return pd.concat(data_rows, ignore_index=True), pd.DataFrame(episode_rows)


def _to_nested_list(value):
    if isinstance(value, np.ndarray):
        return [_to_nested_list(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_to_nested_list(item) for item in value]
    return value


def _normalize_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].map(_to_nested_list)
    return df


def main() -> None:
    info = load_info(ROOT)
    if info.total_episodes != EXPECTED_TOTAL_EPISODES or info.total_frames != EXPECTED_TOTAL_FRAMES:
        raise ValueError(f"Unexpected info totals before repair: {info.total_episodes=}, {info.total_frames=}")

    data = _read_all_parquet(ROOT / "data")
    episodes = _read_all_parquet(ROOT / "meta" / "episodes")
    data_present = set(int(ep) for ep in data["episode_index"].unique())
    episode_present = set(int(ep) for ep in episodes["episode_index"].unique())
    data_missing = EXPECTED_EPISODES - data_present
    episode_missing = EXPECTED_EPISODES - episode_present
    if not data_missing.issubset(EXPECTED_EPISODES) or episode_missing not in (EXPECTED_EPISODES, set()):
        raise ValueError(
            f"Unexpected missing state: {data_missing=}, {episode_missing=}. "
            f"This script only repairs episodes {EXPECTED_EPISODES}."
        )

    manifest = pd.read_csv(MANIFEST)
    source_df = _load_source_data()

    _backup_current()
    missing_data, missing_episodes = _build_missing_rows(source_df, manifest, info.features, episodes)

    repaired_data = (
        pd.concat(
            [data[~data["episode_index"].isin(EXPECTED_EPISODES)], missing_data],
            ignore_index=True,
        )
        .sort_values("index")
        .reset_index(drop=True)
    )
    if len(repaired_data) != EXPECTED_TOTAL_FRAMES:
        raise ValueError(f"Repaired data rows {len(repaired_data)} != {EXPECTED_TOTAL_FRAMES}")
    if not np.array_equal(repaired_data["index"].to_numpy(), np.arange(EXPECTED_TOTAL_FRAMES)):
        raise ValueError("Repaired global index is not contiguous")

    repaired_episodes = (
        pd.concat(
            [episodes[~episodes["episode_index"].isin(EXPECTED_EPISODES)], missing_episodes],
            ignore_index=True,
        )
        .sort_values("episode_index")
        .reset_index(drop=True)
    )
    if len(repaired_episodes) != EXPECTED_TOTAL_EPISODES:
        raise ValueError(f"Repaired episodes rows {len(repaired_episodes)} != {EXPECTED_TOTAL_EPISODES}")
    if repaired_episodes["episode_index"].tolist() != list(range(EXPECTED_TOTAL_EPISODES)):
        raise ValueError("Repaired episode rows are not ordered 0..119")

    bounds = repaired_data.groupby("episode_index")["index"].agg(["min", "max", "count"]).sort_index()
    for ep, row in bounds.iterrows():
        mask = repaired_episodes["episode_index"] == ep
        repaired_episodes.loc[mask, "length"] = int(row["count"])
        repaired_episodes.loc[mask, "dataset_from_index"] = int(row["min"])
        repaired_episodes.loc[mask, "dataset_to_index"] = int(row["max"]) + 1
        repaired_episodes.loc[mask, "data/chunk_index"] = 0
        repaired_episodes.loc[mask, "data/file_index"] = 0 if ep < 50 else 1
        repaired_episodes.loc[mask, "meta/episodes/chunk_index"] = 0
        repaired_episodes.loc[mask, "meta/episodes/file_index"] = 0 if ep < 50 else 1

    repaired_data[repaired_data["episode_index"] < 50].reset_index(drop=True).to_parquet(
        ROOT / "data" / "chunk-000" / "file-000.parquet", index=False
    )
    repaired_data[repaired_data["episode_index"] >= 50].reset_index(drop=True).to_parquet(
        ROOT / "data" / "chunk-000" / "file-001.parquet", index=False
    )
    repaired_episodes = _normalize_nested_columns(repaired_episodes)
    repaired_episodes[repaired_episodes["episode_index"] < 50].reset_index(drop=True).to_parquet(
        ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
    )
    repaired_episodes[repaired_episodes["episode_index"] >= 50].reset_index(drop=True).to_parquet(
        ROOT / "meta" / "episodes" / "chunk-000" / "file-001.parquet", index=False
    )

    info_path = ROOT / "meta" / "info.json"
    info_dict = json.loads(info_path.read_text(encoding="utf-8"))
    info_dict["video_files_size_in_mb"] = DEFAULT_VIDEO_FILES_SIZE_MB
    info_path.write_text(json.dumps(info_dict, indent=4) + "\n", encoding="utf-8")

    if TMP.exists():
        shutil.rmtree(TMP)

    print("repaired missing target episodes 50, 51, 52")
    print(f"backup: {BACKUP}")


if __name__ == "__main__":
    main()
