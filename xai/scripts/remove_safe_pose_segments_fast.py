#!/usr/bin/env python
"""Fast safe-pose cleanup for an already-consolidated LeRobot video dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd


CAMERA_KEYS = ["observation.images.camera1", "observation.images.camera2"]
BOOKKEEPING = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
DEFAULT_BOUNDARIES = [0, 50, 120]
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


def _load_parquets(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(root)
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _normalize_nested(value):
    if isinstance(value, np.ndarray):
        return [_normalize_nested(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_normalize_nested(item) for item in value]
    return value


def _normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].map(_normalize_nested)
    return df


def _episode_file_index(ep: int, boundaries: list[int]) -> int:
    for idx, start in enumerate(boundaries):
        next_start = boundaries[idx + 1] if idx + 1 < len(boundaries) else 10**12
        if start <= ep < next_start:
            return idx
    raise ValueError(f"episode {ep} is before first boundary {boundaries[0]}")


def _keep_frame(ep: int, frame: int, cut_ranges: dict[int, list[tuple[int, int]]]) -> bool:
    return all(not (start <= frame <= end) for start, end in cut_ranges.get(ep, []))


def _stats_for_array(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _stats_for_scalar(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.shape[0])],
        "q01": [float(np.quantile(arr, 0.01))],
        "q10": [float(np.quantile(arr, 0.10))],
        "q50": [float(np.quantile(arr, 0.50))],
        "q90": [float(np.quantile(arr, 0.90))],
        "q99": [float(np.quantile(arr, 0.99))],
    }


def _write_split_parquets(df: pd.DataFrame, base: Path, file_indices: pd.Series) -> None:
    for file_index in sorted(file_indices.unique()):
        part = df[file_indices == file_index].reset_index(drop=True)
        out = base / "chunk-000" / f"file-{int(file_index):03d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out, index=False)


def _encode_filtered_video(
    input_path: Path,
    output_path: Path,
    keep_flags: list[bool],
    fps: int,
    vcodec: str,
    pix_fmt: str,
    crf: float,
    gop: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_container = av.open(str(input_path), "r")
    output_container = av.open(str(output_path), "w", options={"movflags": "faststart"})
    input_stream = input_container.streams.video[0]
    width = input_stream.codec_context.width
    height = input_stream.codec_context.height
    output_stream = output_container.add_stream(
        vcodec,
        rate=fps,
        options={"crf": str(crf), "g": str(gop)},
    )
    output_stream.width = width
    output_stream.height = height
    output_stream.pix_fmt = pix_fmt
    output_stream.time_base = Fraction(1, fps)

    written = 0
    decoded = 0
    for frame in input_container.decode(video=0):
        if decoded >= len(keep_flags):
            break
        if keep_flags[decoded]:
            frame = frame.reformat(width=width, height=height, format="rgb24")
            frame.pts = written
            frame.time_base = Fraction(1, fps)
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
            written += 1
        decoded += 1

    if decoded != len(keep_flags):
        raise ValueError(f"{input_path}: decoded {decoded} frames but expected {len(keep_flags)}")
    for packet in output_stream.encode():
        output_container.mux(packet)
    input_container.close()
    output_container.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--boundaries", default="0,50,120")
    parser.add_argument("--camera-encoder-vcodec", default="h264")
    parser.add_argument("--camera-encoder-crf", type=float, default=30)
    parser.add_argument("--camera-encoder-g", type=int, default=2)
    parser.add_argument("--camera-encoder-pix-fmt", default="yuv420p")
    args = parser.parse_args()

    src_root = Path(args.root)
    out_root = Path(args.output_root)
    if out_root.exists():
        raise FileExistsError(out_root)
    boundaries = [int(item) for item in args.boundaries.split(",") if item.strip()]

    info = json.loads((src_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    data = _load_parquets(src_root / "data").sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    episodes = _load_parquets(src_root / "meta" / "episodes").sort_values("episode_index").reset_index(drop=True)

    keep_mask = np.array(
        [
            _keep_frame(int(row.episode_index), int(row.frame_index), DEFAULT_CUT_RANGES)
            for row in data[["episode_index", "frame_index"]].itertuples(index=False)
        ],
        dtype=bool,
    )
    cut_manifest = []
    for ep, g in data.groupby("episode_index", sort=True):
        local_keep = keep_mask[g.index.to_numpy()]
        cut_manifest.append(
            {
                "episode_index": int(ep),
                "n_frames": int(len(g)),
                "n_keep": int(local_keep.sum()),
                "n_cut": int((~local_keep).sum()),
                "cut_ranges": ";".join(f"{s}-{e}" for s, e in DEFAULT_CUT_RANGES.get(int(ep), [])),
            }
        )
    manifest = pd.DataFrame(cut_manifest)
    print(manifest[manifest["n_cut"] > 0].to_string(index=False))
    print(f"frames: {len(data)} -> {int(keep_mask.sum())} (cut {int((~keep_mask).sum())})")

    out_root.mkdir(parents=True)
    shutil.copytree(src_root / "meta", out_root / "meta")
    (out_root / "data" / "chunk-000").mkdir(parents=True)
    (out_root / "videos").mkdir(parents=True)

    new_data_parts = []
    new_episode_rows = []
    global_index = 0
    for ep, g in data.groupby("episode_index", sort=True):
        ep = int(ep)
        kept = g[keep_mask[g.index.to_numpy()]].copy().reset_index(drop=True)
        if kept.empty:
            raise ValueError(f"Episode {ep} would become empty")
        length = len(kept)
        kept["frame_index"] = np.arange(length, dtype=np.int64)
        kept["timestamp"] = kept["frame_index"].astype(np.float32) / fps
        kept["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
        global_index += length
        new_data_parts.append(kept)

        row = episodes[episodes["episode_index"].astype(int) == ep].iloc[0].copy()
        file_index = _episode_file_index(ep, boundaries)
        row["length"] = length
        row["dataset_from_index"] = int(kept["index"].iloc[0])
        row["dataset_to_index"] = int(kept["index"].iloc[-1]) + 1
        row["data/chunk_index"] = 0
        row["data/file_index"] = file_index
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = file_index
        new_episode_rows.append(row)

    new_data = pd.concat(new_data_parts, ignore_index=True)
    new_episodes = pd.DataFrame(new_episode_rows).sort_values("episode_index").reset_index(drop=True)

    for file_index in sorted({_episode_file_index(int(ep), boundaries) for ep in new_episodes["episode_index"]}):
        cumulative = 0.0
        eps_in_file = new_episodes[new_episodes["episode_index"].astype(int).map(lambda ep: _episode_file_index(ep, boundaries) == file_index)]
        for idx, row in eps_in_file.iterrows():
            length = int(row["length"])
            next_cumulative = cumulative + length / fps
            for camera_key in CAMERA_KEYS:
                new_episodes.loc[idx, f"videos/{camera_key}/chunk_index"] = 0
                new_episodes.loc[idx, f"videos/{camera_key}/file_index"] = file_index
                new_episodes.loc[idx, f"videos/{camera_key}/from_timestamp"] = cumulative
                new_episodes.loc[idx, f"videos/{camera_key}/to_timestamp"] = next_cumulative
            cumulative = next_cumulative

    data_file_indices = new_data["episode_index"].astype(int).map(lambda ep: _episode_file_index(ep, boundaries))
    ep_file_indices = new_episodes["episode_index"].astype(int).map(lambda ep: _episode_file_index(ep, boundaries))
    _write_split_parquets(_normalize_object_columns(new_data), out_root / "data", data_file_indices)
    _write_split_parquets(_normalize_object_columns(new_episodes), out_root / "meta" / "episodes", ep_file_indices)

    for camera_key in CAMERA_KEYS:
        for file_index in sorted(data_file_indices.unique()):
            eps = new_episodes[new_episodes["episode_index"].astype(int).map(lambda ep: _episode_file_index(ep, boundaries) == file_index)]
            flags: list[bool] = []
            for ep in eps["episode_index"].astype(int):
                original = data[data["episode_index"].astype(int) == ep]
                flags.extend(keep_mask[original.index.to_numpy()].tolist())
            input_path = src_root / "videos" / camera_key / "chunk-000" / f"file-{int(file_index):03d}.mp4"
            output_path = out_root / "videos" / camera_key / "chunk-000" / f"file-{int(file_index):03d}.mp4"
            written = _encode_filtered_video(
                input_path,
                output_path,
                flags,
                fps,
                args.camera_encoder_vcodec,
                args.camera_encoder_pix_fmt,
                args.camera_encoder_crf,
                args.camera_encoder_g,
            )
            expected = int(sum(flags))
            if written != expected:
                raise ValueError(f"{output_path}: wrote {written} frames, expected {expected}")
            print(f"{camera_key} file-{int(file_index):03d}: {len(flags)} -> {written}")

    info["total_frames"] = int(len(new_data))
    info["total_episodes"] = int(len(new_episodes))
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    stats = json.loads((src_root / "meta" / "stats.json").read_text(encoding="utf-8"))
    for key in ["action", "observation.state"]:
        stats[key] = _stats_for_array(np.stack(new_data[key].to_numpy()).astype(np.float64))
    for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        stats[key] = _stats_for_scalar(new_data[key].to_numpy())
    (out_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=4), encoding="utf-8")

    manifest["cut_seconds"] = manifest["n_cut"] / fps
    manifest.to_csv(
        Path("xai") / "artifacts" / f"safe_pose_cut_manifest_{args.output_reppo_id if False else args.output_repo_id.split('/')[-1]}.csv",
        index=False,
    )
    print(f"done: {out_root}")


if __name__ == "__main__":
    main()
