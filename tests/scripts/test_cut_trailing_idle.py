#!/usr/bin/env python

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("av", reason="PyAV is required for video encoding/decoding")

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _run_cut_script(args: list[str]) -> None:
    script = Path(__file__).resolve().parents[2] / "xai" / "cut_trailing_idle.py"
    old_argv = sys.argv
    try:
        sys.argv = [str(script), *args]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def _record_deferred_source(root: Path) -> None:
    features = {
        "observation.images.cam": {
            "dtype": "video",
            "shape": (16, 16, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id="test/deferred_source",
        fps=10,
        features=features,
        root=root,
        batch_encoding_size=10,
        defer_video_encoding=True,
    )
    for ep in range(2):
        for frame_idx in range(4):
            state = np.array([float(frame_idx), float(ep)], dtype=np.float32)
            moving = frame_idx < 2
            action = state + np.array([10.0, 0.0], dtype=np.float32) if moving else state.copy()
            image = np.full((16, 16, 3), 20 + ep * 50 + frame_idx, dtype=np.uint8)
            dataset.add_frame(
                {
                    "observation.images.cam": image,
                    "observation.state": state,
                    "action": action,
                    "task": "test task",
                }
            )
        dataset.save_episode()
    dataset.finalize()


def test_cut_trailing_idle_reads_deferred_images_and_writes_video_dataset(tmp_path):
    src_root = tmp_path / "src"
    out_root = tmp_path / "out"
    csv_path = tmp_path / "cuts.csv"
    _record_deferred_source(src_root)

    assert (src_root / "images").exists()
    assert not (src_root / "videos").exists()

    common_args = [
        "--repo-id",
        "test/deferred_source",
        "--root",
        str(src_root),
        "--speed-thresh",
        "5.0",
        "--min-cut-s",
        "0.1",
        "--keep-min-s",
        "0.1",
        "--video-backend",
        "pyav",
        "--csv",
        str(csv_path),
    ]
    _run_cut_script([*common_args, "--dry-run"])

    cuts = pd.read_csv(csv_path)
    assert cuts["n_keep"].tolist() == [2, 2]
    assert cuts["n_cut"].tolist() == [2, 2]

    _run_cut_script(
        [
            *common_args,
            "--output-repo-id",
            "test/deferred_cut",
            "--output-root",
            str(out_root),
            "--camera-encoder-vcodec",
            "h264",
            "--camera-encoder-pix-fmt",
            "yuv420p",
        ]
    )

    output = LeRobotDataset("test/deferred_cut", root=out_root, video_backend="pyav")
    assert output.num_episodes == 2
    assert output.num_frames == 4
    item = output[0]
    assert tuple(item["observation.images.cam"].shape) == (3, 16, 16)
    assert int(output[1]["frame_index"]) == 1
    assert float(output[1]["timestamp"]) == pytest.approx(0.1)
    assert list((out_root / "videos").rglob("*.mp4"))
    assert not list((out_root / "images").rglob("*.png"))
