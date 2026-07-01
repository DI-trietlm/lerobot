#!/usr/bin/env python
"""Rewrite LeRobot data parquet vector columns as list<float32>."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _as_float_list(value):
    return np.asarray(value, dtype=np.float32).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    schema = pa.schema(
        [
            pa.field("action", pa.list_(pa.float32())),
            pa.field("observation.state", pa.list_(pa.float32())),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    )

    for path in sorted((root / "data").glob("chunk-000/file-*.parquet")):
        df = pd.read_parquet(path)
        df = df[["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]]
        df["action"] = df["action"].map(_as_float_list)
        df["observation.state"] = df["observation.state"].map(_as_float_list)
        df["timestamp"] = df["timestamp"].astype("float32")
        for column in ["frame_index", "episode_index", "index", "task_index"]:
            df[column] = df[column].astype("int64")
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(table, path)
        print(f"rewrote {path}")


if __name__ == "__main__":
    main()
