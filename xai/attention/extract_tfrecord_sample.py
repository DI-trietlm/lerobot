#!/usr/bin/env python3
"""Extract one image + instruction from a TFRecord for XAI testing.

This uses the lightweight `tfrecord` package (no TensorFlow dependency).
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tfrecord import reader


def _nth_record(data_path: str, index: int) -> dict:
    for i, record in enumerate(reader.tfrecord_loader(data_path, index_path=None)):
        if i == index:
            return record
    raise IndexError(f"TFRecord index {index} is out of range")


def _get_step_bytes(record: dict, key: str, step: int) -> bytes:
    if key not in record:
        raise KeyError(f"Key not found: {key}")
    arr = record[key]
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray for key {key}, got {type(arr)}")
    if arr.size == 0:
        raise ValueError(f"Key {key} has empty array")
    if step < 0 or step >= arr.shape[0]:
        raise IndexError(f"Step {step} out of range for key {key} (len={arr.shape[0]})")
    return bytes(arr[step])


def _pick_image_key(record: dict, preferred: str) -> str:
    if preferred in record:
        return preferred
    for fallback in [
        "steps/observation/image_0",
        "steps/observation/image_1",
        "steps/observation/image_2",
        "steps/observation/image_3",
    ]:
        if fallback in record:
            return fallback
    raise KeyError("No image key found in TFRecord")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract image + instruction from a TFRecord.")
    parser.add_argument(
        "--tfrecord",
        default="data/bridge_dataset-train.tfrecord-00000-of-01024",
        help="Path to the TFRecord shard",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Record index within the shard (default: 0)",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=0,
        help="Step index inside the episode (default: 0)",
    )
    parser.add_argument(
        "--image-key",
        default="steps/observation/image_2",
        help="Image feature key (default: steps/observation/image_2)",
    )
    parser.add_argument(
        "--text-key",
        default="steps/language_instruction",
        help="Instruction feature key (default: steps/language_instruction)",
    )
    parser.add_argument(
        "--out-dir",
        default="xai/outputs",
        help="Directory to write extracted files",
    )
    args = parser.parse_args()

    tfrecord_path = Path(args.tfrecord)
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")

    record = _nth_record(str(tfrecord_path), args.episode_index)

    image_key = _pick_image_key(record, args.image_key)
    img_bytes = _get_step_bytes(record, image_key, args.step_index)
    text_bytes = _get_step_bytes(record, args.text_key, args.step_index)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_out = out_dir / f"tfds_sample_{args.episode_index:05d}_{args.step_index:03d}.jpg"
    img.save(image_out, format="JPEG", quality=95)

    instruction = text_bytes.decode("utf-8", errors="replace")
    text_out = out_dir / f"tfds_sample_{args.episode_index:05d}_{args.step_index:03d}.txt"
    text_out.write_text(instruction, encoding="utf-8")

    print("Extracted:")
    print(f"  image: {image_out}")
    print(f"  text : {text_out}")
    print(f"  image_key: {image_key}")
    print(f"  text_key : {args.text_key}")
    print(f"  instruction: {instruction}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
