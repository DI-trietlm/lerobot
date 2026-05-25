#!/usr/bin/env python
"""
LeWM Preprocessing — Timestamp-based resampling to uniform grid
===============================================================
Takes a raw session (from collect_lewm_raw.py) and resamples both streams
to a guaranteed 1:5 ratio using nearest-neighbor lookup on timestamps.

No interpolation of pixel values or joint positions — only index selection.

Output per session
------------------
  <session_dir>/processed/
    frames/
      000000.jpg, 000001.jpg, ...   (symlinks or copies of selected raw frames)
    camera_timestamps.npy           (N,)   float64  uniform grid, seconds from t0
    actions.npy                     (N, 5, 6) float32  5 actions per camera step
    states.npy                      (N, 6)    float32  robot state at camera time
    meta.json

Usage
-----
  uv run python scripts/preprocess_lewm.py \\
      --session-dir data/lewm_raw/session_20260515_143022 \\
      --target-cam-hz 15 \\
      --target-rob-hz 75

  # or process all sessions in a directory:
  uv run python scripts/preprocess_lewm.py \\
      --root-dir data/lewm_raw \\
      --target-cam-hz 15 \\
      --target-rob-hz 75
"""

import argparse
import json
import shutil
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeWM preprocessing: resample to uniform grid")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--session-dir", type=Path,
                     help="Path to a single session directory")
    grp.add_argument("--root-dir", type=Path,
                     help="Process all session_* subdirectories found here")
    p.add_argument("--target-cam-hz", type=int, default=15,
                   help="Target camera fps after resampling (default 15)")
    p.add_argument("--target-rob-hz", type=int, default=75,
                   help="Target robot fps after resampling (default 75)")
    p.add_argument("--copy-frames", action="store_true",
                   help="Copy JPEG files instead of symlinking (safer on Windows)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process even if processed/ already exists")
    return p.parse_args()


def _resample_session(
    session_dir: Path,
    target_cam_hz: int,
    target_rob_hz: int,
    copy_frames: bool,
    overwrite: bool,
) -> None:
    out_dir = session_dir / "processed"
    if out_dir.exists() and not overwrite:
        logger.info(f"  Skipping {session_dir.name} (processed/ exists, use --overwrite)")
        return

    # ── Load raw data ─────────────────────────────────────────────────────
    cam_ts_raw = np.load(session_dir / "camera_timestamps.npy")   # (N_cam,)
    rob        = np.load(session_dir / "robot_stream.npz")
    rob_ts     = rob["timestamps"]                                  # (N_rob,)
    rob_act    = rob["actions"]                                     # (N_rob, 6)
    rob_st     = rob["states"]                                      # (N_rob, 6)

    ratio = target_rob_hz // target_cam_hz
    if target_rob_hz % target_cam_hz != 0:
        raise ValueError(
            f"target_rob_hz ({target_rob_hz}) must be an integer multiple "
            f"of target_cam_hz ({target_cam_hz})"
        )

    # ── Build uniform camera grid ─────────────────────────────────────────
    # Grid starts at first camera timestamp, spaced at 1/target_cam_hz.
    # Trim to the time range covered by BOTH streams.
    t_start = max(cam_ts_raw[0],  rob_ts[0])
    t_end   = min(cam_ts_raw[-1], rob_ts[-1])
    cam_grid = np.arange(t_start, t_end, 1.0 / target_cam_hz)

    # ── Select nearest camera frame for each grid point ───────────────────
    # searchsorted gives insertion index; compare neighbours to find nearest.
    raw_cam_idxs = np.searchsorted(cam_ts_raw, cam_grid)
    raw_cam_idxs = np.clip(raw_cam_idxs, 0, len(cam_ts_raw) - 1)

    # Refine: pick left or right neighbour whichever is closer
    left_idxs  = np.maximum(raw_cam_idxs - 1, 0)
    left_err   = np.abs(cam_ts_raw[left_idxs]      - cam_grid)
    right_err  = np.abs(cam_ts_raw[raw_cam_idxs]   - cam_grid)
    cam_idxs   = np.where(left_err < right_err, left_idxs, raw_cam_idxs)

    # ── Select robot actions for each camera window ───────────────────────
    # For camera window [cam_grid[i], cam_grid[i+1]):
    #   place `ratio` uniform slots and pick the nearest robot step to each.
    n_windows  = len(cam_grid) - 1
    act_out    = np.zeros((n_windows, ratio, 6), dtype=np.float32)
    st_out     = np.zeros((n_windows, 6),        dtype=np.float32)

    for i in range(n_windows):
        t_lo, t_hi = cam_grid[i], cam_grid[i + 1]

        # `ratio` evenly-spaced slots inside the window (not at the edges)
        slots = np.linspace(t_lo, t_hi, ratio + 2)[1:-1]   # shape (ratio,)

        # Nearest robot index for each slot
        idxs  = np.searchsorted(rob_ts, slots)
        idxs  = np.clip(idxs, 0, len(rob_ts) - 1)
        act_out[i] = rob_act[idxs]

        # Robot state: nearest step to camera grid point
        cam_rob_idx = np.searchsorted(rob_ts, t_lo)
        cam_rob_idx = np.clip(cam_rob_idx, 0, len(rob_ts) - 1)
        st_out[i]   = rob_st[cam_rob_idx]

    # Trim cam_idxs to match n_windows (drop last grid point)
    cam_idxs_out = cam_idxs[:n_windows]
    cam_ts_out   = cam_grid[:n_windows]

    # ── Write output ──────────────────────────────────────────────────────
    frames_out_dir = out_dir / "frames"
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    frames_src_dir = session_dir / "camera"

    logger.info(f"  Writing {n_windows} frames to {out_dir}…")
    for j, raw_idx in enumerate(cam_idxs_out):
        src  = frames_src_dir / f"{raw_idx:06d}.jpg"
        dst  = frames_out_dir / f"{j:06d}.jpg"
        if dst.exists():
            dst.unlink()
        if copy_frames or not hasattr(dst, "symlink_to"):
            shutil.copy2(src, dst)
        else:
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)

    np.save(str(out_dir / "camera_timestamps.npy"), cam_ts_out.astype(np.float64))
    np.save(str(out_dir / "actions.npy"), act_out)   # (N, ratio, 6)
    np.save(str(out_dir / "states.npy"),  st_out)    # (N, 6)

    # Verify 1:ratio guarantee
    assert act_out.shape == (n_windows, ratio, 6), "shape mismatch"

    # Quality check: mean error between camera grid and selected raw timestamps
    cam_err_ms = np.abs(cam_ts_raw[cam_idxs_out] - cam_ts_out).mean() * 1e3

    # Load original meta
    meta_src = {}
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta_src = json.load(f)

    meta_out = {
        "source_session":   session_dir.name,
        "target_cam_hz":    target_cam_hz,
        "target_rob_hz":    target_rob_hz,
        "ratio":            ratio,
        "n_steps":          n_windows,
        "cam_grid_err_ms":  round(float(cam_err_ms), 3),
        "source_meta":      meta_src,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    logger.info(
        f"  Done: {n_windows} steps  |  "
        f"cam grid error = {cam_err_ms:.1f}ms  |  "
        f"action shape = {act_out.shape}"
    )


def main() -> None:
    args = _parse_args()

    if args.session_dir:
        sessions = [args.session_dir]
    else:
        sessions = sorted(args.root_dir.glob("session_*"))
        if not sessions:
            logger.error(f"No session_* directories found in {args.root_dir}")
            return
        logger.info(f"Found {len(sessions)} sessions in {args.root_dir}")

    for session_dir in sessions:
        logger.info(f"Processing {session_dir.name}…")
        try:
            _resample_session(
                session_dir=session_dir,
                target_cam_hz=args.target_cam_hz,
                target_rob_hz=args.target_rob_hz,
                copy_frames=args.copy_frames,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            logger.error(f"  Failed: {exc}")


if __name__ == "__main__":
    main()
