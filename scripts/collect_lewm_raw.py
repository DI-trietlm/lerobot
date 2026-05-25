#!/usr/bin/env python
"""
LeWM E2E Continuous Raw Data Collection — SO-ARM-101
=====================================================
Collects two fully asynchronous streams in one long continuous session.
NO episode boundaries. NO synchronization at collection time.

Alignment to a clean ratio (e.g. 15 Hz cam / 75 Hz robot → 1:5) is done
in preprocessing using timestamps — see scripts/preprocess_lewm.py.

Threads
-------
  lewm-cam-capture  : async_read() → bounded queue            (~cam_fps Hz)
  lewm-cam-writer   : queue → JPEG files on disk              (disk-bound)
  lewm-robot        : sync_read + get_action + send_action    (~robot_fps Hz)
  lewm-segments     : input() → segment timestamp markers     (user-driven)

Press ENTER during recording to insert a segment boundary marker.
Press Ctrl+C to stop and save everything.

Output layout
-------------
  <output_dir>/session_YYYYMMDD_HHMMSS/
    camera/
      000000.jpg, 000001.jpg, ...     (JPEG, quality=90)
    camera_timestamps.npy             float64, seconds from t0
    robot_stream.npz
      timestamps  : (N,)   float64   seconds from t0
      actions     : (N, 6) float32   commanded joint positions (degrees)
      states      : (N, 6) float32   actual    joint positions (degrees)
      motor_names : (6,)   str
    segments.npy                      (K,) float64, user-marked boundaries from t0
    meta.json                         collection metadata

Usage
-----
  uv run python scripts/collect_lewm_raw.py \\
      --robot-port COM6 --robot-id DI_VLA_FOLLOWER \\
      --teleop-port COM5 --teleop-id DI_VLA_LEADER \\
      --camera-index 0 --camera-width 640 --camera-height 360 \\
      --camera-fps 30 --robot-fps 150 \\
      --output-dir data/lewm_raw
"""

import argparse
import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOTOR_NAMES   = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                 "wrist_flex", "wrist_roll", "gripper"]
JPEG_QUALITY  = 90
WRITER_QUEUE_MAX = 120   # ~4s buffer at 30 Hz; prevents RAM blowup if disk is slow
STATS_INTERVAL   = 5.0  # seconds between live-stats prints


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeWM E2E continuous raw data collection")
    p.add_argument("--robot-port",  required=True,
                   help="Serial port of the SO-101 follower arm (e.g. COM6)")
    p.add_argument("--teleop-port", required=True,
                   help="Serial port of the SO-101 leader arm  (e.g. COM5)")
    p.add_argument("--robot-id",    default=None)
    p.add_argument("--teleop-id",   default=None)
    p.add_argument("--camera-index",   type=int, default=0)
    p.add_argument("--camera-width",   type=int, default=640)
    p.add_argument("--camera-height",  type=int, default=360)
    p.add_argument("--camera-fps",     type=int, default=30,
                   help="Target camera fps (default 30)")
    p.add_argument("--camera-fourcc",  default=None,
                   help="Force FOURCC codec e.g. MJPG")
    p.add_argument("--camera-backend", default="any",
                   choices=["any", "dshow", "msmf", "v4l2"],
                   help="OpenCV backend (use 'dshow' on Windows to reduce burst jitter)")
    p.add_argument("--robot-fps", type=int, default=150,
                   help="Target robot control rate (default 150)")
    p.add_argument("--output-dir", default="data/lewm_raw",
                   help="Root directory; session sub-folder is created automatically")
    p.add_argument("--session-name", default=None,
                   help="Override auto-generated session name (YYYYMMDD_HHMMSS)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Thread: camera capture  (~cam_fps Hz)
# ─────────────────────────────────────────────────────────────────────────────

def _camera_capture_thread(
    camera,
    frame_queue: queue.Queue,
    stop_event:  threading.Event,
    stats:       dict,
) -> None:
    """
    Calls async_read(timeout_ms=100) so the thread exits within 100ms of
    stop_event.set() without blocking indefinitely.
    Pushes (perf_counter_ts, frame_rgb) into frame_queue (bounded).
    """
    while not stop_event.is_set():
        try:
            frame = camera.async_read(timeout_ms=100)
            ts    = time.perf_counter()
            frame_queue.put((ts, frame), block=True, timeout=1.0)
            stats["cam"] += 1
        except TimeoutError:
            continue  # no new frame yet — re-check stop_event
        except queue.Full:
            logger.warning("[cam-capture] frame dropped — writer thread too slow?")
        except Exception as exc:
            if not stop_event.is_set():
                logger.warning(f"[cam-capture] {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Thread: camera writer  (disk I/O, decoupled from capture)
# ─────────────────────────────────────────────────────────────────────────────

def _camera_writer_thread(
    frame_queue:    queue.Queue,
    cam_dir:        Path,
    stop_event:     threading.Event,
    cam_timestamps: list,          # out: appended here in order
) -> None:
    """
    Drains frame_queue and writes JPEG files.
    Keeps running after stop_event until the queue is fully drained so no
    frames buffered before stop are lost.
    """
    idx = 0
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            ts, frame_rgb = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(cam_dir / f"{idx:06d}.jpg"),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        cam_timestamps.append(ts)
        idx += 1


# ─────────────────────────────────────────────────────────────────────────────
# Thread: robot control  (~robot_fps Hz)
# ─────────────────────────────────────────────────────────────────────────────

def _robot_control_thread(
    robot,
    teleop,
    robot_fps:  int,
    stop_event: threading.Event,
    rob_data:   dict,             # out: populated on thread exit
    stats:      dict,
) -> None:
    """
    Reads leader arm → sends to follower → reads follower state.
    Accumulates data locally; writes to rob_data dict on exit so the main
    thread can safely access it after join().
    """
    from lerobot.utils.robot_utils import precise_sleep

    interval  = 1.0 / robot_fps
    ts_list, act_list, st_list = [], [], []

    while not stop_event.is_set():
        loop_start = time.perf_counter()
        try:
            state_raw  = robot.bus.sync_read("Present_Position")
            action_raw = teleop.get_action()
            ts         = time.perf_counter()
            robot.send_action(action_raw)

            ts_list.append(ts)
            act_list.append([action_raw.get(f"{m}.pos", 0.0) for m in MOTOR_NAMES])
            st_list.append([state_raw.get(m, 0.0)           for m in MOTOR_NAMES])
            stats["rob"] += 1
        except Exception as exc:
            if not stop_event.is_set():
                logger.warning(f"[robot] {exc}")

        precise_sleep(max(interval - (time.perf_counter() - loop_start), 0.0))

    rob_data["timestamps"] = np.array(ts_list,  dtype=np.float64)
    rob_data["actions"]    = np.array(act_list,  dtype=np.float32)
    rob_data["states"]     = np.array(st_list,   dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Thread: segment marker  (user-driven via Enter key)
# ─────────────────────────────────────────────────────────────────────────────

def _segment_marker_thread(
    t0:       float,
    segments: list,             # out: appended here
    stop_event: threading.Event,
) -> None:
    """
    Daemon thread: blocks on input(), records timestamp on each ENTER press.
    Killed automatically when main exits (no explicit join needed).
    """
    while not stop_event.is_set():
        try:
            input()
            rel_ts = time.perf_counter() - t0
            segments.append(rel_ts)
            logger.info(f"  [Segment {len(segments)} marked @ {rel_ts:.2f}s]")
        except (EOFError, OSError):
            break


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    _backend_map = {
        "any":   Cv2Backends.ANY,
        "dshow": Cv2Backends.DSHOW,
        "msmf":  Cv2Backends.MSMF,
        "v4l2":  Cv2Backends.V4L2,
    }

    cam_cfg = OpenCVCameraConfig(
        index_or_path=args.camera_index,
        fps=args.camera_fps,
        width=args.camera_width,
        height=args.camera_height,
        fourcc=args.camera_fourcc,
        backend=_backend_map[args.camera_backend],
    )
    robot = SOFollower(SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={"cam": cam_cfg},
    ))
    teleop = SOLeader(SOLeaderTeleopConfig(
        port=args.teleop_port,
        id=args.teleop_id,
    ))

    logger.info("Connecting robot and teleoperator…")
    robot.connect()
    teleop.connect()
    camera = robot.cameras["cam"]

    session_name = args.session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir  = Path(args.output_dir) / session_name
    cam_dir      = session_dir / "camera"
    session_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(exist_ok=True)

    logger.info(
        f"\n{'='*55}\n"
        f"  LeWM E2E Continuous Collection\n"
        f"  Session  : {session_name}\n"
        f"  Output   : {session_dir}\n"
        f"  Camera   : {args.camera_fps} Hz  ({args.camera_width}x{args.camera_height})\n"
        f"  Robot    : {args.robot_fps} Hz\n"
        f"{'='*55}\n"
        f"  Press ENTER to mark a segment boundary.\n"
        f"  Press Ctrl+C to stop and save.\n"
    )

    # Shared state
    stop_event     = threading.Event()
    frame_queue    = queue.Queue(maxsize=WRITER_QUEUE_MAX)
    cam_timestamps: list = []   # populated by writer thread
    rob_data:       dict = {}   # populated by robot thread on exit
    segments:       list = []   # populated by segment thread
    stats = {"cam": 0, "rob": 0}

    t0 = time.perf_counter()

    cam_capture_t = threading.Thread(
        target=_camera_capture_thread,
        args=(camera, frame_queue, stop_event, stats),
        daemon=True, name="lewm-cam-capture",
    )
    cam_writer_t = threading.Thread(
        target=_camera_writer_thread,
        args=(frame_queue, cam_dir, stop_event, cam_timestamps),
        daemon=True, name="lewm-cam-writer",
    )
    robot_t = threading.Thread(
        target=_robot_control_thread,
        args=(robot, teleop, args.robot_fps, stop_event, rob_data, stats),
        daemon=True, name="lewm-robot",
    )
    segment_t = threading.Thread(
        target=_segment_marker_thread,
        args=(t0, segments, stop_event),
        daemon=True, name="lewm-segments",
    )

    cam_capture_t.start()
    cam_writer_t.start()
    robot_t.start()
    segment_t.start()

    logger.info("  Recording…")
    last_stats_t = t0

    try:
        while True:
            time.sleep(0.5)
            now = time.perf_counter()
            if now - last_stats_t >= STATS_INTERVAL:
                elapsed = now - t0
                logger.info(
                    f"  t={elapsed:6.0f}s | "
                    f"cam={stats['cam']:6d} ({stats['cam']/elapsed:5.1f}Hz) | "
                    f"rob={stats['rob']:7d} ({stats['rob']/elapsed:6.1f}Hz) | "
                    f"queue={frame_queue.qsize():3d} | "
                    f"segments={len(segments)}"
                )
                last_stats_t = now
    except KeyboardInterrupt:
        logger.info("\n  Ctrl+C received — stopping…")

    stop_event.set()

    # cam-capture: exits within ~100ms (async_read timeout)
    cam_capture_t.join(timeout=2.0)
    if cam_capture_t.is_alive():
        logger.warning("[cam-capture] thread did not exit cleanly.")

    # robot: exits within one control cycle (~7ms at 150 Hz)
    robot_t.join(timeout=2.0)
    if robot_t.is_alive():
        logger.warning("[robot] thread did not exit cleanly.")

    # writer: must drain the queue fully — allow generous timeout
    logger.info(f"  Flushing {frame_queue.qsize()} buffered frames to disk…")
    cam_writer_t.join(timeout=60.0)
    if cam_writer_t.is_alive():
        logger.warning("[cam-writer] still running — some frames may be lost.")

    # segment thread is daemon; don't join (blocked on input())

    duration = time.perf_counter() - t0

    # ── Save camera timestamps (relative to t0) ───────────────────────────
    cam_ts_arr = np.array(cam_timestamps, dtype=np.float64) - t0
    np.save(str(session_dir / "camera_timestamps.npy"), cam_ts_arr)

    # ── Save robot stream (relative to t0) ───────────────────────────────
    rob_ts  = rob_data.get("timestamps", np.array([], dtype=np.float64)) - t0
    rob_act = rob_data.get("actions",    np.zeros((0, 6), dtype=np.float32))
    rob_st  = rob_data.get("states",     np.zeros((0, 6), dtype=np.float32))
    np.savez_compressed(
        str(session_dir / "robot_stream.npz"),
        timestamps=rob_ts,
        actions=rob_act,
        states=rob_st,
        motor_names=np.array(MOTOR_NAMES),
    )

    # ── Save segment markers ──────────────────────────────────────────────
    np.save(str(session_dir / "segments.npy"),
            np.array(segments, dtype=np.float64))

    # ── Save metadata ─────────────────────────────────────────────────────
    n_cam = len(cam_ts_arr)
    n_rob = len(rob_ts)
    meta = {
        "session":           session_name,
        "duration_s":        round(duration, 3),
        "target_camera_fps": args.camera_fps,
        "target_robot_fps":  args.robot_fps,
        "actual_camera_fps": round(n_cam / duration, 2) if n_cam > 1 else 0.0,
        "actual_robot_fps":  round(n_rob / duration, 2) if n_rob > 1 else 0.0,
        "n_camera_frames":   n_cam,
        "n_robot_steps":     n_rob,
        "n_segments":        len(segments),
        "segment_times_s":   [round(s, 3) for s in segments],
        "camera": {
            "width":   args.camera_width,
            "height":  args.camera_height,
            "fps":     args.camera_fps,
            "fourcc":  args.camera_fourcc,
            "backend": args.camera_backend,
        },
        "motor_names": MOTOR_NAMES,
    }
    with open(session_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"\n  Saved to {session_dir}\n"
        f"  Duration : {duration:.1f}s\n"
        f"  Camera   : {n_cam} frames  ({meta['actual_camera_fps']:.1f} Hz)\n"
        f"  Robot    : {n_rob} steps   ({meta['actual_robot_fps']:.1f} Hz)\n"
        f"  Segments : {len(segments)}"
    )

    robot.disconnect()
    teleop.disconnect()
    logger.info("  Done.")


if __name__ == "__main__":
    main()
