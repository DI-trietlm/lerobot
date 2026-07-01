#!/usr/bin/env python
"""Analyze RTC deploy action traces from server/client JSONL logs.

This script is intentionally offline-only: point it at a `recorded_obs*`
directory containing any of:

- `server_actions.jsonl` from `PolicyServer`
- `client_actions.jsonl` from `RobotClient`
- `metadata.jsonl` from observation recording

It answers the deploy debugging questions we care about:

- Did raw policy/postprocess/RTC produce different actions?
- Did the client receive and execute the same action stream?
- Did the executed/current state move toward a supplied safe pose?

Example:
    uv run python xai/scripts/analyze_action_trace.py --trace-dir recorded_obs-0611 \
        --safe-pose "0,0,0,0,0,0" --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    return arr


def as_2d(value: Any) -> np.ndarray | None:
    arr = as_array(value)
    if arr is None:
        return None
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim > 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def finite(values: Iterable[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def stats(values: Iterable[float | None]) -> dict[str, float | None]:
    vals = sorted(finite(values))
    if not vals:
        return {"mean": None, "median": None, "p95": None, "max": None}
    p95_idx = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
    return {
        "mean": mean(vals),
        "median": median(vals),
        "p95": vals[p95_idx],
        "max": vals[-1],
    }


def l2(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.linalg.norm(a - b))


def parse_keys(text: str | None) -> list[str] | None:
    if not text:
        return None
    keys = [item.strip() for item in text.split(",") if item.strip()]
    return keys or None


def vector_from_dict(value: dict[str, Any] | None, keys: list[str] | None) -> np.ndarray | None:
    if not isinstance(value, dict) or not value:
        return None
    ordered_keys = keys or sorted(value)
    try:
        return np.asarray([float(value[k]) for k in ordered_keys], dtype=np.float64)
    except KeyError:
        return None


def parse_pose(text: str | None, keys: list[str] | None) -> np.ndarray | None:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if isinstance(value, dict):
        return vector_from_dict(value, keys)
    arr = np.asarray(value, dtype=np.float64)
    return arr.reshape(-1)


def infer_action_keys(
    client_executed: list[dict[str, Any]], metadata: list[dict[str, Any]]
) -> list[str] | None:
    for row in client_executed:
        action = row.get("target_action")
        if isinstance(action, dict) and action:
            return list(action.keys())
    for row in metadata:
        state = row.get("state")
        if isinstance(state, dict) and state:
            return list(state.keys())
    return None


def nearest_state_lookup(metadata: list[dict[str, Any]], keys: list[str] | None):
    state_rows: list[tuple[int, np.ndarray]] = []
    for row in metadata:
        timestep = row.get("timestep")
        state = vector_from_dict(row.get("state"), keys)
        if timestep is None or state is None:
            continue
        state_rows.append((int(timestep), state))
    state_rows.sort(key=lambda item: item[0])
    timesteps = [item[0] for item in state_rows]

    def lookup(timestep: int | None) -> tuple[int | None, np.ndarray | None]:
        if timestep is None or not state_rows:
            return None, None
        pos = bisect_left(timesteps, int(timestep))
        candidates = []
        if pos < len(state_rows):
            candidates.append(state_rows[pos])
        if pos > 0:
            candidates.append(state_rows[pos - 1])
        if not candidates:
            return None, None
        best_ts, best_state = min(candidates, key=lambda item: abs(item[0] - int(timestep)))
        return best_ts, best_state

    return lookup


def summarize_server_chunks(server_chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(server_chunks):
        post = as_2d(row.get("postprocessed_action"))
        sent = as_2d(row.get("sent_action"))
        raw = as_2d(row.get("raw_action_norm"))
        diff = None
        diff_max = None
        if post is not None and sent is not None and post.shape == sent.shape:
            per_step = np.linalg.norm(sent - post, axis=1)
            diff = float(per_step.mean())
            diff_max = float(per_step.max())
        rows.append(
            {
                "chunk_idx": idx,
                "obs_timestep": row.get("obs_timestep"),
                "sent_first_timestep": (row.get("sent_timesteps") or [None])[0],
                "sent_last_timestep": (row.get("sent_timesteps") or [None])[-1],
                "chunk_size": None if sent is None else sent.shape[0],
                "action_dim": None if sent is None else sent.shape[-1],
                "raw_shape": None if raw is None else "x".join(map(str, raw.shape)),
                "post_sent_l2_mean": diff,
                "post_sent_l2_max": diff_max,
                "rtc_enabled": row.get("rtc_enabled"),
                "rtc_real_delay": row.get("rtc_real_delay"),
                "inference_ms": row.get("inference_ms"),
                "postprocess_ms": row.get("postprocess_ms"),
                "total_ms": row.get("total_ms"),
            }
        )

    summary = {
        "server_chunks": len(rows),
        "server_rtc_modified_chunks": sum(
            1 for row in rows if (row["post_sent_l2_max"] is not None and row["post_sent_l2_max"] > 1e-6)
        ),
        "server_inference_ms": stats(row["inference_ms"] for row in rows),
        "server_total_ms": stats(row["total_ms"] for row in rows),
        "server_post_sent_l2_max": stats(row["post_sent_l2_max"] for row in rows),
    }
    return rows, summary


def summarize_client_chunks(client_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    incoming_count = sum(len(row.get("actions") or []) for row in client_chunks)
    return {
        "client_chunks_received": len(client_chunks),
        "client_actions_received": incoming_count,
        "server_to_client_latency_ms": stats(row.get("server_to_client_latency_ms") for row in client_chunks),
        "queue_update_ms": stats(row.get("queue_update_ms") for row in client_chunks),
        "client_new_queue_size": stats(row.get("new_queue_size") for row in client_chunks),
    }


def build_execution_timeline(
    executed: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    keys: list[str] | None,
    start_pose: np.ndarray | None,
    safe_pose: np.ndarray | None,
) -> list[dict[str, Any]]:
    lookup_state = nearest_state_lookup(metadata, keys)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(executed):
        timestep = row.get("action_timestep")
        target = vector_from_dict(row.get("target_action"), keys)
        performed = vector_from_dict(row.get("performed_action"), keys)
        state_ts, current_state = lookup_state(int(timestep) if timestep is not None else None)

        out = {
            "exec_idx": idx,
            "wall_time_s": row.get("wall_time_s"),
            "action_timestep": timestep,
            "state_timestep": state_ts,
            "popped_new_timed_action": row.get("popped_new_timed_action"),
            "interpolated_action": row.get("interpolated_action"),
            "queue_size_before_pop": row.get("queue_size_before_pop"),
            "queue_size_after_pop": row.get("queue_size_after_pop"),
            "target_vs_performed_l2": l2(target, performed),
            "target_dist_start": l2(target, start_pose),
            "target_dist_safe": l2(target, safe_pose),
            "performed_dist_start": l2(performed, start_pose),
            "performed_dist_safe": l2(performed, safe_pose),
            "current_dist_start": l2(current_state, start_pose),
            "current_dist_safe": l2(current_state, safe_pose),
        }
        if keys:
            for i, key in enumerate(keys):
                if target is not None and i < target.size:
                    out[f"target.{key}"] = float(target[i])
                if performed is not None and i < performed.size:
                    out[f"performed.{key}"] = float(performed[i])
                if current_state is not None and i < current_state.size:
                    out[f"current.{key}"] = float(current_state[i])
        rows.append(out)
    return rows


def find_safe_drift_windows(
    timeline: list[dict[str, Any]],
    window: int,
    min_safe_improvement: float,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    if window < 2:
        return windows
    for start in range(0, max(0, len(timeline) - window + 1)):
        chunk = timeline[start : start + window]
        safe = finite(row.get("current_dist_safe") for row in chunk)
        start_dist = finite(row.get("current_dist_start") for row in chunk)
        if len(safe) != window or len(start_dist) != window:
            continue
        safe_improvement = safe[0] - safe[-1]
        start_change = start_dist[-1] - start_dist[0]
        safe_monotone_steps = sum(1 for a, b in zip(safe, safe[1:], strict=False) if b <= a)
        if safe_improvement >= min_safe_improvement and start_change >= -0.5:
            windows.append(
                {
                    "start_exec_idx": chunk[0]["exec_idx"],
                    "end_exec_idx": chunk[-1]["exec_idx"],
                    "start_timestep": chunk[0]["action_timestep"],
                    "end_timestep": chunk[-1]["action_timestep"],
                    "safe_improvement": safe_improvement,
                    "start_distance_change": start_change,
                    "safe_monotone_steps": safe_monotone_steps,
                }
            )
    return windows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_stat(value: dict[str, float | None]) -> str:
    if value["mean"] is None:
        return "n/a"
    return (
        f"mean={value['mean']:.3f}, median={value['median']:.3f}, "
        f"p95={value['p95']:.3f}, max={value['max']:.3f}"
    )


def write_report(
    path: Path,
    trace_dir: Path,
    server_summary: dict[str, Any],
    client_summary: dict[str, Any],
    timeline: list[dict[str, Any]],
    drift_windows: list[dict[str, Any]],
    keys: list[str] | None,
    safe_pose: np.ndarray | None,
    start_pose: np.ndarray | None,
) -> None:
    exec_count = len(timeline)
    interpolated = sum(1 for row in timeline if row.get("interpolated_action"))
    popped = sum(1 for row in timeline if row.get("popped_new_timed_action"))
    target_performed = stats(row.get("target_vs_performed_l2") for row in timeline)
    current_safe = stats(row.get("current_dist_safe") for row in timeline)
    current_start = stats(row.get("current_dist_start") for row in timeline)
    target_safe = stats(row.get("target_dist_safe") for row in timeline)

    lines = [
        "# Action Trace Analysis",
        "",
        f"- Trace dir: `{trace_dir}`",
        f"- Action keys: `{', '.join(keys) if keys else 'unknown/vector-indexed'}`",
        f"- Start pose provided: `{start_pose is not None}`",
        f"- Safe pose provided: `{safe_pose is not None}`",
        "",
        "## Server Pipeline",
        "",
        f"- Chunks generated: `{server_summary['server_chunks']}`",
        f"- RTC/postprocess changed chunks: `{server_summary['server_rtc_modified_chunks']}`",
        f"- Inference time: {fmt_stat(server_summary['server_inference_ms'])}",
        f"- Total server time: {fmt_stat(server_summary['server_total_ms'])}",
        f"- L2(sent - postprocessed): {fmt_stat(server_summary['server_post_sent_l2_max'])}",
        "",
        "## Client Receive/Execute",
        "",
        f"- Chunks received: `{client_summary['client_chunks_received']}`",
        f"- Actions received: `{client_summary['client_actions_received']}`",
        f"- Executed loop records: `{exec_count}`",
        f"- Executed popped-new actions: `{popped}`",
        f"- Interpolated/reused actions: `{interpolated}`",
        f"- Server->client latency: {fmt_stat(client_summary['server_to_client_latency_ms'])}",
        f"- Queue update time: {fmt_stat(client_summary['queue_update_ms'])}",
        f"- Queue size after receive: {fmt_stat(client_summary['client_new_queue_size'])}",
        f"- L2(target - performed): {fmt_stat(target_performed)}",
        "",
        "## Safe/Start Pose Signals",
        "",
        f"- Current distance to safe pose: {fmt_stat(current_safe)}",
        f"- Current distance to start pose: {fmt_stat(current_start)}",
        f"- Target action distance to safe pose: {fmt_stat(target_safe)}",
        f"- Suspicious safe-drift windows: `{len(drift_windows)}`",
    ]
    if drift_windows:
        lines.extend(["", "Top drift windows:"])
        for row in drift_windows[:10]:
            lines.append(
                "- "
                f"exec {row['start_exec_idx']}->{row['end_exec_idx']} "
                f"ts {row['start_timestep']}->{row['end_timestep']}: "
                f"safe_dist improves {row['safe_improvement']:.3f}, "
                f"start_dist changes {row['start_distance_change']:.3f}"
            )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- If `L2(sent - postprocessed)` is large, RTC/server-side queueing changed the policy output.",
            "- If `L2(target - performed)` is large, robot execution/clamping/calibration changed the command.",
            "- If `current_dist_safe` decreases while `current_dist_start` does not decrease, that is real safe-pose drift.",
            "- If only `target_dist_safe` decreases, the policy/RTC is asking for safe-pose motion before the robot fully follows it.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot(path: Path, timeline: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x = [row["exec_idx"] for row in timeline]
    series = {
        "current_dist_safe": [row.get("current_dist_safe") for row in timeline],
        "current_dist_start": [row.get("current_dist_start") for row in timeline],
        "target_dist_safe": [row.get("target_dist_safe") for row in timeline],
        "target_vs_performed_l2": [row.get("target_vs_performed_l2") for row in timeline],
    }
    if not any(finite(values) for values in series.values()):
        return

    plt.figure(figsize=(12, 6))
    for label, values in series.items():
        y = [np.nan if value is None else value for value in values]
        if np.isfinite(y).any():
            plt.plot(x, y, label=label)
    plt.xlabel("Executed action index")
    plt.ylabel("L2 distance")
    plt.title("Runtime action trace distances")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default="recorded_obs", help="Directory containing trace JSONL files")
    parser.add_argument("--server-actions", default=None, help="Override path to server_actions.jsonl")
    parser.add_argument("--client-actions", default=None, help="Override path to client_actions.jsonl")
    parser.add_argument("--metadata", default=None, help="Override path to metadata.jsonl")
    parser.add_argument("--out-dir", default=None, help="Directory for summary.md/csv/plot outputs")
    parser.add_argument("--action-keys", default=None, help="Comma-separated joint/action key order")
    parser.add_argument("--start-pose", default=None, help="JSON/list/dict/comma pose; defaults to first metadata state")
    parser.add_argument("--safe-pose", default=None, help="JSON/list/dict/comma pose used for safe-drift checks")
    parser.add_argument("--drift-window-steps", type=int, default=8)
    parser.add_argument("--min-safe-improvement", type=float, default=3.0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    server_path = Path(args.server_actions) if args.server_actions else trace_dir / "server_actions.jsonl"
    client_path = Path(args.client_actions) if args.client_actions else trace_dir / "client_actions.jsonl"
    metadata_path = Path(args.metadata) if args.metadata else trace_dir / "metadata.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else trace_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    server_records = load_jsonl(server_path)
    client_records = load_jsonl(client_path)
    metadata = load_jsonl(metadata_path)
    server_chunks = [row for row in server_records if row.get("event") == "chunk_generated"]
    client_chunks = [row for row in client_records if row.get("event") == "chunk_received"]
    executed = [row for row in client_records if row.get("event") == "action_executed"]

    keys = parse_keys(args.action_keys) or infer_action_keys(executed, metadata)
    start_pose = parse_pose(args.start_pose, keys)
    if start_pose is None and metadata:
        first_state = next((row.get("state") for row in metadata if isinstance(row.get("state"), dict)), None)
        start_pose = vector_from_dict(first_state, keys)
    safe_pose = parse_pose(args.safe_pose, keys)

    if safe_pose is not None and start_pose is not None and safe_pose.shape != start_pose.shape:
        raise ValueError(f"safe_pose shape {safe_pose.shape} != start_pose shape {start_pose.shape}")

    server_rows, server_summary = summarize_server_chunks(server_chunks)
    client_summary = summarize_client_chunks(client_chunks)
    timeline = build_execution_timeline(executed, metadata, keys, start_pose, safe_pose)
    drift_windows = find_safe_drift_windows(
        timeline,
        window=args.drift_window_steps,
        min_safe_improvement=args.min_safe_improvement,
    )

    write_csv(out_dir / "server_chunks.csv", server_rows)
    write_csv(out_dir / "executed_actions.csv", timeline)
    write_csv(out_dir / "safe_drift_windows.csv", drift_windows)
    write_report(
        out_dir / "summary.md",
        trace_dir,
        server_summary,
        client_summary,
        timeline,
        drift_windows,
        keys,
        safe_pose,
        start_pose,
    )
    if args.plot:
        maybe_plot(out_dir / "distances.png", timeline)

    print(f"Wrote {out_dir / 'summary.md'}")
    print(f"server_chunks={len(server_chunks)} client_chunks={len(client_chunks)} executed={len(executed)}")
    print(f"suspicious_safe_drift_windows={len(drift_windows)}")


if __name__ == "__main__":
    main()
