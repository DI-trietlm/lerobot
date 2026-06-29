#!/usr/bin/env python
"""Analyze recorded RTC runtime runs without loading the policy/model."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def arr(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def vec_from_dict(data: dict[str, Any] | None, keys: list[str]) -> np.ndarray | None:
    if not isinstance(data, dict):
        return None
    try:
        return np.asarray([float(data[k]) for k in keys], dtype=np.float64)
    except KeyError:
        return None


def l2(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return float("nan")
    return float(np.linalg.norm(a - b))


def stat(values: list[float]) -> dict[str, float | int | None]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    values = sorted(values)
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": values[0],
        "max": values[-1],
    }


def fmt_stat(s: dict[str, float | int | None]) -> str:
    if not s["n"]:
        return "n/a"
    return (
        f"n={s['n']}, mean={s['mean']:.3f}, med={s['median']:.3f}, "
        f"min={s['min']:.3f}, max={s['max']:.3f}"
    )


def state_lookup(metadata: list[dict[str, Any]], keys: list[str]):
    pairs = []
    for row in metadata:
        state = vec_from_dict(row.get("state"), keys)
        if state is not None:
            pairs.append((int(row["timestep"]), state))
    pairs.sort(key=lambda x: x[0])
    ts = [x[0] for x in pairs]

    def lookup(timestep: int | None) -> tuple[int | None, np.ndarray | None]:
        if timestep is None or not pairs:
            return None, None
        pos = bisect_left(ts, int(timestep))
        candidates = []
        if pos < len(pairs):
            candidates.append(pairs[pos])
        if pos > 0:
            candidates.append(pairs[pos - 1])
        if not candidates:
            return None, None
        return min(candidates, key=lambda x: abs(x[0] - int(timestep)))

    return lookup


def image_stats(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    images_root = run_dir / "images"
    if not images_root.exists():
        return out
    try:
        from PIL import Image
    except ImportError:
        return {"pil_available": False}

    for cam_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        files = sorted(cam_dir.glob("*.png"))
        if not files:
            continue
        means = []
        diffs = []
        prev = None
        sizes = Counter()
        for path in files:
            im = Image.open(path).convert("RGB")
            a = np.asarray(im, dtype=np.float32)
            sizes[f"{a.shape[1]}x{a.shape[0]}"] += 1
            means.append(float(a.mean()))
            small = a[:: max(1, a.shape[0] // 90), :: max(1, a.shape[1] // 160)]
            if prev is not None and prev.shape == small.shape:
                diffs.append(float(np.mean(np.abs(small - prev))))
            prev = small
        out[cam_dir.name] = {
            "count": len(files),
            "first": files[0].name,
            "last": files[-1].name,
            "sizes": dict(sizes),
            "brightness": stat(means),
            "frame_absdiff": stat(diffs),
        }
    return out


def chunk_metrics(
    server: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    keys: list[str],
    start_pose: np.ndarray,
    safe_pose: np.ndarray,
) -> list[dict[str, Any]]:
    lookup = state_lookup(metadata, keys)
    rows = []
    for i, row in enumerate(server):
        raw = arr(row["raw_action_norm"])
        post = arr(row["postprocessed_action"])
        sent = arr(row["sent_action"])
        obs_ts = int(row["obs_timestep"])
        state_ts, current = lookup(obs_ts)
        post_first = post[0] if len(post) else None
        post_end = post[-1] if len(post) else None
        sent_first = sent[0] if len(sent) else None
        sent_end = sent[-1] if len(sent) else None
        sent_timesteps = row.get("sent_timesteps") or []
        rows.append(
            {
                "chunk_idx": i,
                "obs_timestep": obs_ts,
                "state_timestep": state_ts,
                "raw_len": len(raw),
                "post_len": len(post),
                "sent_len": len(sent),
                "sent_ts_first": sent_timesteps[0] if sent_timesteps else None,
                "sent_ts_last": sent_timesteps[-1] if sent_timesteps else None,
                "rtc_delay": row.get("rtc_real_delay"),
                "rtc_index_before": row.get("rtc_action_index_before_inference"),
                "inference_ms": row.get("inference_ms"),
                "total_ms": row.get("total_ms"),
                "post_first_dist_current": l2(post_first, current),
                "post_end_dist_current": l2(post_end, current),
                "sent_first_dist_current": l2(sent_first, current),
                "sent_end_dist_current": l2(sent_end, current),
                "post_first_dist_safe": l2(post_first, safe_pose),
                "sent_first_dist_safe": l2(sent_first, safe_pose),
                "post_first_dist_start": l2(post_first, start_pose),
                "sent_first_dist_start": l2(sent_first, start_pose),
                "post_chunk_step_mean": float(np.mean(np.linalg.norm(np.diff(post, axis=0), axis=1)))
                if len(post) > 1
                else float("nan"),
                "sent_chunk_step_mean": float(np.mean(np.linalg.norm(np.diff(sent, axis=0), axis=1)))
                if len(sent) > 1
                else float("nan"),
                "sent_minus_post_offset_l2": l2(sent_first, post[0]) if len(sent) and len(post) else float("nan"),
            }
        )
    return rows


def client_execution_metrics(
    client: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    keys: list[str],
    start_pose: np.ndarray,
    safe_pose: np.ndarray,
) -> list[dict[str, Any]]:
    lookup = state_lookup(metadata, keys)
    rows = []
    for i, row in enumerate(r for r in client if r.get("event") == "action_executed"):
        target = vec_from_dict(row.get("target_action"), keys)
        performed = vec_from_dict(row.get("performed_action"), keys)
        action_ts = row.get("action_timestep")
        state_ts, current = lookup(action_ts)
        rows.append(
            {
                "exec_idx": i,
                "action_timestep": action_ts,
                "state_timestep": state_ts,
                "popped_new": row.get("popped_new_timed_action"),
                "interpolated": row.get("interpolated_action"),
                "queue_before": row.get("queue_size_before_pop"),
                "queue_after": row.get("queue_size_after_pop"),
                "target_vs_performed": l2(target, performed),
                "target_dist_current": l2(target, current),
                "performed_dist_current": l2(performed, current),
                "current_dist_start": l2(current, start_pose),
                "current_dist_safe": l2(current, safe_pose),
                "target_dist_start": l2(target, start_pose),
                "target_dist_safe": l2(target, safe_pose),
                "performed_dist_start": l2(performed, start_pose),
                "performed_dist_safe": l2(performed, safe_pose),
            }
        )
    return rows


def monotone_summary(values: list[float]) -> dict[str, Any]:
    values = [v for v in values if math.isfinite(v)]
    if len(values) < 2:
        return {"n": len(values)}
    diffs = np.diff(values)
    return {
        "n": len(values),
        "first": float(values[0]),
        "last": float(values[-1]),
        "delta": float(values[-1] - values[0]),
        "decreasing_steps": int(np.sum(diffs < 0)),
        "increasing_steps": int(np.sum(diffs > 0)),
        "max_drop_step": float(np.min(diffs)),
        "max_rise_step": float(np.max(diffs)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def analyze_run(run_dir: Path, safe_pose: np.ndarray, keys: list[str]) -> dict[str, Any]:
    server = load_jsonl(run_dir / "server_actions.jsonl")
    client = load_jsonl(run_dir / "client_actions.jsonl")
    metadata = load_jsonl(run_dir / "metadata.jsonl")
    server_chunks = [r for r in server if r.get("event") == "chunk_generated"]
    client_chunks = [r for r in client if r.get("event") == "chunk_received"]
    start_pose = vec_from_dict(metadata[0].get("state"), keys)
    if start_pose is None:
        raise ValueError(f"{run_dir}: could not read start pose")

    chunks = chunk_metrics(server_chunks, metadata, keys, start_pose, safe_pose)
    execs = client_execution_metrics(client, metadata, keys, start_pose, safe_pose)
    out_dir = run_dir / "analysis_runtime"
    out_dir.mkdir(exist_ok=True)
    write_csv(out_dir / "server_chunk_metrics.csv", chunks)
    write_csv(out_dir / "client_execution_metrics.csv", execs)

    current_safe = [r["current_dist_safe"] for r in execs]
    current_start = [r["current_dist_start"] for r in execs]
    summary = {
        "run": run_dir.name,
        "counts": {
            "server_chunks": len(server_chunks),
            "client_chunks": len(client_chunks),
            "client_events": len(client),
            "metadata_frames": len(metadata),
            "executions": len(execs),
        },
        "metadata_timestep_range": [metadata[0].get("timestep"), metadata[-1].get("timestep")]
        if metadata
        else None,
        "start_pose": dict(zip(keys, start_pose.tolist(), strict=False)),
        "safe_pose": dict(zip(keys, safe_pose.tolist(), strict=False)),
        "start_to_safe_l2": l2(start_pose, safe_pose),
        "server": {
            "raw_len": Counter(r["raw_len"] for r in chunks),
            "post_len": Counter(r["post_len"] for r in chunks),
            "sent_len": Counter(r["sent_len"] for r in chunks),
            "rtc_delay": Counter(r["rtc_delay"] for r in chunks),
            "inference_ms": stat([r["inference_ms"] for r in chunks]),
            "total_ms": stat([r["total_ms"] for r in chunks]),
            "post_first_dist_current": stat([r["post_first_dist_current"] for r in chunks]),
            "sent_first_dist_current": stat([r["sent_first_dist_current"] for r in chunks]),
            "post_chunk_step_mean": stat([r["post_chunk_step_mean"] for r in chunks]),
            "sent_chunk_step_mean": stat([r["sent_chunk_step_mean"] for r in chunks]),
            "post_first_dist_safe": stat([r["post_first_dist_safe"] for r in chunks]),
            "sent_first_dist_safe": stat([r["sent_first_dist_safe"] for r in chunks]),
        },
        "client": {
            "target_vs_performed": stat([r["target_vs_performed"] for r in execs]),
            "target_dist_current": stat([r["target_dist_current"] for r in execs]),
            "performed_dist_current": stat([r["performed_dist_current"] for r in execs]),
            "current_dist_safe": stat(current_safe),
            "current_dist_start": stat(current_start),
            "target_dist_safe": stat([r["target_dist_safe"] for r in execs]),
            "queue_before": stat([r["queue_before"] for r in execs]),
            "queue_after": stat([r["queue_after"] for r in execs]),
            "interpolated_count": sum(1 for r in execs if r["interpolated"]),
            "popped_new_count": sum(1 for r in execs if r["popped_new"]),
            "current_safe_trend": monotone_summary(current_safe),
            "current_start_trend": monotone_summary(current_start),
        },
        "images": image_stats(run_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def save_plots(run_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return
    out_dir = run_dir / "analysis_runtime"
    exec_csv = out_dir / "client_execution_metrics.csv"
    chunk_csv = out_dir / "server_chunk_metrics.csv"
    if exec_csv.exists():
        df = pd.read_csv(exec_csv)
        if not df.empty:
            plt.figure(figsize=(12, 6))
            for col in [
                "current_dist_start",
                "current_dist_safe",
                "target_dist_current",
                "target_dist_safe",
                "target_vs_performed",
            ]:
                if col in df:
                    plt.plot(df["exec_idx"], df[col], label=col)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlabel("execution index")
            plt.ylabel("L2 degrees")
            plt.title(run_dir.name + " client execution/state metrics")
            plt.tight_layout()
            plt.savefig(out_dir / "client_execution_metrics.png", dpi=160)
            plt.close()
    if chunk_csv.exists():
        df = pd.read_csv(chunk_csv)
        if not df.empty:
            plt.figure(figsize=(12, 6))
            for col in [
                "post_first_dist_current",
                "sent_first_dist_current",
                "post_chunk_step_mean",
                "sent_chunk_step_mean",
                "post_first_dist_safe",
                "sent_first_dist_safe",
            ]:
                if col in df:
                    plt.plot(df["chunk_idx"], df[col], label=col)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlabel("server chunk index")
            plt.ylabel("L2 degrees")
            plt.title(run_dir.name + " server chunk metrics")
            plt.tight_layout()
            plt.savefig(out_dir / "server_chunk_metrics.png", dpi=160)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--safe-pose", default="safe_pose.json")
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    args = parser.parse_args()

    keys = [x.strip() for x in args.keys.split(",") if x.strip()]
    safe_data = json.loads(Path(args.safe_pose).read_text(encoding="utf-8"))
    safe_dict = safe_data.get("safe_pose", safe_data)
    safe_pose = vec_from_dict(safe_dict, keys)
    if safe_pose is None:
        raise ValueError("Could not parse safe pose")

    summaries = []
    for run in args.runs:
        run_dir = Path(run)
        summary = analyze_run(run_dir, safe_pose, keys)
        save_plots(run_dir)
        summaries.append(summary)
        print(f"\n=== {run_dir.name} ===")
        print("counts:", summary["counts"])
        print("start_to_safe_l2:", f"{summary['start_to_safe_l2']:.3f}")
        print("server inference:", fmt_stat(summary["server"]["inference_ms"]))
        print("post first dist current:", fmt_stat(summary["server"]["post_first_dist_current"]))
        print("sent first dist current:", fmt_stat(summary["server"]["sent_first_dist_current"]))
        print("client target vs performed:", fmt_stat(summary["client"]["target_vs_performed"]))
        print("current dist safe trend:", summary["client"]["current_safe_trend"])
        print("current dist start trend:", summary["client"]["current_start_trend"])

    out = Path("xai") / "runtime_runs_0629_summary.json"
    out.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
