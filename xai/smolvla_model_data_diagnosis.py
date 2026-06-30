from __future__ import annotations

import gc
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.rtc_inference.helpers import raw_observation_to_observation


JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

OLD_FINAL_REV = "b6f2aafdbdd793046747fad8207459402c33c4b0"
NEW_FINAL_REV = "f7029d03d69e149cb4b7cea8747d7158d35a8fd0"


@dataclass
class DiagnosisConfig:
    repo_root: Path = field(default_factory=lambda: Path(os.environ.get("LEROBOT_ROOT", "/home/trietlm/lerobot")))
    output_dir: Path | None = None
    model_repo_id: str = "di-techinnova/smolvla-pouring-0.3-cutted"
    dataset_repo_id: str = "di-techinnova/so-arm-101-pouring-0.3-cutted"
    dataset_root: Path | None = None
    download_dataset_if_missing: bool = False
    policy_type: str = "smolvla"
    task: str = "Pour from orange cup to blue cup."
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    replay_seeds: list[int] = field(default_factory=lambda: list(range(10)))
    state_scan_seeds: list[int] = field(default_factory=lambda: list(range(5)))
    safe_pull_threshold: float = 0.05
    image_variants: list[str] = field(default_factory=lambda: ["saved_rgb", "server_jpeg_bgr_q90"])
    rename_map: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.camera1": "observation.images.camera1",
            "observation.images.camera2": "observation.images.camera2",
        }
    )
    policy_specs: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"label": "old_50eps_final", "repo_id": "di-techinnova/smolvla-pouring-0.3-cutted", "revision": OLD_FINAL_REV},
            {"label": "new_200eps_final", "repo_id": "di-techinnova/smolvla-pouring-0.3-cutted", "revision": NEW_FINAL_REV},
        ]
    )

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.expanduser()
        if not self.repo_root.exists():
            self.repo_root = Path.cwd()
        if self.output_dir is None:
            self.output_dir = self.repo_root / "xai" / "model_data_diagnosis_smolvla_0629_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.dataset_root is None:
            self.dataset_root = Path(
                os.environ.get(
                    "LEROBOT_DATASET_ROOT",
                    str(self.repo_root / "data" / "so-arm-101-pouring-0.3-cutted"),
                )
            ).expanduser()
        for spec in self.policy_specs:
            spec.setdefault("repo_id", self.model_repo_id)

    @property
    def run_dirs(self) -> list[Path]:
        names = [
            "recorded_obs-0629-01",
            "recorded_obs-0629-02",
            "recorded_obs-0629-nonrtc",
            "recorded_obs-0629-rtc2",
        ]
        return [self.repo_root / name for name in names if (self.repo_root / name).exists()]

    @property
    def safe_pose_path(self) -> Path:
        return self.repo_root / "safe_pose.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def vec_from_state(state: dict[str, Any], keys: list[str] = JOINT_KEYS) -> np.ndarray:
    return np.asarray([float(state[k]) for k in keys], dtype=np.float64)


def state_dict_from_vec(vec: np.ndarray, keys: list[str] = JOINT_KEYS) -> dict[str, float]:
    return {key: float(vec[i]) for i, key in enumerate(keys)}


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def load_safe_pose(cfg: DiagnosisConfig) -> np.ndarray:
    safe_data = json.loads(cfg.safe_pose_path.read_text(encoding="utf-8"))
    return vec_from_state(safe_data.get("safe_pose", safe_data))


def projection_to_safe_axis(x: np.ndarray, start: np.ndarray, safe: np.ndarray) -> float:
    axis = safe - start
    denom = float(np.dot(axis, axis))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(np.asarray(x, dtype=np.float64) - start, axis) / denom)


def nearest_meta(metadata: list[dict[str, Any]], timestep: int) -> dict[str, Any]:
    return min(metadata, key=lambda row: abs(int(row["timestep"]) - int(timestep)))


def image_path_for(run_dir: Path, camera: str, timestep: int) -> Path:
    exact = run_dir / "images" / camera / f"{int(timestep):06d}.png"
    if exact.exists():
        return exact
    folder = run_dir / "images" / camera
    files = sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(folder)
    return min(files, key=lambda p: abs(int(p.stem) - int(timestep)))


def load_rgb_float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def load_image_for_replay(path: Path, image_variant: str) -> np.ndarray:
    rgb_u8 = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if image_variant == "saved_rgb":
        return rgb_u8.astype(np.float32)
    if image_variant == "server_jpeg_bgr_q90":
        import cv2

        ok, encoded = cv2.imencode(".jpg", rgb_u8, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError(f"Failed to JPEG encode {path}")
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            raise RuntimeError(f"Failed to JPEG decode {path}")
        return decoded_bgr.astype(np.float32)
    raise ValueError(f"Unknown image_variant: {image_variant}")


def build_lerobot_features(run_dir: Path) -> dict[str, Any]:
    cam1 = load_rgb_float(sorted((run_dir / "images" / "camera1").glob("*.png"))[0])
    cam2 = load_rgb_float(sorted((run_dir / "images" / "camera2").glob("*.png"))[0])
    return {
        "observation.state": {"dtype": "float32", "shape": (len(JOINT_KEYS),), "names": JOINT_KEYS},
        "observation.images.camera1": {"dtype": "image", "shape": cam1.shape, "names": ["height", "width", "channels"]},
        "observation.images.camera2": {"dtype": "image", "shape": cam2.shape, "names": ["height", "width", "channels"]},
    }


def make_raw_obs(cfg: DiagnosisConfig, run_dir: Path, image_row: dict[str, Any], state_row: dict[str, Any], image_variant: str) -> dict[str, Any]:
    raw = {key: float(state_row["state"][key]) for key in JOINT_KEYS}
    image_ts = int(image_row["timestep"])
    raw["camera1"] = load_image_for_replay(image_path_for(run_dir, "camera1", image_ts), image_variant)
    raw["camera2"] = load_image_for_replay(image_path_for(run_dir, "camera2", image_ts), image_variant)
    raw["task"] = cfg.task
    return raw


def choose_probe_rows(run_dir: Path) -> list[dict[str, Any]]:
    metadata = load_jsonl(run_dir / "metadata.jsonl")
    selected = {0, len(metadata) - 1}
    server_path = run_dir / "server_actions.jsonl"
    if server_path.exists():
        for chunk in load_jsonl(server_path):
            if chunk.get("event") != "chunk_generated":
                continue
            obs_ts = int(chunk["obs_timestep"])
            idx = min(range(len(metadata)), key=lambda i: abs(int(metadata[i]["timestep"]) - obs_ts))
            selected.add(idx)
    return [metadata[i] for i in sorted(selected)]


def snapshot_for_policy(spec: dict[str, Any]) -> Path:
    return Path(snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"]))


def load_policy_bundle(cfg: DiagnosisConfig, spec: dict[str, Any]):
    local_path = snapshot_for_policy(spec)
    policy_class = get_policy_class(cfg.policy_type)
    policy = policy_class.from_pretrained(str(local_path))
    policy.to(cfg.device).eval()
    device_override = {"device": cfg.device}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(local_path),
        preprocessor_overrides={
            "device_processor": device_override,
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
        postprocessor_overrides={"device_processor": device_override},
    )
    return local_path, policy, preprocessor, postprocessor


def release_policy_bundle(bundle) -> None:
    if bundle is None:
        return
    try:
        _, policy, _, _ = bundle
        del policy
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_chunk(cfg: DiagnosisConfig, bundle, run_dir: Path, image_row: dict[str, Any], state_row: dict[str, Any], seed: int, image_variant: str) -> np.ndarray:
    _, policy, preprocessor, postprocessor = bundle
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    raw_obs = make_raw_obs(cfg, run_dir, image_row, state_row, image_variant)
    obs = raw_observation_to_observation(
        raw_obs,
        build_lerobot_features(run_dir),
        policy.config.image_features,
        rename_map=cfg.rename_map,
    )
    obs = preprocessor(obs)
    with torch.inference_mode():
        raw_action = policy.predict_action_chunk(obs)
    if raw_action.ndim != 3:
        raw_action = raw_action.unsqueeze(0)
    _, chunk_size, _ = raw_action.shape
    processed = [postprocessor(raw_action[:, i, :]) for i in range(chunk_size)]
    return torch.stack(processed, dim=1).squeeze(0).detach().cpu().numpy()


def summarize_chunk(chunk: np.ndarray, current: np.ndarray, start: np.ndarray, safe: np.ndarray) -> dict[str, float]:
    ps = np.asarray([projection_to_safe_axis(action, start, safe) for action in chunk], dtype=np.float64)
    steps = np.linalg.norm(np.diff(chunk, axis=0), axis=1) if len(chunk) > 1 else np.asarray([])
    return {
        "p_first": float(ps[0]),
        "p_end": float(ps[-1]),
        "p_max": float(ps.max()),
        "p_mean": float(ps.mean()),
        "first_dist_current": l2(chunk[0], current),
        "end_dist_current": l2(chunk[-1], current),
        "first_dist_safe": l2(chunk[0], safe),
        "end_dist_safe": l2(chunk[-1], safe),
        "chunk_step_mean": float(steps.mean()) if len(steps) else np.nan,
    }


ABLATIONS = [
    ("current_image_current_state", "current", "current"),
    ("start_image_current_state", "start", "current"),
    ("current_image_start_state", "current", "start"),
    ("start_image_start_state", "start", "start"),
]


def run_replay(cfg: DiagnosisConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    safe = load_safe_pose(cfg)
    records: list[dict[str, Any]] = []
    for spec in cfg.policy_specs:
        print(f"\\n=== Loading policy: {spec['label']} {spec['revision']} ===")
        bundle = load_policy_bundle(cfg, spec)
        try:
            for run_dir in cfg.run_dirs:
                metadata = load_jsonl(run_dir / "metadata.jsonl")
                start_row = metadata[0]
                start_state = vec_from_state(start_row["state"])
                probes = choose_probe_rows(run_dir)
                print(spec["label"], run_dir.name, "n_probes", len(probes))
                for probe in probes:
                    current_state = vec_from_state(probe["state"])
                    current_p = projection_to_safe_axis(current_state, start_state, safe)
                    rows_by_name = {"current": probe, "start": start_row}
                    for image_variant in cfg.image_variants:
                        for ablation, image_key, state_key in ABLATIONS:
                            for seed in cfg.replay_seeds:
                                rec = {
                                    "run": run_dir.name,
                                    "policy": spec["label"],
                                    "policy_revision": spec["revision"],
                                    "obs_timestep": int(probe["timestep"]),
                                    "elapsed_s": float(probe.get("elapsed_s", np.nan)),
                                    "current_p": current_p,
                                    "image_variant": image_variant,
                                    "ablation": ablation,
                                    "seed": seed,
                                    "error": None,
                                }
                                try:
                                    chunk = predict_chunk(
                                        cfg,
                                        bundle,
                                        run_dir,
                                        rows_by_name[image_key],
                                        rows_by_name[state_key],
                                        seed,
                                        image_variant,
                                    )
                                    rec.update(summarize_chunk(chunk, current_state, start_state, safe))
                                except Exception as exc:
                                    rec["error"] = repr(exc)
                                records.append(rec)
        finally:
            release_policy_bundle(bundle)

    replay_df = pd.DataFrame(records)
    replay_df.to_csv(cfg.output_dir / "A_replay_predictions.csv", index=False)

    ok = replay_df[replay_df["error"].isna()].copy()
    summary = (
        ok.groupby(
            ["run", "policy", "policy_revision", "obs_timestep", "elapsed_s", "current_p", "image_variant", "ablation"],
            dropna=False,
        )
        .agg(
            p_first_mean=("p_first", "mean"),
            p_first_std=("p_first", "std"),
            p_end_mean=("p_end", "mean"),
            p_end_std=("p_end", "std"),
            p_max_mean=("p_max", "mean"),
            first_dist_current_mean=("first_dist_current", "mean"),
            end_dist_safe_mean=("end_dist_safe", "mean"),
            chunk_step_mean=("chunk_step_mean", "mean"),
            n=("seed", "count"),
        )
        .reset_index()
    )
    summary["end_minus_current_p"] = summary["p_end_mean"] - summary["current_p"]
    summary["first_minus_current_p"] = summary["p_first_mean"] - summary["current_p"]
    summary["safe_pull"] = summary["end_minus_current_p"] > cfg.safe_pull_threshold
    summary.to_csv(cfg.output_dir / "A_replay_summary.csv", index=False)

    run_policy_agg = (
        summary[summary["ablation"].eq("current_image_current_state") & summary["image_variant"].eq("saved_rgb")]
        .groupby(["run", "policy"])
        .agg(
            n_obs=("obs_timestep", "count"),
            safe_pull_count=("safe_pull", "sum"),
            safe_pull_rate=("safe_pull", "mean"),
            mean_end_minus_current=("end_minus_current_p", "mean"),
            max_end_minus_current=("end_minus_current_p", "max"),
            mean_chunk_step=("chunk_step_mean", "mean"),
        )
        .reset_index()
    )
    run_policy_agg.to_csv(cfg.output_dir / "A_run_policy_agg.csv", index=False)

    server_compare = build_server_compare(cfg, summary, safe)
    make_replay_plots(cfg, summary)
    return replay_df, summary, run_policy_agg, server_compare


def build_server_compare(cfg: DiagnosisConfig, summary: pd.DataFrame, safe: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for run_dir in cfg.run_dirs:
        server_path = run_dir / "server_actions.jsonl"
        if not server_path.exists():
            continue
        metadata = load_jsonl(run_dir / "metadata.jsonl")
        start_state = vec_from_state(metadata[0]["state"])
        chunks = [row for row in load_jsonl(server_path) if row.get("event") == "chunk_generated"]
        for chunk in chunks:
            obs_ts = int(chunk["obs_timestep"])
            meta = nearest_meta(metadata, obs_ts)
            current = vec_from_state(meta["state"])
            current_p = projection_to_safe_axis(current, start_state, safe)
            post = np.asarray(chunk["postprocessed_action"], dtype=np.float64)
            server_end_p = projection_to_safe_axis(post[-1], start_state, safe)
            matches = summary[
                (summary["run"] == run_dir.name)
                & (summary["obs_timestep"] == obs_ts)
                & (summary["ablation"] == "current_image_current_state")
            ]
            for _, row in matches.iterrows():
                records.append(
                    {
                        "run": run_dir.name,
                        "obs_timestep": obs_ts,
                        "server_rtc_enabled": chunk.get("rtc_enabled"),
                        "server_rtc_real_delay": chunk.get("rtc_real_delay"),
                        "current_p": current_p,
                        "server_end_p": server_end_p,
                        "server_end_minus_current_p": server_end_p - current_p,
                        "policy": row["policy"],
                        "image_variant": row["image_variant"],
                        "offline_end_mean": row["p_end_mean"],
                        "offline_end_minus_current_p": row["end_minus_current_p"],
                        "delta_end_server_minus_offline": server_end_p - row["p_end_mean"],
                    }
                )
    df = pd.DataFrame(records)
    if not df.empty:
        df["server_safe_pull"] = df["server_end_minus_current_p"] > cfg.safe_pull_threshold
        df["offline_safe_pull"] = df["offline_end_minus_current_p"] > cfg.safe_pull_threshold
    df.to_csv(cfg.output_dir / "A_server_compare.csv", index=False)
    return df


def make_replay_plots(cfg: DiagnosisConfig, summary: pd.DataFrame) -> None:
    plot_dir = cfg.output_dir / "plots_A_replay"
    plot_dir.mkdir(exist_ok=True)
    for run in sorted(summary["run"].unique()):
        for policy in sorted(summary["policy"].unique()):
            for image_variant in cfg.image_variants:
                sub = summary[(summary["run"] == run) & (summary["policy"] == policy) & (summary["image_variant"] == image_variant)]
                if sub.empty:
                    continue
                fig, ax = plt.subplots(figsize=(11, 5))
                for ablation, _, _ in ABLATIONS:
                    s = sub[sub["ablation"] == ablation].sort_values("current_p")
                    ax.plot(s["current_p"], s["p_end_mean"], marker="o", label=ablation)
                ax.plot([0, 1], [0, 1], "--", color="gray", label="p_end = current_p")
                ax.axhline(1.0, color="red", alpha=0.2)
                ax.set_title(f"{run} / {policy} / {image_variant}: ablation p_end")
                ax.set_xlabel("current state projection p")
                ax.set_ylabel("predicted chunk end projection p")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(plot_dir / f"{run}_{policy}_{image_variant}_ablation_p_end.png", dpi=140)
                plt.close(fig)


def pick_scan_image_rows(run_dir: Path, max_images: int = 4) -> list[dict[str, Any]]:
    probes = choose_probe_rows(run_dir)
    if len(probes) <= max_images:
        return probes
    idxs = sorted({0, 1, min(3, len(probes) - 1), min(5, len(probes) - 1)})
    return [probes[i] for i in idxs[:max_images]]


def run_state_scan(cfg: DiagnosisConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    safe = load_safe_pose(cfg)
    scan_runs = [p for p in cfg.run_dirs if p.name in {"recorded_obs-0629-nonrtc", "recorded_obs-0629-rtc2"}] or cfg.run_dirs[:2]
    alphas = np.linspace(-0.05, 1.10, 48)
    records: list[dict[str, Any]] = []
    for spec in cfg.policy_specs:
        print(f"\\n=== State scan policy: {spec['label']} ===")
        bundle = load_policy_bundle(cfg, spec)
        try:
            for run_dir in scan_runs:
                metadata = load_jsonl(run_dir / "metadata.jsonl")
                start_row = metadata[0]
                start_state = vec_from_state(start_row["state"])
                axis = safe - start_state
                image_rows = pick_scan_image_rows(run_dir)
                print(spec["label"], run_dir.name, "image timesteps", [r["timestep"] for r in image_rows])
                for image_row in image_rows:
                    for alpha in alphas:
                        fake_state = start_state + float(alpha) * axis
                        state_row = {"timestep": int(image_row["timestep"]), "state": state_dict_from_vec(fake_state)}
                        for seed in cfg.state_scan_seeds:
                            rec = {
                                "run": run_dir.name,
                                "policy": spec["label"],
                                "policy_revision": spec["revision"],
                                "image_timestep": int(image_row["timestep"]),
                                "image_elapsed_s": float(image_row.get("elapsed_s", np.nan)),
                                "alpha_current_p": float(alpha),
                                "seed": seed,
                                "image_variant": "saved_rgb",
                                "error": None,
                            }
                            try:
                                chunk = predict_chunk(cfg, bundle, run_dir, image_row, state_row, seed, "saved_rgb")
                                rec.update(summarize_chunk(chunk, fake_state, start_state, safe))
                            except Exception as exc:
                                rec["error"] = repr(exc)
                            records.append(rec)
        finally:
            release_policy_bundle(bundle)
    df = pd.DataFrame(records)
    df.to_csv(cfg.output_dir / "B_state_scan_predictions.csv", index=False)
    ok = df[df["error"].isna()].copy()
    ok["end_minus_current_p"] = ok["p_end"] - ok["alpha_current_p"]
    ok["safe_pull"] = ok["end_minus_current_p"] > cfg.safe_pull_threshold
    summary = (
        ok.groupby(["run", "policy", "policy_revision", "image_timestep", "image_elapsed_s", "alpha_current_p"], dropna=False)
        .agg(
            p_first_mean=("p_first", "mean"),
            p_end_mean=("p_end", "mean"),
            p_max_mean=("p_max", "mean"),
            end_minus_current_p=("end_minus_current_p", "mean"),
            safe_pull_rate=("safe_pull", "mean"),
            chunk_step_mean=("chunk_step_mean", "mean"),
            n=("seed", "count"),
        )
        .reset_index()
    )
    summary["safe_pull"] = summary["end_minus_current_p"] > cfg.safe_pull_threshold
    summary.to_csv(cfg.output_dir / "B_state_scan_summary.csv", index=False)
    danger = (
        summary[summary["safe_pull"]]
        .groupby(["run", "policy", "image_timestep"], dropna=False)
        .agg(first_danger_alpha=("alpha_current_p", "min"), max_delta=("end_minus_current_p", "max"))
        .reset_index()
    )
    danger.to_csv(cfg.output_dir / "B_state_scan_danger_points.csv", index=False)
    make_state_scan_plots(cfg, summary)
    return df, summary, danger


def make_state_scan_plots(cfg: DiagnosisConfig, summary: pd.DataFrame) -> None:
    plot_dir = cfg.output_dir / "plots_B_state_scan"
    plot_dir.mkdir(exist_ok=True)
    for run in sorted(summary["run"].unique()):
        for image_ts in sorted(summary[summary["run"] == run]["image_timestep"].unique()):
            fig, ax = plt.subplots(figsize=(11, 5))
            sub = summary[(summary["run"] == run) & (summary["image_timestep"] == image_ts)]
            for policy in sorted(sub["policy"].unique()):
                s = sub[sub["policy"] == policy].sort_values("alpha_current_p")
                ax.plot(s["alpha_current_p"], s["p_end_mean"], marker="o", markersize=3, label=f"{policy} p_end")
            ax.plot([-0.05, 1.1], [-0.05, 1.1], "--", color="gray", alpha=0.35, label="p_end=current_p")
            ax.axhline(1.0, color="red", alpha=0.2)
            ax.set_title(f"State scan {run}, fixed image timestep {image_ts}")
            ax.set_xlabel("fake current state projection alpha: start=0, safe=1")
            ax.set_ylabel("predicted chunk end projection p")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{run}_image{int(image_ts):06d}_state_scan.png", dpi=140)
            plt.close(fig)


def ensure_dataset_root(cfg: DiagnosisConfig) -> Path | None:
    if cfg.dataset_root.exists():
        return cfg.dataset_root
    if cfg.download_dataset_if_missing:
        cfg.dataset_root = Path(snapshot_download(repo_id=cfg.dataset_repo_id, repo_type="dataset"))
        return cfg.dataset_root
    print("DATASET_ROOT missing. Set LEROBOT_DATASET_ROOT or cfg.download_dataset_if_missing=True.")
    return None


def to_vec(value: Any) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(arr) < len(JOINT_KEYS):
        return None
    return arr[: len(JOINT_KEYS)]


def read_train_table(dataset_root: Path) -> pd.DataFrame:
    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet")) or sorted(dataset_root.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {dataset_root}")
    parts = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        df["_source_parquet"] = str(path.relative_to(dataset_root))
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def run_dataset_audit(cfg: DiagnosisConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    safe = load_safe_pose(cfg)
    root = ensure_dataset_root(cfg)
    if root is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train_df = read_train_table(root)
    records: list[dict[str, Any]] = []
    for ep, ep_df in train_df.groupby("episode_index", dropna=False):
        sort_col = "frame_index" if "frame_index" in ep_df.columns else None
        ep_df = ep_df.sort_values(sort_col) if sort_col else ep_df
        first_state = next((to_vec(v) for v in ep_df["observation.state"] if to_vec(v) is not None), None)
        if first_state is None:
            continue
        axis = safe - first_state
        denom = float(np.dot(axis, axis))
        if denom <= 1e-12:
            continue
        for _, row in ep_df.iterrows():
            state = to_vec(row["observation.state"])
            action = to_vec(row["action"])
            if state is None or action is None:
                continue
            p_state = float(np.dot(state - first_state, axis) / denom)
            p_action = float(np.dot(action - first_state, axis) / denom)
            delta_p = p_action - p_state
            records.append(
                {
                    "episode_index": int(ep) if not pd.isna(ep) else np.nan,
                    "frame_index": int(row["frame_index"]) if "frame_index" in row and not pd.isna(row["frame_index"]) else np.nan,
                    "timestamp": float(row["timestamp"]) if "timestamp" in row and not pd.isna(row["timestamp"]) else np.nan,
                    "source_parquet": row.get("_source_parquet"),
                    "p_state": p_state,
                    "p_action": p_action,
                    "delta_p_action": delta_p,
                    "state_dist_safe": l2(state, safe),
                    "action_dist_safe": l2(action, safe),
                    "safe_pull_action": delta_p > cfg.safe_pull_threshold,
                    "early_or_mid": p_state < 0.70,
                }
            )
    audit = pd.DataFrame(records)
    audit.to_csv(cfg.output_dir / "C_dataset_frame_audit.csv", index=False)
    if audit.empty:
        return train_df, audit, pd.DataFrame(), pd.DataFrame()
    audit["p_bucket"] = pd.cut(audit["p_state"], bins=np.linspace(-0.2, 1.2, 29), include_lowest=True)
    bucket = (
        audit.groupby("p_bucket", observed=False)
        .agg(
            n=("p_state", "count"),
            safe_pull_count=("safe_pull_action", "sum"),
            safe_pull_rate=("safe_pull_action", "mean"),
            mean_delta_p=("delta_p_action", "mean"),
            max_delta_p=("delta_p_action", "max"),
            mean_action_dist_safe=("action_dist_safe", "mean"),
        )
        .reset_index()
    )
    suspects = (
        audit[audit["early_or_mid"]]
        .groupby("episode_index", dropna=False)
        .agg(
            n=("p_state", "count"),
            safe_pull_count=("safe_pull_action", "sum"),
            safe_pull_rate=("safe_pull_action", "mean"),
            max_delta_p=("delta_p_action", "max"),
            min_p_state=("p_state", "min"),
            max_p_state=("p_state", "max"),
        )
        .reset_index()
        .sort_values(["safe_pull_count", "max_delta_p"], ascending=False)
    )
    bucket.to_csv(cfg.output_dir / "C_dataset_bucket_summary.csv", index=False)
    suspects.to_csv(cfg.output_dir / "C_dataset_episode_suspects.csv", index=False)
    make_dataset_plot(cfg, bucket)
    return train_df, audit, bucket, suspects


def make_dataset_plot(cfg: DiagnosisConfig, bucket: pd.DataFrame) -> None:
    if bucket.empty:
        return
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(bucket))
    ax1.bar(x, bucket["n"], alpha=0.25, label="n frames")
    ax1.set_ylabel("n frames")
    ax2 = ax1.twinx()
    ax2.plot(x, bucket["safe_pull_rate"], marker="o", color="red", label="safe_pull_rate")
    ax2.plot(x, bucket["mean_delta_p"], marker="o", color="blue", label="mean_delta_p")
    ax2.set_ylabel("rate / delta_p")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([str(v) for v in bucket["p_bucket"].astype(str).iloc[::2]], rotation=60, ha="right")
    ax1.set_title("Dataset bucket audit along episode_start -> safe_pose axis")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(cfg.output_dir / "C_dataset_bucket_audit.png", dpi=150)
    plt.close(fig)


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten_json(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        if len(obj) <= 20 and all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = obj
        else:
            out[f"{prefix}.__len__"] = len(obj)
            for i, value in enumerate(obj[:20]):
                out.update(flatten_json(value, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def run_config_audit(cfg: DiagnosisConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshot_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for spec in cfg.policy_specs:
        snap = snapshot_for_policy(spec)
        files = sorted([p for p in snap.rglob("*") if p.is_file()])
        snapshot_rows.append({"policy": spec["label"], "revision": spec["revision"], "snapshot_path": str(snap), "n_files": len(files)})
        for path in files:
            rel = str(path.relative_to(snap)).replace("\\", "/")
            if any(token in rel.lower() for token in ["config", "processor", "readme", "train"]) or rel.endswith(".json"):
                file_rows.append({"policy": spec["label"], "relpath": rel, "size": path.stat().st_size, "sha256": file_sha256(path)})
            if path.suffix == ".json" and any(token in rel.lower() for token in ["config", "processor", "train"]):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for key, value in flatten_json(data).items():
                    config_rows.append({"policy": spec["label"], "file": rel, "key": key, "value": json.dumps(value, ensure_ascii=False, sort_keys=True)})
    snapshot_df = pd.DataFrame(snapshot_rows)
    file_manifest = pd.DataFrame(file_rows)
    config_flat = pd.DataFrame(config_rows)
    snapshot_df.to_csv(cfg.output_dir / "D_snapshot_summary.csv", index=False)
    file_manifest.to_csv(cfg.output_dir / "D_file_manifest.csv", index=False)
    config_flat.to_csv(cfg.output_dir / "D_config_flat.csv", index=False)
    policy_cols = [spec["label"] for spec in cfg.policy_specs]
    config_diff = pd.DataFrame()
    if not config_flat.empty:
        pivot = config_flat.pivot_table(index=["file", "key"], columns="policy", values="value", aggfunc="first").reset_index()
        for col in policy_cols:
            if col not in pivot.columns:
                pivot[col] = np.nan
        pivot["changed"] = pivot[policy_cols[0]].astype(str) != pivot[policy_cols[1]].astype(str)
        config_diff = pivot[pivot["changed"]].sort_values(["file", "key"])
        config_diff.to_csv(cfg.output_dir / "D_config_diff.csv", index=False)
    file_diff = pd.DataFrame()
    if not file_manifest.empty:
        fp = file_manifest.pivot_table(index="relpath", columns="policy", values="sha256", aggfunc="first").reset_index()
        for col in policy_cols:
            if col not in fp.columns:
                fp[col] = np.nan
        fp["changed_or_missing"] = fp[policy_cols[0]].astype(str) != fp[policy_cols[1]].astype(str)
        file_diff = fp[fp["changed_or_missing"]].sort_values("relpath")
        file_diff.to_csv(cfg.output_dir / "D_file_diff.csv", index=False)
    return snapshot_df, file_manifest, config_flat, config_diff, file_diff


def write_eval_report(
    cfg: DiagnosisConfig,
    run_policy_agg: pd.DataFrame | None = None,
    danger: pd.DataFrame | None = None,
    suspects: pd.DataFrame | None = None,
    config_diff: pd.DataFrame | None = None,
    file_diff: pd.DataFrame | None = None,
) -> str:
    def table(df: pd.DataFrame, max_rows: int | None = None) -> str:
        if max_rows is not None:
            df = df.head(max_rows)
        if df.empty:
            return "_empty_"
        # Avoid pandas.to_markdown because the server environment may not have tabulate installed.
        return "```text\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# SmolVLA Offline Model/Data Diagnosis Report",
        "",
        f"- Model repo: `{cfg.model_repo_id}`",
        f"- Old final revision: `{OLD_FINAL_REV}`",
        f"- New final revision: `{NEW_FINAL_REV}`",
        f"- Recorded runs: `{[p.name for p in cfg.run_dirs]}`",
        f"- Safe-pull threshold: `p_end - current_p > {cfg.safe_pull_threshold}`",
        "",
        "## A. Replay Summary (`current_image_current_state`, `saved_rgb`)",
    ]
    if run_policy_agg is not None and not run_policy_agg.empty:
        lines.append(table(run_policy_agg))
    else:
        lines.append("_No replay summary available._")
    lines += ["", "## B. State Scan Danger Points"]
    if danger is not None and not danger.empty:
        lines.append(table(danger))
    else:
        lines.append("_No state-scan danger points found or section not run._")
    lines += ["", "## C. Top Dataset Episode Suspects"]
    if suspects is not None and not suspects.empty:
        lines.append(table(suspects, max_rows=30))
    else:
        lines.append("_No dataset audit available._")
    lines += ["", "## D. Config/File Diff"]
    if config_diff is not None:
        lines.append(f"- Changed flattened config keys: `{len(config_diff)}`")
    if file_diff is not None:
        lines.append(f"- Changed/missing tracked config/processor files: `{len(file_diff)}`")
    if run_policy_agg is not None and not run_policy_agg.empty:
        score = (
            run_policy_agg.groupby("policy")
            .agg(
                total_obs=("n_obs", "sum"),
                total_safe_pull=("safe_pull_count", "sum"),
                mean_safe_pull_rate=("safe_pull_rate", "mean"),
                mean_end_minus_current=("mean_end_minus_current", "mean"),
                max_end_minus_current=("max_end_minus_current", "max"),
            )
            .reset_index()
        )
        score.to_csv(cfg.output_dir / "E_policy_score.csv", index=False)
        lines += ["", "## E. Policy Score", table(score)]
    text = "\n".join(lines)
    (cfg.output_dir / "E_offline_eval_report.md").write_text(text, encoding="utf-8")
    return text
