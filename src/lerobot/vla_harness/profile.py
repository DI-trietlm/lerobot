from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .envelopes import build_speed_envelope_profile
from .invariants import InvariantMiner
from .mode import build_mode_profile
from .rescue import RescueIndex, build_rescue_index
from .schemas import HarnessDiagnostics, HarnessProfile, RescueIndexMetadata

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    q25 = np.quantile(values, 0.25, axis=0)
    q75 = np.quantile(values, 0.75, axis=0)
    scale = q75 - q25
    scale[scale == 0] = 1.0
    return scale


@dataclass
class HarnessProfileBundle:
    profile: HarnessProfile
    rescue_index: RescueIndex | None = None
    rescue_frame_table: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HarnessProfileMinerConfig:
    dataset_repo_id: str
    dataset_root: str | None = None
    dataset_revision: str | None = None
    output_dir: str = "outputs/harness_profiles/default"
    fps: int | None = None
    rescue_horizon_steps: int = 8
    percentile_low: float = 0.005
    percentile_high: float = 0.995
    min_support: float = 0.95
    max_train_violation_rate: float = 0.02


class HarnessProfileMiner:
    def __init__(self, cfg: HarnessProfileMinerConfig):
        self.cfg = cfg

    def _extract_arrays_from_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], int]:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            repo_id=self.cfg.dataset_repo_id,
            root=self.cfg.dataset_root,
            revision=self.cfg.dataset_revision,
            download_videos=False,
        )
        state_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        episode_ids: list[int] = []

        for row in dataset.select_columns(["observation.state", "action", "episode_index"]):
            state_rows.append(np.asarray(row["observation.state"], dtype=np.float64))
            action_rows.append(np.asarray(row["action"], dtype=np.float64))
            episode_ids.append(int(row["episode_index"]))

        state_keys = [f"state_{idx}" for idx in range(len(state_rows[0]))]
        action_keys = [f"action_{idx}" for idx in range(len(action_rows[0]))]
        fps = int(self.cfg.fps or dataset.fps)
        return (
            np.stack(state_rows, axis=0),
            np.stack(action_rows, axis=0),
            np.asarray(episode_ids, dtype=np.int64),
            state_keys,
            action_keys,
            fps,
        )

    def build(self) -> HarnessProfileBundle:
        states, actions, episode_ids, state_keys, action_keys, fps = self._extract_arrays_from_dataset()
        state_scales = _robust_scale(states)
        action_scales = _robust_scale(actions)

        mode_profile = build_mode_profile(states, actions, episode_ids)
        invariants = InvariantMiner(
            min_support=self.cfg.min_support,
            max_train_violation_rate=self.cfg.max_train_violation_rate,
        ).mine(states, actions, episode_ids, mode_profile)
        speed_envelopes = build_speed_envelope_profile(
            actions,
            mode_profile.mode_ids,
            percentile_low=self.cfg.percentile_low,
            percentile_high=self.cfg.percentile_high,
            states=states,
            episode_ids=episode_ids,
        )
        rescue_index = build_rescue_index(
            states,
            actions,
            episode_ids,
            state_scales=state_scales,
            mode_ids=mode_profile.mode_ids,
            horizon_steps=self.cfg.rescue_horizon_steps,
        )
        rescue_frame_table = [
            {
                "episode_index": int(rescue_index.episode_ids[idx]),
                "frame_index": int(rescue_index.frame_indices[idx]),
                "snippet_start": int(rescue_index.snippet_starts[idx]),
                "snippet_end": int(rescue_index.snippet_ends[idx]),
                "future_progress_score": float(rescue_index.future_progress_scores[idx]),
                "mode_id": str(rescue_index.mode_ids[idx]),
            }
            for idx in range(len(rescue_index.frame_indices))
        ]

        profile = HarnessProfile(
            schema_version=1,
            dataset_repo_id=self.cfg.dataset_repo_id,
            dataset_revision=self.cfg.dataset_revision,
            fps=fps,
            state_keys=state_keys,
            action_keys=action_keys,
            scales={
                "state": state_scales.tolist(),
                "action": action_scales.tolist(),
            },
            mode_profile=mode_profile,
            invariants=invariants,
            speed_envelopes=speed_envelopes,
            rescue_index=RescueIndexMetadata(
                type="state_knn",
                index_path="rescue_index.npz",
                frame_table_path="rescue_frames.parquet",
                state_dim=int(states.shape[1]),
                snippet_horizon_steps=self.cfg.rescue_horizon_steps,
                num_entries=int(len(states)),
            ),
            diagnostics=HarnessDiagnostics(
                num_episodes=int(len(np.unique(episode_ids))),
                num_frames=int(len(states)),
                miner_config={
                    "rescue_horizon_steps": self.cfg.rescue_horizon_steps,
                    "percentile_low": self.cfg.percentile_low,
                    "percentile_high": self.cfg.percentile_high,
                    "min_support": self.cfg.min_support,
                    "max_train_violation_rate": self.cfg.max_train_violation_rate,
                },
            ),
        )
        return HarnessProfileBundle(
            profile=profile,
            rescue_index=rescue_index,
            rescue_frame_table=rescue_frame_table,
        )

    def export(self) -> HarnessProfileBundle:
        bundle = self.build()
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        profile_path = output_dir / "harness_profile.json"
        with profile_path.open("w", encoding="utf-8") as handle:
            json.dump(bundle.profile.to_dict(), handle, indent=2)

        if bundle.rescue_index is not None:
            np.savez_compressed(
                output_dir / "rescue_index.npz",
                normalized_states=bundle.rescue_index.normalized_states,
                episode_ids=bundle.rescue_index.episode_ids,
                frame_indices=bundle.rescue_index.frame_indices,
                snippet_starts=bundle.rescue_index.snippet_starts,
                snippet_ends=bundle.rescue_index.snippet_ends,
                future_progress_scores=bundle.rescue_index.future_progress_scores,
                action_snippets=bundle.rescue_index.action_snippets,
                mode_ids=bundle.rescue_index.mode_ids,
                scales=bundle.rescue_index.scales,
            )

        self._write_csv(
            output_dir / "invariant_report.csv",
            [
                {
                    "invariant_id": item.invariant_id,
                    "kind": item.kind,
                    "category": item.category,
                    "support": item.support,
                    "train_violation_rate": item.train_violation_rate,
                }
                for item in bundle.profile.invariants
            ],
        )
        self._write_csv(
            output_dir / "mode_report.csv",
            [
                {
                    "mode_id": mode.mode_id,
                    "label": mode.label,
                    "support": mode.support,
                    "min_duration_steps": mode.min_duration_steps,
                }
                for mode in bundle.profile.mode_profile.modes
            ],
        )
        self._write_csv(
            output_dir / "speed_envelope_report.csv",
            [
                {
                    "band": band_name,
                    "low": getattr(bundle.profile.speed_envelopes, band_name).low,
                    "high": getattr(bundle.profile.speed_envelopes, band_name).high,
                }
                for band_name in ("value", "delta", "acceleration")
            ],
        )
        self._write_markdown(output_dir / "profile_diagnostics.md", bundle)
        self._write_rescue_frame_table(output_dir / "rescue_frames.parquet", bundle.rescue_frame_table)
        return bundle

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_markdown(path: Path, bundle: HarnessProfileBundle) -> None:
        diagnostics = bundle.profile.diagnostics
        lines = [
            "# Harness Profile Diagnostics",
            "",
            f"- Dataset: `{bundle.profile.dataset_repo_id}`",
            f"- Frames: `{diagnostics.num_frames}`",
            f"- Episodes: `{diagnostics.num_episodes}`",
            f"- Modes: `{len(bundle.profile.mode_profile.modes)}`",
            f"- Invariants: `{len(bundle.profile.invariants)}`",
            f"- Rescue entries: `{bundle.profile.rescue_index.num_entries if bundle.profile.rescue_index else 0}`",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_rescue_frame_table(path: Path, rows: list[dict[str, Any]]) -> None:
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "Exporting rescue_frames.parquet requires pandas with parquet support."
            ) from exc
        pd.DataFrame(rows).to_parquet(path, index=False)


def load_harness_profile(profile_path: str | Path) -> HarnessProfileBundle:
    profile_path = Path(profile_path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = HarnessProfile.from_dict(payload)
    rescue_index = None
    rescue_frame_table: list[dict[str, Any]] = []
    if profile.rescue_index and profile.rescue_index.index_path:
        index_path = profile_path.parent / profile.rescue_index.index_path
        if index_path.exists():
            npz = np.load(index_path, allow_pickle=True)
            rescue_index = RescueIndex(
                normalized_states=np.asarray(npz["normalized_states"], dtype=np.float64),
                episode_ids=np.asarray(npz["episode_ids"], dtype=np.int64),
                frame_indices=np.asarray(npz["frame_indices"], dtype=np.int64),
                snippet_starts=np.asarray(npz["snippet_starts"], dtype=np.int64),
                snippet_ends=np.asarray(npz["snippet_ends"], dtype=np.int64),
                future_progress_scores=np.asarray(npz["future_progress_scores"], dtype=np.float64),
                action_snippets=np.asarray(npz["action_snippets"], dtype=np.float64),
                mode_ids=np.asarray(npz["mode_ids"], dtype=object),
                scales=np.asarray(npz["scales"], dtype=np.float64),
            )
    if profile.rescue_index and profile.rescue_index.frame_table_path:
        frame_table_path = profile_path.parent / profile.rescue_index.frame_table_path
        if frame_table_path.exists():
            try:
                import pandas as pd
            except Exception:  # pragma: no cover - optional runtime dependency
                rescue_frame_table = []
            else:
                rescue_frame_table = pd.read_parquet(frame_table_path).to_dict(orient="records")
    return HarnessProfileBundle(
        profile=profile,
        rescue_index=rescue_index,
        rescue_frame_table=rescue_frame_table,
    )
