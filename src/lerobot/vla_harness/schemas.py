from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: dataclass_to_dict(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): dataclass_to_dict(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


@dataclass
class ModeRecord:
    mode_id: str
    label: str
    support: float
    centroid_state: list[float]
    feature_centroid: list[float]
    min_duration_steps: int = 1


@dataclass
class TransitionEdge:
    source_mode_id: str
    target_mode_id: str
    support: float
    count: int


@dataclass
class ModeProfile:
    modes: list[ModeRecord] = field(default_factory=list)
    transitions: list[TransitionEdge] = field(default_factory=list)
    feature_keys: list[str] = field(default_factory=list)
    mode_ids: list[str] = field(default_factory=list)
    stable: bool = False


@dataclass
class InvariantSpec:
    invariant_id: str
    kind: str
    category: str
    support: float
    train_violation_rate: float
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvelopeBand:
    low: list[float]
    high: list[float]


@dataclass
class SpeedEnvelopeProfile:
    value: EnvelopeBand
    delta: EnvelopeBand
    acceleration: EnvelopeBand
    per_mode: dict[str, dict[str, EnvelopeBand]] = field(default_factory=dict)


@dataclass
class RescueIndexMetadata:
    type: str
    index_path: str | None = None
    frame_table_path: str | None = None
    state_dim: int = 0
    snippet_horizon_steps: int = 0
    num_entries: int = 0


@dataclass
class HarnessDiagnostics:
    num_episodes: int
    num_frames: int
    miner_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessProfile:
    schema_version: int
    dataset_repo_id: str
    dataset_revision: str | None
    fps: int
    state_keys: list[str]
    action_keys: list[str]
    scales: dict[str, list[float]]
    mode_profile: ModeProfile = field(default_factory=ModeProfile)
    invariants: list[InvariantSpec] = field(default_factory=list)
    speed_envelopes: SpeedEnvelopeProfile | None = None
    rescue_index: RescueIndexMetadata | None = None
    diagnostics: HarnessDiagnostics = field(
        default_factory=lambda: HarnessDiagnostics(num_episodes=0, num_frames=0)
    )

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HarnessProfile":
        mode_profile_payload = payload.get("mode_profile") or {}
        invariants_payload = payload.get("invariants") or []
        speed_payload = payload.get("speed_envelopes")
        rescue_payload = payload.get("rescue_index")
        diagnostics_payload = payload.get("diagnostics") or {}

        mode_profile = ModeProfile(
            modes=[ModeRecord(**mode) for mode in mode_profile_payload.get("modes", [])],
            transitions=[
                TransitionEdge(**edge) for edge in mode_profile_payload.get("transitions", [])
            ],
            feature_keys=list(mode_profile_payload.get("feature_keys", [])),
            mode_ids=list(mode_profile_payload.get("mode_ids", [])),
            stable=bool(mode_profile_payload.get("stable", False)),
        )

        speed_profile = None
        if speed_payload:
            speed_profile = SpeedEnvelopeProfile(
                value=EnvelopeBand(**speed_payload["value"]),
                delta=EnvelopeBand(**speed_payload["delta"]),
                acceleration=EnvelopeBand(**speed_payload["acceleration"]),
                per_mode={
                    mode_id: {
                        band_name: EnvelopeBand(**band_payload)
                        for band_name, band_payload in band_map.items()
                    }
                    for mode_id, band_map in speed_payload.get("per_mode", {}).items()
                },
            )

        rescue_meta = RescueIndexMetadata(**rescue_payload) if rescue_payload else None
        diagnostics = HarnessDiagnostics(**diagnostics_payload)

        return cls(
            schema_version=int(payload["schema_version"]),
            dataset_repo_id=str(payload["dataset_repo_id"]),
            dataset_revision=payload.get("dataset_revision"),
            fps=int(payload["fps"]),
            state_keys=list(payload.get("state_keys", [])),
            action_keys=list(payload.get("action_keys", [])),
            scales={str(key): list(values) for key, values in (payload.get("scales") or {}).items()},
            mode_profile=mode_profile,
            invariants=[InvariantSpec(**item) for item in invariants_payload],
            speed_envelopes=speed_profile,
            rescue_index=rescue_meta,
            diagnostics=diagnostics,
        )
