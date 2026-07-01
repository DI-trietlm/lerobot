from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import HarnessConfig
from .schemas import EnvelopeBand, SpeedEnvelopeProfile


@dataclass
class EnvelopeViolation:
    kind: str
    severity: str
    reason: str
    metadata: dict[str, Any]


def build_speed_envelope_profile(
    actions: np.ndarray,
    mode_ids: list[str] | np.ndarray | None,
    percentile_low: float,
    percentile_high: float,
    states: np.ndarray | None = None,
    episode_ids: np.ndarray | None = None,
) -> SpeedEnvelopeProfile:
    values = np.asarray(actions, dtype=np.float64)
    if states is not None:
        deltas = values - np.asarray(states, dtype=np.float64)
    else:
        deltas = np.diff(values, axis=0, prepend=values[:1])
        if episode_ids is not None:
            episode_ids = np.asarray(episode_ids)
            boundary_mask = np.r_[True, episode_ids[1:] != episode_ids[:-1]]
            deltas[boundary_mask] = 0.0
    accelerations = np.diff(deltas, axis=0, prepend=deltas[:1])
    if episode_ids is not None:
        episode_ids = np.asarray(episode_ids)
        boundary_mask = np.r_[True, episode_ids[1:] != episode_ids[:-1]]
        accelerations[boundary_mask] = 0.0

    def make_band(array: np.ndarray) -> EnvelopeBand:
        return EnvelopeBand(
            low=np.quantile(array, percentile_low, axis=0).tolist(),
            high=np.quantile(array, percentile_high, axis=0).tolist(),
        )

    per_mode: dict[str, dict[str, EnvelopeBand]] = {}
    if mode_ids is not None:
        mode_ids = np.asarray(mode_ids, dtype=object)
        for mode_id in dict.fromkeys(mode_ids.tolist()):
            mask = mode_ids == mode_id
            if not np.any(mask):
                continue
            per_mode[str(mode_id)] = {
                "value": make_band(values[mask]),
                "delta": make_band(deltas[mask]),
                "acceleration": make_band(accelerations[mask]),
            }

    return SpeedEnvelopeProfile(
        value=make_band(values),
        delta=make_band(deltas),
        acceleration=make_band(accelerations),
        per_mode=per_mode,
    )


class ActionEnvelopeGuard:
    def __init__(self, profile: SpeedEnvelopeProfile | None, cfg: HarnessConfig):
        self.profile = profile
        self.cfg = cfg
        self._consecutive_clamps = 0

    @property
    def consecutive_clamps(self) -> int:
        return self._consecutive_clamps

    def _select_band(self, mode_id: str | None, band_name: str) -> EnvelopeBand | None:
        if self.profile is None:
            return None
        if (
            self.cfg.speed_envelope.mode_conditioned
            and mode_id
            and mode_id in self.profile.per_mode
            and band_name in self.profile.per_mode[mode_id]
        ):
            return self.profile.per_mode[mode_id][band_name]
        return getattr(self.profile, band_name)

    def evaluate(
        self,
        current_state: np.ndarray,
        action_chunk: np.ndarray,
        mode_id: str | None,
    ) -> tuple[np.ndarray, list[EnvelopeViolation]]:
        if not self.cfg.effective_enabled(self.cfg.speed_envelope.enable) or self.profile is None:
            return np.asarray(action_chunk), []

        current_state = np.asarray(current_state, dtype=np.float64)
        candidate = np.asarray(action_chunk, dtype=np.float64).copy()
        deltas = candidate - current_state
        accelerations = np.diff(deltas, axis=0, prepend=deltas[:1])
        violations: list[EnvelopeViolation] = []
        clamped = False

        for band_name, values in (
            ("value", candidate),
            ("delta", deltas),
            ("acceleration", accelerations),
        ):
            band = self._select_band(mode_id, band_name)
            if band is None:
                continue
            low = np.asarray(band.low, dtype=np.float64)
            high = np.asarray(band.high, dtype=np.float64)
            outside = np.logical_or(values < low, values > high)
            if not np.any(outside):
                continue

            severity = "shadow" if (self.cfg.shadow_mode or self.cfg.speed_envelope.shadow_mode) else "soft"
            violations.append(
                EnvelopeViolation(
                    kind=band_name,
                    severity=severity,
                    reason=f"{band_name}_outside_envelope",
                    metadata={"outside_dims": np.where(np.any(outside, axis=0))[0].tolist()},
                )
            )
            if severity != "shadow":
                clamped = True
                if band_name == "value":
                    candidate = np.clip(candidate, low, high)
                elif band_name == "delta":
                    candidate = np.clip(candidate - current_state, low, high) + current_state

        if clamped:
            self._consecutive_clamps += 1
        else:
            self._consecutive_clamps = 0

        if (
            self._consecutive_clamps >= self.cfg.speed_envelope.max_consecutive_clamps
            and self.cfg.speed_envelope.flush_after_repeated_clamp
        ):
            violations.append(
                EnvelopeViolation(
                    kind="repeated_clamp",
                    severity="shadow"
                    if (self.cfg.shadow_mode or self.cfg.speed_envelope.shadow_mode)
                    else "hard",
                    reason="repeated_speed_clamp_requires_flush",
                    metadata={"consecutive_clamps": self._consecutive_clamps},
                )
            )

        return candidate, violations
