from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import HarnessConfig
from .mode import ModeEstimate
from .schemas import InvariantSpec, ModeProfile


def _episode_slices(episode_ids: np.ndarray) -> list[np.ndarray]:
    return [np.where(episode_ids == episode)[0] for episode in np.unique(episode_ids)]


def _run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for flag in mask:
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


@dataclass
class InvariantViolation:
    invariant_id: str
    kind: str
    category: str
    severity: str
    reason: str
    metadata: dict[str, Any]


class InvariantMiner:
    def __init__(self, min_support: float = 0.95, max_train_violation_rate: float = 0.02):
        self.min_support = min_support
        self.max_train_violation_rate = max_train_violation_rate

    def mine(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ids: np.ndarray,
        mode_profile: ModeProfile,
    ) -> list[InvariantSpec]:
        invariants: list[InvariantSpec] = []
        mode_ids = np.asarray(mode_profile.mode_ids, dtype=object)
        if len(mode_ids) != len(states):
            mode_ids = np.full(len(states), "mode_plateau", dtype=object)
        episode_slices = _episode_slices(episode_ids)
        num_episodes = max(len(episode_slices), 1)

        transition_counts: dict[tuple[str, str], int] = {}
        total_mode_transitions = 0
        for idx in range(1, len(mode_ids)):
            if episode_ids[idx] != episode_ids[idx - 1]:
                continue
            src = str(mode_ids[idx - 1])
            dst = str(mode_ids[idx])
            if src == dst:
                continue
            total_mode_transitions += 1
            transition_counts[(src, dst)] = transition_counts.get((src, dst), 0) + 1

        for mode in mode_profile.modes:
            mask = mode_ids == mode.mode_id
            if not np.any(mask):
                continue

            action_subset = actions[mask]
            state_subset = states[mask]
            low = np.quantile(action_subset, 0.01, axis=0).tolist()
            high = np.quantile(action_subset, 0.99, axis=0).tolist()
            outside = np.logical_or(action_subset < np.asarray(low), action_subset > np.asarray(high))
            violation_rate = float(np.mean(np.any(outside, axis=1))) if len(action_subset) else 0.0
            invariants.append(
                InvariantSpec(
                    invariant_id=f"value_envelope::{mode.mode_id}",
                    kind="value_envelope",
                    category="value_envelope",
                    support=max(0.0, 1.0 - violation_rate),
                    train_violation_rate=violation_rate,
                    parameters={
                        "mode_id": mode.mode_id,
                        "low": low,
                        "high": high,
                    },
                )
            )

            velocity = action_subset - state_subset
            vel_low = np.quantile(velocity, 0.01, axis=0).tolist()
            vel_high = np.quantile(velocity, 0.99, axis=0).tolist()
            vel_outside = np.logical_or(velocity < np.asarray(vel_low), velocity > np.asarray(vel_high))
            vel_violation_rate = float(np.mean(np.any(vel_outside, axis=1))) if len(velocity) else 0.0
            invariants.append(
                InvariantSpec(
                    invariant_id=f"velocity_envelope::{mode.mode_id}",
                    kind="velocity_envelope",
                    category="velocity_envelope",
                    support=max(0.0, 1.0 - vel_violation_rate),
                    train_violation_rate=vel_violation_rate,
                    parameters={
                        "mode_id": mode.mode_id,
                        "low": vel_low,
                        "high": vel_high,
                    },
                )
            )

            durations: list[int] = []
            current = 0
            for idx, frame_mode in enumerate(mode_ids):
                if frame_mode == mode.mode_id:
                    current += 1
                elif current:
                    durations.append(current)
                    current = 0
                if idx < len(mode_ids) - 1 and episode_ids[idx] != episode_ids[idx + 1] and current:
                    durations.append(current)
                    current = 0
            if current:
                durations.append(current)
            if durations:
                min_duration = int(max(1, np.quantile(durations, 0.1)))
                short_segments = sum(duration < min_duration for duration in durations)
                violation_rate = short_segments / max(len(durations), 1)
                episodes_with_mode = sum(np.any(mode_ids[indices] == mode.mode_id) for indices in episode_slices)
                invariants.append(
                    InvariantSpec(
                        invariant_id=f"plateau_min_duration::{mode.mode_id}",
                        kind="plateau_min_duration",
                        category="plateau_min_duration",
                        support=float(episodes_with_mode / num_episodes),
                        train_violation_rate=violation_rate,
                        parameters={"mode_id": mode.mode_id, "min_duration_steps": min_duration},
                    )
                )

        for (source_mode, target_mode), count in transition_counts.items():
            support = count / max(total_mode_transitions, 1)
            reverse_count = transition_counts.get((target_mode, source_mode), 0)
            reverse_support = reverse_count / max(total_mode_transitions, 1)
            invariants.append(
                InvariantSpec(
                    invariant_id=f"no_backtrack::{source_mode}->{target_mode}",
                    kind="no_backtrack_transition",
                    category="no_backtrack",
                    support=max(0.0, 1.0 - reverse_support),
                    train_violation_rate=reverse_support,
                    parameters={
                        "source_mode_id": source_mode,
                        "target_mode_id": target_mode,
                        "reverse_count": reverse_count,
                    },
                )
            )

        invariants.extend(self._mine_actuator_hold_release(states, actions, episode_slices))

        return invariants

    def _mine_actuator_hold_release(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_slices: list[np.ndarray],
    ) -> list[InvariantSpec]:
        mined: list[InvariantSpec] = []
        if actions.ndim != 2 or states.shape[1] != actions.shape[1]:
            return mined

        min_hold_steps = 5
        for dim in range(actions.shape[1]):
            values = actions[:, dim]
            value_range = float(np.quantile(values, 0.99) - np.quantile(values, 0.01))
            if value_range <= 1e-6:
                continue

            low_threshold = float(np.quantile(values, 0.2))
            high_threshold = float(np.quantile(values, 0.8))
            midpoint_threshold = float((low_threshold + high_threshold) / 2.0)
            candidates = [
                ("low", values <= low_threshold, low_threshold, midpoint_threshold),
                ("high", values >= high_threshold, high_threshold, midpoint_threshold),
            ]

            best: tuple[str, np.ndarray, float, float, float, float] | None = None
            for side, mask, hold_threshold, release_threshold in candidates:
                episodes_with_hold = 0
                hold_frames = 0
                release_frames = 0
                for indices in episode_slices:
                    episode_mask = mask[indices]
                    if max(_run_lengths(episode_mask) or [0]) >= min_hold_steps:
                        episodes_with_hold += 1
                    for local_idx, global_idx in enumerate(indices[:-1]):
                        if not mask[global_idx]:
                            continue
                        hold_frames += 1
                        next_value = values[indices[local_idx + 1]]
                        leaves_hold = next_value > release_threshold if side == "low" else next_value < release_threshold
                        if leaves_hold:
                            release_frames += 1

                support = episodes_with_hold / max(len(episode_slices), 1)
                violation_rate = release_frames / max(hold_frames, 1)
                score = support - violation_rate
                if best is None or score > best[4] - best[5]:
                    best = (side, mask, hold_threshold, release_threshold, support, violation_rate)

            if best is None:
                continue
            side, _, hold_threshold, release_threshold, support, violation_rate = best
            if support < self.min_support or violation_rate > self.max_train_violation_rate:
                continue
            mined.append(
                InvariantSpec(
                    invariant_id=f"actuator_hold_release::dim_{dim}",
                    kind="actuator_hold_release",
                    category="catastrophic_actuator_release",
                    support=float(support),
                    train_violation_rate=float(violation_rate),
                    parameters={
                        "dim": dim,
                        "hold_side": side,
                        "hold_threshold": hold_threshold,
                        "release_threshold": release_threshold,
                        "min_hold_steps": min_hold_steps,
                    },
                )
            )
        return mined


class InvariantGuard:
    def __init__(self, invariants: list[InvariantSpec], cfg: HarnessConfig):
        self.cfg = cfg
        self.invariants = [invariant for invariant in invariants if invariant.enabled]
        self._transition_memory: list[str] = []

    def _severity_for(self, invariant: InvariantSpec, mode_confidence: float) -> str:
        if invariant.category in self.cfg.invariant_guard.hard_guard_categories:
            if self.cfg.invariant_guard.shadow_mode or self.cfg.shadow_mode:
                return "shadow"
            return "hard"
        if mode_confidence < self.cfg.invariant_guard.min_mode_confidence:
            return "shadow"
        if invariant.category in self.cfg.invariant_guard.soft_guard_categories:
            return "soft"
        if self.cfg.invariant_guard.shadow_mode or self.cfg.shadow_mode:
            return "shadow"
        return "soft"

    def evaluate_chunk(
        self,
        current_state: np.ndarray,
        action_chunk: np.ndarray,
        mode_estimate: ModeEstimate,
        mode_history: list[str] | None = None,
    ) -> list[InvariantViolation]:
        if not self.cfg.effective_enabled(self.cfg.invariant_guard.enable):
            return []

        current_state = np.asarray(current_state, dtype=np.float64)
        action_chunk = np.asarray(action_chunk, dtype=np.float64)
        current_mode = mode_estimate.mode_id
        history = mode_history or []
        violations: list[InvariantViolation] = []

        for invariant in self.invariants:
            if invariant.support < self.cfg.invariant_guard.min_support:
                continue
            if invariant.train_violation_rate > self.cfg.invariant_guard.max_train_violation_rate:
                continue

            severity = self._severity_for(invariant, mode_estimate.confidence)
            params = invariant.parameters
            if invariant.kind == "value_envelope" and params.get("mode_id") == current_mode:
                low = np.asarray(params["low"], dtype=np.float64)
                high = np.asarray(params["high"], dtype=np.float64)
                outside = np.logical_or(action_chunk < low, action_chunk > high)
                if np.any(outside):
                    violations.append(
                        InvariantViolation(
                            invariant_id=invariant.invariant_id,
                            kind=invariant.kind,
                            category=invariant.category,
                            severity=severity,
                            reason="action_outside_mode_envelope",
                            metadata={
                                "mode_id": current_mode,
                                "outside_dims": np.where(np.any(outside, axis=0))[0].tolist(),
                            },
                        )
                    )

            elif invariant.kind == "velocity_envelope" and params.get("mode_id") == current_mode:
                deltas = action_chunk - current_state
                low = np.asarray(params["low"], dtype=np.float64)
                high = np.asarray(params["high"], dtype=np.float64)
                outside = np.logical_or(deltas < low, deltas > high)
                if np.any(outside):
                    violations.append(
                        InvariantViolation(
                            invariant_id=invariant.invariant_id,
                            kind=invariant.kind,
                            category=invariant.category,
                            severity=severity,
                            reason="delta_outside_mode_velocity_envelope",
                            metadata={
                                "mode_id": current_mode,
                                "outside_dims": np.where(np.any(outside, axis=0))[0].tolist(),
                            },
                        )
                    )

            elif invariant.kind == "no_backtrack_transition":
                source_mode_id = str(params.get("source_mode_id"))
                target_mode_id = str(params.get("target_mode_id"))
                if len(history) >= 1 and history[-1] == target_mode_id and current_mode == source_mode_id:
                    violations.append(
                        InvariantViolation(
                            invariant_id=invariant.invariant_id,
                            kind=invariant.kind,
                            category=invariant.category,
                            severity=severity,
                            reason="reverse_transition_detected",
                            metadata={
                                "previous_mode": history[-1],
                                "current_mode": current_mode,
                            },
                        )
                    )

            elif invariant.kind == "plateau_min_duration" and params.get("mode_id") == current_mode:
                min_duration = int(params["min_duration_steps"])
                streak = 0
                for mode_id in reversed(history + [current_mode]):
                    if mode_id != current_mode:
                        break
                    streak += 1
                if 0 < streak < min_duration:
                    next_action = action_chunk[0]
                    if np.linalg.norm(next_action - current_state) > 0:
                        violations.append(
                            InvariantViolation(
                                invariant_id=invariant.invariant_id,
                                kind=invariant.kind,
                                category=invariant.category,
                                severity="shadow" if severity == "hard" else severity,
                                reason="premature_mode_exit",
                                metadata={
                                    "mode_id": current_mode,
                                    "required_duration_steps": min_duration,
                                    "observed_duration_steps": streak,
                                },
                        )
                    )

            elif invariant.kind == "actuator_hold_release":
                dim = int(params["dim"])
                if dim >= current_state.shape[0] or dim >= action_chunk.shape[1]:
                    continue
                hold_side = str(params["hold_side"])
                hold_threshold = float(params["hold_threshold"])
                release_threshold = float(params["release_threshold"])
                current_value = current_state[dim]
                action_values = action_chunk[:, dim]
                if hold_side == "low":
                    in_hold = current_value <= hold_threshold
                    leaves_hold = np.any(action_values > release_threshold)
                else:
                    in_hold = current_value >= hold_threshold
                    leaves_hold = np.any(action_values < release_threshold)
                if in_hold and leaves_hold:
                    violations.append(
                        InvariantViolation(
                            invariant_id=invariant.invariant_id,
                            kind=invariant.kind,
                            category=invariant.category,
                            severity=severity,
                            reason="actuator_release_from_stable_hold",
                            metadata={
                                "dim": dim,
                                "hold_side": hold_side,
                                "hold_threshold": hold_threshold,
                                "release_threshold": release_threshold,
                            },
                        )
                    )

        if mode_estimate.mode_id != "unknown":
            self._transition_memory.append(mode_estimate.mode_id)
            self._transition_memory = self._transition_memory[-32:]

        return violations
