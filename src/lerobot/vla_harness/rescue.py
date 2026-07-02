from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import HarnessConfig


def _normalize_rows(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    scales = np.where(scales == 0, 1.0, scales)
    return np.asarray(values, dtype=np.float64) / scales


def compute_future_progress_scores(
    states: np.ndarray,
    episode_ids: np.ndarray,
    horizon_steps: int,
) -> np.ndarray:
    scores = np.zeros(len(states), dtype=np.float64)
    for idx in range(len(states)):
        end_idx = min(len(states) - 1, idx + horizon_steps)
        if episode_ids[end_idx] != episode_ids[idx]:
            while end_idx > idx and episode_ids[end_idx] != episode_ids[idx]:
                end_idx -= 1
        displacement = np.linalg.norm(states[end_idx] - states[idx])
        local_scale = np.linalg.norm(np.std(states[max(0, idx - horizon_steps) : end_idx + 1], axis=0)) + 1e-6
        scores[idx] = float(displacement / local_scale)
    if np.max(scores) > 0:
        scores = scores / np.max(scores)
    return scores


@dataclass
class RescueIndex:
    normalized_states: np.ndarray
    episode_ids: np.ndarray
    frame_indices: np.ndarray
    snippet_starts: np.ndarray
    snippet_ends: np.ndarray
    future_progress_scores: np.ndarray
    action_snippets: np.ndarray
    mode_ids: np.ndarray
    scales: np.ndarray

    def query(
        self,
        current_state: np.ndarray,
        k_neighbors: int,
    ) -> list[dict[str, Any]]:
        normalized_state = _normalize_rows(np.asarray(current_state)[None, :], self.scales)[0]
        distances = np.linalg.norm(self.normalized_states - normalized_state, axis=1)
        order = np.argsort(distances)[: max(1, k_neighbors)]
        return [
            {
                "index": int(index),
                "distance": float(distances[index]),
                "future_progress_score": float(self.future_progress_scores[index]),
                "episode_index": int(self.episode_ids[index]),
                "frame_index": int(self.frame_indices[index]),
                "snippet_start": int(self.snippet_starts[index]),
                "snippet_end": int(self.snippet_ends[index]),
                "mode_id": str(self.mode_ids[index]),
                "action_snippet": self.action_snippets[index].copy(),
            }
            for index in order
        ]


@dataclass
class MicroRescueDecision:
    accepted: bool
    severity: str
    reason: str
    snippet: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


def build_rescue_index(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    state_scales: np.ndarray,
    mode_ids: list[str] | np.ndarray | None,
    horizon_steps: int,
) -> RescueIndex:
    normalized_states = _normalize_rows(states, state_scales)
    progress_scores = compute_future_progress_scores(states, episode_ids, horizon_steps)
    global_indices = np.arange(len(states), dtype=np.int64)
    frame_indices = np.zeros(len(states), dtype=np.int64)
    episode_end_exclusive: dict[int, int] = {}
    for episode in np.unique(episode_ids):
        indices = np.where(episode_ids == episode)[0]
        frame_indices[indices] = np.arange(len(indices), dtype=np.int64)
        episode_end_exclusive[int(episode)] = int(indices[-1]) + 1

    snippet_starts = global_indices.copy()
    snippet_ends = np.zeros(len(states), dtype=np.int64)
    snippets = np.zeros((len(states), horizon_steps, actions.shape[1]), dtype=np.float64)
    for idx in range(len(states)):
        episode_end = episode_end_exclusive[int(episode_ids[idx])]
        snippet_ends[idx] = min(idx + horizon_steps, episode_end)
        end = int(snippet_ends[idx])
        snippet = actions[idx:end]
        snippets[idx, : len(snippet)] = snippet
        if len(snippet) and len(snippet) < horizon_steps:
            snippets[idx, len(snippet) :] = snippet[-1]
    return RescueIndex(
        normalized_states=normalized_states,
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        frame_indices=frame_indices,
        snippet_starts=snippet_starts,
        snippet_ends=np.asarray(snippet_ends, dtype=np.int64),
        future_progress_scores=progress_scores,
        action_snippets=snippets,
        mode_ids=np.asarray(mode_ids if mode_ids is not None else ["unknown"] * len(states), dtype=object),
        scales=np.asarray(state_scales, dtype=np.float64),
    )


class MicroRescuePlanner:
    def __init__(self, rescue_index: RescueIndex | None, cfg: HarnessConfig, fps: int = 15):
        self.rescue_index = rescue_index
        self.cfg = cfg
        self.fps = max(1, int(fps))
        self._rescues_this_episode = 0
        self._last_rescue_time = 0.0

    def reset_episode(self) -> None:
        self._rescues_this_episode = 0
        self._last_rescue_time = 0.0

    def _prepend_ramp_in(
        self,
        current_state: np.ndarray,
        snippet: np.ndarray,
        max_total_steps: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if len(snippet) == 0:
            return snippet, {"ramp_in_steps": 0}

        current = np.asarray(current_state, dtype=np.float64)
        if current.shape != snippet[0].shape:
            return snippet, {"ramp_in_steps": 0, "ramp_in_skipped": "shape_mismatch"}

        requested_steps = max(0, int(self.cfg.micro_rescue.ramp_in_steps))
        max_joint_delta = self.cfg.micro_rescue.ramp_in_max_joint_delta
        if max_joint_delta is not None and max_joint_delta > 0:
            max_delta = float(np.max(np.abs(snippet[0] - current)))
            requested_steps = max(requested_steps, int(np.ceil(max_delta / max_joint_delta)))

        if requested_steps <= 1:
            first_step_delta = float(np.linalg.norm(snippet[0] - current))
            return snippet, {
                "ramp_in_steps": 0,
                "pre_ramp_first_step_l2": first_step_delta,
                "post_ramp_first_step_l2": first_step_delta,
            }

        fractions = np.linspace(1.0 / requested_steps, 1.0, requested_steps, dtype=np.float64)
        bridge = current[None, :] + fractions[:, None] * (snippet[0][None, :] - current[None, :])
        snippet = np.concatenate([bridge, snippet[1:]], axis=0)
        if len(snippet) > max_total_steps:
            snippet = snippet[:max_total_steps]

        return snippet, {
            "ramp_in_steps": int(len(bridge)),
            "ramp_in_requested_steps": int(requested_steps),
            "ramp_in_max_joint_delta": max_joint_delta,
            "pre_ramp_first_step_l2": float(np.linalg.norm(bridge[-1] - current)),
            "post_ramp_first_step_l2": float(np.linalg.norm(snippet[0] - current)),
        }

    def query(self, current_state: np.ndarray, now_s: float | None = None) -> MicroRescueDecision:
        if not self.cfg.effective_enabled(self.cfg.micro_rescue.enable) or self.rescue_index is None:
            return MicroRescueDecision(False, "shadow", "micro_rescue_disabled")
        if not self.cfg.micro_rescue.state_knn_enable:
            return MicroRescueDecision(False, "shadow", "state_knn_rescue_disabled")
        if self._rescues_this_episode >= self.cfg.micro_rescue.max_rescues_per_episode:
            return MicroRescueDecision(False, "shadow", "micro_rescue_budget_exhausted")
        if now_s is not None and (now_s - self._last_rescue_time) < self.cfg.micro_rescue.cooldown_s:
            return MicroRescueDecision(False, "shadow", "micro_rescue_cooldown_active")

        neighbors = self.rescue_index.query(current_state, self.cfg.micro_rescue.k_neighbors)
        filtered = []
        for neighbor in neighbors:
            if neighbor["future_progress_score"] < self.cfg.micro_rescue.min_future_progress_score:
                continue
            if (
                self.cfg.micro_rescue.max_state_distance is not None
                and neighbor["distance"] > self.cfg.micro_rescue.max_state_distance
            ):
                continue
            filtered.append(neighbor)

        if not filtered:
            return MicroRescueDecision(False, "shadow", "no_rescue_neighbor_passed_filters")

        selected = max(
            filtered,
            key=lambda item: item["future_progress_score"] - 0.1 * item["distance"],
        )
        snippet = np.asarray(selected["action_snippet"], dtype=np.float64)
        max_steps_from_duration = max(1, int(round(self.cfg.micro_rescue.max_duration_s * self.fps)))
        horizon = min(self.cfg.micro_rescue.snippet_horizon_steps, max_steps_from_duration, len(snippet))
        snippet = snippet[:horizon]
        if self.cfg.micro_rescue.blend_alpha < 1.0:
            snippet = self.cfg.micro_rescue.blend_alpha * snippet + (
                1.0 - self.cfg.micro_rescue.blend_alpha
            ) * np.asarray(current_state, dtype=np.float64)
        snippet, ramp_metadata = self._prepend_ramp_in(
            current_state=np.asarray(current_state, dtype=np.float64),
            snippet=snippet,
            max_total_steps=max_steps_from_duration,
        )

        self._rescues_this_episode += 1
        if now_s is not None:
            self._last_rescue_time = now_s

        severity = "shadow" if (self.cfg.shadow_mode or self.cfg.micro_rescue.shadow_mode) else "soft"
        return MicroRescueDecision(
            accepted=True,
            severity=severity,
            reason="micro_rescue_selected",
            snippet=snippet,
            metadata={
                "selected_neighbor": {
                    key: value
                    for key, value in selected.items()
                    if key != "action_snippet"
                },
                "neighbor_count": len(neighbors),
                **ramp_metadata,
            },
        )
