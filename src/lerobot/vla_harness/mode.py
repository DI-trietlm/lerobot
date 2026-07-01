from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import ModeProfile, ModeRecord, TransitionEdge


def _robust_normalize(features: np.ndarray) -> np.ndarray:
    median = np.median(features, axis=0)
    q25 = np.quantile(features, 0.25, axis=0)
    q75 = np.quantile(features, 0.75, axis=0)
    scale = q75 - q25
    scale[scale == 0] = 1.0
    return (features - median) / scale


def _deterministic_kmeans(features: np.ndarray, n_clusters: int, iterations: int = 25) -> np.ndarray:
    if len(features) == 0:
        return np.asarray([], dtype=np.int64)
    n_clusters = max(1, min(n_clusters, len(features)))
    motion_score = features[:, 0] + features[:, 1]
    seeds = np.quantile(motion_score, np.linspace(0.0, 1.0, n_clusters))
    seed_indices = [int(np.argmin(np.abs(motion_score - seed))) for seed in seeds]
    centroids = features[seed_indices].copy()
    labels = np.zeros(len(features), dtype=np.int64)

    for _ in range(iterations):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1)
        next_centroids = centroids.copy()
        for cluster_id in range(n_clusters):
            mask = next_labels == cluster_id
            if np.any(mask):
                next_centroids[cluster_id] = features[mask].mean(axis=0)
        if np.array_equal(next_labels, labels):
            centroids = next_centroids
            break
        labels = next_labels
        centroids = next_centroids
    return labels


def _motion_features(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    action_norm = np.linalg.norm(actions, axis=1)
    deltas = np.diff(states, axis=0, prepend=states[:1])
    if episode_ids is not None:
        episode_ids = np.asarray(episode_ids)
        boundary_mask = np.r_[True, episode_ids[1:] != episode_ids[:-1]]
        deltas[boundary_mask] = 0.0
    state_velocity_norm = np.linalg.norm(deltas, axis=1)
    action_abs_mean = np.mean(np.abs(actions), axis=1)
    features = np.stack([action_norm, state_velocity_norm, action_abs_mean], axis=1)
    return features, ["action_norm", "state_velocity_norm", "action_abs_mean"]


def discover_mode_ids(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features, feature_keys = _motion_features(states, actions, episode_ids)
    normalized_features = _robust_normalize(features)
    labels = _deterministic_kmeans(normalized_features, n_clusters=3)
    motion_score = features[:, 0] + features[:, 1]
    label_scores = {
        label: float(np.mean(motion_score[labels == label]))
        for label in np.unique(labels)
    }
    ordered_labels = [label for label, _ in sorted(label_scores.items(), key=lambda item: item[1])]
    names = ["mode_plateau", "mode_transition", "mode_excursion"]
    label_to_name = {
        label: names[min(rank, len(names) - 1)]
        for rank, label in enumerate(ordered_labels)
    }
    mode_ids = np.asarray([label_to_name[label] for label in labels], dtype=object)
    return np.asarray(mode_ids, dtype=object), features, feature_keys


def build_mode_profile(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
) -> ModeProfile:
    mode_ids, features, feature_keys = discover_mode_ids(states, actions, episode_ids)
    unique_modes = list(dict.fromkeys(mode_ids.tolist()))
    modes: list[ModeRecord] = []
    transitions: dict[tuple[str, str], int] = {}

    for mode_id in unique_modes:
        mask = mode_ids == mode_id
        support = float(np.mean(mask)) if len(mask) else 0.0
        centroid_state = states[mask].mean(axis=0).tolist() if np.any(mask) else [0.0] * states.shape[1]
        centroid_feature = (
            features[mask].mean(axis=0).tolist() if np.any(mask) else [0.0] * features.shape[1]
        )
        min_duration = 1
        current = 0
        durations: list[int] = []
        for idx, current_mode in enumerate(mode_ids):
            if current_mode == mode_id:
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
            min_duration = max(1, int(np.quantile(durations, 0.1)))

        modes.append(
            ModeRecord(
                mode_id=mode_id,
                label=mode_id.replace("mode_", ""),
                support=support,
                centroid_state=centroid_state,
                feature_centroid=centroid_feature,
                min_duration_steps=min_duration,
            )
        )

    for idx in range(1, len(mode_ids)):
        if episode_ids[idx] != episode_ids[idx - 1]:
            continue
        prev_mode = str(mode_ids[idx - 1])
        next_mode = str(mode_ids[idx])
        if prev_mode == next_mode:
            continue
        transitions[(prev_mode, next_mode)] = transitions.get((prev_mode, next_mode), 0) + 1

    total_transitions = max(sum(transitions.values()), 1)
    edges = [
        TransitionEdge(
            source_mode_id=source,
            target_mode_id=target,
            support=count / total_transitions,
            count=count,
        )
        for (source, target), count in sorted(transitions.items())
    ]
    return ModeProfile(
        modes=modes,
        transitions=edges,
        feature_keys=feature_keys,
        mode_ids=[str(mode_id) for mode_id in mode_ids.tolist()],
        stable=len(unique_modes) > 1,
    )


@dataclass
class ModeEstimate:
    mode_id: str
    confidence: float
    distances: dict[str, float]


class ModeEstimator:
    def __init__(self, profile: ModeProfile):
        self.profile = profile

    def estimate(self, current_state: np.ndarray) -> ModeEstimate:
        if not self.profile.modes:
            return ModeEstimate(mode_id="unknown", confidence=0.0, distances={})

        current_state = np.asarray(current_state, dtype=np.float64)
        distances: dict[str, float] = {}
        for mode in self.profile.modes:
            centroid = np.asarray(mode.centroid_state, dtype=np.float64)
            distances[mode.mode_id] = float(np.linalg.norm(current_state - centroid))

        best_mode_id = min(distances, key=distances.get)
        sorted_distances = sorted(distances.values())
        best = sorted_distances[0]
        second = sorted_distances[1] if len(sorted_distances) > 1 else best + 1.0
        confidence = 1.0 if second <= 0 else float(max(0.0, 1.0 - best / (second + 1e-6)))
        return ModeEstimate(mode_id=best_mode_id, confidence=confidence, distances=distances)
