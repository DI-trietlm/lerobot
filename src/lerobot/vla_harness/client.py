from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict
from typing import Any

import numpy as np

from .config import HarnessConfig
from .envelopes import ActionEnvelopeGuard, EnvelopeViolation
from .invariants import InvariantGuard, InvariantViolation
from .mode import ModeEstimator
from .profile import HarnessProfileBundle
from .protocol import InterventionEvent
from .runtime import FlushCoordinator, HarnessRuntimeState, InterventionLedger
from .trace import HarnessTraceWriter


class ClientHarnessController:
    def __init__(self, cfg: HarnessConfig, bundle: HarnessProfileBundle | None, trace_path: str | None = None):
        self.cfg = cfg
        self.bundle = bundle
        self.runtime = HarnessRuntimeState()
        self.ledger = InterventionLedger()
        self.flush = FlushCoordinator(cfg)
        self.trace = HarnessTraceWriter(trace_path) if trace_path else None
        profile = bundle.profile if bundle is not None else None
        self.mode_estimator = ModeEstimator(profile.mode_profile) if profile is not None else None
        self.invariant_guard = InvariantGuard(profile.invariants, cfg) if profile is not None else None
        self.envelope_guard = (
            ActionEnvelopeGuard(profile.speed_envelopes, cfg) if profile is not None else None
        )
        self._mode_history: list[str] = []
        self._state_history: deque[np.ndarray] = deque(maxlen=16)
        self._current_chunk_id: str | None = None
        self._current_inference_id: str | None = None
        self._blocked_chunk_id: str | None = None

    def close(self) -> None:
        if self.trace is not None:
            self.trace.close()

    @property
    def execution_blocked(self) -> bool:
        return self.runtime.blocked_until_new_chunk

    def on_chunk_received(self, chunk_id: str | None, inference_id: str | None) -> None:
        if chunk_id is None:
            return
        self._current_chunk_id = chunk_id
        self._current_inference_id = inference_id
        if self._blocked_chunk_id is None or self._blocked_chunk_id != chunk_id:
            self.runtime.blocked_until_new_chunk = False

    def observe_state(self, current_state: np.ndarray) -> InterventionEvent | None:
        current_state = np.asarray(current_state, dtype=np.float64)
        self._state_history.append(current_state.copy())
        if not self.cfg.effective_enabled(self.cfg.client.enable):
            return None
        if not self.cfg.client.tracking_monitor_enable:
            return None
        if len(self._state_history) < self._state_history.maxlen:
            return None
        displacement = float(np.linalg.norm(self._state_history[-1] - self._state_history[0]))
        scale = float(np.linalg.norm(np.std(np.stack(self._state_history, axis=0), axis=0)))
        stuck = displacement < max(1e-3, 0.1 * scale)
        if not stuck or self._current_chunk_id is None or self._current_inference_id is None:
            return None
        if self.cfg.shadow_mode:
            self._trace_intervention_like(
                severity="shadow",
                reason="stuck_candidate",
                current_state=current_state,
                original_action=current_state,
                executed_action=current_state,
                metadata={"displacement": displacement, "scale": scale, "would_flush": True},
            )
            return None
        return self._make_event(
            severity="soft",
            reason="stuck_candidate",
            original_action=current_state,
            executed_action=current_state,
            current_state=current_state,
            metadata={"displacement": displacement, "scale": scale},
        )

    def evaluate_action(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        meta: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, InterventionEvent | None, list[InvariantViolation], list[EnvelopeViolation]]:
        current_state = np.asarray(current_state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        if not self.cfg.effective_enabled(self.cfg.client.enable) or not self.cfg.client.execution_guard_enable:
            return action, None, [], []
        chunk = action[None, :]
        mode_estimate = (
            self.mode_estimator.estimate(current_state)
            if self.mode_estimator is not None
            else None
        )
        invariant_violations = (
            self.invariant_guard.evaluate_chunk(
                current_state=current_state,
                action_chunk=chunk,
                mode_estimate=mode_estimate,
                mode_history=self._mode_history,
            )
            if self.cfg.client.hard_invariant_guard_enable
            and self.invariant_guard is not None
            and mode_estimate is not None
            else []
        )
        adjusted_chunk, envelope_violations = (
            self.envelope_guard.evaluate(
                current_state=current_state,
                action_chunk=chunk,
                mode_id=mode_estimate.mode_id if mode_estimate is not None else None,
            )
            if self.cfg.client.speed_envelope_enable and self.envelope_guard is not None
            else (chunk, [])
        )
        adjusted = adjusted_chunk[0]

        event = None
        all_violations = [*invariant_violations, *envelope_violations]
        hard_violation = next((violation for violation in all_violations if violation.severity == "hard"), None)
        if hard_violation is not None and self._current_chunk_id is not None and self._current_inference_id is not None:
            if isinstance(hard_violation, InvariantViolation):
                adjusted = current_state.copy()
            event = self._make_event(
                severity=hard_violation.severity,
                reason="invariant_violation" if isinstance(hard_violation, InvariantViolation) else "speed_clamp",
                original_action=action,
                executed_action=adjusted,
                current_state=current_state,
                metadata={"violation": asdict(hard_violation), "timed_action_meta": meta or {}},
            )

        if mode_estimate is not None:
            self._mode_history.append(mode_estimate.mode_id)
            self._mode_history = self._mode_history[-64:]

        if self.trace is not None:
            self.trace.write(
                {
                    "timestamp": time.time(),
                    "episode_id": "runtime",
                    "chunk_id": self._current_chunk_id or "unknown",
                    "event_type": "execute",
                    "current_state": current_state.tolist(),
                    "raw_action": action.tolist(),
                    "postprocessed_action": adjusted.tolist(),
                    "executed_action": adjusted.tolist(),
                    "mode_estimate": asdict(mode_estimate) if mode_estimate is not None else None,
                    "violations": [asdict(violation) for violation in all_violations],
                    "rescue": None,
                }
            )
        return adjusted, event, invariant_violations, envelope_violations

    def register_intervention(self, event: InterventionEvent) -> None:
        self.ledger.append(event)
        if event.severity != "shadow" and self.flush.intervention_requires_flush(event):
            self.runtime.invalidate_chunk(event.chunk_id)
            self._blocked_chunk_id = event.chunk_id

    def _make_event(
        self,
        *,
        severity: str,
        reason: str,
        original_action: np.ndarray,
        executed_action: np.ndarray,
        current_state: np.ndarray,
        metadata: dict[str, Any],
    ) -> InterventionEvent:
        event = InterventionEvent.create(
            chunk_id=self._current_chunk_id or "unknown",
            inference_id=self._current_inference_id or "unknown",
            timestamp=time.time(),
            component="client.execution_guard",
            severity=severity,
            reason=reason,
            original_action=np.asarray(original_action).tolist(),
            executed_action=np.asarray(executed_action).tolist(),
            current_state=np.asarray(current_state).tolist(),
            queue_cleared=self.cfg.client.clear_queue_on_intervention,
            requires_reinfer=self.cfg.client.request_reinfer_on_intervention,
            metadata=metadata,
        )
        self.register_intervention(event)
        return event

    def _trace_intervention_like(
        self,
        *,
        severity: str,
        reason: str,
        original_action: np.ndarray,
        executed_action: np.ndarray,
        current_state: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        if self.trace is None:
            return
        self.trace.write(
            {
                "timestamp": time.time(),
                "episode_id": "runtime",
                "chunk_id": self._current_chunk_id or "unknown",
                "event_type": "intervention_shadow",
                "current_state": np.asarray(current_state).tolist(),
                "raw_action": np.asarray(original_action).tolist(),
                "postprocessed_action": np.asarray(executed_action).tolist(),
                "executed_action": np.asarray(executed_action).tolist(),
                "mode_estimate": None,
                "violations": [{"severity": severity, "reason": reason}],
                "rescue": metadata,
            }
        )
