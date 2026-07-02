from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np

from .config import HarnessConfig
from .envelopes import ActionEnvelopeGuard, EnvelopeViolation
from .invariants import InvariantGuard, InvariantViolation
from .mode import ModeEstimator
from .profile import HarnessProfileBundle
from .protocol import ActionChunkEnvelope, HarnessDecision, InterventionEvent, PolicyMetadata
from .rescue import MicroRescuePlanner
from .runtime import FlushCoordinator, HarnessRuntimeState, InterventionLedger
from .trace import HarnessTraceWriter


class ServerHarnessController:
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
        self.rescue_planner = (
            MicroRescuePlanner(bundle.rescue_index, cfg, fps=profile.fps)
            if bundle is not None and profile is not None
            else None
        )
        self._mode_history: list[str] = []

    def close(self) -> None:
        if self.trace is not None:
            self.trace.close()

    def register_intervention(self, event: InterventionEvent) -> None:
        self.ledger.append(event)
        if event.severity != "shadow" and self.flush.intervention_requires_flush(event):
            self.runtime.invalidate_chunk(event.chunk_id)
        if (
            event.requires_reinfer
            and self.cfg.effective_enabled(self.cfg.server.enable)
            and self.cfg.server.micro_rescue_proposal_enable
        ):
            self.runtime.pending_rescue = event.reason in {"stuck_candidate", "tracking_error"}
        if self.trace is not None:
            self.trace.write(
                {
                    "timestamp": event.timestamp,
                    "episode_id": "runtime",
                    "chunk_id": event.chunk_id,
                    "event_type": "intervention",
                    "current_state": event.current_state,
                    "raw_action": event.original_action,
                    "postprocessed_action": event.executed_action,
                    "executed_action": event.executed_action,
                    "mode_estimate": None,
                    "violations": [event.reason],
                    "rescue": event.metadata,
                }
            )

    def maybe_replace_with_rescue(self, current_state: np.ndarray) -> tuple[np.ndarray | None, dict | None]:
        if (
            not self.cfg.effective_enabled(self.cfg.server.enable)
            or not self.cfg.server.micro_rescue_proposal_enable
            or not self.runtime.pending_rescue
            or self.rescue_planner is None
        ):
            return None, None
        decision = self.rescue_planner.query(current_state, now_s=time.time())
        self.runtime.pending_rescue = False
        if not decision.accepted or decision.snippet is None:
            return None, {"reason": decision.reason, "severity": decision.severity}
        if decision.severity == "shadow":
            return None, {
                "reason": decision.reason,
                "severity": decision.severity,
                "would_rescue": True,
                **(decision.metadata or {}),
            }
        return decision.snippet, {
            "reason": decision.reason,
            "severity": decision.severity,
            **(decision.metadata or {}),
        }

    def build_envelope(
        self,
        *,
        current_state: np.ndarray,
        action_chunk: np.ndarray,
        timestamp: float,
        policy_metadata: PolicyMetadata,
        raw_chunk_ref: str | None = None,
        resample_count: int = 0,
    ) -> tuple[np.ndarray, ActionChunkEnvelope, list[InvariantViolation], list[EnvelopeViolation]]:
        current_state = np.asarray(current_state, dtype=np.float64)
        candidate = np.asarray(action_chunk, dtype=np.float64)
        validators_enabled = self.cfg.effective_enabled(self.cfg.server.enable) and self.cfg.server.chunk_validator_enable
        mode_estimate = (
            self.mode_estimator.estimate(current_state)
            if validators_enabled and self.mode_estimator is not None
            else None
        )
        invariant_violations = (
            self.invariant_guard.evaluate_chunk(
                current_state=current_state,
                action_chunk=candidate,
                mode_estimate=mode_estimate,
                mode_history=self._mode_history,
            )
            if (
                validators_enabled
                and self.cfg.server.invariant_guard_enable
                and self.invariant_guard is not None
                and mode_estimate is not None
            )
            else []
        )
        candidate, envelope_violations = (
            self.envelope_guard.evaluate(
                current_state=current_state,
                action_chunk=candidate,
                mode_id=mode_estimate.mode_id if mode_estimate is not None else None,
            )
            if validators_enabled and self.envelope_guard is not None
            else (candidate, [])
        )
        hard_violations = [violation for violation in [*invariant_violations, *envelope_violations] if violation.severity == "hard"]
        server_valid = not hard_violations
        if hard_violations and self.cfg.server.reject_resample_enable and not self.cfg.shadow_mode:
            candidate = np.repeat(current_state[None, :], repeats=len(candidate), axis=0)

        decision = HarnessDecision(
            server_valid=server_valid,
            server_shadow_violations=[
                asdict(violation)
                for violation in [*invariant_violations, *envelope_violations]
                if violation.severity == "shadow"
            ],
            resample_count=resample_count,
            intervention_required=bool(hard_violations),
            rescue_suggested=any(violation.reason == "repeated_speed_clamp_requires_flush" for violation in envelope_violations),
        )
        envelope = ActionChunkEnvelope.new(
            timestamp=timestamp,
            policy_metadata=policy_metadata,
            postprocessed_actions=candidate.tolist(),
        )
        envelope.harness_decision = decision
        envelope.raw_chunk_ref = raw_chunk_ref
        self.runtime.register_chunk(envelope)
        if mode_estimate is not None:
            self._mode_history.append(mode_estimate.mode_id)
            self._mode_history = self._mode_history[-64:]

        if self.trace is not None:
            self.trace.write(
                {
                    "timestamp": timestamp,
                    "episode_id": "runtime",
                    "chunk_id": envelope.chunk_id,
                    "event_type": "validate",
                    "current_state": current_state.tolist(),
                    "raw_action": action_chunk.tolist(),
                    "postprocessed_action": candidate.tolist(),
                    "executed_action": None,
                    "mode_estimate": asdict(mode_estimate) if mode_estimate is not None else None,
                    "violations": [
                        asdict(violation) for violation in [*invariant_violations, *envelope_violations]
                    ],
                    "rescue": None,
                }
            )
        return candidate, envelope, invariant_violations, envelope_violations
