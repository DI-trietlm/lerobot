from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class PolicyMetadata:
    policy_id: str
    policy_revision: str | None
    profile_id: str | None = None


@dataclass
class HarnessDecision:
    server_valid: bool = True
    server_shadow_violations: list[dict[str, Any]] = field(default_factory=list)
    resample_count: int = 0
    intervention_required: bool = False
    rescue_suggested: bool = False


@dataclass
class ActionChunkEnvelope:
    chunk_id: str
    inference_id: str
    timestamp: float
    policy_metadata: PolicyMetadata
    postprocessed_actions: list[list[float]]
    harness_decision: HarnessDecision = field(default_factory=HarnessDecision)
    raw_chunk_ref: str | None = None

    @classmethod
    def new(
        cls,
        timestamp: float,
        policy_metadata: PolicyMetadata,
        postprocessed_actions: list[list[float]],
        chunk_id: str | None = None,
        inference_id: str | None = None,
    ) -> "ActionChunkEnvelope":
        return cls(
            chunk_id=chunk_id or str(uuid.uuid4()),
            inference_id=inference_id or str(uuid.uuid4()),
            timestamp=timestamp,
            policy_metadata=policy_metadata,
            postprocessed_actions=postprocessed_actions,
        )


@dataclass
class InterventionEvent:
    event_id: str
    chunk_id: str
    inference_id: str
    timestamp: float
    component: str
    severity: str
    reason: str
    original_action: list[float]
    executed_action: list[float]
    current_state: list[float]
    queue_cleared: bool
    requires_reinfer: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        chunk_id: str,
        inference_id: str,
        timestamp: float,
        component: str,
        severity: str,
        reason: str,
        original_action: list[float],
        executed_action: list[float],
        current_state: list[float],
        queue_cleared: bool,
        requires_reinfer: bool,
        metadata: dict[str, Any] | None = None,
    ) -> "InterventionEvent":
        return cls(
            event_id=str(uuid.uuid4()),
            chunk_id=chunk_id,
            inference_id=inference_id,
            timestamp=timestamp,
            component=component,
            severity=severity,
            reason=reason,
            original_action=original_action,
            executed_action=executed_action,
            current_state=current_state,
            queue_cleared=queue_cleared,
            requires_reinfer=requires_reinfer,
            metadata=metadata or {},
        )


@dataclass
class ReinferRequest:
    chunk_id: str
    inference_id: str
    timestamp: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class HarnessMessageCodec:
    @staticmethod
    def encode_intervention(event: InterventionEvent) -> bytes:
        return json.dumps(_jsonable(asdict(event)), separators=(",", ":")).encode("utf-8")

    @staticmethod
    def decode_intervention(payload: bytes) -> InterventionEvent:
        data = json.loads(payload.decode("utf-8"))
        return InterventionEvent(**data)

    @staticmethod
    def encode_json(payload: dict[str, Any]) -> str:
        return json.dumps(_jsonable(payload), separators=(",", ":"))

    @staticmethod
    def decode_json(payload: str) -> dict[str, Any]:
        return json.loads(payload)
