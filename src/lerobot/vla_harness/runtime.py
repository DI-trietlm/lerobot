from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import HarnessConfig
from .protocol import ActionChunkEnvelope, InterventionEvent, ReinferRequest


@dataclass
class HarnessRuntimeState:
    active_chunks: dict[str, ActionChunkEnvelope] = field(default_factory=dict)
    invalidated_chunk_ids: set[str] = field(default_factory=set)
    executed_chunk_ids: set[str] = field(default_factory=set)
    blocked_until_new_chunk: bool = False
    pending_reinfer: ReinferRequest | None = None
    pending_rescue: bool = False

    def register_chunk(self, envelope: ActionChunkEnvelope) -> None:
        self.active_chunks[envelope.chunk_id] = envelope
        self.blocked_until_new_chunk = False

    def invalidate_chunk(self, chunk_id: str) -> None:
        self.invalidated_chunk_ids.add(chunk_id)
        self.active_chunks.pop(chunk_id, None)
        self.blocked_until_new_chunk = True

    def mark_executed(self, chunk_id: str) -> None:
        self.executed_chunk_ids.add(chunk_id)

    def is_invalidated(self, chunk_id: str) -> bool:
        return chunk_id in self.invalidated_chunk_ids


@dataclass
class InterventionLedger:
    events: list[InterventionEvent] = field(default_factory=list)

    def append(self, event: InterventionEvent) -> None:
        self.events.append(event)

    def summary(self) -> dict[str, Any]:
        summary: dict[str, int] = {}
        for event in self.events:
            key = f"{event.component}:{event.reason}:{event.severity}"
            summary[key] = summary.get(key, 0) + 1
        return summary


class FlushCoordinator:
    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg

    def intervention_requires_flush(self, event: InterventionEvent) -> bool:
        if not self.cfg.sync.enable:
            return self.cfg.fail_closed and event.severity in {"hard", "emergency"}
        if event.reason == "micro_rescue":
            return self.cfg.sync.flush_on_rescue
        if event.reason == "speed_clamp":
            return self.cfg.sync.flush_on_repeated_speed_clamp
        if event.severity in {"hard", "emergency"}:
            return self.cfg.sync.flush_on_hard_clamp
        return event.queue_cleared
