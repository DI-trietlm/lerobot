from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TRACE_REQUIRED_FIELDS = {
    "timestamp",
    "episode_id",
    "chunk_id",
    "event_type",
    "current_state",
    "raw_action",
    "postprocessed_action",
    "executed_action",
    "mode_estimate",
    "violations",
    "rescue",
}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class TraceSchemaValidator:
    def validate(self, payload: dict[str, Any]) -> None:
        missing = TRACE_REQUIRED_FIELDS - set(payload)
        if missing:
            raise ValueError(f"Trace payload missing required fields: {sorted(missing)}")


class HarnessTraceWriter:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.file_path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._validator = TraceSchemaValidator()

    def write(self, payload: dict[str, Any]) -> None:
        self._validator.validate(payload)
        with self._lock:
            self._handle.write(json.dumps(_jsonable(payload), separators=(",", ":")) + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.flush()
            self._handle.close()


class HarnessTraceReader:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def read(self) -> Iterable[dict[str, Any]]:
        validator = TraceSchemaValidator()
        with self.file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                validator.validate(payload)
                yield payload
