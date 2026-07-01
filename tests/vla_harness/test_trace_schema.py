from __future__ import annotations

import pytest

from lerobot.vla_harness.trace import HarnessTraceReader, HarnessTraceWriter, TraceSchemaValidator


def test_trace_round_trip(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    writer = HarnessTraceWriter(trace_path)
    payload = {
        "timestamp": 1.0,
        "episode_id": "episode-1",
        "chunk_id": "chunk-1",
        "event_type": "execute",
        "current_state": [0.0],
        "raw_action": [0.1],
        "postprocessed_action": [0.1],
        "executed_action": [0.1],
        "mode_estimate": None,
        "violations": [],
        "rescue": None,
    }
    writer.write(payload)
    writer.close()

    rows = list(HarnessTraceReader(trace_path).read())
    assert rows == [payload]


def test_trace_validator_rejects_missing_fields():
    validator = TraceSchemaValidator()
    with pytest.raises(ValueError):
        validator.validate({"timestamp": 1.0})
