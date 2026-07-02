import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.vla_harness.trace import HarnessTraceReader


@dataclass
class ReplayHarnessProfileConfig:
    trace_path: str


@draccus.wrap()
def main(cfg: ReplayHarnessProfileConfig) -> None:
    reader = HarnessTraceReader(Path(cfg.trace_path))
    summary = {
        "events": 0,
        "interventions": 0,
        "violations": 0,
        "event_types": {},
    }
    for payload in reader.read():
        summary["events"] += 1
        summary["event_types"][payload["event_type"]] = (
            summary["event_types"].get(payload["event_type"], 0) + 1
        )
        summary["violations"] += len(payload["violations"])
        if payload["event_type"] == "intervention":
            summary["interventions"] += 1

    print(pformat(summary))
    output_path = Path(cfg.trace_path).with_suffix(".summary.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
