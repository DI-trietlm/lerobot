from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.vla_harness.profile import load_harness_profile


@dataclass
class InspectHarnessProfileConfig:
    profile_path: str


@draccus.wrap()
def main(cfg: InspectHarnessProfileConfig) -> None:
    bundle = load_harness_profile(Path(cfg.profile_path))
    print(
        pformat(
            {
                "dataset_repo_id": bundle.profile.dataset_repo_id,
                "dataset_revision": bundle.profile.dataset_revision,
                "fps": bundle.profile.fps,
                "state_keys": bundle.profile.state_keys,
                "action_keys": bundle.profile.action_keys,
                "mode_count": len(bundle.profile.mode_profile.modes),
                "invariant_count": len(bundle.profile.invariants),
                "rescue_entries": bundle.profile.rescue_index.num_entries
                if bundle.profile.rescue_index
                else 0,
            }
        )
    )


if __name__ == "__main__":
    main()
