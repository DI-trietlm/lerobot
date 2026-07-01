from __future__ import annotations

from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.vla_harness.profile import HarnessProfileMiner, HarnessProfileMinerConfig


@dataclass
class BuildHarnessProfileConfig:
    dataset: HarnessProfileMinerConfig


@draccus.wrap()
def main(cfg: BuildHarnessProfileConfig) -> None:
    print(pformat(asdict(cfg)))
    bundle = HarnessProfileMiner(cfg.dataset).export()
    print(
        pformat(
            {
                "profile_path": f"{cfg.dataset.output_dir}/harness_profile.json",
                "num_modes": len(bundle.profile.mode_profile.modes),
                "num_invariants": len(bundle.profile.invariants),
                "num_rescue_entries": bundle.profile.rescue_index.num_entries
                if bundle.profile.rescue_index
                else 0,
            }
        )
    )


if __name__ == "__main__":
    main()
