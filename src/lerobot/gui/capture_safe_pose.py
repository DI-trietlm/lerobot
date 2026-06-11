# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture a SAFE rest pose for the RTC GUI's auto-reset cycle.

The "safe pose" is a mechanically stable configuration the arm can hold once torque
is removed at disconnect (i.e. it will NOT free-fall). The GUI moves the arm here
*before* disconnecting during its auto-reset cycle.

Usage (run in your own terminal, it is interactive):

    uv run python -m lerobot.gui.capture_safe_pose --config_path rtc_xvla_config.json
    uv run python -m lerobot.gui.capture_safe_pose --config_path rtc_xvla_config.json --output safe_pose.json

It reuses the robot block of an exported RTC config (Export Config in the GUI),
disables torque so you can move the arm by hand, then records its joint positions.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import draccus

from lerobot.robots import (  # noqa: F401  -- import side-effects register robot choice classes
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins


def capture(config_path: str, output: str) -> dict:
    register_third_party_plugins()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if "robot" not in payload:
        raise ValueError(f"{config_path} has no 'robot' block (export it from the RTC GUI).")

    robot_payload = dict(payload["robot"])
    robot_payload["cameras"] = {}  # cameras are irrelevant for a pose capture
    robot_cfg = draccus.decode(RobotConfig, robot_payload)

    robot = make_robot_from_config(robot_cfg)
    print(f"Connecting to {robot_cfg.type} on {getattr(robot_cfg, 'port', '?')} ...")
    robot.connect(calibrate=False)
    try:
        robot.bus.disable_torque()
        print("\nTorque DISABLED. Move the arm BY HAND to a SAFE rest pose")
        print("(low/folded enough that it holds itself when torque is off).")
        input("Press ENTER to record the current pose... ")

        obs = robot.get_observation()
        pose = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        if not pose:
            raise RuntimeError(f"No '*.pos' joints found in observation keys: {list(obs)}")

        data = {
            "safe_pose": pose,
            "robot_type": robot_cfg.type,
            "robot_id": getattr(robot_cfg, "id", None),
            "use_degrees": bool(getattr(robot_cfg, "use_degrees", True)),
            "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_config": config_path,
        }
        Path(output).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nSaved safe pose to {output}:")
        for k, v in pose.items():
            print(f"  {k} = {v:.2f}")
        return data
    finally:
        robot.disconnect()


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture a safe rest pose for the RTC GUI auto-reset cycle.")
    ap.add_argument("--config_path", required=True, help="Exported RTC config JSON (uses its 'robot' block).")
    ap.add_argument("--output", default="safe_pose.json", help="Where to write the safe-pose file.")
    args = ap.parse_args()
    capture(args.config_path, args.output)


if __name__ == "__main__":
    main()
