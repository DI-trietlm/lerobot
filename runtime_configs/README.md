# Runtime Configs

This folder keeps local runtime JSON files out of the repository root.

- `rtc_*.json` files are GUI/orchestrator configs for robot inference probes.
- `safe_pose.json` is the captured safe reset pose used by the RTC GUI auto-reset cycle.

Typical commands:

```bash
uv run python -m lerobot.gui.capture_safe_pose --config_path runtime_configs/rtc_xvla_config.json
uv run python scripts/orchestrator/orchestrator_rtc_client_only.py --config_path runtime_configs/rtc_smolvla_3.json
```
