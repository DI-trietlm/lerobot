# Runtime Trace-Sync Analysis 2026-07-02

## Files Analyzed

Local trace-sync run:

- `recorded_obs-harness-trace-sync/metadata.jsonl`
- `recorded_obs-harness-trace-sync/client_actions.jsonl`
- `recorded_obs-harness-trace-sync/images/camera1/*.png`
- `recorded_obs-harness-trace-sync/images/camera2/*.png`
- `harness_traces/trace_sync/client_trace.jsonl`

Reference runs:

- `recorded_obs-0629-01`
- `recorded_obs-0629-02`
- `recorded_obs-0629-nonrtc`
- `recorded_obs-0629-rtc2`

Generated artifacts:

- `xai/runtime_trace_sync_0702_keyframes.jpg`
- `xai/runtime_trace_sync_0702_analysis/summary.csv`
- `xai/runtime_trace_sync_0702_analysis/summary.json`
- `xai/runtime_trace_sync_0702_analysis/dist_from_start.png`
- `xai/runtime_trace_sync_0702_analysis/dist_to_safe.png`

Server files were not fetched because non-interactive SSH failed with `Permission denied (publickey,password)`. Expected server-side files:

- `/home/trietlm/lerobot/harness_traces/trace_sync/server_trace.jsonl`
- `/home/trietlm/lerobot/recorded_obs-harness-trace-sync/server_actions.jsonl`
- `/home/trietlm/lerobot/logs/rtc_audit/policy_server_obs_interarrival_1782957514.csv`
- `/home/trietlm/lerobot/logs/rtc_audit/policy_server_queue_latency_1782957514.csv`

## Main Conclusion

This run is not the same failure as "policy drives back to safe pose".

The 2026-07-02 trace-sync run is a **start-pose basin / start-loop failure**:

- The robot does move in joint space, but it never enters the strong approach/safe-like trajectory that the 2026-06-29 runs entered.
- After 45.1s, the robot ends only `8.91 deg` L2 arm distance from the initial state.
- The closest it ever gets to the saved safe pose is still `95.14 deg` away.
- The action targets themselves also never get close to safe pose: closest target-to-safe is `93.99 deg`.

In contrast, the older runs all moved much farther away from start and much closer to the safe/approach basin:

| Run | Duration | Max State Dist From Start | Last State Dist From Start | Min State Dist To Safe | Min Target Dist To Safe |
|---|---:|---:|---:|---:|---:|
| 0702 trace-sync | 45.10s | 63.29 | 8.91 | 95.14 | 93.99 |
| 0629 nonrtc | 15.32s | 137.92 | 134.75 | 19.70 | 15.03 |
| 0629 rtc2 | 10.94s | 143.16 | 133.16 | 9.25 | 8.66 |
| 0629 run01 | 22.95s | 129.97 | 129.97 | 29.31 | 24.48 |
| 0629 run02 | 15.20s | 134.96 | 130.05 | 18.95 | 12.81 |

## Trace-Sync Specific Findings

`harness_traces/trace_sync/client_trace.jsonl` contains:

- `663` trace rows.
- All rows are `event_type="execute"`.
- `violations=[]` for every row.
- `rescue=null` for every row.
- `mode_estimate=null` for every row.
- `chunk_id="unknown"` for every row.

`recorded_obs-harness-trace-sync/client_actions.jsonl` contains:

- `81` `chunk_received` events.
- `663` `action_executed` events.
- Median server-to-client latency: `257.64 ms`.
- Max server-to-client latency: `696.23 ms`.
- Queue size median after chunk update: about `28`.

Important sync observation:

- All sampled action metadata has `chunk_id=null`.
- All sampled action metadata has `inference_id=null`.
- This is also true for the older runs.

So the current `trace_sync` configuration is useful for logging, but the runtime action payload still does not carry a real chunk/inference identity. That means `require_chunk_id=true` is not yet backed by a complete identity handshake in the observed logs.

## Image/State Sync Check

The new run has:

- `80` unique metadata timesteps.
- `80` images in `camera1`.
- `80` images in `camera2`.
- No missing image for any metadata timestep.
- No extra image outside metadata timesteps.

There are duplicate metadata rows for timestep `0`, but image-state mapping by unique timestep is consistent.

## Visual Check

`xai/runtime_trace_sync_0702_keyframes.jpg` shows the same qualitative pattern as the numeric trace:

- The 2026-07-02 trace-sync run remains around the start/preparation region.
- The 2026-06-29 non-RTC/RTC2/reference runs move toward the cup region within the first several seconds.
- The new run does not reproduce that transition even after a much longer runtime.

## Interpretation

Current evidence points away from these explanations:

- Not primarily RTC pulling the robot back. The non-RTC and RTC2 old runs both entered the approach/safe-like basin; the new trace-sync run did not.
- Not primarily client-side harness intervention. Harness trace shows no violations, no rescue, and no postprocessed-vs-executed difference.
- Not an image-state file mismatch. Unique timesteps match images in both cameras.
- Not a server-to-client action corruption issue. `81/81` server-generated chunk signatures match the client-received chunk signatures.

Current evidence points toward:

- Raw policy instability / multimodal action selection around the initial observation.
- The policy sometimes selects a trajectory family that performs small preparatory motions but never commits to the approach/grasp phase.
- The new run is a stronger example of the "dead point near start pose" that micro-rescue is meant to escape.

## Server Action Addendum

Server-side `server_actions.jsonl` was fetched and analyzed after the first local-only pass.

There is no `harness_traces/trace_sync/server_trace.jsonl` on the server because the server was started with `harness.enable=false`. The server still recorded `server_actions.jsonl`, so raw policy/chunk behavior can be analyzed.

Key result:

- Server generated `81` chunks.
- Client received `81` chunks.
- All `81` server chunk action signatures match the client-received chunk signatures.
- Therefore the client received exactly what the server sent.

The important difference is **where the useful actions are inside each chunk**.

For the new 2026-07-02 trace-sync run:

| Action Set | Count | Max Dist From Start | Min Dist To Safe | Count Dist From Start > 80 | Count Dist To Safe < 80 |
|---|---:|---:|---:|---:|---:|
| Server all sent actions | 2588 | 103.2 | 55.3 | 23 | 5 |
| Server head-1 actions only | 81 | 58.0 | 100.8 | 0 | 0 |
| Server head-3 actions only | 243 | 58.0 | 100.8 | 0 | 0 |
| Client executed actions | 663 | 65.3 | 94.0 | 0 | 0 |

So the server occasionally predicts a future action that starts moving toward the safe/approach basin, but those actions appear very late in the predicted horizon.

Example:

- First server action with `dist_to_safe < 80`: chunk `43`, horizon index `27`.
- Closest server action to safe: chunk `43`, horizon index `31`, `dist_to_safe=55.3`.
- Closest client-executed action to safe: `dist_to_safe=94.0`.

The client never executes those late-horizon candidates because the loop keeps receiving newer chunks and executing/replacing only the head region. This makes the runtime behavior a local start-pose loop even though a few late-horizon server predictions contain partial escape candidates.

Comparison with older runs:

| Run | Server Head-1 Count Dist From Start > 80 | Client Executed Count Dist From Start > 80 | Client Executed Min Dist To Safe |
|---|---:|---:|---:|
| 0702 trace-sync | 0 / 81 | 0 / 663 | 94.0 |
| 0629 nonrtc | 16 / 21 | 173 / 222 | 15.0 |
| 0629 rtc2 | 13 / 19 | 116 / 153 | 8.7 |
| 0629 run01 | 18 / 40 | 132 / 336 | 24.5 |
| 0629 run02 | 20 / 27 | 182 / 238 | 12.8 |

This is the clearest evidence so far:

- In old runs, the **head of the server chunks** already contained committed approach/safe-like actions.
- In the new run, the **head of every server chunk** stays in the start/preparation basin.
- Therefore the observed failure is upstream in raw policy chunk generation / receding-horizon commitment, not in transport, client deserialization, or basic queue execution.

## Remaining Gap

The available evidence is enough to rule out server-to-client action corruption for this run.

The remaining missing artifact is `harness_traces/trace_sync/server_trace.jsonl`, but it was not created because the server process was running with `harness.enable=false`. To get server harness trace in a future run, the server must be started with a matching harness-enabled server config, or the protocol must explicitly send/merge harness config from client to server.
