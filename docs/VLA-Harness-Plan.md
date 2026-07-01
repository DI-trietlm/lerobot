# VLA Harness Implementation Plan

Mục tiêu: implement đầy đủ VLA Harness theo kiến trúc hybrid server/client trong
`docs/VLA-Harness.md`, không đơn giản hóa thành một stuck detector nhỏ. Tất cả
thành phần phải bật/tắt độc lập bằng config để có thể rollout theo từng mức rủi
ro.

Harness cuối cùng gồm bốn nhóm:

1. Dataset-Manifold Micro-Rescue.
2. Data-Derived Invariant Guard.
3. Action Stability And Speed Envelope Guard.
4. Runtime Synchronization, Re-Infer, And Trace Harness.

---

## 1. Non-Goals Và Guardrails

Không làm:

- Không thay VLA bằng planner/controller cổ điển.
- Không hard-code logic riêng cho pouring làm mặc định.
- Không cấm tuyệt đối các pose gần safe/folded pose, vì đó có thể là waypoint
  hợp lệ.
- Không clamp action liên tục mà server không biết.
- Không để client can thiệp mạnh mà không flush/re-infer.

Phải làm:

- Mọi harness component có thể bật/tắt độc lập.
- Mọi intervention có trace đầy đủ.
- Mọi intervention đáng kể phải invalidate queue/chunk và re-infer từ
  observation mới.
- Offline profile phải sinh ra từ dataset, không từ assumption thủ công.

---

## 2. Kiến Trúc Tổng Thể

```text
Dataset
  -> Offline Harness Profile Miner
  -> harness_profile.json

Server runtime
  -> policy inference
  -> server-side harness validators/rescue proposal
  -> chunk_id + action_chunk + metadata

Client runtime
  -> receives chunk
  -> final execution guard
  -> RTC/interpolation/execution
  -> actual state + intervention events

Synchronization
  -> intervention_event
  -> queue clear
  -> server context invalidation
  -> fresh observation
  -> re-infer
```

Suggested package layout:

```text
src/lerobot/vla_harness/
  __init__.py
  config.py
  schemas.py
  profile.py
  mode.py
  invariants.py
  rescue.py
  envelopes.py
  runtime.py
  trace.py
  protocol.py
  server.py
  client.py

scripts/vla_harness/
  build_harness_profile.py
  inspect_harness_profile.py
  replay_harness_profile.py

tests/vla_harness/
  test_profile_miner.py
  test_mode_discovery.py
  test_invariant_guard.py
  test_micro_rescue.py
  test_speed_envelope.py
  test_protocol_flush_reinfer.py
  test_trace_schema.py
```

---

## 3. Config Design

Add one top-level config object to orchestrator/server/client configs:

```json
{
  "harness": {
    "enable": false,
    "profile_path": null,
    "shadow_mode": true,
    "fail_closed": false,
    "log_dir": "harness_traces",

    "server": {
      "enable": true,
      "chunk_validator_enable": true,
      "invariant_guard_enable": true,
      "micro_rescue_proposal_enable": true,
      "reject_resample_enable": true,
      "max_resample_attempts": 1,
      "re_infer_on_intervention": true
    },

    "client": {
      "enable": true,
      "execution_guard_enable": true,
      "hard_invariant_guard_enable": true,
      "speed_envelope_enable": true,
      "tracking_monitor_enable": true,
      "clear_queue_on_intervention": true,
      "request_reinfer_on_intervention": true
    },

    "micro_rescue": {
      "enable": true,
      "shadow_mode": true,
      "state_knn_enable": true,
      "image_knn_enable": false,
      "k_neighbors": 16,
      "snippet_horizon_steps": 8,
      "max_duration_s": 1.0,
      "blend_alpha": 1.0,
      "min_future_progress_score": 0.2,
      "max_state_distance": null,
      "cooldown_s": 2.0,
      "max_rescues_per_episode": 3
    },

    "invariant_guard": {
      "enable": true,
      "shadow_mode": true,
      "min_support": 0.95,
      "max_train_violation_rate": 0.02,
      "min_mode_confidence": 0.7,
      "hard_guard_categories": ["catastrophic_actuator_release"],
      "soft_guard_categories": ["value_envelope", "no_backtrack"],
      "flush_on_hard_guard": true
    },

    "speed_envelope": {
      "enable": true,
      "shadow_mode": true,
      "percentile_low": 0.005,
      "percentile_high": 0.995,
      "mode_conditioned": true,
      "max_consecutive_clamps": 3,
      "flush_after_repeated_clamp": true
    },

    "sync": {
      "enable": true,
      "require_chunk_id": true,
      "flush_on_reject": true,
      "flush_on_rescue": true,
      "flush_on_hard_clamp": true,
      "flush_on_repeated_speed_clamp": true,
      "block_execution_until_fresh_chunk": true
    },

    "trace": {
      "enable": true,
      "record_images": false,
      "record_raw_chunks": true,
      "record_postprocessed_chunks": true,
      "record_executed_actions": true,
      "record_mode_estimates": true,
      "record_rescue_neighbors": true
    }
  }
}
```

Rules:

- `harness.enable=false` disables everything.
- Each subcomponent has its own `enable`.
- Each risk-bearing component supports `shadow_mode`.
- Client hard safety may run even when server harness is off.
- If `sync.enable=false`, any component requiring flush/re-infer must refuse
  hard intervention unless `fail_closed=false` and it is explicitly in shadow.

---

## 4. Data Schemas

### 4.1 Harness Profile

`harness_profile.json`:

```json
{
  "schema_version": 1,
  "dataset_repo_id": "...",
  "dataset_revision": "...",
  "fps": 15,
  "state_keys": ["..."],
  "action_keys": ["..."],
  "scales": {
    "state": [],
    "action": []
  },
  "mode_profile": {},
  "invariants": [],
  "speed_envelopes": {},
  "rescue_index": {
    "type": "state_knn",
    "index_path": "rescue_index.npz",
    "frame_table_path": "rescue_frames.parquet"
  },
  "diagnostics": {
    "num_episodes": 0,
    "num_frames": 0,
    "miner_config": {}
  }
}
```

### 4.2 Runtime Chunk Message

Server -> client:

```json
{
  "chunk_id": "uuid-or-monotonic-id",
  "inference_id": "uuid",
  "timestamp": 0.0,
  "policy_metadata": {
    "policy_id": "...",
    "policy_revision": "...",
    "profile_id": "..."
  },
  "raw_chunk_ref": null,
  "postprocessed_actions": [],
  "harness_decision": {
    "server_valid": true,
    "server_shadow_violations": [],
    "resample_count": 0
  }
}
```

### 4.3 Intervention Event

Client -> server:

```json
{
  "event_id": "uuid",
  "chunk_id": "same-as-server",
  "inference_id": "same-as-server",
  "timestamp": 0.0,
  "component": "client.execution_guard",
  "severity": "shadow|soft|hard|emergency",
  "reason": "invariant_violation|speed_clamp|micro_rescue|tracking_error|stop",
  "original_action": [],
  "executed_action": [],
  "current_state": [],
  "queue_cleared": true,
  "requires_reinfer": true,
  "metadata": {}
}
```

### 4.4 Trace Event

Append-only JSONL:

```json
{
  "timestamp": 0.0,
  "episode_id": "...",
  "chunk_id": "...",
  "event_type": "infer|validate|execute|intervention|reinfer",
  "current_state": [],
  "raw_action": null,
  "postprocessed_action": null,
  "executed_action": null,
  "mode_estimate": null,
  "violations": [],
  "rescue": null
}
```

---

## 5. Offline Harness Profile Miner

Command:

```bash
uv run python scripts/vla_harness/build_harness_profile.py \
  --dataset.repo_id=di-techinnova/so-arm-101-pouring-0.3-cutted \
  --dataset.root=... \
  --output_dir=outputs/harness_profiles/pouring_0_3 \
  --fps=15
```

Pipeline:

1. Load LeRobot dataset parquet metadata and action/state arrays.
2. Validate schema, fps, episode/frame continuity.
3. Compute robust state/action scales.
4. Mine mode candidates:
   - state/action recent-history windows;
   - plateau/transition/excursion features;
   - clustering or segmentation.
5. Mine invariants:
   - duration support;
   - transition graph support;
   - value envelope support;
   - velocity envelope support;
   - train violation rate.
6. Build speed/action envelopes:
   - global;
   - mode-conditioned if modes are stable.
7. Build micro-rescue index:
   - frame embeddings from state;
   - optional image embedding;
   - future progress score;
   - action snippet metadata.
8. Export profile files and diagnostic report.

Outputs:

```text
profile/
  harness_profile.json
  rescue_index.npz
  rescue_frames.parquet
  invariant_report.csv
  mode_report.csv
  speed_envelope_report.csv
  profile_diagnostics.md
```

Acceptance checks:

- All episodes accounted for.
- Invariants list includes support and violation rate.
- Rescue index references valid dataset frames.
- Profile can be loaded without dataset present, except for optional debug.

---

## 6. Harness 1: Dataset-Manifold Micro-Rescue

Implementation modules:

- `rescue.py`
  - `RescueIndex`
  - `FutureProgressScorer`
  - `MicroRescuePlanner`
  - `MicroRescueDecision`

Offline:

- Compute future progress:
  - distance from current state after `N` steps;
  - mode transition within `N` steps;
  - action magnitude above low threshold;
  - optional task-specific proxy stored as extra signal, not required.
- Store action snippets:
  - `episode_index`;
  - `frame_index`;
  - `snippet_start`;
  - `snippet_end`;
  - normalized state embedding;
  - future progress score.

Runtime:

1. `StuckMonitor` raises `stuck_candidate`.
2. `MicroRescuePlanner.query(current_state, optional_image_embedding)`.
3. Filter candidates:
   - distance <= threshold;
   - future progress >= threshold;
   - not from forbidden/low-quality episode;
   - action snippet within safety envelope.
4. Select candidate:
   - highest future progress;
   - or weighted by state distance/progress.
5. Return rescue action snippet.
6. Client executes bounded snippet.
7. Sync harness flushes queue and re-infers.

Required toggles:

- `micro_rescue.enable`
- `micro_rescue.shadow_mode`
- `micro_rescue.state_knn_enable`
- `micro_rescue.image_knn_enable`
- `micro_rescue.blend_alpha`
- `micro_rescue.max_rescues_per_episode`

Tests:

- KNN returns expected neighbor on synthetic data.
- Rescue refuses OOD state.
- Rescue respects horizon and bounds.
- Rescue event requires flush/re-infer when not shadow.

---

## 7. Harness 2: Data-Derived Invariant Guard

Implementation modules:

- `mode.py`
  - `ModeProfile`
  - `ModeEstimator`
  - `TransitionGraph`
- `invariants.py`
  - `Invariant`
  - `InvariantMiner`
  - `InvariantGuard`
  - `InvariantViolation`

Invariant types:

- `plateau_min_duration`
- `no_backtrack_transition`
- `one_shot_event`
- `value_envelope`
- `velocity_envelope`
- `catastrophic_actuator_release`

Mining algorithm:

1. Detect candidate plateaus per dimension.
2. Detect transitions/excursions.
3. Build per-episode symbolic sequences.
4. Learn transition support.
5. Promote invariant if:
   - support >= `min_support`;
   - train violation <= `max_train_violation_rate`;
   - duration/envelope robust across episodes;
   - component category is enabled.

Runtime:

1. Estimate current mode from recent history.
2. Evaluate proposed action chunk.
3. Emit one of:
   - `pass`;
   - `shadow_violation`;
   - `soft_violation`;
   - `hard_violation`.
4. For hard violations:
   - server: reject/resample before client;
   - client: clamp/reject at final execution;
   - sync: flush/re-infer.

Required toggles:

- `invariant_guard.enable`
- `invariant_guard.shadow_mode`
- `hard_guard_categories`
- `soft_guard_categories`
- `flush_on_hard_guard`
- per-invariant `enabled` in profile override.

Tests:

- Miner recovers known plateau/no-backtrack invariant on synthetic dataset.
- Guard stays quiet when action follows invariant.
- Guard fires when action violates high-confidence invariant.
- Guard does not hard-fire when mode confidence is low unless configured.
- Hard guard emits `requires_reinfer=true`.

---

## 8. Harness 3: Action Stability And Speed Envelope Guard

Implementation modules:

- `envelopes.py`
  - `SpeedEnvelopeProfile`
  - `ActionEnvelopeGuard`
  - `EnvelopeViolation`

Offline:

- Compute per-dim action delta percentiles.
- Compute acceleration/jerk percentiles.
- Compute endpoint jump distribution.
- Compute mode-conditioned envelopes if mode profile quality is high.

Runtime:

1. Evaluate next action or chunk path.
2. If value/speed/jerk outside envelope:
   - shadow log if shadow;
   - soft clamp if configured;
   - reject if severe.
3. Track consecutive clamps.
4. If clamp count exceeds threshold:
   - clear queue;
   - request re-infer or micro-rescue.

Required toggles:

- `speed_envelope.enable`
- `speed_envelope.shadow_mode`
- `speed_envelope.mode_conditioned`
- `speed_envelope.max_consecutive_clamps`
- `speed_envelope.flush_after_repeated_clamp`

Tests:

- Allows in-distribution synthetic trajectories.
- Flags spikes.
- Consecutive clamp counter resets after normal action.
- Repeated clamp emits flush/re-infer requirement.

---

## 9. Harness 4: Runtime Synchronization, Re-Infer, And Trace

Implementation modules:

- `protocol.py`
  - `ChunkId`
  - `InterventionEvent`
  - `ReinferRequest`
  - `HarnessMessageCodec`
- `trace.py`
  - `HarnessTraceWriter`
  - `HarnessTraceReader`
  - `TraceSchemaValidator`
- `runtime.py`
  - `HarnessRuntimeState`
  - `InterventionLedger`
  - `FlushCoordinator`

Server integration:

- Assign `chunk_id` to every chunk.
- Keep chunk registry:
  - active;
  - invalidated;
  - executed;
  - intervened.
- On intervention event:
  - mark chunk invalidated;
  - stop aggregation reuse;
  - request fresh observation;
  - re-infer if enabled.

Client integration:

- Attach current `chunk_id` to execution queue.
- If intervention:
  - clear local queue/RTC pending actions;
  - write trace;
  - send intervention event to server;
  - block execution until fresh chunk if configured.

Trace requirements:

- Every action has:
  - source chunk id;
  - original action;
  - executed action;
  - guard decisions;
  - current state before/after when available.

Required toggles:

- `sync.enable`
- `sync.require_chunk_id`
- `sync.flush_on_*`
- `sync.block_execution_until_fresh_chunk`
- `trace.enable`
- `trace.record_*`

Tests:

- Client intervention invalidates server chunk.
- Server refuses to reuse invalidated chunk.
- Trace event round-trips through JSONL schema.
- Hard intervention with `sync.enable=false` fails closed or stays shadow,
  depending on config.

---

## 10. Server Integration Plan

Likely integration points:

- policy server script that receives observations and sends chunks;
- SmolVLA/RTC server action generation path;
- postprocessor path after model raw output.

Steps:

1. Add `HarnessConfig` to server config.
2. Load `harness_profile.json` if enabled.
3. Wrap inference output:
   - raw chunk;
   - postprocessed chunk;
   - chunk id;
   - metadata.
4. Run server-side guards:
   - invariant guard;
   - speed/action envelope check on chunk;
   - micro-rescue proposal only if requested by runtime state.
5. If reject:
   - resample up to configured limit;
   - if still reject, emit stop/rescue decision.
6. Send chunk plus harness metadata to client.
7. Receive intervention events.
8. Invalidate server-side active chunk/context.
9. Re-infer from fresh observation.

Must preserve behavior when harness disabled.

---

## 11. Client Integration Plan

Likely integration points:

- GUI/orchestrator client execution loop;
- RTC queue/interpolation code;
- robot send-action function.

Steps:

1. Add `HarnessConfig` to client config.
2. Receive chunk id and metadata.
3. Store local execution queue with chunk id.
4. Before each action execute:
   - run final invariant hard guard;
   - run speed envelope guard;
   - log shadow violations.
5. If hard intervention:
   - modify/reject/stop action;
   - clear queue/RTC pending actions;
   - send intervention event;
   - request re-infer;
   - block or hold until fresh chunk depending config.
6. Track actual robot state and tracking error.
7. Write trace JSONL.

Must preserve behavior when harness disabled.

---

## 12. Rollout Milestones

### Milestone 0: Config + Schemas + Trace

- Add configs.
- Add schemas.
- Add chunk id.
- Add trace writer/reader.
- No intervention yet.

Exit criteria:

- Existing infer works with `harness.enable=false`.
- With `trace.enable=true`, logs chunks/actions without changing behavior.

### Milestone 1: Offline Profile Miner

- Build profile from dataset.
- Export reports.
- Add synthetic tests.

Exit criteria:

- `harness_profile.json` loads.
- Profile diagnostics are deterministic enough for tests.

### Milestone 2: Shadow Runtime Monitor

- Server/client evaluate guards in shadow mode.
- Log would-reject/would-clamp/would-rescue.

Exit criteria:

- Runtime unaffected.
- Offline replay can summarize false positives.

### Milestone 3: Synchronization Protocol

- Implement intervention event.
- Implement queue clear.
- Implement server invalidation.
- Implement re-infer handshake.

Exit criteria:

- Synthetic client intervention forces server re-infer.
- Invalidated chunk cannot be reused.

### Milestone 4: Hard Invariant Guard

- Enable hard guard for high-confidence actuator invariant.
- Start with gripper-like catastrophic release profile from data, but through
  generic invariant path.

Exit criteria:

- Guard can prevent known bad gripper release in replay.
- Guard triggers flush/re-infer.

### Milestone 5: Speed Envelope Guard

- Enable spike/OOD clamp or reject.
- Add repeated-clamp escalation.

Exit criteria:

- Spike test is blocked.
- Normal replay has low false positive.

### Milestone 6: Micro-Rescue

- Implement state KNN rescue.
- Execute short snippets.
- Flush/re-infer after rescue.

Exit criteria:

- Offline stuck states retrieve plausible snippets.
- Online shadow suggests neighbors.
- Non-shadow rescue can be tested in a controlled run.

### Milestone 7: Full Hybrid Harness

- Combine all four harnesses.
- Add reports.
- Add config presets:
  - `off`;
  - `trace_only`;
  - `shadow_all`;
  - `guard_gripper_only`;
  - `guard_and_speed`;
  - `full_with_micro_rescue`.

Exit criteria:

- Can run any preset.
- Can disable each harness independently.
- Trace is sufficient to reconstruct intervention timeline.

---

## 13. Test Plan

Unit tests:

- Config defaults and toggles.
- Profile schema validation.
- Mode/invariant mining on synthetic data.
- Invariant guard decisions.
- Speed envelope decisions.
- Micro-rescue KNN and refusal cases.
- Protocol serialization.
- Flush/re-infer state machine.
- Trace JSONL round-trip.

Integration tests:

- Harness disabled preserves existing behavior.
- Shadow mode never modifies action.
- Hard guard modifies/rejects and emits intervention event.
- Client intervention invalidates server chunk.
- Repeated speed clamp escalates to flush.
- Micro-rescue executes bounded snippet and re-infers.

Replay tests:

- Replay recorded runtime logs through harness.
- Measure would-interventions.
- Verify false-positive rate before enabling hard mode.

Manual robot tests:

1. `trace_only`.
2. `shadow_all`.
3. hard invariant guard only.
4. speed guard only.
5. micro-rescue in controlled stuck setup.
6. full harness.

---

## 14. Logging And Artifacts

Per run:

```text
harness_traces/<run_id>/
  client_trace.jsonl
  server_trace.jsonl
  interventions.jsonl
  summary.json
  config.json
  profile_snapshot.json
  optional_images/
```

Summary metrics:

- number of chunks;
- number of shadow violations;
- number of hard interventions;
- number of flush/re-infer events;
- number of rescues;
- rescue success/fail;
- speed clamps;
- invariant violations;
- latency overhead;
- final task outcome if labeled.

---

## 15. Risks And Mitigations

False positives:

- Use shadow rollout.
- Require high invariant confidence.
- Use mode confidence gates.

Confusing the model:

- Flush/re-infer after hard intervention.
- Do not continue old chunk after modified execution.

Over-controlling VLA:

- Keep rescue short.
- Limit interventions per episode.
- Prefer re-infer over long scripted execution.

Latency:

- Keep client hard guards lightweight.
- Put heavy KNN/mode reasoning on server/offline profile.

Task-specific leakage:

- Store human-readable phase labels only for reports.
- Runtime uses generic mode/invariant ids.

---

## 16. Definition Of Done

Implementation is complete when:

- Harness can be fully disabled and existing runtime is unchanged.
- Each of the four harness groups can be enabled/disabled independently.
- Offline profile miner produces a reusable profile from a LeRobot dataset.
- Server-side validators can reject/resample chunks in shadow and hard mode.
- Client-side final guards can intervene safely.
- Any hard client intervention clears local queue and triggers server
  flush/re-infer.
- Trace logs can reconstruct raw action, guarded action, executed action, and
  resulting observation/state.
- Tests cover config, profile, guards, rescue, sync, and trace.
- At least one controlled online run demonstrates:
  - trace-only mode;
  - hard invariant guard;
  - micro-rescue with flush/re-infer.
