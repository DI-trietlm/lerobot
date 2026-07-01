from __future__ import annotations

import numpy as np

from lerobot.vla_harness.client import ClientHarnessController
from lerobot.vla_harness.config import HarnessConfig
from lerobot.vla_harness.mode import build_mode_profile
from lerobot.vla_harness.profile import HarnessProfileBundle
from lerobot.vla_harness.protocol import HarnessMessageCodec, InterventionEvent, PolicyMetadata
from lerobot.vla_harness.rescue import build_rescue_index
from lerobot.vla_harness.schemas import HarnessDiagnostics, HarnessProfile, InvariantSpec, RescueIndexMetadata
from lerobot.vla_harness.server import ServerHarnessController


def _make_bundle() -> HarnessProfileBundle:
    states = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]], dtype=np.float64)
    actions = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1]], dtype=np.float64)
    episode_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    mode_profile = build_mode_profile(states, actions, episode_ids)
    rescue_index = build_rescue_index(
        states,
        actions,
        episode_ids,
        state_scales=np.array([1.0]),
        mode_ids=mode_profile.mode_ids,
        horizon_steps=2,
    )
    profile = HarnessProfile(
        schema_version=1,
        dataset_repo_id="synthetic/test",
        dataset_revision=None,
        fps=15,
        state_keys=["s0"],
        action_keys=["a0"],
        scales={"state": [1.0], "action": [1.0]},
        mode_profile=mode_profile,
        invariants=[],
        speed_envelopes=None,
        rescue_index=RescueIndexMetadata(
            type="state_knn",
            index_path="rescue_index.npz",
            frame_table_path="rescue_frames.parquet",
            state_dim=1,
            snippet_horizon_steps=2,
            num_entries=len(states),
        ),
        diagnostics=HarnessDiagnostics(num_episodes=2, num_frames=6),
    )
    return HarnessProfileBundle(profile=profile, rescue_index=rescue_index)


def test_client_intervention_invalidates_server_chunk():
    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.micro_rescue.shadow_mode = False
    bundle = _make_bundle()
    server = ServerHarnessController(cfg, bundle)
    client = ClientHarnessController(cfg, bundle)

    _, envelope, _, _ = server.build_envelope(
        current_state=np.array([0.0]),
        action_chunk=np.array([[0.1], [0.2]]),
        timestamp=1.0,
        policy_metadata=PolicyMetadata("test", None, None),
    )
    client.on_chunk_received(envelope.chunk_id, envelope.inference_id)
    event = InterventionEvent.create(
        chunk_id=envelope.chunk_id,
        inference_id=envelope.inference_id,
        timestamp=2.0,
        component="client.execution_guard",
        severity="hard",
        reason="tracking_error",
        original_action=[0.1],
        executed_action=[0.0],
        current_state=[0.0],
        queue_cleared=True,
        requires_reinfer=True,
    )
    client.register_intervention(event)
    server.register_intervention(event)

    assert client.execution_blocked
    assert server.runtime.is_invalidated(envelope.chunk_id)
    rescue, _ = server.maybe_replace_with_rescue(np.array([0.05]))
    assert rescue is not None


def test_intervention_codec_round_trips_without_pickle_object_channel():
    event = InterventionEvent.create(
        chunk_id="chunk",
        inference_id="infer",
        timestamp=1.0,
        component="client.execution_guard",
        severity="hard",
        reason="speed_clamp",
        original_action=[0.1],
        executed_action=[0.0],
        current_state=[0.0],
        queue_cleared=True,
        requires_reinfer=True,
        metadata={"source": "test"},
    )

    payload = HarnessMessageCodec.encode_intervention(event)
    assert payload.startswith(b"{")
    decoded = HarnessMessageCodec.decode_intervention(payload)

    assert decoded == event


def test_shadow_micro_rescue_does_not_replace_actions():
    cfg = HarnessConfig(enable=True, shadow_mode=True)
    bundle = _make_bundle()
    server = ServerHarnessController(cfg, bundle)

    event = InterventionEvent.create(
        chunk_id="chunk",
        inference_id="infer",
        timestamp=2.0,
        component="client.execution_guard",
        severity="soft",
        reason="stuck_candidate",
        original_action=[0.1],
        executed_action=[0.0],
        current_state=[0.0],
        queue_cleared=True,
        requires_reinfer=True,
    )
    server.register_intervention(event)
    rescue, metadata = server.maybe_replace_with_rescue(np.array([0.05]))

    assert rescue is None
    assert metadata is not None
    assert metadata["severity"] == "shadow"
    assert metadata["would_rescue"] is True


def test_server_hard_invariant_rejects_to_hold_action():
    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.invariant_guard.shadow_mode = False
    bundle = _make_bundle()
    bundle.profile.invariants = [
        InvariantSpec(
            invariant_id="actuator_hold_release::dim_0",
            kind="actuator_hold_release",
            category="catastrophic_actuator_release",
            support=1.0,
            train_violation_rate=0.0,
            parameters={
                "dim": 0,
                "hold_side": "low",
                "hold_threshold": 0.2,
                "release_threshold": 0.8,
            },
        )
    ]
    server = ServerHarnessController(cfg, bundle)

    candidate, envelope, invariant_violations, _ = server.build_envelope(
        current_state=np.array([0.0]),
        action_chunk=np.array([[1.0], [1.0]]),
        timestamp=1.0,
        policy_metadata=PolicyMetadata("test", None, None),
    )

    assert invariant_violations
    assert envelope.harness_decision.server_valid is False
    assert envelope.harness_decision.intervention_required is True
    assert np.allclose(candidate, np.array([[0.0], [0.0]]))


def test_client_shadow_tracking_does_not_block_or_emit_intervention():
    cfg = HarnessConfig(enable=True, shadow_mode=True)
    client = ClientHarnessController(cfg, _make_bundle())
    client.on_chunk_received("chunk", "infer")

    events = [client.observe_state(np.array([0.0])) for _ in range(16)]

    assert all(event is None for event in events)
    assert not client.execution_blocked


def test_client_component_toggle_disables_hard_invariant_guard():
    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.client.hard_invariant_guard_enable = False
    cfg.invariant_guard.shadow_mode = False
    bundle = _make_bundle()
    bundle.profile.invariants = [
        InvariantSpec(
            invariant_id="actuator_hold_release::dim_0",
            kind="actuator_hold_release",
            category="catastrophic_actuator_release",
            support=1.0,
            train_violation_rate=0.0,
            parameters={
                "dim": 0,
                "hold_side": "low",
                "hold_threshold": 0.2,
                "release_threshold": 0.8,
            },
        )
    ]
    client = ClientHarnessController(cfg, bundle)
    client.on_chunk_received("chunk", "infer")

    adjusted, event, invariant_violations, _ = client.evaluate_action(
        current_state=np.array([0.0]),
        action=np.array([1.0]),
    )

    assert event is None
    assert not invariant_violations
    assert np.allclose(adjusted, [1.0])


def test_server_component_toggle_disables_invariant_validation():
    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.server.invariant_guard_enable = False
    cfg.invariant_guard.shadow_mode = False
    bundle = _make_bundle()
    bundle.profile.invariants = [
        InvariantSpec(
            invariant_id="actuator_hold_release::dim_0",
            kind="actuator_hold_release",
            category="catastrophic_actuator_release",
            support=1.0,
            train_violation_rate=0.0,
            parameters={
                "dim": 0,
                "hold_side": "low",
                "hold_threshold": 0.2,
                "release_threshold": 0.8,
            },
        )
    ]
    server = ServerHarnessController(cfg, bundle)

    candidate, envelope, invariant_violations, _ = server.build_envelope(
        current_state=np.array([0.0]),
        action_chunk=np.array([[1.0]]),
        timestamp=1.0,
        policy_metadata=PolicyMetadata("test", None, None),
    )

    assert not invariant_violations
    assert envelope.harness_decision.server_valid is True
    assert np.allclose(candidate, np.array([[1.0]]))


def test_server_component_toggle_disables_micro_rescue_proposal():
    cfg = HarnessConfig(enable=True, shadow_mode=False)
    cfg.server.micro_rescue_proposal_enable = False
    cfg.micro_rescue.shadow_mode = False
    server = ServerHarnessController(cfg, _make_bundle())
    event = InterventionEvent.create(
        chunk_id="chunk",
        inference_id="infer",
        timestamp=2.0,
        component="client.execution_guard",
        severity="soft",
        reason="stuck_candidate",
        original_action=[0.1],
        executed_action=[0.0],
        current_state=[0.0],
        queue_cleared=True,
        requires_reinfer=True,
    )

    server.register_intervention(event)
    rescue, metadata = server.maybe_replace_with_rescue(np.array([0.05]))

    assert rescue is None
    assert metadata is None


def test_fail_closed_when_sync_disabled():
    cfg = HarnessConfig(enable=True, shadow_mode=False, fail_closed=True)
    cfg.sync.enable = False
    event = InterventionEvent.create(
        chunk_id="chunk",
        inference_id="infer",
        timestamp=1.0,
        component="client.execution_guard",
        severity="hard",
        reason="speed_clamp",
        original_action=[0.0],
        executed_action=[0.0],
        current_state=[0.0],
        queue_cleared=True,
        requires_reinfer=True,
    )
    assert ClientHarnessController(cfg, _make_bundle()).flush.intervention_requires_flush(event)
