from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HarnessServerConfig:
    enable: bool = True
    chunk_validator_enable: bool = True
    invariant_guard_enable: bool = True
    micro_rescue_proposal_enable: bool = True
    reject_resample_enable: bool = True
    max_resample_attempts: int = 1
    re_infer_on_intervention: bool = True


@dataclass
class HarnessClientConfig:
    enable: bool = True
    execution_guard_enable: bool = True
    hard_invariant_guard_enable: bool = True
    speed_envelope_enable: bool = True
    tracking_monitor_enable: bool = True
    tracking_monitor_window_steps: int = 45
    tracking_monitor_state_radius: float = 18.0
    tracking_monitor_min_path_length: float = 0.0
    tracking_monitor_cooldown_steps: int = 45
    tracking_monitor_dims: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    clear_queue_on_intervention: bool = True
    request_reinfer_on_intervention: bool = True


@dataclass
class MicroRescueConfig:
    enable: bool = True
    shadow_mode: bool = True
    state_knn_enable: bool = True
    image_knn_enable: bool = False
    k_neighbors: int = 16
    snippet_horizon_steps: int = 8
    max_duration_s: float = 1.0
    blend_alpha: float = 1.0
    min_future_progress_score: float = 0.2
    max_state_distance: float | None = None
    cooldown_s: float = 2.0
    max_rescues_per_episode: int = 3


@dataclass
class InvariantGuardConfig:
    enable: bool = True
    shadow_mode: bool = True
    min_support: float = 0.95
    max_train_violation_rate: float = 0.02
    min_mode_confidence: float = 0.7
    hard_guard_categories: list[str] = field(
        default_factory=lambda: ["catastrophic_actuator_release"]
    )
    soft_guard_categories: list[str] = field(
        default_factory=lambda: ["value_envelope", "no_backtrack"]
    )
    flush_on_hard_guard: bool = True


@dataclass
class SpeedEnvelopeConfig:
    enable: bool = True
    shadow_mode: bool = True
    percentile_low: float = 0.005
    percentile_high: float = 0.995
    mode_conditioned: bool = True
    max_consecutive_clamps: int = 3
    flush_after_repeated_clamp: bool = True


@dataclass
class HarnessSyncConfig:
    enable: bool = True
    require_chunk_id: bool = True
    flush_on_reject: bool = True
    flush_on_rescue: bool = True
    flush_on_hard_clamp: bool = True
    flush_on_repeated_speed_clamp: bool = True
    block_execution_until_fresh_chunk: bool = True


@dataclass
class HarnessTraceConfig:
    enable: bool = True
    record_images: bool = False
    record_raw_chunks: bool = True
    record_postprocessed_chunks: bool = True
    record_executed_actions: bool = True
    record_mode_estimates: bool = True
    record_rescue_neighbors: bool = True


@dataclass
class HarnessConfig:
    enable: bool = False
    profile_path: str | None = None
    shadow_mode: bool = True
    fail_closed: bool = False
    log_dir: str = "harness_traces"
    server: HarnessServerConfig = field(default_factory=HarnessServerConfig)
    client: HarnessClientConfig = field(default_factory=HarnessClientConfig)
    micro_rescue: MicroRescueConfig = field(default_factory=MicroRescueConfig)
    invariant_guard: InvariantGuardConfig = field(default_factory=InvariantGuardConfig)
    speed_envelope: SpeedEnvelopeConfig = field(default_factory=SpeedEnvelopeConfig)
    sync: HarnessSyncConfig = field(default_factory=HarnessSyncConfig)
    trace: HarnessTraceConfig = field(default_factory=HarnessTraceConfig)

    def effective_enabled(self, component_enabled: bool) -> bool:
        return self.enable and component_enabled


def harness_preset(name: str) -> HarnessConfig:
    cfg = HarnessConfig()
    match name:
        case "off":
            return cfg
        case "trace_only":
            cfg.enable = True
            cfg.shadow_mode = True
            cfg.server.chunk_validator_enable = False
            cfg.server.invariant_guard_enable = False
            cfg.server.micro_rescue_proposal_enable = False
            cfg.client.execution_guard_enable = False
            cfg.client.hard_invariant_guard_enable = False
            cfg.client.speed_envelope_enable = False
            cfg.client.tracking_monitor_enable = False
            return cfg
        case "shadow_all":
            cfg.enable = True
            cfg.shadow_mode = True
            return cfg
        case "guard_gripper_only":
            cfg.enable = True
            cfg.shadow_mode = False
            cfg.speed_envelope.enable = False
            cfg.micro_rescue.enable = False
            cfg.invariant_guard.shadow_mode = False
            cfg.invariant_guard.hard_guard_categories = ["catastrophic_actuator_release"]
            return cfg
        case "guard_and_speed":
            cfg.enable = True
            cfg.shadow_mode = False
            cfg.micro_rescue.enable = False
            cfg.invariant_guard.shadow_mode = False
            cfg.speed_envelope.shadow_mode = False
            return cfg
        case "full_with_micro_rescue":
            cfg.enable = True
            cfg.shadow_mode = False
            cfg.invariant_guard.shadow_mode = False
            cfg.speed_envelope.shadow_mode = False
            cfg.micro_rescue.shadow_mode = False
            return cfg
        case _:
            raise ValueError(f"Unknown harness preset: {name}")
