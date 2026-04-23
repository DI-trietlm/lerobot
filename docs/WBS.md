# XAI Integration - Work Breakdown Structure

## Overview

This document defines the implementation tasks for integrating 7 XAI methods into LeRobot. Each task has:
- **Deliverables**: What must be produced
- **Tests**: How to verify correctness
- **Exit Criteria**: When the task is considered complete

### Principles
1. **Non-breaking**: No changes to existing code behavior
2. **Optional**: XAI disabled by default
3. **Additive only**: New files only, no modifications to existing files except integration points
4. **Testable**: Each task has tests before/after

---

## Phase 1: Infrastructure

### Task 1.1: Project Structure & Dependencies

**Location**: `src/lerobot/policies/xai/`

**Deliverables**:
```
src/lerobot/policies/xai/
├── __init__.py                    # Empty, exports nothing yet
├── config.py                      # XAIConfig dataclass
└── buffer.py                      # StepRecord, EpisodeXAIBuffer
```

**Implementation**:
```python
# config.py
@dataclass
class XAIConfig:
    # Flags
    use_p0_v_attention: bool = False
    use_p1_v_gmar: bool = False
    use_p1_a_denoising: bool = False
    use_p2_a_bundle: bool = False
    use_p2_x_integrated_gradients: bool = False
    use_p3_a_correlation: bool = False
    use_p3_rtc_smoothness: bool = False

    # Common
    output_dir: str = "xai_outputs"
    entropy_threshold: float = 3.5
    quality_threshold: float = 0.80
    boundary_low_threshold: float = 0.75
    boundary_critical_threshold: float = 0.5
```

**Tests**:
```python
# tests/policies/xai/test_xai_config.py
def test_xai_config_defaults():
    cfg = XAIConfig()
    assert cfg.use_p0_v_attention is False
    assert cfg.use_p3_rtc_smoothness is False
    assert cfg.output_dir == "xai_outputs"

def test_xai_config_custom():
    cfg = XAIConfig(use_p0_v_attention=True, entropy_threshold=4.0)
    assert cfg.use_p0_v_attention is True
    assert cfg.entropy_threshold == 4.0
```

**Exit Criteria**:
- [ ] `XAIConfig` instantiates with defaults
- [ ] All 7 method flags exist and default to False
- [ ] Tests pass

---

### Task 1.2: Data Structures - StepRecord & EpisodeXAIBuffer

**Deliverables**:
```python
# buffer.py
@dataclass
class StepRecord:
    step_idx: int
    timestamp: float
    attn_entropy: float = 0.0
    attn_compressed: torch.Tensor | None = None
    convergence_speed: list[float] = field(default_factory=list)
    boundary_sim: float | None = None
    flagged: bool = False
    flag_reason: str | None = None

@dataclass
class EpisodeXAIBuffer:
    episode_id: str
    step_records: list[StepRecord] = field(default_factory=list)
    flagged_steps: list[int] = field(default_factory=list)
    episode_quality: float | None = None
    mean_entropy: float | None = None

    def summary(self) -> dict: ...
    def should_include_in_training(self, threshold: float) -> bool: ...
    def save(self, output_dir: Path) -> None: ...
```

**Tests**:
```python
# tests/policies/xai/test_buffer.py
def test_step_record_creation():
    record = StepRecord(step_idx=0, timestamp=1.0)
    assert record.step_idx == 0
    assert record.flagged is False

def test_episode_xai_buffer_add_step():
    buffer = EpisodeXAIBuffer(episode_id="ep_001")
    buffer.add_step(StepRecord(step_idx=0, timestamp=1.0, flagged=False))
    buffer.add_step(StepRecord(step_idx=1, timestamp=2.0, flagged=True, flag_reason="high_entropy"))
    assert len(buffer.step_records) == 2
    assert buffer.flagged_steps == [1]

def test_should_include_in_training():
    buffer = EpisodeXAIBuffer(episode_id="ep_001")
    buffer.episode_quality = 0.9
    assert buffer.should_include_in_training(0.8) is True
    buffer.episode_quality = 0.7
    assert buffer.should_include_in_training(0.8) is False
```

**Exit Criteria**:
- [ ] `StepRecord` dataclass works
- [ ] `EpisodeXAIBuffer.add_step()` tracks flagged steps
- [ ] `should_include_in_training()` returns correct bool
- [ ] Tests pass

---

### Task 1.3: XAIMethod Base Class

**Deliverables**:
```python
# methods/base.py
class XAIMethod(abc.ABC):
    """Base class for all XAI methods."""

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy):
        self.config = config
        self.policy = policy

    @abc.abstractmethod
    def name(self) -> str:
        """Return method name, e.g. 'p0_v_attention'."""
        ...

    @abc.abstractmethod
    def is_realtime(self) -> bool:
        """True for real-time methods, False for offline."""
        ...

    def start_episode(self, episode_id: str): ...
    def on_step(self, batch: dict, action_chunk: torch.Tensor, step_idx: int): ...
    def end_episode(self) -> EpisodeXAIBuffer: ...
    def reset(self): ...
```

**Tests**:
```python
# tests/policies/xai/test_base.py
def test_xai_method_base():
    class DummyMethod(XAIMethod):
        def name(self) -> str: return "dummy"
        def is_realtime(self) -> bool: return True

    cfg = XAIConfig()
    # Need a real policy or mock - use unit test with mock
    method = DummyMethod(cfg, mock_policy)
    assert method.name() == "dummy"
    assert method.is_realtime() is True
```

**Exit Criteria**:
- [ ] `XAIMethod` is abstract base
- [ ] All methods implement `name()` and `is_realtime()`
- [ ] Tests pass

---

## Phase 2: Real-time Methods

### Task 2.1: P3-RTC Chunk Boundary Smoothness Monitor

**Priority**: P3-RTC is simplest, lowest risk - good first real-time method

**Deliverables**:
```python
# methods/p3_rtc_smoothness.py
class P3RTCSmoothnessMonitor(XAIMethod):
    """
    Computes cosine similarity between chunk boundary transitions.
    Real-time: ~0 compute cost.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy):
        super().__init__(config, policy)
        self.overlap_steps = config.p3_rtc_overlap_steps
        self.low_threshold = config.boundary_low_threshold
        self.critical_threshold = config.boundary_critical_threshold
        self._prev_chunk: torch.Tensor | None = None
        self._history: list[float] = []

    def name(self) -> str: return "p3_rtc_smoothness"
    def is_realtime(self) -> bool: return True

    def on_step(self, batch: dict, action_chunk: torch.Tensor, step_idx: int):
        if self._prev_chunk is None:
            self._prev_chunk = action_chunk.detach().clone()
            return

        prev_tail = self._prev_chunk[:, -self.overlap_steps:, :]
        new_head = action_chunk[:, :self.overlap_steps, :]

        sim = F.cosine_similarity(
            prev_tail.flatten(1),
            new_head.flatten(1),
            dim=1
        ).mean().item()

        self._history.append(sim)
        self._prev_chunk = action_chunk.detach().clone()

    def get_status(self, sim: float) -> str:
        if sim < self.critical_threshold: return "critical_jerk"
        if sim < self.low_threshold: return "warning_jerk"
        return "ok"

    def episode_quality(self) -> float:
        if not self._history: return 1.0
        good = sum(s >= self.low_threshold for s in self._history)
        return good / len(self._history)

    def reset(self):
        self._prev_chunk = None
        self._history = []
```

**Tests**:
```python
# tests/policies/xai/test_p3_rtc.py
def test_smoothness_monitor_no_jerk():
    monitor = P3RTCSmoothnessMonitor(cfg, mock_policy)
    chunk = torch.randn(2, 10, 7)  # [B, chunk_size, dim_action]
    monitor.on_step({}, chunk, 0)  # First call - no comparison
    chunk2 = chunk + 0.001  # Similar chunk
    monitor.on_step({}, chunk2, 1)
    assert len(monitor._history) == 1
    assert monitor.get_status(monitor._history[0]) == "ok"

def test_smoothness_monitor_jerk():
    monitor = P3RTCSmoothnessMonitor(cfg, mock_policy)
    chunk1 = torch.randn(2, 10, 7)
    chunk2 = -chunk1  # Opposite chunk = jerk
    monitor.on_step({}, chunk1, 0)
    monitor.on_step({}, chunk2, 1)
    assert monitor.get_status(monitor._history[0]) == "critical_jerk"

def test_episode_quality():
    monitor = P3RTCSmoothnessMonitor(cfg, mock_policy)
    monitor._history = [0.9, 0.85, 0.95, 0.8, 0.7]
    assert monitor.episode_quality() == 0.8  # 4/5 good
```

**Exit Criteria**:
- [ ] `P3RTCSmoothnessMonitor` computes cosine similarity
- [ ] Status returned: `ok`, `warning_jerk`, `critical_jerk`
- [ ] `episode_quality()` returns correct ratio
- [ ] No modifications to existing code
- [ ] Tests pass

---

### Task 2.2: P0-V Raw Attention Map

**Deliverables**:
```python
# methods/p0_v_attention_map.py
class P0VAttentionMap(XAIMethod):
    """
    Captures attention weights from Florence-2 encoder.
    Real-time: ~0 compute, only memory bandwidth.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy):
        super().__init__(config, policy)
        self.layer_indices = config.p0_v_layer_indices
        self.patch_grid = config.p0_v_patch_grid
        self._hooks: list = []
        self._attention_maps: list[torch.Tensor] = []

    def name(self) -> str: return "p0_v_attention"
    def is_realtime(self) -> bool: return True

    def _find_encoder_layers(self):
        """Find Florence-2 encoder layers."""
        vlm = self.policy.model.vlm
        return vlm.language_model.model.encoder.layers

    def register(self):
        """Register forward hooks on encoder layers."""
        encoder_layers = self._find_encoder_layers()
        for idx in self.layer_indices:
            layer = encoder_layers[idx]
            hook = layer.self_attn.register_forward_hook(self._hook_fn)
            self._hooks.append(hook)

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            self._attention_maps.append(output[1].detach().cpu())

    def compute_entropy(self) -> tuple[float, torch.Tensor]:
        """
        Returns:
            entropy: scalar attention entropy
            compressed: [7, 7] heatmap tensor
        """
        if not self._attention_maps:
            return 0.0, torch.zeros(7, 7)

        attn = self._attention_maps[-1]  # Last layer: [B, H, seq, seq]

        # Assume first P tokens are image patches
        num_img_tokens = self.patch_grid[0] * self.patch_grid[1]  # 196
        num_lang_tokens = attn.shape[-1] - num_img_tokens

        # Language-to-image attention
        lang_to_img = attn[:, :, num_lang_tokens:, :num_img_tokens]
        img_attn = lang_to_img.mean(dim=(1, 2))  # [B, P]

        # Entropy
        p = F.softmax(img_attn[0], dim=-1)
        entropy = -(p * (p + 1e-8).log()).sum().item()

        # Compress to 7x7
        heatmap = img_attn[0].reshape(self.patch_grid)
        compressed = F.avg_pool2d(heatmap[None, None], 2)[0, 0]

        return entropy, compressed

    def clear(self):
        self._attention_maps.clear()

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def reset(self):
        self.clear()
        self.remove()
        self._hooks = []
```

**Tests**:
```python
# tests/policies/xai/test_p0_v.py
def test_attention_map_initialization():
    # Mock policy with fake encoder structure
    monitor = P0VAttentionMap(cfg, mock_policy)
    assert monitor.name() == "p0_v_attention"
    assert monitor.is_realtime() is True
    assert len(monitor._hooks) == 0  # Not registered yet

def test_compute_entropy_uniform():
    """Test entropy is high when attention is uniform."""
    monitor = P0VAttentionMap.__new__(P0VAttentionMap)
    monitor._attention_maps = []
    # Uniform attention = max entropy
    uniform_attn = torch.ones(196) / 196
    p = F.softmax(uniform_attn, dim=-1)
    entropy = -(p * (p + 1e-8).log()).sum().item()
    # Max entropy for 196 tokens = -log(1/196) ≈ 5.28
    assert entropy > 5.0

def test_compute_entropy_focused():
    """Test entropy is low when attention is focused."""
    focused_attn = torch.zeros(196)
    focused_attn[0] = 1.0  # All on one token
    p = F.softmax(focused_attn, dim=-1)
    entropy = -(p * (p + 1e-8).log()).sum().item()
    assert entropy < 0.1
```

**Exit Criteria**:
- [ ] Hook registration works on Florence-2 encoder layers
- [ ] `compute_entropy()` returns valid entropy and compressed heatmap
- [ ] Hooks properly removed on `remove()`
- [ ] Works only with XVLA policy (Florence-2 based) - graceful error for others
- [ ] Tests pass

---

### Task 2.3: P1-A Denoising Trajectory Tracker

**Deliverables**:
```python
# methods/p1_a_denoising.py
class P1ADenoisingTracker(XAIMethod):
    """
    Tracks denoising trajectory convergence.
    Real-time: log only, ~0 compute.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy):
        super().__init__(config, policy)
        self._trajectory: list[torch.Tensor] = []
        self._enabled = False

    def name(self) -> str: return "p1_a_denoising"
    def is_realtime(self) -> bool: return True

    def enable(self):
        self._enabled = True

    def track(self, x_t: torch.Tensor):
        """Call from within generate_actions() loop."""
        if self._enabled:
            self._trajectory.append(x_t.detach().cpu().clone())

    def get_convergence_speed(self) -> list[float]:
        """Compute delta between consecutive x_t."""
        if len(self._trajectory) < 2:
            return []
        deltas = []
        for i in range(1, len(self._trajectory)):
            delta = (self._trajectory[i] - self._trajectory[i-1]).pow(2).mean().sqrt().item()
            deltas.append(delta)
        return deltas

    def get_final_std(self) -> float | None:
        """Std of final x_t = action consistency."""
        if not self._trajectory:
            return None
        final = self._trajectory[-1]
        return final.std(dim=1).mean().item()

    def clear(self):
        self._trajectory.clear()

    def reset(self):
        self.clear()
        self._enabled = False
```

**Tests**:
```python
# tests/policies/xai/test_p1_a.py
def test_tracker_basic():
    tracker = P1ADenoisingTracker(cfg, mock_policy)
    assert tracker.name() == "p1_a_denoising"
    assert tracker.is_realtime() is True

def test_convergence_speed():
    tracker = P1ADenoisingTracker(cfg, mock_policy)
    # Simulate converging trajectory
    x0 = torch.randn(2, 10, 7)
    x1 = x0 + 0.5
    x2 = x1 + 0.1
    x3 = x2 + 0.01
    tracker.track(x0)
    tracker.track(x1)
    tracker.track(x2)
    tracker.track(x3)
    deltas = tracker.get_convergence_speed()
    assert len(deltas) == 3
    assert deltas[0] > deltas[1] > deltas[2]  # Converging

def test_final_std():
    tracker = P1ADenoisingTracker(cfg, mock_policy)
    final = torch.randn(2, 10, 7)
    tracker.track(final)
    std = tracker.get_final_std()
    assert isinstance(std, float)
```

**Exit Criteria**:
- [ ] `P1ADenoisingTracker` tracks x_t
- [ ] `get_convergence_speed()` computes deltas
- [ ] Integration point identified in `generate_actions()`
- [ ] Tests pass

---

### Task 2.4: XAIPipeline Integration (Real-time Only)

**Deliverables**:
```python
# pipeline.py
class XAIPipeline:
    def __init__(self, policy: PreTrainedPolicy, config: XAIConfig):
        self.config = config
        self.policy = policy
        self.current_episode: EpisodeXAIBuffer | None = None

        # Initialize real-time methods
        self.p0_v: P0VAttentionMap | None = None
        self.p1_a: P1ADenoisingTracker | None = None
        self.p3_rtc: P3RTCSmoothnessMonitor | None = None

        self._verify_policy_supports()

    def _verify_policy_supports(self):
        """Check if policy has required architecture for XAI methods."""
        if not hasattr(self.policy, 'model'):
            raise ValueError(f"Policy {type(self.policy)} does not have 'model' attribute")
        if not hasattr(self.policy.model, 'vlm'):
            raise ValueError(f"Policy model does not have VLM (Florence-2)")

    def start_episode(self, episode_id: str, episode_index: int = 0):
        self.current_episode = EpisodeXAIBuffer(
            episode_id=episode_id,
            episode_index=episode_index,
            timestamp_start=time.time(),
        )
        if self.config.use_p0_v_attention:
            self.p0_v = P0VAttentionMap(self.config, self.policy)
            self.p0_v.register()
        if self.config.use_p1_a_denoising:
            self.p1_a = P1ADenoisingTracker(self.config, self.policy)
            self.p1_a.enable()
        if self.config.use_p3_rtc_smoothness:
            self.p3_rtc = P3RTCSmoothnessMonitor(self.config, self.policy)

    def on_after_action(self, batch: dict, action_chunk: torch.Tensor, step_idx: int):
        if self.current_episode is None:
            return
        record = StepRecord(step_idx=step_idx, timestamp=time.time())
        # P0-V
        if self.p0_v:
            entropy, compressed = self.p0_v.compute_entropy()
            record.attn_entropy = entropy
            record.attn_compressed = compressed
            self.p0_v.clear()
        # P1-A
        if self.p1_a:
            record.convergence_speed = self.p1_a.get_convergence_speed()
            self.p1_a.clear()
        # P3-RTC
        if self.p3_rtc:
            record.boundary_sim = self.p3_rtc.update(action_chunk)
            record.status = self.p3_rtc.get_status(record.boundary_sim)
        # Flagging
        if record.attn_entropy > self.config.entropy_threshold:
            record.flagged = True
            record.flag_reason = "high_entropy"
        elif getattr(record, 'status', None) == "critical_jerk":
            record.flagged = True
            record.flag_reason = "critical_jerk"
        elif getattr(record, 'status', None) == "warning_jerk":
            record.flagged = True
            record.flag_reason = "warning_jerk"
        self.current_episode.add_step(record)

    def end_episode(self) -> EpisodeXAIBuffer | None:
        if self.current_episode is None:
            return None
        # Cleanup
        if self.p0_v:
            self.p0_v.remove()
            self.p0_v = None
        if self.p1_a:
            self.p1_a.reset()
            self.p1_a = None
        # Episode quality
        if self.p3_rtc:
            self.current_episode.episode_quality = self.p3_rtc.episode_quality()
            self.p3_rtc = None
        episode = self.current_episode
        self.current_episode = None
        return episode
```

**Tests**:
```python
# tests/policies/xai/test_pipeline.py
def test_pipeline_initialization():
    policy = create_mock_xvla_policy()
    cfg = XAIConfig(use_p3_rtc_smoothness=True)
    pipeline = XAIPipeline(policy, cfg)
    assert pipeline.config == cfg

def test_pipeline_start_end_episode():
    policy = create_mock_xvla_policy()
    cfg = XAIConfig(use_p3_rtc_smoothness=True)
    pipeline = XAIPipeline(policy, cfg)
    pipeline.start_episode("ep_001", 0)
    assert pipeline.current_episode is not None
    assert pipeline.current_episode.episode_id == "ep_001"
    episode = pipeline.end_episode()
    assert episode is not None
    assert episode.episode_id == "ep_001"
    assert pipeline.current_episode is None

def test_pipeline_on_after_action():
    policy = create_mock_xvla_policy()
    cfg = XAIConfig(
        use_p3_rtc_smoothness=True,
        boundary_low_threshold=0.75
    )
    pipeline = XAIPipeline(policy, cfg)
    pipeline.start_episode("ep_001", 0)
    action = torch.randn(2, 10, 7)
    pipeline.on_after_action({}, action, 0)
    assert len(pipeline.current_episode.step_records) == 1
```

**Exit Criteria**:
- [ ] `XAIPipeline` initializes with config
- [ ] `start_episode()` creates buffer
- [ ] `on_after_action()` records step data
- [ ] `end_episode()` returns buffer and cleans up
- [ ] Real-time methods (P0-V, P1-A, P3-RTC) work together
- [ ] Tests pass

---

## Phase 3: Offline Methods

### Task 3.1: P1-V GMAR

**Deliverables**:
```python
# methods/p1_v_gmar.py
class P1VGMAR(XAIMethod):
    """
    Gradient-weighted Multi-head Attention Rollout.
    Offline: 1.5x forward compute. Triggered on flagged episodes.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy):
        super().__init__(config, policy)
        self.target_action_dim = config.p1_v_target_action_dim

    def name(self) -> str: return "p1_v_gmar"
    def is_realtime(self) -> bool: return False

    def compute(self, batch: dict) -> torch.Tensor:
        """
        Compute GMAR heatmap for given batch.

        Returns: heatmap [B, 14, 14]
        """
        # 1. Register hooks
        # 2. Forward with grad
        # 3. Backward from target
        # 4. Compute rollout
        # 5. Cleanup
        ...
```

**Tests**:
```python
# tests/policies/xai/test_p1_v.py
def test_gmar_initialization():
    gmar = P1VGMAR(cfg, mock_policy)
    assert gmar.name() == "p1_v_gmar"
    assert gmar.is_realtime() is False
```

**Exit Criteria**:
- [ ] GMAR heatmap computed correctly
- [ ] Uses action-conditioned gradient target
- [ ] Handles both single and batch inputs
- [ ] Tests pass

---

### Task 3.2: P2-A Action Sample Bundle

**Deliverables**:
```python
# methods/p2_a_bundle.py
class P2ABundle(XAIMethod):
    """
    Action Sample Bundle - uncertainty analysis.
    Offline: N x forward compute. Expensive.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy):
        super().__init__(config, policy)
        self.n_samples = config.p2_a_n_samples

    def name(self) -> str: return "p2_a_bundle"
    def is_realtime(self) -> bool: return False

    def compute(self, batch: dict) -> dict:
        """
        Sample N trajectories and compute statistics.
        Returns: {mean, std, cv, is_multimodal}
        """
        ...
```

**Tests**:
```python
# tests/policies/xai/test_p2_a.py
def test_bundle_initialization():
    bundle = P2ABundle(cfg, mock_policy)
    assert bundle.name() == "p2_a_bundle"
    assert bundle.is_realtime() is False
```

**Exit Criteria**:
- [ ] Samples N trajectories
- [ ] Computes mean, std, cv
- [ ] Detects multimodal distribution
- [ ] Tests pass

---

### Task 3.3: P2-X Integrated Gradients

**Deliverables**:
```python
# methods/p2_x_integrated_gradients.py
class P2XIntegratedGradients(XAIMethod):
    """
    Cross-modal attribution via Integrated Gradients.
    Offline: 50x forward compute. Most expensive.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy):
        super().__init__(config, policy)
        self.n_steps = config.p2_x_n_steps

    def name(self) -> str: return "p2_x_integrated_gradients"
    def is_realtime(self) -> bool: return False

    def compute(self, batch: dict) -> dict:
        """
        Returns: {vision_pct, language_pct, proprio_pct}
        """
        ...
```

**Tests**:
```python
# tests/policies/xai/test_p2_x.py
def test_ig_initialization():
    ig = P2XIntegratedGradients(cfg, mock_policy)
    assert ig.name() == "p2_x_integrated_gradients"
    assert ig.is_realtime() is False
```

**Exit Criteria**:
- [ ] Computes IG attribution for vision, language, proprio
- [ ] Attribution percentages sum to ~100%
- [ ] Tests pass

---

### Task 3.4: P3-A Action Dimension Correlation

**Deliverables**:
```python
# methods/p3_a_correlation.py
class P3ACorrelation(XAIMethod):
    """
    Action Dimension Correlation Heatmap.
    Offline: Low compute. Uses P2-A bundle as input.
    """

    def __init__(self, config: XAIConfig, policy: PreTrainedPolicy):
        super().__init__(config, policy)

    def name(self) -> str: return "p3_a_correlation"
    def is_realtime(self) -> bool: return False

    def compute(self, observations: list[dict], bundle_results: dict) -> torch.Tensor:
        """
        Returns: correlation_matrix [dim_action, dim_action]
        """
        ...
```

**Tests**:
```python
# tests/policies/xai/test_p3_a.py
def test_correlation_initialization():
    corr = P3ACorrelation(cfg, mock_policy)
    assert corr.name() == "p3_a_correlation"
    assert corr.is_realtime() is False
```

**Exit Criteria**:
- [ ] Computes correlation matrix
- [ ] Detects spurious correlations
- [ ] Tests pass

---

## Phase 4: Full Integration

### Task 4.1: lerobot_eval.py Integration

**Deliverables**:
```python
# Optional: create_xai_pipeline function
def create_xai_pipeline(policy, cfg: XAIConfig) -> XAIPipeline | None:
    if not any([
        cfg.use_p0_v_attention,
        cfg.use_p1_v_gmar,
        cfg.use_p1_a_denoising,
        cfg.use_p2_a_bundle,
        cfg.use_p2_x_integrated_gradients,
        cfg.use_p3_a_correlation,
        cfg.use_p3_rtc_smoothness,
    ]):
        return None
    return XAIPipeline(policy, cfg)
```

**Integration point in lerobot_eval.py**:
```python
# Add near policy creation (line ~543):
xai_pipeline = create_xai_pipeline(policy, cfg.xai) if cfg.xai else None

# Modify rollout loop to call XAI hooks:
# After action = policy.select_action(observation) (line ~186)
# Add:
if xai_pipeline and xai_pipeline.current_episode is not None:
    xai_pipeline.on_after_action(observation, action, step)
```

**Tests**:
```python
# tests/policies/xai/test_integration.py
def test_eval_integration_no_xai():
    """When XAI disabled, eval works normally."""
    cfg = EvalPipelineConfig(...)
    cfg.xai = None
    # Run eval, verify no errors

def test_eval_integration_with_xai():
    """When XAI enabled, data is collected."""
    cfg = EvalPipelineConfig(...)
    cfg.xai = XAIConfig(use_p3_rtc_smoothness=True)
    # Run eval, verify xai data collected
```

**Exit Criteria**:
- [ ] `create_xai_pipeline()` returns None when no XAI enabled
- [ ] XAI hooks called during eval
- [ ] Existing eval without XAI works unchanged
- [ ] Tests pass

---

### Task 4.2: Final Verification & Documentation

**Deliverables**:
- All unit tests pass
- Integration tests pass
- No regressions in existing tests
- Updated `docs/PLAN.md` with actual implementation notes

**Tests to run**:
```bash
uv run pytest tests/policies/xai/ -v
uv run pytest tests/policies/ -v --ignore=tests/policies/test_pi0_new.py
```

**Exit Criteria**:
- [ ] All XAI tests pass
- [ ] All existing policy tests pass
- [ ] lerobot_eval runs with and without XAI
- [ ] No breaking changes

---

## Summary Checklist

### Phase 1: Infrastructure
- [ ] Task 1.1: Project structure, XAIConfig
- [ ] Task 1.2: StepRecord, EpisodeXAIBuffer
- [ ] Task 1.3: XAIMethod base class

### Phase 2: Real-time Methods
- [ ] Task 2.1: P3-RTC Smoothness
- [ ] Task 2.2: P0-V Attention Map
- [ ] Task 2.3: P1-A Denoising Tracker
- [ ] Task 2.4: XAIPipeline (real-time integration)

### Phase 3: Offline Methods
- [ ] Task 3.1: P1-V GMAR
- [ ] Task 3.2: P2-A Action Bundle
- [ ] Task 3.3: P2-X Integrated Gradients
- [ ] Task 3.4: P3-A Correlation

### Phase 4: Integration
- [ ] Task 4.1: lerobot_eval.py integration
- [ ] Task 4.2: Final verification

---

## File List (All New Files)

```
src/lerobot/policies/xai/
├── __init__.py
├── config.py
├── buffer.py
├── pipeline.py
├── utils.py
└── methods/
    ├── __init__.py
    ├── base.py
    ├── p0_v_attention_map.py
    ├── p1_v_gmar.py
    ├── p1_a_denoising.py
    ├── p2_a_bundle.py
    ├── p2_x_integrated_gradients.py
    ├── p3_a_correlation.py
    └── p3_rtc_smoothness.py

tests/policies/xai/
├── __init__.py
├── test_config.py
├── test_buffer.py
├── test_base.py
├── test_p0_v.py
├── test_p1_a.py
├── test_p3_rtc.py
├── test_p1_v.py
├── test_p2_a.py
├── test_p2_x.py
├── test_p3_a.py
├── test_pipeline.py
└── test_integration.py
```
