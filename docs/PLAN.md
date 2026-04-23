# XAI Integration Plan for LeRobot

## 1. Overview

Tích hợp 7 phương pháp XAI (Explainable AI) vào LeRobot inference/eval pipeline. Các methods có thể bật/tắt độc lập qua config flags. Output được lưu cả in-memory buffer và local files.

### 7 XAI Methods

| ID | Name | Target | Timing | Cost |
|----|------|--------|--------|------|
| P0-V | Raw Attention Map | Florence-2 Encoder | Real-time | ~0 |
| P1-V | GMAR (Gradient x Attn Rollout) | Florence-2 Encoder | Offline | 1.5x forward |
| P1-A | Denoising Trajectory Plot | Flow Matching ODE | Real-time log | ~0 |
| P2-A | Action Sample Bundle | Flow Matching Output | Offline | N x forward |
| P2-X | Integrated Gradients | Cross-modal | Offline | 50x forward |
| P3-A | Action Dim Correlation | Action Space | Offline | Low |
| P3-RTC | Chunk Boundary Smoothness | RTC Pipeline | Real-time | ~0 |

### Design Goals

- **Optional**: Mỗi method có flag riêng (`use_p0_v`, `use_p1_v`, etc.)
- **Inference-only**: Tích hợp vào eval pipeline, không ảnh hưởng training
- **Dual storage**: In-memory buffer + optional save to disk (JSON/PNG)
- **Per-method config**: Thresholds và parameters tùy chỉnh được per method

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Eval Pipeline                                   │
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐   │
│  │  Env     │───▶│ Preprocessor │───▶│   Policy   │───▶│ Postprocessor│   │
│  └──────────┘    └──────────────┘    └────────────┘    └──────────────┘   │
│                                             │                               │
│                          ┌──────────────────┴──────────────────┐           │
│                          │       XAI Pipeline (optional)       │           │
│                          └─────────────────────────────────────┘           │
│                                              │                             │
│                   ┌──────────────────────────┼──────────────────────────┐ │
│                   │ Real-time Methods         │ Offline Methods          │ │
│                   │ (P0-V, P1-A, P3-RTC)     │ (P1-V, P2-A, P2-X, P3-A)│ │
│                   │ Hook vào select_action()  │ Triggered sau episode   │ │
│                   └───────────────────────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Flow

```
Episode starts
      │
      ▼
XAIPipeline.start_episode(episode_id)
      │
      ├─[P0-V] Register attention hooks on Florence-2 encoder
      ├─[P1-A] Initialize DenoisingTracker
      ├─[P3-RTC] Initialize BoundarySmoothnessMonitor
      │
      ▼
Each step in rollout:
      │
      ├─[P0-V] Hook captures attention → compute entropy → flag if high
      ├─[P1-A] Log convergence speed from denoising steps (lightweight)
      ├─[P3-RTC] Compute boundary smoothness
      │
      ▼
XAIPipeline.end_episode()
      │
      ├─[P3-RTC] Compute episode_quality
      ├─[Real-time methods done] → Save to EpisodeXAIBuffer
      │
      ├─[Triggered: flagged > 10%] → Schedule offline methods:
      │     ├─[P1-V] GMAR on flagged steps
      │     ├─[P2-A] Action bundle analysis
      │     ├─[P2-X] Integrated gradients
      │     └─[P3-A] Correlation analysis
      │
      ▼
EpisodeXAIBuffer saved to output_dir + in-memory
```

---

## 3. File Structure

```
src/lerobot/policies/xai/
├── __init__.py                          # Exports: XAIConfig, XAIPipeline, XAIMethod
├── config.py                            # XAIConfig dataclass with per-method flags
├── buffer.py                            # StepRecord, EpisodeXAIBuffer dataclasses
├── pipeline.py                          # XAIPipeline orchestration class
├── utils.py                             # Shared utilities (attention_entropy, etc.)
└── methods/
    ├── __init__.py
    ├── base.py                          # XAIMethod base class (abc.ABC)
    ├── p0_v_attention_map.py             # Raw Attention Map
    ├── p1_v_gmar.py                      # GMAR (Gradient-weighted Multi-head Attention Rollout)
    ├── p1_a_denoising.py                 # Denoising Trajectory
    ├── p2_a_bundle.py                    # Action Sample Bundle
    ├── p2_x_integrated_gradients.py      # Integrated Gradients
    ├── p3_a_correlation.py               # Action Dimension Correlation
    └── p3_rtc_smoothness.py              # Chunk Boundary Smoothness
```

---

## 4. XAIConfig

```python
@dataclass
class XAIConfig:
    # =========================================================
    # Real-time methods (default: all off)
    # =========================================================
    use_p0_v_attention: bool = False          # Raw Attention Map
    use_p1_a_denoising: bool = False          # Denoising Trajectory
    use_p3_rtc_smoothness: bool = False        # Chunk Boundary Smoothness

    # =========================================================
    # Offline methods (default: all off)
    # =========================================================
    use_p1_v_gmar: bool = False                # GMAR
    use_p2_a_bundle: bool = False              # Action Sample Bundle
    use_p2_x_integrated_gradients: bool = False # Integrated Gradients
    use_p3_a_correlation: bool = False         # Action Dimension Correlation

    # =========================================================
    # Common settings
    # =========================================================
    output_dir: str = "xai_outputs"
    save_videos: bool = False                  # Save annotated videos
    save_heatmaps: bool = True                 # Save heatmap images

    # =========================================================
    # P0-V settings
    # =========================================================
    p0_v_layer_indices: list = field(default_factory=lambda: [-1, -3, -6])
    p0_v_patch_grid: tuple = (14, 14)
    entropy_threshold: float = 3.5

    # =========================================================
    # P1-V settings
    # =========================================================
    p1_v_target_action_dim: int | None = None  # None = avg all joints

    # =========================================================
    # P1-A settings
    # =========================================================
    p1_a_log_full_trajectory: bool = False     # Default: scalar only

    # =========================================================
    # P2-A settings
    # =========================================================
    p2_a_n_samples: int = 50
    p2_a_trigger_on_flagged: bool = True

    # =========================================================
    # P2-X settings
    # =========================================================
    p2_x_n_steps: int = 50
    p2_x_target_joints: list | None = None

    # =========================================================
    # P3-RTC settings
    # =========================================================
    p3_rtc_overlap_steps: int = 3
    boundary_low_threshold: float = 0.75
    boundary_critical_threshold: float = 0.5

    # =========================================================
    # Episode quality settings
    # =========================================================
    quality_threshold: float = 0.80            # For demo filtering
    flagged_episode_threshold: float = 0.1     # % flagged steps to trigger offline
```

---

## 5. Data Structures

### StepRecord

```python
@dataclass
class StepRecord:
    step_idx: int
    timestamp: float

    # P0-V: Raw Attention
    attn_entropy: float = 0.0
    attn_compressed: torch.Tensor | None = None  # [7, 7] compressed heatmap

    # P1-A: Denoising
    convergence_speed: list[float] = field(default_factory=list)

    # P3-RTC: Boundary
    boundary_sim: float | None = None

    # Flagging
    flagged: bool = False
    flag_reason: str | None = None
```

### EpisodeXAIBuffer

```python
@dataclass
class EpisodeXAIBuffer:
    episode_id: str
    episode_index: int = 0
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0

    # Step-level data
    step_records: list[StepRecord] = field(default_factory=list)
    flagged_steps: list[int] = field(default_factory=list)

    # Computed post-episode (from real-time methods)
    episode_quality: float | None = None        # From P3-RTC
    mean_entropy: float | None = None          # From P0-V
    attention_stability: float | None = None   # Std of attn centroid

    # Offline results (filled after episode ends)
    gmar_heatmaps: dict[int, torch.Tensor] | None = None  # step_idx -> [B, 14, 14]
    action_bundle: dict | None = None                    # P2-A results
    integrated_gradients: dict | None = None            # P2-X results
    correlation_matrix: torch.Tensor | None = None       # P3-A: [dim, dim]

    # Summary
    should_include_in_training: bool = True

    def summary(self) -> dict:
        """Return a JSON-serializable summary dict."""
        ...

    def save(self, output_dir: Path):
        """Save buffer to disk."""
        ...
```

---

## 6. XAIPipeline Class

```python
class XAIPipeline:
    def __init__(self, policy: PreTrainedPolicy, config: XAIConfig):
        self.config = config
        self.policy = policy
        self.current_episode: EpisodeXAIBuffer | None = None

        # Initialize enabled methods
        self.enabled_methods = self._get_enabled_methods()

        # Real-time method handlers
        self.p0_v: P0VAttentionMap | None = None
        self.p1_a: P1ADenoisingTracker | None = None
        self.p3_rtc: P3RTCSmoothnessMonitor | None = None

    def _get_enabled_methods(self) -> dict[str, bool]:
        """Return dict of method_name -> enabled."""
        ...

    def start_episode(self, episode_id: str, episode_index: int = 0):
        """Called at start of each episode."""
        self.current_episode = EpisodeXAIBuffer(
            episode_id=episode_id,
            episode_index=episode_index,
            timestamp_start=time.time(),
        )

        # Initialize real-time methods
        if self.config.use_p0_v_attention:
            self.p0_v = P0VAttentionMap(self.policy, self.config)
            self.p0_v.register()
        if self.config.use_p1_a_denoising:
            self.p1_a = P1ADenoisingTracker()
        if self.config.use_p3_rtc_smoothness:
            self.p3_rtc = P3RTCSmoothnessMonitor(self.config)

    def on_after_action(self, batch: dict, action_chunk: torch.Tensor, step_idx: int):
        """Called after each policy.select_action() call."""
        if self.current_episode is None:
            return

        # Capture real-time metrics
        record = StepRecord(step_idx=step_idx, timestamp=time.time())

        # P0-V: Attention entropy
        if self.p0_v:
            entropy, compressed = self.p0_v.compute_entropy()
            record.attn_entropy = entropy
            record.attn_compressed = compressed
            self.p0_v.clear()

        # P1-A: Denoising convergence
        if self.p1_a:
            record.convergence_speed = self.p1_a.get_convergence_speed()
            self.p1_a.clear()

        # P3-RTC: Boundary smoothness
        if self.p3_rtc:
            record.boundary_sim = self.p3_rtc.update(action_chunk)
            record.status = self.p3_rtc.get_status(record.boundary_sim)

        # Flagging logic
        if record.attn_entropy > self.config.entropy_threshold:
            record.flagged = True
            record.flag_reason = "high_entropy"
        elif record.status == "critical_jerk":
            record.flagged = True
            record.flag_reason = "critical_jerk"
        elif record.status == "warning_jerk":
            record.flagged = True
            record.flag_reason = "warning_jerk"

        self.current_episode.step_records.append(record)
        if record.flagged:
            self.current_episode.flagged_steps.append(step_idx)

    def end_episode(self) -> EpisodeXAIBuffer:
        """Called at end of episode. Triggers offline methods if needed."""
        if self.current_episode is None:
            return None

        self.current_episode.timestamp_end = time.time()

        # Cleanup hooks
        if self.p0_v:
            self.p0_v.remove()
            self.p0_v = None

        # Compute episode-level metrics
        if self.p3_rtc:
            self.current_episode.episode_quality = self.p3_rtc.episode_quality()

        self.current_episode.mean_entropy = self._compute_mean_entropy()

        # Decision: should we run offline methods?
        flagged_ratio = len(self.current_episode.flagged_steps) / max(1, len(self.current_episode.step_records))
        should_run_offline = flagged_ratio > self.config.flagged_episode_threshold

        if should_run_offline and self._has_offline_methods():
            self._schedule_offline_analysis(self.current_episode)

        # Determine training inclusion
        self.current_episode.should_include_in_training = (
            self.current_episode.episode_quality >= self.config.quality_threshold
        )

        episode = self.current_episode
        self.current_episode = None
        return episode

    def _schedule_offline_analysis(self, episode: EpisodeXAIBuffer):
        """Trigger offline XAI methods asynchronously."""
        ...

    def wrap_policy(self, policy: PreTrainedPolicy) -> PreTrainedPolicy:
        """Return a policy wrapped with XAI hooks."""
        ...
```

---

## 7. Method Implementations

### P0-V: Raw Attention Map

**Hook Point**: `model.vlm.language_model.model.encoder.layers[i].self_attn`

**Behavior**:
- Register forward hooks on specified encoder layers
- Capture attention weights `[B, num_heads, seq_len, seq_len]`
- Compute `lang_to_img` attention (language tokens attending to image patches)
- Compute entropy as: `entropy = -(p * log(p)).sum()` where p = softmax(attn)

**Output per step**:
- `entropy`: scalar
- `compressed`: `[7, 7]` tensor (14x14 pooled 2x2)

### P1-A: Denoising Trajectory

**Hook Point**: Inside `XVLAModel.generate_actions()` - patch to track `x_t` at each step

**Behavior**:
- In non-RTC path: track `action` tensor after each denoising iteration
- Compute delta between consecutive `x_t`: `delta = ||x_t[i] - x_t[i-1]||_2`
- Convergence speed = list of deltas

**Output per step**:
- `convergence_speed`: list of deltas (or empty if not enough steps)

### P3-RTC: Chunk Boundary Smoothness

**Hook Point**: After `policy.select_action()` returns

**Behavior**:
- Store previous action chunk's tail
- Compare with current chunk's head using cosine similarity
- `overlap_steps = 3` (configurable)

**Output per step**:
- `boundary_sim`: scalar (0-1)
- Status: `ok` | `warning_jerk` | `critical_jerk`

### P1-V: GMAR (Offline)

**Hook Point**: Full backward pass through Florence-2 encoder

**Behavior**:
1. Forward pass with hooks to capture attention weights
2. Create noisy action at t=0.5
3. Forward through transformer to get action prediction
4. Backward from target action dimension to get gradients w.r.t. attention
5. Compute: `head_weights = grad.abs().mean(dim=(-2, -1))`
6. Rollout: `weighted_attn = attn * head_weights`, then matmul across layers

**Output**:
- `gmar_heatmaps`: dict mapping step_idx -> `[B, 14, 14]` heatmap

### P2-A: Action Sample Bundle (Offline)

**Hook Point**: Called after episode ends for flagged episodes

**Behavior**:
1. Cache VLM encoding (don't recompute per sample)
2. Sample N=50 action trajectories with different noise
3. Compute mean, std, coefficient of variation
4. Detect multimodal via GMM on action magnitude

**Output**:
- `mean`: `[B, chunk, dim]`
- `std`: `[B, chunk, dim]`
- `cv`: coefficient of variation
- `is_multimodal`: bool

### P2-X: Integrated Gradients (Offline)

**Hook Point**: Called after episode ends, monthly audit

**Behavior**:
1. Compute baseline (zero image, empty language, zero proprio)
2. Interpolate 50 steps from baseline to actual
3. Accumulate gradients at each step
4. IG = (actual - baseline) * avg_gradient

**Output**:
- `vision_pct`: percentage attribution to vision
- `language_pct`: percentage attribution to language
- `proprio_pct`: percentage attribution to proprio

### P3-A: Action Dimension Correlation (Offline)

**Hook Point**: Called after episode ends

**Behavior**:
1. Use action bundle from P2-A (or compute if not available)
2. Collect samples across multiple observations
3. Compute covariance matrix: `cov = torch.cov(samples.T)`
4. Normalize to correlation: `corr = cov / (std.outer(std) + eps)`

**Output**:
- `correlation_matrix`: `[dim_action, dim_action]`

---

## 8. Config Integration

### Eval Config Extension

```yaml
# In EvalPipelineConfig (lerobot/configs/eval.py)
xai:
  type: xai
  use_p0_v_attention: false
  use_p1_a_denoising: false
  use_p3_rtc_smoothness: false
  use_p1_v_gmar: false
  use_p2_a_bundle: false
  use_p2_x_integrated_gradients: false
  use_p3_a_correlation: false
  output_dir: "xai_outputs"
  entropy_threshold: 3.5
  quality_threshold: 0.80
```

### Usage in lerobot_eval.py

```python
from lerobot.policies.xai import XAIConfig, XAIPipeline

# In eval_main():
if cfg.xai is not None:
    xai_pipeline = XAIPipeline(policy, cfg.xai)
    xai_pipeline.wrap_policy(policy)

    # Or use hooks on preprocessor:
    # preprocessor.after_step_hooks.append(xai_pipeline.on_after_step)
```

---

## 9. Output Format

```
xai_outputs/
└── episode_0000/
    ├── summary.json              # Episode-level metrics
    ├── step_records.json         # Per-step data (lightweight)
    ├── heatmaps/
    │   ├── step_0000_p0_v.png   # Attention heatmap
    │   ├── step_0010_p0_v.png
    │   └── ...
    ├── gmar/
    │   ├── step_0005_gmar.png
    │   └── ...
    ├── denoising/
    │   └── convergence.png       # If full trajectory logged
    ├── bundle/
    │   └── analysis.json         # Action bundle stats
    └── ig/
        └── attribution.json       # Integrated gradients summary
```

---

## 10. Implementation Order

| Phase | Task | Duration | Methods |
|-------|------|----------|---------|
| 1 | Infrastructure | 1 day | - |
| | Create `src/lerobot/policies/xai/` structure | | |
| | Implement `XAIConfig` dataclass | | |
| | Implement `StepRecord`, `EpisodeXAIBuffer` | | |
| | Implement `XAIMethod` base class | | |
| | Implement `XAIPipeline` orchestration | | |
| 2 | Real-time methods | 2 days | P3-RTC, P0-V |
| | P3-RTC Smoothness Monitor | | |
| | P0-V Attention Map with hooks | | |
| 3 | P1-A Denoising | 1 day | P1-A |
| | P1-A Denoising Tracker | | |
| 4 | Offline: P1-V GMAR | 2 days | P1-V |
| | P1-V GMAR implementation | | |
| 5 | Offline: P2-A Bundle | 2 days | P2-A |
| | P2-A Action Sample Bundle | | |
| 6 | Offline: P2-X IG | 3 days | P2-X |
| | P2-X Integrated Gradients | | |
| 7 | Offline: P3-A Correlation | 1 day | P3-A |
| | P3-A Correlation Heatmap | | |
| 8 | Integration & Testing | 2 days | All |
| | Hook into lerobot_eval.py | | |
| | Write tests | | |
| | Documentation | | |

**Total estimated**: ~14 days

---

## 11. Dependencies

- **No new dependencies** for real-time methods (P0-V, P1-A, P3-RTC)
- **scikit-learn** for P2-A GMM (already in test deps)
- **No changes to existing code** - all integration via hooks/callbacks

---

## 12. Backward Compatibility

- XAI disabled by default (`XAIConfig(use_xai=False)`)
- Existing eval scripts work without modification
- No changes to policy classes themselves
- Optional: can be enabled per-eval via config

---

## 13. Considerations

### Florence-2 Attention Hook Path

The attention hook must access:
```python
model.vlm.language_model.model.encoder.layers[i].self_attn
```

This path may need adjustment based on exact Florence-2 model structure. Verify during implementation.

### RTC vs Non-RTC

`XVLAModel.generate_actions()` has two code paths:
1. **Non-RTC**: Standard denoising loop
2. **RTC**: Uses `rtc_processor.denoise_step()`

P1-A (Denoising Tracker) must handle both paths.

### Memory Management

- P2-A with N=50 samples is memory-intensive
- Consider batching samples or using CPU offload
- Clear caches after each episode

### Thread Safety

If using async/multithreaded eval:
- XAIPipeline should be episode-level (not shared across threads)
- Or use locks for shared data structures
