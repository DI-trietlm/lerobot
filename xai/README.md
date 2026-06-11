# XAI Scripts — XVLA Analysis

Explainability scripts for the XVLA fine-tuned model (`xvla-pouring-0.1`).  
Runs on a GPU server. Does **not** require the full LeRobot framework.

## Scripts

| File | Description |
|------|-------------|
| `load_model_test.py` | Smoke test — loads the model and prints DaViT feature-map shapes + VRAM stats |
| `xai_feature_maps.py` | Phase 1a — visualizes mean-channel activation maps at each DaViT stage |
| `xai_p0v_raw_attention.py` | Phase 1b — extracts language→image attention from the Florence-2 encoder (single image) |
| `xai_p0v_raw_attention_video.py` | Phase 1c — same as above, processed frame-by-frame over an MP4 video |
| `xai_smolvla_attention_video.py` | **SmolVLA** — cross-attention (text→image) + self-attention (SVD) heatmaps over a `recorded_obs-*` frame sequence, one video per camera |
| `xai_smolvla_occlusion.py` | **SmolVLA** — *causal* occlusion sensitivity: hide image regions, measure predicted-action change → heatmap of regions the model actually relies on (static figures on key frames) |
| `xai_weight_diff.py` | Phase 4 — compares weight tensors between two XVLA checkpoints, reports per-component divergence |
| `xai_utils.py` | Shared bootstrap: lerobot stubs, model loading, image preprocessing |
| `check_environment.py` | Verifies that all dependencies are installed correctly |

## Requirements

```
torch >= 2.0
safetensors
transformers
opencv-python
matplotlib
Pillow
numpy
```

Install on the server:

```bash
pip install safetensors transformers opencv-python matplotlib Pillow numpy
```

## Directory structure expected

By default the scripts resolve paths **relative to this `xai/` directory**:

```
<project_root>/
├── xai/                         ← this directory
│   └── *.py
├── XVLA original source/        ← XVLA_SOURCE_DIR (default)
│   ├── modeling_florence2.py
│   ├── modeling_xvla.py
│   └── ...
└── xvla-pouring-0.1/           ← XVLA_MODEL_DIR (default)
    ├── model.safetensors
    └── config.json
```

If your layout is different (e.g., scripts placed inside another repo), set the two environment variables below.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `XVLA_SOURCE_DIR` | `../XVLA original source` (relative to `xai/`) | Absolute path to the XVLA source directory containing `modeling_florence2.py` etc. |
| `XVLA_MODEL_DIR` | `../xvla-pouring-0.1` (relative to `xai/`) | Absolute path to the fine-tuned model directory containing `model.safetensors` and `config.json` |

### Setting env vars (Linux / bash)

```bash
export XVLA_SOURCE_DIR=/path/to/XVLA_original_source
export XVLA_MODEL_DIR=/path/to/xvla-pouring-0.1
```

Or inline for a single run:

```bash
XVLA_SOURCE_DIR=/data/xvla/src XVLA_MODEL_DIR=/data/xvla/model \
    python3 xai_feature_maps.py --image my_image.png
```

## Running the scripts

All commands should be run from the `xai/` directory.

### 1. Smoke test (verify model loads correctly)

```bash
python3 load_model_test.py
```

Expected output: DaViT stage shapes, VRAM usage, and `SMOKE TEST PASSED`.

### 2. Feature map visualization

```bash
python3 xai_feature_maps.py --image /path/to/image.png --alpha 0.55 --cmap turbo
```

Outputs saved to `xai/outputs/`:
- `feature_maps_grid.png` — 2×5 grid (raw + overlay per stage)
- `feature_maps_combined.png` — averaged heatmap across all stages
- `feature_maps_stage{0-3}.png` — individual per-stage overlays

### 3. Language→image attention (single image)

```bash
python3 xai_p0v_raw_attention.py \
    --image /path/to/image.png \
    --instruction "Pour coffee from the orange cup into the light blue cup."
```

Outputs saved to `xai/outputs/`:
- `p0v_per_layer_grid.png` — attention per encoder layer
- `p0v_head_analysis_layer11.png` — per-head analysis (last layer)
- `p0v_aggregated.png` — mean across all layers
- `p0v_overlay.png` — last-layer overlay

### 4. Language→image attention (video)

```bash
python3 xai_p0v_raw_attention_video.py \
    --video /path/to/video.mp4 \
    --instruction "Pour coffee from the orange cup into the light blue cup."
```

Output: `xai/outputs/p0v_video_<videoname>.mp4`

### 4b. SmolVLA cross/self attention video (recorded_obs frames)

Unlike the XVLA scripts above, this one loads a **SmolVLA** policy via LeRobot
(`SmolVLAPolicy.from_pretrained`) and reads a `recorded_obs-*` directory
(`images/<camera>/*.png` + `metadata.jsonl`). It writes **one MP4 per camera**;
each frame is a 2-row figure like the reference paper image:

- Row 1 — cross-attention: source + per-word (`Put`, `orange`, `cup`, …) text→image maps
- Row 2 — self-attention: source + top-K image→image SVD components

```bash
# smoke test — load model, map cameras, print token layout, exit
uv run python xai_smolvla_attention_video.py \
    --obs-dir ../recorded_obs-05-06-03 --dry-run

# full run (outputs to xai/outputs/smolvla_attn_<dir>_<camera>.mp4)
uv run python xai_smolvla_attention_video.py \
    --obs-dir ../recorded_obs-05-06-03 \
    --model di-techinnova/smolvla-pick-cup-0.2 \
    --instruction "Put orange cup into white box." \
    --fps 5 --topk 6 --max-words 6
```

Interpretability flags (`--cross-mode`, `--self-mode`, `--layer`) and an honest caveat:

Raw SmolVLA attention is dominated by **query-independent "attention-sink" saliency**
— a few image patches get most of the attention regardless of which token is asking.
The prefix `[images | language | state]` is one bidirectional (prefix-LM) block, and
SmolVLA was **not** trained with a per-token text→image grounding objective (unlike
Stable Diffusion's cross-attention). So the per-word cross-attention rows all collapse
onto the same saliency pattern (measured pairwise correlation ≈ 0.96), and the dominant
self-attention SVD mode is that same pattern (corr ≈ 0.99). With raw attention the
cross- and self-attention rows therefore look almost identical — **and that is the
honest picture: the model is not attending per-word.**

Defaults are **raw** (`--cross-mode raw --self-mode raw`):

- `--cross-mode raw` (default): unprocessed language→image attention. Words look alike;
  truthful.
- `--cross-mode contrast`: divides each patch by saliency and subtracts the
  mean-over-words baseline. ⚠️ **Misleading on this model** — when the raw per-word maps
  are near-identical, the residual is mostly *noise*, so words appear to localise to
  different objects when they actually don't (e.g. "cup"/"box" land on wrong regions).
  Kept only for experimentation.
- `--self-mode raw` (default): plain SVD; `Top-0` is the saliency mode.
  `--self-mode centered` mean-centers before SVD to surface structural modes (this one
  is a legitimate decomposition, not noise amplification).
- `--layer mean|last|<int>`: which VLM layer to read (default mean over all 16).

**For genuine per-word grounding**, raw attention is the wrong tool (well documented:
"attention is not explanation"). The proper approach is gradient-based relevancy
(Grad-CAM / Chefer et al. attention-rollout-with-gradients) — not implemented here.

Notes:
- Downloads the SmolVLA checkpoint + the SmolVLM2-500M VLM backbone on first run.
  Runs on GPU or **CPU** (`--device cpu`); CPU is slow but works (~a few seconds/frame).
- `di-techinnova/smolvla-pick-cup-0.2` has **3 camera inputs** (camera1/2/3). With only
  two recorded cameras, the third is fed a black frame (kept so the model sees its
  expected input count) and is not rendered. Override mapping with `--cameras`.
- Each camera image becomes an 8×8 grid of patch tokens; the prefix sequence is
  `[img_cam1(64) | img_cam2(64) | img_cam3(64) | language(8) | state(1)] = 201` tokens.
- The recorded data has no robot-state vector, so the single state token is fed as
  zeros — it does not affect the text↔image attention being visualised.
- The whole policy is cast to float32 (the VLM ships as bfloat16 but the action-expert
  projections are float32); float32 is also the right choice for CPU.
- Heatmaps are overlaid on the model-input image (resize-with-pad to 512×512) so the
  attention grid aligns spatially with what the model actually sees.

### 4c. SmolVLA occlusion grounding (causal, recommended for "does the model use X")

Raw attention on this checkpoint is **not** per-word grounded (see 4b). To actually
*prove* the model relies on a region/word, use occlusion: hide a region, measure how
much the sampled action changes. The same fixed noise tensor is reused for the baseline
and every occluded run, so the action delta reflects the occlusion, not sampling noise.

```bash
# region grounding: where does the action depend on the image?
uv run python xai_smolvla_occlusion.py --device cpu \
    --obs-dir ../recorded_obs-05-06-03 --frames 0 16 32 --cameras camera1 camera2 --grid 8

# word grounding: where is a specific word grounded? (full vs ablated instruction)
uv run python xai_smolvla_occlusion.py --device cpu \
    --frames 0 --cameras camera1 camera2 --grid 6 --word box
```

Outputs: `xai/outputs/smolvla_occlusion[_word-<w>]_<dir>_f<NNN>.png` — top row source,
bottom row the sensitivity heatmap overlaid (bright = the model relies on this region).

Cost: each occluded cell needs a full action sample (~10 s/forward on CPU), so this is
**static figures on a few key frames**, not a video. `--grid N` = N² occlusions/camera;
`--word` doubles it (full + ablated instruction per cell). Use a GPU to go faster.

### 5. Weight comparison between checkpoints

```bash
python3 xai_weight_diff.py \
    --model_a ../xvla-pouring-0.1 \
    --model_b ../xvla-pouring-0.2
```

Outputs:
- Terminal report with per-component divergence stats
- `weight_diff_summary.png`
- `weight_diff_heatmap.png`
- `weight_diff_histogram.png`

Use `--dry-run` on any script to verify model loading without running the full pipeline.

## Notes

- Most scripts require a CUDA-capable GPU. `xai_weight_diff.py` runs on CPU only.
- The model checkpoint (`model.safetensors`) is **not** included in this repository. Obtain it separately and point `XVLA_MODEL_DIR` to its location.
- `xai_utils.py` bootstraps the lerobot dependency stubs at import time — it must not be run directly.
