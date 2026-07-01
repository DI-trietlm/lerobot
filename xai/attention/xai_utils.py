#!/usr/bin/env python3
"""xai_utils

Shared utilities for the XAI scripts in this directory.

This module supports two execution modes:

1) **In-repo LeRobot mode (default in this workspace)**
     Uses Florence-2 / XVLA sources from `src/lerobot/policies/xvla/`.
     Model weights can be loaded from either:
     - a local checkpoint directory, or
     - a Hugging Face model repo id (downloaded via `huggingface_hub`).

2) **Standalone mode (legacy)**
     If LeRobot sources are not importable, it falls back to loading from an
     external XVLA source directory using lightweight `lerobot.*` stubs.

Environment variables (optional):
    - `XVLA_MODEL` / `XVLA_MODEL_ID`: default HF repo id to load.
    - `XVLA_MODEL_DIR`: default local checkpoint directory to load.
    - `XVLA_MODEL_REVISION`: optional HF revision.
    - `XVLA_LOCAL_FILES_ONLY`: set to "1" to avoid network downloads.
    - `XVLA_SOURCE_DIR`: legacy standalone mode source directory.

All scripts in `xai/` import from this module instead of duplicating setup.
"""

import importlib.util
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(UTILS_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    # Allow running scripts from the repo root without installing the package.
    sys.path.insert(0, SRC_DIR)

LEGACY_SOURCE_DIR = os.environ.get(
    "XVLA_SOURCE_DIR",
    os.path.join(PROJECT_DIR, "XVLA original source"),
)
LEGACY_MODEL_DIR = os.environ.get(
    "XVLA_MODEL_DIR",
    os.path.join(PROJECT_DIR, "xvla-pouring-0.1"),
)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "artifacts", "attention_outputs")
PACKAGE_NAME = "xvla_src"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
DAVIT_INPUT_SIZE = (224, 224)
CAMERA_INPUT_SIZE = (256, 256)


def _install_lerobot_stubs() -> None:
    """Registers no-op stub modules for every lerobot.* symbol used by XVLA sources."""

    def _make(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    _make("lerobot")

    cfg = _make("lerobot.configs")

    class _PreTrainedConfig:
        pass

    class _FeatureType:
        VISUAL = "VISUAL"
        STATE = "STATE"
        ACTION = "ACTION"

    class _NormalizationMode:
        IDENTITY = "IDENTITY"
        MEAN_STD = "MEAN_STD"

    _PreTrainedConfig.register_subclass = staticmethod(lambda key: lambda cls: cls)

    cfg.PreTrainedConfig = _PreTrainedConfig
    cfg.PolicyFeature = type("PolicyFeature", (), {})
    cfg.FeatureType = _FeatureType
    cfg.NormalizationMode = _NormalizationMode
    cfg.PipelineFeatureType = _FeatureType

    _make("lerobot.utils")
    ucc = _make("lerobot.utils.constants")
    ucc.ACTION = "action"
    ucc.OBS_LANGUAGE_TOKENS = "observation.language_tokens"
    ucc.OBS_STATE = "observation.state"
    ucc.OBS_IMAGES = "observation.images"
    ucc.OBS_PREFIX = "observation."
    ucc.IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    ucc.POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
    ucc.POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

    uiu = _make("lerobot.utils.import_utils")
    uiu._transformers_available = True
    uiu.require_package = lambda *a, **kw: None

    _make("lerobot.utils.utils")
    sys.modules["lerobot.utils"].populate_queues = lambda queues, batch, **kw: queues

    opt = _make("lerobot.optim")
    opt.CosineDecayWithWarmupSchedulerConfig = object
    opt.XVLAAdamWConfig = object

    pre = _make("lerobot.pretrained")

    class _PreTrainedPolicy(torch.nn.Module):
        def __init__(self, config, **kw):
            super().__init__()
            self.config = config

    pre.PreTrainedPolicy = _PreTrainedPolicy
    pre.T = None
    _make("lerobot.policies").PreTrainedPolicy = _PreTrainedPolicy

    proc = _make("lerobot.processor")

    class _NoOpStep:
        def __init__(self, *a, **kw): pass
        def __call__(self, x): return x
        def transform_features(self, f): return f
        def get_config(self): return {}

    class _NoOpPipeline:
        def __init__(self, *a, **kw): pass
        def __class_getitem__(cls, item): return cls

    class _StepRegistry:
        @staticmethod
        def register(name):
            return lambda cls: cls

    for _name in [
        "AddBatchDimensionProcessorStep", "DeviceProcessorStep",
        "NormalizerProcessorStep", "ObservationProcessorStep",
        "PolicyAction", "ProcessorStep", "RenameObservationsProcessorStep",
        "TokenizerProcessorStep", "UnnormalizerProcessorStep",
    ]:
        setattr(proc, _name, _NoOpStep)

    proc.PolicyProcessorPipeline = _NoOpPipeline
    proc.ProcessorStepRegistry = _StepRegistry
    proc.policy_action_to_transition = lambda x: x
    proc.transition_to_policy_action = lambda x: x

    typ = _make("lerobot.types")

    class _EnvTransition(dict):
        pass

    class _TransitionKey:
        OBSERVATION = "observation"
        ACTION = "action"
        COMPLEMENTARY_DATA = "complementary_data"

    typ.EnvTransition = _EnvTransition
    typ.TransitionKey = _TransitionKey


def _register_source_package(source_dir: str) -> None:
    """
    Registers XVLA original source/ as xvla_src nested under a fake parent (xvla_parent)
    so that relative imports inside modeling_xvla.py resolve without errors.
    """
    parent_name = "xvla_parent"
    parent_pkg = types.ModuleType(parent_name)
    parent_pkg.__path__ = []
    parent_pkg.__package__ = parent_name
    sys.modules[parent_name] = parent_pkg

    pre_stub = types.ModuleType(f"{parent_name}.pretrained")
    pre_stub.__package__ = parent_name

    class _PreTrainedPolicy(torch.nn.Module):
        def __init__(self, config, **kw):
            super().__init__()
            self.config = config

    pre_stub.PreTrainedPolicy = _PreTrainedPolicy
    pre_stub.T = None
    sys.modules[f"{parent_name}.pretrained"] = pre_stub

    utils_stub = types.ModuleType(f"{parent_name}.utils")
    utils_stub.__package__ = parent_name
    utils_stub.populate_queues = lambda queues, batch, **kw: queues
    sys.modules[f"{parent_name}.utils"] = utils_stub
    setattr(parent_pkg, "utils", utils_stub)

    full_pkg_name = f"{parent_name}.{PACKAGE_NAME}"
    pkg = types.ModuleType(full_pkg_name)
    pkg.__path__ = [source_dir]
    pkg.__package__ = full_pkg_name
    pkg.__spec__ = importlib.util.spec_from_file_location(
        full_pkg_name,
        os.path.join(source_dir, "__init__.py"),
        submodule_search_locations=[source_dir],
    )
    sys.modules[full_pkg_name] = pkg
    sys.modules[PACKAGE_NAME] = pkg
    setattr(parent_pkg, PACKAGE_NAME, pkg)

    py_files = [
        f[:-3] for f in os.listdir(source_dir)
        if f.endswith(".py") and f != "__init__.py"
    ]
    sub_specs: dict[str, importlib.machinery.ModuleSpec] = {}
    for mod_name in py_files:
        nested = f"{full_pkg_name}.{mod_name}"
        flat = f"{PACKAGE_NAME}.{mod_name}"
        path = os.path.join(source_dir, f"{mod_name}.py")
        spec = importlib.util.spec_from_file_location(nested, path)
        sub = importlib.util.module_from_spec(spec)
        sub.__package__ = full_pkg_name
        sub.__name__ = nested
        sys.modules[nested] = sub
        sys.modules[flat] = sub
        sub_specs[mod_name] = spec

    SKIP_EXEC = {"modeling_xvla", "processor_xvla"}
    load_order = [
        "utils",
        "configuration_florence2",
        "configuration_xvla",
        "action_hub",
        "soft_transformer",
        "modeling_florence2",
    ]
    remaining = [m for m in py_files if m not in load_order and m not in SKIP_EXEC]
    for mod_name in load_order + remaining:
        if mod_name in sub_specs:
            try:
                sub_specs[mod_name].loader.exec_module(
                    sys.modules[f"{PACKAGE_NAME}.{mod_name}"]
                )
            except Exception as exc:
                print(f"  [warn] could not load xvla_src.{mod_name}: {exc}")
def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y"}


def _repo_xvla_source_available() -> bool:
    base = os.path.join(PROJECT_DIR, "src", "lerobot", "policies", "xvla")
    return (
        os.path.isfile(os.path.join(base, "modeling_florence2.py"))
        and os.path.isfile(os.path.join(base, "configuration_florence2.py"))
    )


def _ensure_repo_package_chain() -> None:
    """Create lightweight package modules for lerobot.policies.xvla.

    This avoids executing lerobot.policies.__init__ (which imports many optional policies).
    """
    repo_root = os.path.join(PROJECT_DIR, "src")
    lerobot_dir = os.path.join(repo_root, "lerobot")
    policies_dir = os.path.join(lerobot_dir, "policies")
    xvla_dir = os.path.join(policies_dir, "xvla")

    def _ensure_pkg(name: str, path: str) -> None:
        if name in sys.modules:
            return
        pkg = types.ModuleType(name)
        pkg.__path__ = [path]
        pkg.__package__ = name
        sys.modules[name] = pkg

    _ensure_pkg("lerobot", lerobot_dir)
    _ensure_pkg("lerobot.policies", policies_dir)
    _ensure_pkg("lerobot.policies.xvla", xvla_dir)


def _import_repo_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_name.rsplit(".", 1)[0]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _import_florence2_classes():
    """Return (Florence2Config, Florence2ForConditionalGeneration).

    Prefers in-repo LeRobot sources; falls back to the legacy standalone mode.
    """
    if _env_flag("XVLA_USE_LEGACY_SOURCE"):
        if not os.path.isdir(LEGACY_SOURCE_DIR):
            raise FileNotFoundError(
                "XVLA_USE_LEGACY_SOURCE is set but XVLA_SOURCE_DIR does not exist: "
                f"{LEGACY_SOURCE_DIR}"
            )
        _install_lerobot_stubs()
        _register_source_package(LEGACY_SOURCE_DIR)
        from xvla_src.configuration_florence2 import Florence2Config
        from xvla_src.modeling_florence2 import Florence2ForConditionalGeneration
        return Florence2Config, Florence2ForConditionalGeneration

    if _repo_xvla_source_available():
        try:
            _ensure_repo_package_chain()
            base = os.path.join(PROJECT_DIR, "src", "lerobot", "policies", "xvla")
            cfg_mod = _import_repo_module(
                "lerobot.policies.xvla.configuration_florence2",
                os.path.join(base, "configuration_florence2.py"),
            )
            mdl_mod = _import_repo_module(
                "lerobot.policies.xvla.modeling_florence2",
                os.path.join(base, "modeling_florence2.py"),
            )
            return cfg_mod.Florence2Config, mdl_mod.Florence2ForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "Failed to import LeRobot XVLA sources. "
                "Install XVLA/transformers deps (e.g., `uv sync --locked --extra xvla`) "
                "or set XVLA_USE_LEGACY_SOURCE=1 with a valid XVLA_SOURCE_DIR."
            ) from exc

    if os.path.isdir(LEGACY_SOURCE_DIR):
        _install_lerobot_stubs()
        _register_source_package(LEGACY_SOURCE_DIR)
        from xvla_src.configuration_florence2 import Florence2Config
        from xvla_src.modeling_florence2 import Florence2ForConditionalGeneration
        return Florence2Config, Florence2ForConditionalGeneration

    raise FileNotFoundError(
        "XVLA source code not found. Either install from this repo (src/lerobot/policies/xvla) "
        "or set XVLA_SOURCE_DIR to the external XVLA source directory."
    )


def resize_with_pad(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Letterbox-resize maintaining aspect ratio; padding value = 0."""
    if img.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {img.shape}")
    ch, cw = img.shape[2], img.shape[3]
    if ch == height and cw == width:
        return img
    ratio = max(cw / width, ch / height)
    rh, rw = int(ch / ratio), int(cw / ratio)
    resized = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
    ph, pw = max(0, height - rh), max(0, width - rw)
    return F.pad(resized, (pw, 0, ph, 0), value=0.0)


def load_image_pil(path: str):
    """Returns a PIL Image (RGB) from the given path."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def pil_to_tensor(pil_img, device: torch.device) -> torch.Tensor:
    """Converts a PIL RGB image to a normalized, letterboxed (1, 3, 224, 224) float32 tensor."""
    import numpy as np
    arr = torch.from_numpy(np.array(pil_img)).float() / 255.0
    arr = arr.permute(2, 0, 1).unsqueeze(0)
    mean = IMAGENET_MEAN.view(1, 3, 1, 1)
    std = IMAGENET_STD.view(1, 3, 1, 1)
    arr = (arr - mean) / std
    return resize_with_pad(arr, DAVIT_INPUT_SIZE[0], DAVIT_INPUT_SIZE[1]).to(device)

def _default_model_spec() -> str:
    """Return a default model identifier.

    Priority:
      1) `XVLA_MODEL_DIR` if it exists locally
      2) `./xvla-pouring-0.1` if it exists locally
      3) `XVLA_MODEL` / `XVLA_MODEL_ID`
      4) `lerobot/xvla-base`
    """
    if os.path.isdir(LEGACY_MODEL_DIR):
        return LEGACY_MODEL_DIR
    local_default = os.path.join(PROJECT_DIR, "xvla-pouring-0.1")
    if os.path.isdir(local_default):
        return local_default
    return (
        os.environ.get("XVLA_MODEL")
        or os.environ.get("XVLA_MODEL_ID")
        or "lerobot/xvla-base"
    )


def resolve_model_dir(
    model_id_or_path: str | None = None,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool | None = None,
) -> str:
    """Resolve a model source (local dir or HF repo id) to a local directory."""
    model_id_or_path = (model_id_or_path or _default_model_spec()).strip()
    if os.path.isdir(model_id_or_path):
        return model_id_or_path

    # Treat as Hugging Face repo id.
    from huggingface_hub import snapshot_download

    if revision is None:
        revision = os.environ.get("XVLA_MODEL_REVISION")
    if local_files_only is None:
        local_files_only = _env_flag("XVLA_LOCAL_FILES_ONLY")

    return snapshot_download(
        repo_id=model_id_or_path,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "*.json",
        ],
    )


def load_florence2_config(model_id_or_path: str | None = None):
    Florence2Config, _ = _import_florence2_classes()
    model_dir = resolve_model_dir(model_id_or_path)
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    if "florence_config" not in raw:
        raise KeyError(
            "config.json is missing 'florence_config'. "
            "Make sure you are pointing to a LeRobot XVLA checkpoint (policy config)."
        )
    return Florence2Config(**raw["florence_config"])


def load_raw_state_dict(model_dir_or_repo_id: str) -> dict[str, torch.Tensor]:
    model_dir = resolve_model_dir(model_dir_or_repo_id)
    model_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.safetensors not found at {model_path}")
    import safetensors.torch as st
    return st.load_file(model_path)


def load_normalizer_stats(model_dir_or_repo_id: str) -> dict[str, torch.Tensor]:
    model_dir = resolve_model_dir(model_dir_or_repo_id)
    stats_path = os.path.join(
        model_dir,
        "policy_preprocessor_step_7_normalizer_processor.safetensors",
    )
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"normalizer stats not found at {stats_path}")
    import safetensors.torch as st
    return st.load_file(stats_path)


def load_vision_tower(
    device: torch.device,
    keep_language_encoder: bool = False,
    model_id_or_path: str | None = None,
):
    """
    Loads the Florence-2 model from xvla-pouring-0.1/model.safetensors.

    By default the language decoder is deleted to save VRAM.
    Set keep_language_encoder=True for text-guided attention extraction.
    Returns the model in eval mode on the specified device.
    """
    _, Florence2ForConditionalGeneration = _import_florence2_classes()

    model_dir = resolve_model_dir(model_id_or_path)
    config = load_florence2_config(model_dir)
    model = Florence2ForConditionalGeneration(config)

    if not keep_language_encoder and hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
            del lm.model.decoder
        if hasattr(lm, "lm_head"):
            del lm.lm_head

    import safetensors.torch as st
    state_dict = st.load_file(os.path.join(model_dir, "model.safetensors"))

    prefix = "model.vlm."
    florence_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    enc_key = "language_model.model.encoder.embed_tokens.weight"
    shared_key = "language_model.model.shared.weight"
    if enc_key in florence_sd and shared_key not in florence_sd:
        florence_sd[shared_key] = florence_sd[enc_key]

    missing, _ = model.load_state_dict(florence_sd, strict=False)
    non_decoder_missing = [k for k in missing if "decoder" not in k and "lm_head" not in k]
    if non_decoder_missing:
        print(f"  [WARN] Non-decoder missing keys: {non_decoder_missing[:5]}")

    model.vision_tower = model.vision_tower.to(dtype=torch.float32)
    model.image_projection.data = model.image_projection.data.to(dtype=torch.float32)
    model.image_proj_norm = model.image_proj_norm.to(dtype=torch.float32)

    if keep_language_encoder:
        model.language_model.model.encoder.to(dtype=torch.float32)

    return model.to(device).eval()


def report_vram(device: torch.device, label: str = "") -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / (1024 ** 3)
    free, _ = torch.cuda.mem_get_info(device)
    free_gb = free / (1024 ** 3)
    tag = f" [{label}]" if label else ""
    print(f"VRAM{tag}  alloc={alloc:.2f}GB  reserved={reserved:.2f}GB  "
          f"free={free_gb:.2f}GB  total={total:.1f}GB")


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
