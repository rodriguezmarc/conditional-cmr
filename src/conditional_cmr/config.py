"""Run configuration: one frozen dataclass, filled from a per-setup YAML file."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml

SETUP_IDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Techniques:
    """Metadata-aware techniques enabled by a setup (paper Table I columns)."""

    train_text_encoder: bool
    metadata_free_cfg: bool
    inverse_frequency_sampling: bool
    contrastive_batching: bool


@dataclass(frozen=True)
class RunConfig:
    """Every value that defines a training, inference or evaluation run."""

    setup: int
    name: str
    techniques: Techniques

    # Paths
    ukbb_root: Path = Path("data/ukbb")
    outputs_root: Path = Path("outputs")
    results_root: Path = Path("results")

    # Base model (Stable Diffusion v1.4, pinned to the revision used for the paper)
    base_model_id: str = "CompVis/stable-diffusion-v1-4"
    base_model_revision: str = "133a221b8aa7292a167afc5127cb63fb5005638b"
    local_files_only: bool = False
    hf_cache_dir: Path | None = None

    # Runtime
    device: Literal["auto", "cuda", "cpu"] = "auto"
    precision: Literal["auto", "fp16", "bf16"] = "bf16"
    enable_xformers: bool = True
    seed: int = 42
    resolution: int = 120

    # Training
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 50
    max_train_steps: int | None = 12000
    checkpoint_every_steps: int = 1000
    unet_learning_rate: float = 5e-6
    text_encoder_learning_rate: float = 1e-7
    lr_scheduler: str = "cosine"
    lr_warmup_ratio: float = 0.05
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    metadata_dropout_probability: float = 0.15
    inverse_frequency_max_weight: float = 2.0      # tau in Eq. 1
    compile_unet: bool = True
    gradient_checkpointing: bool = True
    latent_cache_batch_size: int = 8

    # Checkpoint selection score: MAE + (1 - MS-SSIM) + worst subgroup gap
    selection_mae_weight: float = 1.0
    selection_ms_ssim_weight: float = 1.0
    selection_subgroup_penalty_weight: float = 1.0
    selection_min_subgroup_count: int = 2
    # Generation settings for checkpoint validation and training-search scoring. The
    # paper's checkpoints were selected with these, before the inference search ran.
    validation_num_inference_steps: int = 60
    validation_guidance_scale: float = 2.0

    # Inference (final evaluation and `cmr infer`)
    num_inference_steps: int = 40
    guidance_scale: float = 3.0
    generation_batch_size: int = 32
    real_metrics_seed: int = 42

    # Optional hyperparameter search spaces ({field: [values]})
    training_search: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.setup not in SETUP_IDS:
            raise ValueError(f"setup must be one of {SETUP_IDS}, got {self.setup}.")
        if self.resolution % 8:
            raise ValueError(f"resolution must be divisible by 8, got {self.resolution}.")
        positive = ("train_batch_size", "gradient_accumulation_steps", "num_train_epochs", "checkpoint_every_steps",
                    "num_inference_steps", "generation_batch_size", "latent_cache_batch_size")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}.")
        if self.max_train_steps is not None and self.max_train_steps < 1:
            raise ValueError(f"max_train_steps must be at least 1, got {self.max_train_steps}.")
        for name in ("unet_learning_rate", "text_encoder_learning_rate", "adam_eps", "max_grad_norm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if not 0.0 <= self.lr_warmup_ratio < 1.0:
            raise ValueError(f"lr_warmup_ratio must be in [0, 1), got {self.lr_warmup_ratio}.")
        if not 0.0 <= self.metadata_dropout_probability < 1.0:
            raise ValueError(f"metadata_dropout_probability must be in [0, 1), got {self.metadata_dropout_probability}.")
        if self.inverse_frequency_max_weight < 1.0:
            raise ValueError(f"inverse_frequency_max_weight must be at least 1, got {self.inverse_frequency_max_weight}.")
        if self.guidance_scale < 1.0:
            raise ValueError(f"guidance_scale must be at least 1, got {self.guidance_scale}.")
        unknown = set(self.training_search) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ValueError(f"Unknown search fields: {sorted(unknown)}")

    def replace(self, **changes: Any) -> RunConfig:
        """Return a copy with some fields changed."""
        return dataclasses.replace(self, **changes)

    # ------------------------------------------------------------------ paths
    @property
    def dataset_dir(self) -> Path:
        """Preprocessed images, manifest and metadata CSV."""
        return self.outputs_root / "ukbb"

    @property
    def cache_dir(self) -> Path:
        """Reusable VAE latents, prompt embeddings and the real-vs-real FID baseline."""
        return self.dataset_dir / "cache"

    @property
    def run_dir(self) -> Path:
        """All artifacts of this setup."""
        return self.results_root / f"setup{self.setup}"

    @property
    def model_dir(self) -> Path:
        """Selected fine-tuned weights (unet/ and, when trained, text_encoder/)."""
        return self.run_dir / "model"

    @property
    def train_dir(self) -> Path:
        return self.run_dir / "train"

    @property
    def checkpoints_dir(self) -> Path:
        return self.train_dir / "checkpoints"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval"

    @property
    def infer_dir(self) -> Path:
        return self.run_dir / "infer"

    @property
    def resolved_config_path(self) -> Path:
        """Configuration actually used to train the model in model_dir."""
        return self.run_dir / "config.yaml"


# Layout of the setup YAML files: per-setup hyperparameters go in these sections and
# every other field in "runtime". Loading and saving both follow this schema.
YAML_SECTIONS: dict[str, tuple[str, ...]] = {
    "training": ("unet_learning_rate", "text_encoder_learning_rate", "gradient_accumulation_steps",
                 "metadata_dropout_probability", "inverse_frequency_max_weight"),
    "inference": ("num_inference_steps", "guidance_scale"),
}
PATH_FIELDS = tuple(f.name for f in dataclasses.fields(RunConfig) if "Path" in str(f.type))


def load_config(setup: int, path: Path | None = None, **overrides: Any) -> RunConfig:
    """
    Build the configuration of one setup.

    Values come from, in increasing priority: the dataclass defaults, the packaged
    configs/setupN.yaml (or `path`), and keyword overrides.
    """
    if path is None:
        text = resources.files("conditional_cmr").joinpath(f"configs/setup{setup}.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    values: dict[str, Any] = {
        "setup": setup,
        "name": document["name"],
        "techniques": Techniques(**document["techniques"]),
        "training_search": document.get("search", {}).get("training", {}),
    }
    for section, allowed in YAML_SECTIONS.items():
        entries = document.get(section, {})
        misplaced = set(entries) - set(allowed)
        if misplaced:
            raise ValueError(f"Section {section!r} only takes {allowed}; move {sorted(misplaced)} to 'runtime'.")
        values.update(entries)
    values.update(document.get("runtime", {}))
    values.update({key: value for key, value in overrides.items() if value is not None})
    for key in PATH_FIELDS:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    return RunConfig(**values)


def save_config(config: RunConfig, path: Path) -> None:
    """Write the full configuration with the same layout as configs/setupN.yaml."""
    flat = dataclasses.asdict(config)
    document: dict[str, Any] = {
        "name": flat.pop("name"),
        "techniques": flat.pop("techniques"),
        **{section: {key: flat.pop(key) for key in keys} for section, keys in YAML_SECTIONS.items()},
        "search": {"training": flat.pop("training_search")},
    }
    flat.pop("setup")
    document["runtime"] = {key: str(value) if isinstance(value, Path) else value for key, value in flat.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
