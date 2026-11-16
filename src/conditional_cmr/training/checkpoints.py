"""
Intermediate checkpoints and checkpoint selection.

Every `checkpoint_every_steps`, the weights are saved and the full testing split is
generated with the validation settings. The checkpoint with the lowest score

    score = w_mae * MAE + w_ssim * (1 - MS-SSIM) + w_gap * (worst subgroup MAE gap + worst subgroup MS-SSIM gap)

becomes the final model. Only the best and the latest checkpoint keep their weights.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import UNet2DConditionModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import CLIPTextModel

from conditional_cmr.config import RunConfig
from conditional_cmr.data.manifest import DatasetRow
from conditional_cmr.evaluation.paired import METRICS_BATCH_SIZE, PAIR_METRICS, SummaryAccumulator, per_image_metrics
from conditional_cmr.generation import GenerationRequest, generate
from conditional_cmr.models import build_pipeline, release_memory
from conditional_cmr.prompts import group_by_attribute
from conditional_cmr.utils import load_json, save_json

log = logging.getLogger(__name__)

STATE_FILE = "training_state.pt"


@dataclass(frozen=True)
class Candidate:
    step: int
    path: Path
    score: float
    components: dict[str, float]


# ---------------------------------------------------------------------- weights
def save_weights(unet: UNet2DConditionModel, text_encoder: CLIPTextModel, output_dir: Path, *, with_text_encoder: bool) -> None:
    """Save the UNet and, when it is trained, the text encoder in diffusers format."""
    unet.save_pretrained(output_dir / "unet")
    if with_text_encoder:
        text_encoder.save_pretrained(output_dir / "text_encoder")


def load_weights(
    unet: UNet2DConditionModel, text_encoder: CLIPTextModel, checkpoint_dir: Path, dtype: torch.dtype, *, with_text_encoder: bool,
) -> None:
    """Load checkpoint weights into the live training modules."""
    unet.load_state_dict(UNet2DConditionModel.from_pretrained(checkpoint_dir / "unet", torch_dtype=dtype).state_dict())
    if with_text_encoder:
        text_encoder.load_state_dict(CLIPTextModel.from_pretrained(checkpoint_dir / "text_encoder", torch_dtype=dtype).state_dict())


def has_weights(checkpoint_dir: Path, *, with_text_encoder: bool) -> bool:
    return (checkpoint_dir / "unet").is_dir() and (not with_text_encoder or (checkpoint_dir / "text_encoder").is_dir())


def delete_weights(checkpoint_dir: Path) -> None:
    shutil.rmtree(checkpoint_dir / "unet", ignore_errors=True)
    shutil.rmtree(checkpoint_dir / "text_encoder", ignore_errors=True)


def save_state(checkpoint_dir: Path, *, step: int, epoch: int, optimizer: Optimizer, lr_scheduler: LRScheduler) -> None:
    """Save optimizer and scheduler state so training can resume from this step."""
    path = checkpoint_dir / STATE_FILE
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"step": step, "epoch": epoch, "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict()}, temporary)
    temporary.replace(path)


def load_state(checkpoint_dir: Path) -> dict[str, Any]:
    # Written by save_state on this machine; optimizer states need full unpickling.
    return torch.load(checkpoint_dir / STATE_FILE, map_location="cpu", weights_only=False)


def step_dirs(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    """(step, directory) of every checkpoint on disk, sorted by step."""
    found = []
    for directory in checkpoints_dir.glob("step_*"):
        try:
            found.append((int(directory.name.removeprefix("step_")), directory))
        except ValueError:
            continue
    return sorted(found)


# ---------------------------------------------------------------------- validation and score
def validate_checkpoint(checkpoint_dir: Path, rows: list[DatasetRow], config: RunConfig) -> dict[str, object]:
    """Generate the validation rows with the checkpoint weights and return paired/subgroup metrics."""
    metrics_path = checkpoint_dir / "validation_metrics.json"
    if metrics_path.is_file():
        return load_json(metrics_path)
    pipeline = build_pipeline(config, checkpoint_dir, compile_unet=False)
    output_dir = checkpoint_dir / "validation"
    requests = [GenerationRequest(row.prompt, output_dir / f"validation_{i:03d}.png", i) for i, row in enumerate(rows)]
    try:
        generate(
            pipeline, requests, batch_size=config.generation_batch_size, seed=config.seed,
            num_inference_steps=config.validation_num_inference_steps, guidance_scale=config.validation_guidance_scale,
            resolution=config.resolution, metadata_free_cfg=config.techniques.metadata_free_cfg,
            desc=f"Validating {checkpoint_dir.name}",
        )
        del pipeline
        release_memory()
        metrics = validation_metrics(rows, [r.output_path for r in requests], config)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
    save_json(metrics, metrics_path)
    return metrics


def validation_metrics(rows: list[DatasetRow], generated_paths: list[Path], config: RunConfig) -> dict[str, object]:
    """Global and per-subgroup MAE / MS-SSIM, plus the selection score."""
    values = per_image_metrics([row.image_path for row in rows], generated_paths, config.resolution)
    report: dict[str, object] = {"rows": len(rows)}
    for name in PAIR_METRICS:
        accumulator = SummaryAccumulator()
        for start in range(0, len(rows), METRICS_BATCH_SIZE):     # same float32 summation order as the paper code
            accumulator.update(values[name][start:start + METRICS_BATCH_SIZE])
        report[name] = accumulator.finalize()

    subgroups: dict[str, dict[str, object]] = {}
    for attribute, groups in sorted(group_by_attribute([row.metadata for row in rows]).items()):
        subgroups[attribute] = {}
        for value, indices in sorted(groups.items()):
            if len(indices) < config.selection_min_subgroup_count:
                subgroups[attribute][value] = {"count": len(indices), "skipped": True}
                continue
            subgroups[attribute][value] = {"count": len(indices)}
            for name in PAIR_METRICS:
                accumulator = SummaryAccumulator()
                for index in indices:
                    accumulator.update(values[name][index:index + 1])
                subgroups[attribute][value][name] = accumulator.finalize()
    report["subgroup"] = subgroups
    report["score"], report["score_components"] = selection_score(report, config)
    return report


def selection_score(report: dict[str, object], config: RunConfig) -> tuple[float, dict[str, float]]:
    """Lower is better: weighted MAE + (1 - MS-SSIM) + worst subgroup degradation."""
    mae, ms_ssim = _mean(report, "mae"), _mean(report, "ms_ssim")
    worst_mae_gap = worst_ms_ssim_gap = 0.0
    for groups in report.get("subgroup", {}).values():
        for value_report in groups.values():
            if "skipped" in value_report:
                continue
            worst_mae_gap = max(worst_mae_gap, _mean(value_report, "mae") - mae)
            worst_ms_ssim_gap = max(worst_ms_ssim_gap, ms_ssim - _mean(value_report, "ms_ssim"))
    gap = worst_mae_gap + worst_ms_ssim_gap
    score = (config.selection_mae_weight * mae + config.selection_ms_ssim_weight * (1.0 - ms_ssim)
             + config.selection_subgroup_penalty_weight * gap)
    return score, {"mae": mae, "ms_ssim": ms_ssim, "worst_subgroup_mae_gap": worst_mae_gap,
                   "worst_subgroup_ms_ssim_gap": worst_ms_ssim_gap}


def _mean(report: dict[str, Any], name: str) -> float:
    metric = report.get(name)
    return float(metric["mean"]) if isinstance(metric, dict) and metric.get("mean") is not None else 0.0


# ---------------------------------------------------------------------- bookkeeping
class CheckpointTracker:
    """Keeps checkpoint scores, prunes weights and exports the best checkpoint."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.with_text_encoder = config.techniques.train_text_encoder
        self.candidates: list[Candidate] = []
        for step, directory in step_dirs(config.checkpoints_dir):        # restored when resuming
            metrics_path = directory / "validation_metrics.json"
            if metrics_path.is_file():
                self._register(step, directory, load_json(metrics_path))

    @property
    def best(self) -> Candidate | None:
        with_weights = [c for c in self.candidates if has_weights(c.path, with_text_encoder=self.with_text_encoder)]
        return min(with_weights, key=lambda c: c.score, default=None)

    def add(self, step: int, directory: Path, metrics: dict[str, object]) -> None:
        candidate = self._register(step, directory, metrics)
        if self.best is candidate:
            log.info("Best checkpoint so far: step %d (score %.6f)", step, candidate.score)
        keep = {directory, self.best.path if self.best else directory}
        for other in self.candidates:
            if other.path not in keep:
                delete_weights(other.path)
                (other.path / STATE_FILE).unlink(missing_ok=True)

    def export_best(self, model_dir: Path) -> Candidate | None:
        """Copy the best checkpoint's weights to model_dir and keep only its weights."""
        best = self.best
        if best is None:
            return None
        shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True)
        shutil.copytree(best.path / "unet", model_dir / "unet")
        if self.with_text_encoder:
            shutil.copytree(best.path / "text_encoder", model_dir / "text_encoder")
        for other in self.candidates:
            if other.path != best.path:
                delete_weights(other.path)
        save_json({
            "selected_step": best.step,
            "score": best.score,
            "score_components": best.components,
            "checkpoints": [{"step": c.step, "score": c.score, "score_components": c.components} for c in self.candidates],
        }, self.config.train_dir / "best_checkpoint.json")
        return best

    def _register(self, step: int, directory: Path, metrics: dict[str, object]) -> Candidate:
        score, components = selection_score(metrics, self.config)
        candidate = Candidate(step, directory, score, components)
        self.candidates.append(candidate)
        return candidate
