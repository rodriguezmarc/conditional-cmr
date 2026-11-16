"""Paired fidelity: MAE and MS-SSIM between each generated image and its reference (paper §III-C)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from tqdm.auto import tqdm

METRICS_BATCH_SIZE = 32
# Four scales instead of the standard five: 120 px images are too small for five.
MS_SSIM_BETAS = (0.40, 0.30, 0.20, 0.10)


@dataclass
class SummaryAccumulator:
    """Running mean and (population) standard deviation of a metric."""

    count: int = 0
    total: float = 0.0
    total_square: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().cpu().float().reshape(-1)
        self.count += int(values.numel())
        self.total += float(values.sum().item())
        self.total_square += float(values.pow(2).sum().item())

    def finalize(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"mean": None, "std": None, "n": 0}
        mean = self.total / self.count
        return {"mean": mean, "std": max(self.total_square / self.count - mean**2, 0.0) ** 0.5, "n": self.count}


def mae(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Per-image mean absolute error for [N, 3, H, W] tensors in [0, 1]."""
    _check_pair(left, right)
    return (left - right).abs().flatten(1).mean(dim=1)


def ms_ssim(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Per-image multi-scale SSIM for [N, 3, H, W] tensors in [0, 1]."""
    _check_pair(left, right)
    return multiscale_structural_similarity_index_measure(
        left, right, data_range=1.0, reduction="none", betas=MS_SSIM_BETAS,
    ).reshape(-1)


PAIR_METRICS = {"mae": mae, "ms_ssim": ms_ssim}


def load_image(path: Path, resolution: int) -> torch.Tensor:
    """Load an image as an RGB [3, H, W] tensor in [0, 1], bilinearly resized to resolution."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((resolution, resolution), Image.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_images(paths: list[Path], resolution: int, desc: str = "Loading images") -> torch.Tensor:
    """Stack several images into an [N, 3, H, W] tensor."""
    return torch.stack([load_image(path, resolution) for path in tqdm(paths, desc=desc, unit="image")])


def paired_metrics(reference: torch.Tensor, generated: torch.Tensor) -> dict[str, dict[str, float | int | None]]:
    """Mean and std of MAE and MS-SSIM over aligned image pairs."""
    _check_pair(reference, generated)
    accumulators = {name: SummaryAccumulator() for name in PAIR_METRICS}
    for start in range(0, reference.shape[0], METRICS_BATCH_SIZE):
        chunk = slice(start, start + METRICS_BATCH_SIZE)
        for name, metric in PAIR_METRICS.items():
            accumulators[name].update(metric(reference[chunk], generated[chunk]))
    return {name: accumulator.finalize() for name, accumulator in accumulators.items()}


def per_image_metrics(reference_paths: list[Path], generated_paths: list[Path], resolution: int) -> dict[str, torch.Tensor]:
    """Per-image MAE and MS-SSIM computed on CPU from image files, in bounded batches."""
    if len(reference_paths) != len(generated_paths):
        raise ValueError(f"Got {len(reference_paths)} reference and {len(generated_paths)} generated images.")
    values: dict[str, list[torch.Tensor]] = {name: [] for name in PAIR_METRICS}
    for start in range(0, len(reference_paths), METRICS_BATCH_SIZE):
        reference = torch.stack([load_image(p, resolution) for p in reference_paths[start:start + METRICS_BATCH_SIZE]])
        generated = torch.stack([load_image(p, resolution) for p in generated_paths[start:start + METRICS_BATCH_SIZE]])
        for name, metric in PAIR_METRICS.items():
            values[name].append(metric(reference, generated).detach().cpu().reshape(-1))
    return {name: torch.cat(chunks) for name, chunks in values.items()}


def _check_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    for batch in (left, right):
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError(f"Expected an image batch [N, 3, H, W], got {tuple(batch.shape)}.")
    if left.shape != right.shape:
        raise ValueError(f"Expected equal shapes, got {tuple(left.shape)} and {tuple(right.shape)}.")
