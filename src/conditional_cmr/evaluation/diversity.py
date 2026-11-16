"""
Diversity ratios (paper Table I).

For a cohort, diversity is the mean pairwise distance (MAE, or 1 - MS-SSIM) over image
pairs drawn uniformly without replacement. The ratio compares generated with real
diversity; values near 1 mean realistic variation. Its standard deviation uses
first-order error propagation:  sigma_R = R * sqrt((sigma_g/mu_g)^2 + (sigma_r/mu_r)^2).
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from conditional_cmr.evaluation.paired import METRICS_BATCH_SIZE, PAIR_METRICS, SummaryAccumulator


def diversity_ratios(generated: torch.Tensor, reference: torch.Tensor, *, seed: int) -> dict[str, object]:
    """Generated-vs-real diversity ratios for MAE and MS-SSIM distance."""
    generated_summary = cohort_diversity(generated, pair_count=reference.shape[0], seed=seed, desc="Generated diversity")
    real_summary = cohort_diversity(reference, pair_count=generated.shape[0], seed=seed, desc="Real diversity")
    report: dict[str, object] = {}
    for name, key in (("mae", "mae"), ("ms_ssim", "ms_ssim_distance")):
        g, r = generated_summary[name], real_summary[name]
        if name == "ms_ssim":   # similarity -> distance; the std is unchanged
            g, r = _as_distance(g), _as_distance(r)
        report[key] = {
            "generated": g,
            "real_reference": r,
            "ratio": _ratio(g, r),
            "ratio_std": _ratio_std(g, r),
        }
    return report


def cohort_diversity(images: torch.Tensor, *, pair_count: int, seed: int, desc: str) -> dict[str, dict[str, object]]:
    """Mean pairwise MAE and MS-SSIM over `pair_count` uniformly sampled image pairs."""
    total = images.shape[0] * (images.shape[0] - 1) // 2
    pairs = sample_unique_pairs(images.shape[0], min(pair_count, total), seed=seed)
    accumulators = {name: SummaryAccumulator() for name in PAIR_METRICS}
    for start in tqdm(range(0, len(pairs), METRICS_BATCH_SIZE), desc=desc, unit="batch"):
        chunk = pairs[start:start + METRICS_BATCH_SIZE]
        left, right = images[[i for i, _ in chunk]], images[[j for _, j in chunk]]
        for name, metric in PAIR_METRICS.items():
            accumulators[name].update(metric(left, right))
    return {
        name: {**accumulator.finalize(), "total_pair_count": total, "sampled_pair_count": len(pairs)}
        for name, accumulator in accumulators.items()
    }


def sample_unique_pairs(image_count: int, pair_count: int, *, seed: int) -> list[tuple[int, int]]:
    """
    Draw distinct unordered pairs (i < j) uniformly at random. The same seed gives the
    same index pairs for the real and generated cohorts, whose rows are aligned.
    """
    total = image_count * (image_count - 1) // 2
    if pair_count <= 0 or total == 0:
        return []
    flat = np.random.default_rng(seed).choice(total, size=min(pair_count, total), replace=False)
    return [_pair_from_flat_index(int(k), image_count) for k in flat]


def _pair_from_flat_index(k: int, n: int) -> tuple[int, int]:
    """Map k in [0, n(n-1)/2) to the k-th pair (i, j), i < j, in row-major order."""
    i = n - 2 - int(((-8 * k + 4 * n * (n - 1) - 7) ** 0.5 - 1) // 2)
    j = k + i + 1 - n * (n - 1) // 2 + (n - i) * (n - i - 1) // 2
    return i, j


def _as_distance(summary: dict[str, object]) -> dict[str, object]:
    mean = summary.get("mean")
    return {**summary, "mean": None if mean is None else 1.0 - float(mean)}


def _ratio(numerator: dict[str, object], denominator: dict[str, object]) -> float | None:
    top, bottom = numerator.get("mean"), denominator.get("mean")
    if top is None or bottom in (None, 0.0):
        return None
    return float(top) / float(bottom)


def _ratio_std(numerator: dict[str, object], denominator: dict[str, object]) -> float | None:
    ratio = _ratio(numerator, denominator)
    if ratio is None:
        return None
    terms = []
    for summary in (numerator, denominator):
        mean, std = summary.get("mean"), summary.get("std")
        if mean in (None, 0.0) or std is None:
            return None
        terms.append((float(std) / float(mean)) ** 2)
    return abs(ratio) * sum(terms) ** 0.5
