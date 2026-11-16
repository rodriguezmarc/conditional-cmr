"""Subgroup FID matrices (paper Fig. 3 and §IV-B)."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import torch
from matplotlib.axes import Axes

from conditional_cmr.evaluation.distribution import fid_from_paths
from conditional_cmr.prompts import group_by_attribute

AXIS_TITLES = {
    "age_group": "Age",
    "bmi_group": "BMI",
    "pathology": "Pathology",
    "sex": "Sex",
    "slice_position": "Slice Position",
}


def evaluate_subgroup_fid(
    reference_paths: list[Path],
    generated_paths: list[Path],
    metadata: list[dict[str, str]],
    device: torch.device,
) -> dict[str, object]:
    """
    For every metadata attribute, compute the reference-subgroup x generated-subgroup
    FID matrix and the mean diagonal / off-diagonal values reported in §IV-B.
    """
    groups = group_by_attribute(metadata)
    report: dict[str, object] = {}
    for column, value_groups in sorted(groups.items()):
        labels = sorted(value_groups)
        matrix: dict[str, dict[str, object]] = {}
        diagonal: list[float] = []
        off_diagonal: list[float] = []
        for real_label in labels:
            real_paths = [reference_paths[i] for i in value_groups[real_label]]
            row: dict[str, object] = {}
            for generated_label in labels:
                fake_paths = [generated_paths[i] for i in value_groups[generated_label]]
                fid = fid_from_paths(real_paths, fake_paths, device)
                row[generated_label] = fid
                value = fid.get("value")
                if isinstance(value, (int, float)):
                    (diagonal if real_label == generated_label else off_diagonal).append(float(value))
            matrix[real_label] = row
        diagonal_mean = _mean_or_none(diagonal)
        off_diagonal_mean = _mean_or_none(off_diagonal)
        report[column] = {
            "labels": labels,
            "counts": {label: len(value_groups[label]) for label in labels},
            "matrix": matrix,
            "diagonal_mean": diagonal_mean,
            "off_diagonal_mean": off_diagonal_mean,
            "separation_score": (
                None if diagonal_mean is None or off_diagonal_mean is None
                else off_diagonal_mean - diagonal_mean
            ),
        }
    return report


def plot_subgroup_fid_matrices(
    report: dict[str, object],
    output_path: Path,
    title: str,
) -> None:
    """Draw one FID heatmap per attribute: rows are reference, columns generated subgroups."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = [_matrix_values(axis, axis_report) for axis, axis_report in report.items()]
    matrices = [m for m in matrices if m is not None]
    if not matrices:
        return

    columns = min(3, len(matrices))
    rows = ceil(len(matrices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 5.6 * rows), squeeze=False)
    finite = np.concatenate([m["values"][np.isfinite(m["values"])] for m in matrices])
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (None, None)

    image = None
    for axis_plot, matrix in zip(axes.flat, matrices):
        labels = [_short_label(label) for label in matrix["labels"]]
        image = axis_plot.imshow(matrix["values"], cmap="magma_r", aspect="auto", vmin=vmin, vmax=vmax)
        axis_plot.set_title(AXIS_TITLES.get(matrix["axis"], matrix["axis"]), fontsize=12, pad=14)
        axis_plot.set_xlabel("Generated subgroup", fontsize=10, labelpad=10)
        axis_plot.set_ylabel("Reference subgroup", fontsize=10, labelpad=16)
        axis_plot.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        axis_plot.set_yticks(np.arange(len(labels)), labels)
        _annotate(axis_plot, matrix["values"])
    for empty_axis in list(axes.flat)[len(matrices):]:
        empty_axis.axis("off")

    figure.suptitle(title, fontsize=16, y=0.98)
    figure.subplots_adjust(left=0.08, right=0.84, top=0.88, bottom=0.12, hspace=0.72, wspace=0.74)
    colorbar = figure.colorbar(image, cax=figure.add_axes([0.89, 0.20, 0.018, 0.60]))
    colorbar.set_label("FID", fontsize=11, labelpad=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _matrix_values(axis: str, axis_report: object) -> dict[str, object] | None:
    """Convert one attribute report into a NaN-padded numeric matrix."""
    if not isinstance(axis_report, dict):
        return None
    labels = axis_report["labels"]
    values = np.full((len(labels), len(labels)), np.nan)
    for i, real_label in enumerate(labels):
        for j, generated_label in enumerate(labels):
            value = axis_report["matrix"][real_label][generated_label].get("value")
            if isinstance(value, (int, float)):
                values[i, j] = float(value)
    if np.isnan(values).all():
        return None
    return {"axis": axis, "labels": labels, "values": values}


def _annotate(axis_plot: Axes, values: np.ndarray) -> None:
    """Write each FID value inside its cell."""
    finite = values[np.isfinite(values)]
    midpoint = float(np.nanmedian(finite)) if finite.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                axis_plot.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=7,
                               color="white" if values[i, j] > midpoint else "black")


def _short_label(label: str) -> str:
    """Compact tick label."""
    return label.replace(" patient", "").replace(" heart", "")


def _mean_or_none(values: list[float]) -> float | None:
    """Arithmetic mean, or None for an empty list."""
    return sum(values) / len(values) if values else None
