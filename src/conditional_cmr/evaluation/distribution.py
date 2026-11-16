"""
Distributional fidelity: FID (pytorch-fid) and FRD (frd-score) between image sets (paper §III-C).

Both backends read the image files directly: real images at their preprocessed (native)
resolution and generated images at the training resolution. This matches the protocol
used for the paper's numbers.
"""

from __future__ import annotations

import shutil
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import torch

FID_FEATURE_DIMS = 2048
FID_BATCH_SIZE = 32


def fid_from_paths(reference_paths: list[Path], generated_paths: list[Path], device: torch.device) -> dict[str, object]:
    """Fréchet Inception Distance between two image sets."""
    from pytorch_fid.fid_score import calculate_fid_given_paths

    if not reference_paths or not generated_paths:
        return {"value": None, "reference_n": len(reference_paths), "generated_n": len(generated_paths)}
    with tempfile.TemporaryDirectory(prefix="cmr_fid_") as tmp:
        value = calculate_fid_given_paths(
            [str(_link_images(reference_paths, Path(tmp) / "reference")), str(_link_images(generated_paths, Path(tmp) / "generated"))],
            batch_size=FID_BATCH_SIZE,
            device=str(device),
            dims=FID_FEATURE_DIMS,
            num_workers=0,
        )
    return {"value": float(value), "reference_n": len(reference_paths), "generated_n": len(generated_paths)}


def frd_from_paths(reference_paths: list[Path], generated_paths: list[Path]) -> dict[str, object]:
    """Fréchet Radiomics Distance (frd-score v1) between two image sets."""
    from frd_score import compute_frd
    import frd_score.frd as frd_backend

    # frd-score reads its YAML config with the locale encoding; an ASCII copy avoids
    # decode errors on systems whose locale is not UTF-8.
    config_path = Path(frd_backend.V1_CONFIG_2D)
    with tempfile.TemporaryDirectory(prefix="cmr_frd_") as tmp:
        ascii_config = Path(tmp) / config_path.name
        text = unicodedata.normalize("NFKD", config_path.read_bytes().decode("utf-8"))
        ascii_config.write_text(text.encode("ascii", "ignore").decode("ascii"), encoding="ascii")
        value = compute_frd(
            [[str(p) for p in reference_paths], [str(p) for p in generated_paths]],
            frd_version="v1", verbose=True, match_sample_count=False, config_path=str(ascii_config),
        )
    return {"value": float(value), "reference_n": len(reference_paths), "generated_n": len(generated_paths)}


def split_real_reference(paths: list[Path], *, seed: int) -> tuple[list[Path], list[Path]]:
    """Shuffle a real cohort and split it into two equal halves (real-vs-real FID baseline)."""
    shuffled = list(paths)
    np.random.default_rng(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:2 * half]


def _link_images(paths: list[Path], directory: Path) -> Path:
    """pytorch-fid takes a directory; symlink (or copy) the files into one."""
    directory.mkdir(parents=True)
    for index, path in enumerate(paths):
        link = directory / f"{index:08d}{path.suffix or '.png'}"
        try:
            link.symlink_to(path.resolve())
        except OSError:
            shutil.copy2(path, link)
    return directory
