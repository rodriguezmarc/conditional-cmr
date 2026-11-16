"""`cmr eval`: generate the testing split and compute the Table I and Fig. 3 metrics."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch

from conditional_cmr.config import RunConfig
from conditional_cmr.data.manifest import DatasetRow, load_rows
from conditional_cmr.evaluation.distribution import fid_from_paths, frd_from_paths, split_real_reference
from conditional_cmr.evaluation.diversity import diversity_ratios
from conditional_cmr.evaluation.paired import load_images, paired_metrics
from conditional_cmr.evaluation.subgroup import evaluate_subgroup_fid, plot_subgroup_fid_matrices
from conditional_cmr.generation import GenerationRequest, generate
from conditional_cmr.models import build_pipeline, release_memory, select_device_and_dtype
from conditional_cmr.utils import load_json, save_json

log = logging.getLogger(__name__)


def run_evaluation(config: RunConfig, *, metrics_only: bool = False) -> dict[str, object]:
    """Generate one image per testing row (seed + row index), then write eval/metrics.json."""
    rows = load_rows(config.dataset_dir, "testing")
    generated_paths = [config.eval_dir / "generated" / f"{row.image_path.stem}_generated.png" for row in rows]
    _generate_missing(config, rows, generated_paths, metrics_only=metrics_only)

    device, _ = select_device_and_dtype(config)
    reference_paths = [row.image_path for row in rows]
    reference = load_images(reference_paths, config.resolution, desc="Loading reference images")
    generated = load_images(generated_paths, config.resolution, desc="Loading generated images")
    fid = fid_from_paths(reference_paths, generated_paths, device)
    real_fid = _real_vs_real_fid(config, reference_paths, device)
    metrics: dict[str, object] = {
        "setup": config.setup,
        "inference": {"num_inference_steps": config.num_inference_steps, "guidance_scale": config.guidance_scale},
        "paired": paired_metrics(reference, generated),
        "diversity_ratio": diversity_ratios(generated, reference, seed=config.real_metrics_seed),
        "distribution": {
            "fid": {
                "generated_vs_real": fid,
                "real_vs_real_reference": real_fid,
                "ratio": fid["value"] / real_fid["value"] if real_fid.get("value") else None,
            },
            "frd": {"generated_vs_real": frd_from_paths(reference_paths, generated_paths)},
        },
        "subgroup_fid": evaluate_subgroup_fid(reference_paths, generated_paths, [row.metadata for row in rows], device),
    }
    save_json(metrics, config.eval_dir / "metrics.json")
    plot_subgroup_fid_matrices(metrics["subgroup_fid"], config.eval_dir / "subgroup_fid_matrices.png", f"Setup {config.setup} Subgroup FID")
    log.info("Metrics saved to %s", config.eval_dir / "metrics.json")
    return metrics


def _generate_missing(config: RunConfig, rows: list[DatasetRow], generated_paths: list[Path], *, metrics_only: bool) -> None:
    """Reuse generated images only if they come from the same weights and settings."""
    fingerprint = {
        "setup": config.setup,
        "model_dir": str(config.model_dir),
        "model_mtime_ns": (config.model_dir / "unet").stat().st_mtime_ns if (config.model_dir / "unet").exists() else None,
        "seed": config.seed,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "resolution": config.resolution,
    }
    fingerprint_path = config.eval_dir / "generated_fingerprint.json"
    if not fingerprint_path.is_file() or load_json(fingerprint_path) != fingerprint:
        if metrics_only:
            raise ValueError("The generated images do not match this model and settings. Run `cmr eval` without --metrics-only.")
        shutil.rmtree(config.eval_dir / "generated", ignore_errors=True)
    requests = [
        GenerationRequest(row.prompt, path, index)
        for index, (row, path) in enumerate(zip(rows, generated_paths))
        if not path.is_file()
    ]
    if requests and metrics_only:
        raise FileNotFoundError(f"{len(requests)} generated images are missing; run `cmr eval` without --metrics-only.")
    if requests:
        pipeline = build_pipeline(config, config.model_dir, compile_unet=True)
        generate(
            pipeline, requests, batch_size=config.generation_batch_size, seed=config.seed,
            num_inference_steps=config.num_inference_steps, guidance_scale=config.guidance_scale,
            resolution=config.resolution, metadata_free_cfg=config.techniques.metadata_free_cfg,
            desc=f"Generating testing split (setup {config.setup})",
        )
        del pipeline
        release_memory(reset_compiler=True)
    save_json(fingerprint, fingerprint_path)


def _real_vs_real_fid(config: RunConfig, reference_paths: list[Path], device: torch.device) -> dict[str, object]:
    """FID between two random halves of the real testing split, cached across setups."""
    cache_path = config.cache_dir / "real_vs_real_fid.json"
    signature = {"sample_count": len(reference_paths), "seed": config.real_metrics_seed}
    if cache_path.is_file():
        cached = load_json(cache_path)
        if cached.get("signature") == signature:
            return cached["fid"]
    left, right = split_real_reference(reference_paths, seed=config.real_metrics_seed)
    fid = fid_from_paths(left, right, device)
    save_json({"signature": signature, "fid": fid}, cache_path)
    return fid
