"""
Optional hyperparameter search used by `cmr run --search` (thesis §5.2.3 and §5.3.1).

The shipped configs already hold the values this search selected for the paper; the
search exists to re-derive them, or to tune a new dataset.
- Training search: each candidate of the setup's search space trains 3,000 steps with
  batch size 2 and no intermediate checkpoints, then generates the first 32 testing rows
  and is scored with the checkpoint-selection score.
- Inference search: a grid over denoising steps and guidance scale, scored the same way
  on the first 20 % of the testing split.
"""

from __future__ import annotations

import itertools
import logging
import shutil

from conditional_cmr.config import RunConfig
from conditional_cmr.data.manifest import DatasetRow, load_rows
from conditional_cmr.generation import GenerationRequest, generate
from conditional_cmr.models import build_pipeline, release_memory
from conditional_cmr.training.checkpoints import validation_metrics
from conditional_cmr.training.loop import train
from conditional_cmr.utils import save_json

log = logging.getLogger(__name__)

TRAINING_SEARCH_STEPS = 3000
TRAINING_SEARCH_BATCH_SIZE = 2
TRAINING_SEARCH_ROWS = 32
TRAINING_SEARCH_GENERATION_BATCH_SIZE = 8
INFERENCE_SEARCH_ROW_FRACTION = 0.20
INFERENCE_SEARCH_STEPS = (15, 20, 30, 40, 50)
INFERENCE_SEARCH_GUIDANCE = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)


def training_search(config: RunConfig) -> RunConfig:
    """Return `config` with the best values of its training search space."""
    space = config.training_search
    if not space:
        return config
    rows = load_rows(config.dataset_dir, "training")
    probe_rows = load_rows(config.dataset_dir, "testing")[:TRAINING_SEARCH_ROWS]
    results = []
    for index, values in enumerate(itertools.product(*space.values())):
        changes = dict(zip(space, values))
        candidate = config.replace(
            **changes,
            max_train_steps=TRAINING_SEARCH_STEPS,
            train_batch_size=TRAINING_SEARCH_BATCH_SIZE,
            results_root=config.run_dir / "search" / "training" / f"candidate_{index:02d}",
        )
        log.info("Training search candidate %d: %s", index, changes)
        train(candidate, rows, [], save_checkpoints=False)
        score = _score(candidate, probe_rows, candidate.validation_num_inference_steps,
                       candidate.validation_guidance_scale, TRAINING_SEARCH_GENERATION_BATCH_SIZE)
        shutil.rmtree(candidate.results_root, ignore_errors=True)
        results.append({"changes": changes, "score": score})
    best = min(results, key=lambda result: result["score"])
    save_json({"selected": best, "candidates": results}, config.run_dir / "search" / "training.json")
    log.info("Training search selected %s", best["changes"])
    return config.replace(**best["changes"])


def inference_search(config: RunConfig) -> RunConfig:
    """Return `config` with the best number of denoising steps and guidance scale."""
    rows = load_rows(config.dataset_dir, "testing")
    rows = rows[:max(1, round(len(rows) * INFERENCE_SEARCH_ROW_FRACTION))]
    results = []
    for steps, guidance in itertools.product(INFERENCE_SEARCH_STEPS, INFERENCE_SEARCH_GUIDANCE):
        score = _score(config, rows, steps, guidance, config.generation_batch_size)
        results.append({"num_inference_steps": steps, "guidance_scale": guidance, "score": score})
    best = min(results, key=lambda result: result["score"])
    save_json({"selected": best, "candidates": results}, config.run_dir / "search" / "inference.json")
    log.info("Inference search selected %d steps, guidance %.1f", best["num_inference_steps"], best["guidance_scale"])
    return config.replace(num_inference_steps=best["num_inference_steps"], guidance_scale=best["guidance_scale"])


def _score(config: RunConfig, rows: list[DatasetRow], steps: int, guidance: float, batch_size: int) -> float:
    """Generate `rows` with the model of `config` and return the checkpoint-selection score."""
    output_dir = config.run_dir / "search" / "probe"
    requests = [GenerationRequest(row.prompt, output_dir / f"{i:04d}.png", i) for i, row in enumerate(rows)]
    pipeline = build_pipeline(config, config.model_dir, compile_unet=False)
    try:
        generate(pipeline, requests, batch_size=batch_size, seed=config.seed, num_inference_steps=steps,
                 guidance_scale=guidance, resolution=config.resolution,
                 metadata_free_cfg=config.techniques.metadata_free_cfg, desc=f"Search probe {steps} steps, s={guidance}")
        del pipeline
        release_memory()
        return float(validation_metrics(rows, [r.output_path for r in requests], config)["score"])
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
