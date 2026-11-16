"""`cmr run`: preprocess if needed, (optionally search), train and evaluate one setup."""

from __future__ import annotations

import logging

from conditional_cmr.config import RunConfig, save_config
from conditional_cmr.data.ukbb import preprocess
from conditional_cmr.workflows.evaluate import run_evaluation
from conditional_cmr.workflows.search import inference_search, training_search
from conditional_cmr.workflows.train import run_training

log = logging.getLogger(__name__)


def run_all(config: RunConfig, *, search: bool = False) -> dict[str, object]:
    """
    Full workflow. By default it uses the paper's hyperparameters from the setup config;
    with `search`, it re-runs the training and inference searches first.
    """
    if not (config.dataset_dir / "manifest.csv").is_file():
        preprocess(config.ukbb_root, config.dataset_dir)
    if search:
        config = training_search(config)
    run_training(config)
    if search:
        config = inference_search(config)
        save_config(config, config.resolved_config_path)
    return run_evaluation(config)
