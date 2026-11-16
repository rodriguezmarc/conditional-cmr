"""`cmr train`: fine-tune one setup on the training split."""

from __future__ import annotations

import logging
import shutil

from conditional_cmr.config import RunConfig, save_config
from conditional_cmr.data.manifest import load_rows
from conditional_cmr.training.loop import train

log = logging.getLogger(__name__)


def run_training(config: RunConfig, *, resume: bool = False) -> None:
    """
    Train setup `config.setup`. Checkpoints are validated on the testing split, as in the
    paper (there is no separate validation split). Without `resume`, earlier results of
    this setup are deleted first.
    """
    if not resume:
        for path in (config.model_dir, config.train_dir, config.eval_dir, config.infer_dir):
            shutil.rmtree(path, ignore_errors=True)
    rows = load_rows(config.dataset_dir, "training")
    validation_rows = load_rows(config.dataset_dir, "testing")
    log.info("Setup %d (%s): %d training rows, %d validation rows.", config.setup, config.name, len(rows), len(validation_rows))
    save_config(config, config.resolved_config_path)
    train(config, rows, validation_rows, resume=resume)
    log.info("Model saved to %s", config.model_dir)
