"""Command line interface: `cmr {preprocess,train,infer,eval,run}`."""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

from conditional_cmr.config import SETUP_IDS, RunConfig, load_config


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ukbb-root", type=Path, default=os.environ.get("CMR_UKBB_ROOT"),
                        help="UK Biobank folder with prompt.json, target/ and source/ (or set CMR_UKBB_ROOT).")
    common.add_argument("--outputs-root", type=Path, help="Preprocessed data and caches (default: outputs).")
    common.add_argument("--results-root", type=Path, help="Models and metrics (default: results).")
    common.add_argument("--device", choices=("auto", "cuda", "cpu"))
    common.add_argument("--local-files-only", action="store_true", default=None,
                        help="Load Stable Diffusion from the local Hugging Face cache only.")
    with_setup = argparse.ArgumentParser(add_help=False, parents=[common])
    with_setup.add_argument("-s", "--setup", type=int, choices=SETUP_IDS, required=True,
                            help="1 Fine-tuned SD, 2 MF-CFG, 3 MF-CFG+IFS, 4 MF-CFG+CB, 5 MF-CFG+IFS+CB.")

    parser = argparse.ArgumentParser(prog="cmr", description="Metadata-conditioned cardiac MRI synthesis.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preprocess", parents=[common], help="Export UK Biobank images, prompts and metadata.")
    train = commands.add_parser("train", parents=[with_setup], help="Fine-tune one setup.")
    train.add_argument("--resume", action="store_true", help="Continue from the latest checkpoint.")
    infer = commands.add_parser("infer", parents=[with_setup], help="Generate one image from a prompt.")
    infer.add_argument("-p", "--prompt", required=True)
    evaluate = commands.add_parser("eval", parents=[with_setup], help="Generate the testing split and compute metrics.")
    evaluate.add_argument("--metrics-only", action="store_true", help="Reuse generated images; only recompute metrics.")
    run = commands.add_parser("run", parents=[with_setup], help="Preprocess if needed, train and evaluate.")
    run.add_argument("--search", action="store_true", help="Re-run the hyperparameter searches instead of using the paper values.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

    if args.command == "preprocess":
        from conditional_cmr.data.ukbb import preprocess
        if args.ukbb_root is None:
            raise SystemExit("Pass --ukbb-root or set CMR_UKBB_ROOT.")
        config = _config(args, setup=1)
        count = preprocess(config.ukbb_root, config.dataset_dir)
        logging.info("Exported %d samples to %s", count, config.dataset_dir)
        return

    config = _config(args, setup=args.setup)
    if args.command == "train":
        from conditional_cmr.workflows.train import run_training
        run_training(config, resume=args.resume)
    elif args.command == "infer":
        from conditional_cmr.workflows.infer import run_inference
        logging.info("Image saved to %s", run_inference(config, args.prompt))
    elif args.command == "eval":
        from conditional_cmr.workflows.evaluate import run_evaluation
        run_evaluation(config, metrics_only=args.metrics_only)
    elif args.command == "run":
        from conditional_cmr.workflows.run import run_all
        if args.ukbb_root is None and not (config.dataset_dir / "manifest.csv").is_file():
            raise SystemExit("The dataset is not preprocessed yet: pass --ukbb-root or set CMR_UKBB_ROOT.")
        run_all(config, search=args.search)


def _config(args: argparse.Namespace, setup: int) -> RunConfig:
    """Setup config, or the config saved with an already trained model of this setup."""
    overrides = {
        "ukbb_root": args.ukbb_root,
        "outputs_root": args.outputs_root,
        "results_root": args.results_root,
        "device": args.device,
        "local_files_only": args.local_files_only,
    }
    packaged = load_config(setup, **overrides)
    trained = packaged.resolved_config_path
    if args.command in ("infer", "eval") and trained.is_file():
        return load_config(setup, path=trained, **overrides)
    return packaged


if __name__ == "__main__":
    main()
