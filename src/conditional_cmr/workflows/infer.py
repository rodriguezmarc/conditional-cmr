"""`cmr infer`: generate one image from a free-text prompt."""

from __future__ import annotations

from pathlib import Path

from conditional_cmr.config import RunConfig
from conditional_cmr.generation import GenerationRequest, generate
from conditional_cmr.models import build_pipeline
from conditional_cmr.utils import save_json


def run_inference(config: RunConfig, prompt: str) -> Path:
    """Generate one image from a free-text prompt with the trained model of this setup."""
    if not prompt.strip():
        raise ValueError("The prompt must not be empty.")
    pipeline = build_pipeline(config, config.model_dir, compile_unet=False)
    output_path = config.infer_dir / "generated.png"
    generate(
        pipeline, [GenerationRequest(prompt.strip(), output_path, 0)], batch_size=1, seed=config.seed,
        num_inference_steps=config.num_inference_steps, guidance_scale=config.guidance_scale,
        resolution=config.resolution, metadata_free_cfg=config.techniques.metadata_free_cfg,
    )
    save_json({"setup": config.setup, "prompt": prompt.strip(), "seed": config.seed,
               "num_inference_steps": config.num_inference_steps, "guidance_scale": config.guidance_scale},
              config.infer_dir / "inference.json")
    return output_path
