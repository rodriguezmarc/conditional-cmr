"""
Batched image generation with Metadata-Free Classifier-Free Guidance (paper §II-A).

With MF-CFG the unconditional branch receives the metadata-free prompt, so the guided
prediction eps_neg + s * (eps_pos - eps_neg) emphasises the metadata. Setup 1 uses
standard CFG with a fixed negative prompt instead.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm.auto import tqdm

from conditional_cmr.prompts import NEGATIVE_PROMPT, metadata_free_prompt

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationRequest:
    """One image to generate; its noise seed is base_seed + seed_offset."""

    prompt: str
    output_path: Path
    seed_offset: int


def generate(
    pipeline: StableDiffusionPipeline,
    requests: list[GenerationRequest],
    *,
    batch_size: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    resolution: int,
    metadata_free_cfg: bool,
    desc: str = "Generating",
) -> None:
    """Generate every request in batches and save each image to its output path."""
    batches = [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]
    for batch in tqdm(batches, desc=desc, unit="batch"):
        images = _generate_batch(
            pipeline, batch, seed=seed, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, resolution=resolution, metadata_free_cfg=metadata_free_cfg,
        )
        for request, image in zip(batch, images, strict=True):
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(request.output_path)


def _generate_batch(
    pipeline: StableDiffusionPipeline,
    requests: list[GenerationRequest],
    *,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    resolution: int,
    metadata_free_cfg: bool,
) -> list[Image.Image]:
    prompts = [request.prompt for request in requests]
    negative_prompts = [metadata_free_prompt(p) if metadata_free_cfg else NEGATIVE_PROMPT for p in prompts]
    device = pipeline.device
    dtype = getattr(pipeline.unet, "dtype", torch.float32)
    generators = [torch.Generator(device=device).manual_seed(seed + request.seed_offset) for request in requests]
    use_autocast = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode(), (torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else nullcontext()):
        result = pipeline(
            prompt=prompts,
            negative_prompt=negative_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=generators,
            height=resolution,
            width=resolution,
            output_type="pil",
            return_dict=True,
        )
    return result.images
