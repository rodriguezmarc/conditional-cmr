"""Stable Diffusion v1.4 components and the generation pipeline (paper §III-B)."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from conditional_cmr.config import RunConfig

log = logging.getLogger(__name__)


@dataclass
class Components:
    """Pretrained parts of the latent diffusion model used for fine-tuning."""

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler
    device: torch.device
    dtype: torch.dtype


def select_device_and_dtype(config: RunConfig) -> tuple[torch.device, torch.dtype]:
    """Resolve the execution device and the half-precision dtype (fp32 on CPU)."""
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but CUDA is not available.")
    use_cuda = config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available())
    if not use_cuda:
        return torch.device("cpu"), torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda"), torch.float16 if config.precision == "fp16" else torch.bfloat16


def load_components(config: RunConfig) -> Components:
    """Load tokenizer, text encoder, VAE, UNet and DDPM noise scheduler from the base model."""
    device, dtype = select_device_and_dtype(config)
    kwargs = _hub_kwargs(config)
    components = Components(
        tokenizer=CLIPTokenizer.from_pretrained(config.base_model_id, subfolder="tokenizer", **kwargs),
        text_encoder=CLIPTextModel.from_pretrained(config.base_model_id, subfolder="text_encoder", torch_dtype=dtype, **kwargs),
        vae=AutoencoderKL.from_pretrained(config.base_model_id, subfolder="vae", torch_dtype=dtype, **kwargs),
        unet=UNet2DConditionModel.from_pretrained(config.base_model_id, subfolder="unet", torch_dtype=dtype, **kwargs),
        noise_scheduler=DDPMScheduler.from_pretrained(config.base_model_id, subfolder="scheduler", **kwargs),
        device=device,
        dtype=dtype,
    )
    for module in (components.text_encoder, components.vae, components.unet):
        module.to(device=device, dtype=dtype)
    return components


def build_pipeline(config: RunConfig, weights_dir: Path, *, compile_unet: bool) -> StableDiffusionPipeline:
    """
    Build the inference pipeline from fine-tuned weights: weights_dir/unet and, when the
    text encoder was trained, weights_dir/text_encoder (otherwise the base one is used).
    """
    device, dtype = select_device_and_dtype(config)
    if not (weights_dir / "unet").is_dir():
        raise FileNotFoundError(f"No fine-tuned UNet in {weights_dir}. Run `cmr train --setup {config.setup}` first.")
    kwargs = _hub_kwargs(config)
    unet = UNet2DConditionModel.from_pretrained(weights_dir, subfolder="unet", torch_dtype=dtype)
    if (weights_dir / "text_encoder").is_dir():
        text_encoder = CLIPTextModel.from_pretrained(weights_dir, subfolder="text_encoder", torch_dtype=dtype)
    else:
        text_encoder = CLIPTextModel.from_pretrained(config.base_model_id, subfolder="text_encoder", torch_dtype=dtype, **kwargs)
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.base_model_id,
        unet=unet,
        text_encoder=text_encoder,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=dtype,
        **kwargs,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True,
    )
    _enable_memory_efficient_attention(pipeline, config, device)
    if compile_unet and device.type == "cuda":
        try:
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=False)
        except Exception as error:   # compilation is an optimisation only
            log.warning("UNet compilation unavailable: %s", error)
    return pipeline


def release_memory(*, reset_compiler: bool = False) -> None:
    """Free cached GPU memory between heavy stages."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if reset_compiler:
        torch._dynamo.reset()


def _enable_memory_efficient_attention(pipeline: StableDiffusionPipeline, config: RunConfig, device: torch.device) -> None:
    if config.enable_xformers and device.type == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            return
        except Exception as error:
            log.warning("xFormers unavailable, falling back to attention slicing: %s", error)
    pipeline.enable_attention_slicing()


def _hub_kwargs(config: RunConfig) -> dict[str, object]:
    return {
        "revision": config.base_model_revision,
        "local_files_only": config.local_files_only,
        "cache_dir": config.hf_cache_dir,
    }
