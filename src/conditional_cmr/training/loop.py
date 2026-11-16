"""
Fine-tuning loop (paper §III-B).

The VAE stays frozen; the UNet is fine-tuned to predict the noise added to CMR latents,
conditioned on CLIP embeddings of the metadata prompt. Setups 2-5 also fine-tune the text
encoder. Batches follow the CB / IFS sampling order and prompts go through metadata dropout.

The order of random operations (seeding, latent encoding, noise, timesteps) matches the
code used for the paper, so a seeded run follows the same stream.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from conditional_cmr.config import RunConfig
from conditional_cmr.data.manifest import DatasetRow
from conditional_cmr.models import Components, load_components, release_memory
from conditional_cmr.prompts import drop_metadata
from conditional_cmr.sampling import sampling_order
from conditional_cmr.training import checkpoints
from conditional_cmr.training.caches import LatentDataset, PromptEmbeddingCache, collate, encode_prompts, load_latents

log = logging.getLogger(__name__)


def train(
    config: RunConfig,
    rows: list[DatasetRow],
    validation_rows: list[DatasetRow],
    *,
    save_checkpoints: bool = True,
    resume: bool = False,
) -> None:
    """Fine-tune one setup and write the selected weights to config.model_dir."""
    torch.manual_seed(config.seed)
    parts = load_components(config)
    train_text_encoder = config.techniques.train_text_encoder
    try:
        _prepare_modules(parts, config)
        if parts.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        trainable = [p for p in parts.unet.parameters() if p.requires_grad]
        text_encoder_params = [p for p in parts.text_encoder.parameters() if p.requires_grad]
        trainable += text_encoder_params
        log.info("Trainable parameters: %d", sum(p.numel() for p in trainable))

        forward_unet = parts.unet
        if config.compile_unet and parts.device.type == "cuda" and not text_encoder_params:
            try:
                forward_unet = torch.compile(parts.unet, mode="reduce-overhead", fullgraph=False)
            except Exception as error:   # compilation is an optimisation only
                log.warning("UNet compilation unavailable: %s", error)

        latents = load_latents(rows, parts.vae, parts.device, parts.dtype, config)
        order = sampling_order(
            [row.metadata for row in rows],
            batch_size=config.train_batch_size,
            seed=config.seed,
            tau=config.inverse_frequency_max_weight,
            inverse_frequency=config.techniques.inverse_frequency_sampling,
            contrastive_batching=config.techniques.contrastive_batching,
        )
        pin_memory = parts.device.type == "cuda"
        loader = DataLoader(
            LatentDataset(latents, [row.prompt for row in rows]),
            batch_size=config.train_batch_size, sampler=order, collate_fn=collate, num_workers=0, pin_memory=pin_memory,
        )

        optimizer = _build_optimizer(parts, text_encoder_params, config)
        scaler = torch.amp.GradScaler("cuda", enabled=parts.device.type == "cuda" and parts.dtype == torch.float16)
        use_autocast = parts.device.type == "cuda" and parts.dtype in (torch.float16, torch.bfloat16)
        max_steps = config.max_train_steps or config.num_train_epochs * math.ceil(len(loader) / config.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            config.lr_scheduler, optimizer=optimizer,
            num_warmup_steps=warmup_steps(max_steps, config.lr_warmup_ratio), num_training_steps=max_steps,
        )
        tracker = checkpoints.CheckpointTracker(config)
        step = _resume(parts, optimizer, lr_scheduler, config) if resume else 0

        parts.unet.train()
        parts.text_encoder.train() if text_encoder_params else parts.text_encoder.eval()
        prompt_cache = None
        if not text_encoder_params:
            prompt_cache = PromptEmbeddingCache(parts.tokenizer, parts.text_encoder, parts.device, config)
            prompt_cache.preload([row.prompt for row in rows])

        optimizer.zero_grad(set_to_none=True)
        dropout_generator = torch.Generator().manual_seed(config.seed + 17)
        log_rows: list[dict[str, float]] = []
        micro_losses: list[float] = []
        progress = tqdm(total=max_steps, initial=step, desc=f"Training setup {config.setup}", unit="step")

        for epoch in range(config.num_train_epochs):
            for batch_index, batch in enumerate(loader):
                batch_latents = batch["latents"].to(device=parts.device, dtype=parts.dtype, non_blocking=pin_memory)
                prompts = drop_metadata(batch["prompts"], config.metadata_dropout_probability, dropout_generator)
                if prompt_cache is not None:
                    hidden_states = prompt_cache(prompts)
                else:
                    hidden_states = encode_prompts(parts.tokenizer, parts.text_encoder, prompts, parts.device, grad=True)

                noise = torch.randn_like(batch_latents)
                timesteps = torch.randint(
                    0, parts.noise_scheduler.config.num_train_timesteps, (batch_latents.shape[0],), device=parts.device,
                ).long()
                noisy_latents = parts.noise_scheduler.add_noise(batch_latents, noise, timesteps)
                with torch.autocast(device_type=parts.device.type, dtype=parts.dtype, enabled=use_autocast):
                    prediction = forward_unet(noisy_latents, timesteps, encoder_hidden_states=hidden_states).sample
                    loss = F.mse_loss(prediction.float(), _target(parts, batch_latents, noise, timesteps).float())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss {loss.item()} at step {step}.")
                micro_losses.append(float(loss.detach().item()))
                scaler.scale(loss / config.gradient_accumulation_steps).backward()

                if (batch_index + 1) % config.gradient_accumulation_steps and batch_index + 1 != len(loader):
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm, error_if_nonfinite=True)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not scaler.is_enabled() or scaler.get_scale() >= scale_before:
                    lr_scheduler.step()      # skipped when fp16 overflow made the scaler skip the step
                optimizer.zero_grad(set_to_none=True)
                step += 1
                log_rows.append({"step": step, "epoch": epoch, "loss": sum(micro_losses) / len(micro_losses),
                                 "unet_learning_rate": optimizer.param_groups[0]["lr"]})
                micro_losses.clear()
                progress.update(1)
                progress.set_postfix(loss=f"{log_rows[-1]['loss']:.4f}")

                if save_checkpoints and step % config.checkpoint_every_steps == 0:
                    _checkpoint(parts, step, epoch, optimizer, lr_scheduler, tracker, validation_rows, config)
                if step >= max_steps:
                    break
            if step >= max_steps:
                break
        progress.close()

        if tracker.export_best(config.model_dir) is None:
            checkpoints.save_weights(parts.unet, parts.text_encoder, config.model_dir, with_text_encoder=train_text_encoder)
        _save_loss_log(log_rows, config.train_dir / "loss_log.csv")
    finally:
        for module in (parts.unet, parts.text_encoder, parts.vae):
            module.to("cpu")
        release_memory(reset_compiler=True)


def warmup_steps(max_steps: int, ratio: float) -> int:
    """Warmup length as a fraction of the schedule, at most max_steps - 1."""
    if max_steps <= 1 or ratio <= 0.0:
        return 0
    return max(0, min(max_steps - 1, int(round(max_steps * ratio))))


def _prepare_modules(parts: Components, config: RunConfig) -> None:
    """Freeze the VAE, unfreeze the UNet and, if the setup says so, the text encoder."""
    if config.enable_xformers and parts.device.type == "cuda":
        try:
            parts.unet.enable_xformers_memory_efficient_attention()
        except Exception as error:
            log.warning("xFormers unavailable: %s", error)
    parts.vae.requires_grad_(False)
    parts.vae.eval()
    parts.unet.requires_grad_(True)
    parts.text_encoder.requires_grad_(config.techniques.train_text_encoder)
    parts.text_encoder.train() if config.techniques.train_text_encoder else parts.text_encoder.eval()
    if config.gradient_checkpointing and config.techniques.train_text_encoder:
        parts.unet.enable_gradient_checkpointing()
        parts.text_encoder.gradient_checkpointing_enable()
    # Trainable weights stay in fp32; activations use bf16/fp16 autocast.
    parts.unet.float()
    parts.text_encoder.float()


def _build_optimizer(parts: Components, text_encoder_params: list[torch.nn.Parameter], config: RunConfig) -> Optimizer:
    """8-bit paged AdamW on CUDA (memory), standard AdamW on CPU; the text encoder has its own LR."""
    groups = [{"params": [p for p in parts.unet.parameters() if p.requires_grad], "lr": config.unet_learning_rate}]
    if text_encoder_params:
        groups.append({"params": text_encoder_params, "lr": config.text_encoder_learning_rate})
    if parts.device.type == "cuda":
        from bitsandbytes.optim import PagedAdamW8bit
        return PagedAdamW8bit(groups, lr=config.unet_learning_rate, eps=config.adam_eps)
    return torch.optim.AdamW(groups, eps=config.adam_eps, foreach=False)


def _target(parts: Components, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """What the UNet predicts: noise (epsilon, used by SD 1.4), velocity, or the clean latent."""
    prediction_type = parts.noise_scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return parts.noise_scheduler.get_velocity(latents, noise, timesteps)
    if prediction_type == "sample":
        return latents
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def _checkpoint(
    parts: Components,
    step: int,
    epoch: int,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    tracker: checkpoints.CheckpointTracker,
    validation_rows: list[DatasetRow],
    config: RunConfig,
) -> None:
    """Save weights and state, then validate with the live modules temporarily moved to CPU."""
    directory = config.checkpoints_dir / f"step_{step:06d}"
    with_text_encoder = config.techniques.train_text_encoder
    checkpoints.save_weights(parts.unet, parts.text_encoder, directory, with_text_encoder=with_text_encoder)
    checkpoints.save_state(directory, step=step, epoch=epoch, optimizer=optimizer, lr_scheduler=lr_scheduler)
    if not validation_rows:
        return
    training_modes = (parts.unet.training, parts.text_encoder.training)
    parts.unet.eval()
    parts.text_encoder.eval()
    devices = [(module, next(module.parameters()).device) for module in (parts.unet, parts.text_encoder, parts.vae)]
    if parts.device.type == "cuda":
        for module, _ in devices:
            module.to("cpu")
        release_memory()
    try:
        tracker.add(step, directory, checkpoints.validate_checkpoint(directory, validation_rows, config))
    finally:
        release_memory()
        for module, device in devices:
            module.to(device)
        if training_modes[0]:
            parts.unet.train()
        if training_modes[1]:
            parts.text_encoder.train()


def _resume(parts: Components, optimizer: Optimizer, lr_scheduler: LRScheduler, config: RunConfig) -> int:
    """Restore weights, optimizer and scheduler from the latest checkpoint that has them."""
    with_text_encoder = config.techniques.train_text_encoder
    for step, directory in reversed(checkpoints.step_dirs(config.checkpoints_dir)):
        if not (directory / checkpoints.STATE_FILE).is_file() or not checkpoints.has_weights(directory, with_text_encoder=with_text_encoder):
            continue
        state = checkpoints.load_state(directory)
        checkpoints.load_weights(parts.unet, parts.text_encoder, directory, parts.dtype, with_text_encoder=with_text_encoder)
        parts.unet.to(parts.device).float()
        parts.text_encoder.to(parts.device).float()
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        log.info("Resumed from step %d (%s).", state["step"], directory)
        release_memory(reset_compiler=True)
        return int(state["step"])
    log.info("No resumable checkpoint found; starting from step 0.")
    return 0


def _save_loss_log(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "epoch", "loss", "unet_learning_rate"])
        writer.writeheader()
        writer.writerows(rows)
