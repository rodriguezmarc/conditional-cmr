"""
Disk caches for training: VAE latents of the frozen VAE and, when the text encoder is
frozen, CLIP prompt embeddings. Keys include the image identity, resolution and base model.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from conditional_cmr.config import RunConfig
from conditional_cmr.data.manifest import DatasetRow

log = logging.getLogger(__name__)


class LatentDataset(Dataset):
    """In-memory (latent, prompt) pairs."""

    def __init__(self, latents: list[torch.Tensor], prompts: list[str]) -> None:
        self.latents, self.prompts = latents, prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        return self.latents[index], self.prompts[index]


def collate(batch: list[tuple[torch.Tensor, str]]) -> dict[str, object]:
    return {"latents": torch.stack([latent for latent, _ in batch]), "prompts": [prompt for _, prompt in batch]}


def image_transform(resolution: int) -> transforms.Compose:
    """Resize and map an RGB image to [-1, 1], the VAE input range."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load_latents(
    rows: Sequence[DatasetRow], vae: AutoencoderKL, device: torch.device, dtype: torch.dtype, config: RunConfig,
) -> list[torch.Tensor]:
    """
    Encode every training image once with the frozen VAE (one posterior sample, scaled)
    and keep the latents in RAM; encodings are reused across runs from disk.
    """
    paths = [_latent_path(row, config) for row in rows]
    missing = [(row, path) for row, path in zip(rows, paths) if not path.is_file()]
    log.info("Latent cache: %d of %d available.", len(rows) - len(missing), len(rows))
    transform = image_transform(config.resolution)
    vae.eval()
    for start in tqdm(range(0, len(missing), config.latent_cache_batch_size), desc="Building latent cache", unit="batch"):
        chunk = missing[start:start + config.latent_cache_batch_size]
        pixels = torch.stack([_load_rgb(row.image_path, transform) for row, _ in chunk]).to(device=device, dtype=dtype)
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        for (_, path), latent in zip(chunk, latents.detach().cpu()):
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(latent, path)
    loaded: dict[Path, torch.Tensor] = {}
    for path in tqdm(paths, desc="Loading latents", unit="latent"):
        if path not in loaded:
            loaded[path] = torch.load(path, map_location="cpu", weights_only=True)
    return [loaded[path] for path in paths]


def encode_prompts(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, prompts: list[str], device: torch.device, *, grad: bool) -> torch.Tensor:
    """CLIP last hidden states of the prompts (77 tokens, padded and truncated)."""
    tokens = tokenizer(prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.set_grad_enabled(grad):
        return text_encoder(tokens.input_ids.to(device))[0]


class PromptEmbeddingCache:
    """Embeddings of a frozen text encoder, kept in RAM and on disk."""

    def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: torch.device, config: RunConfig) -> None:
        self.tokenizer, self.text_encoder, self.device, self.config = tokenizer, text_encoder, device, config
        self.memory: dict[str, torch.Tensor] = {}

    def preload(self, prompts: Sequence[str]) -> None:
        """Load or compute the embedding of every distinct training prompt."""
        self.text_encoder.eval()
        for prompt in tqdm(sorted(set(prompts)), desc="Preparing prompt cache", unit="prompt"):
            path = self._path(prompt)
            if path.is_file():
                self.memory[prompt] = torch.load(path, map_location="cpu", weights_only=True)
            else:
                self.memory[prompt] = self._encode(prompt)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.memory[prompt], path)

    def __call__(self, prompts: list[str]) -> torch.Tensor:
        """Embeddings of a batch; prompts changed by metadata dropout are encoded on demand."""
        self.text_encoder.eval()
        for prompt in prompts:
            if prompt not in self.memory:
                self.memory[prompt] = self._encode(prompt)
        return torch.cat([self.memory[prompt].to(self.device) for prompt in prompts], dim=0)

    def _encode(self, prompt: str) -> torch.Tensor:
        return encode_prompts(self.tokenizer, self.text_encoder, [prompt], self.device, grad=False).detach().cpu()

    def _path(self, prompt: str) -> Path:
        key = "|".join([prompt, self.config.base_model_id, str(self.config.resolution)])
        return self.config.cache_dir / "embedding" / str(self.config.resolution) / f"{hashlib.sha1(key.encode()).hexdigest()}.pt"


def _latent_path(row: DatasetRow, config: RunConfig) -> Path:
    image_path = row.image_path.resolve()
    stat = image_path.stat()
    key = "|".join([image_path.as_posix(), str(stat.st_size), str(stat.st_mtime_ns), str(config.resolution), config.base_model_id])
    return config.cache_dir / "vae" / str(config.resolution) / f"{hashlib.sha1(key.encode()).hexdigest()}.pt"


def _load_rgb(path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB"))
