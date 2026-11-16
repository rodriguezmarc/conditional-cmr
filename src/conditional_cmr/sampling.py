"""
Training sample order (paper §II-B and §II-C).

Each batch is assigned one metadata attribute c, cycling through the attributes.
- Contrastive Batching (CB): the batch takes distinct values of c, drawn without
  replacement; if the batch is larger than the number of values, they are reused
  cyclically in the same order.
- Inverse-Frequency Sampling (IFS): values are drawn with probability proportional to
  w_x = min(N̄_c / N_x, tau)  (Eq. 1), where N̄_c is the mean bucket size over all values of c.
For every drawn value, one sample is picked uniformly from its bucket (Eq. 2).
Without CB or IFS the order is a seeded shuffle of all rows.

The order is computed once and reused every epoch.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from conditional_cmr.prompts import group_by_attribute


def sampling_order(
    metadata: Sequence[Mapping[str, str]],
    *,
    batch_size: int,
    seed: int,
    tau: float,
    inverse_frequency: bool,
    contrastive_batching: bool,
) -> list[int]:
    """Return one epoch of row indices for the DataLoader."""
    if not metadata:
        raise ValueError("Cannot build a sampling order from an empty dataset.")
    if batch_size < 1 or tau < 1.0:
        raise ValueError(f"Invalid batch_size={batch_size} or tau={tau}.")

    rng = random.Random(seed)
    if not inverse_frequency and not contrastive_batching:
        indices = list(range(len(metadata)))
        rng.shuffle(indices)
        return indices

    buckets = group_by_attribute(metadata)
    attributes = sorted(buckets)
    if not attributes:
        return list(range(len(metadata)))

    order: list[int] = []
    for batch_index in range((len(metadata) + batch_size - 1) // batch_size):
        value_buckets = buckets[attributes[batch_index % len(attributes)]]
        values = _batch_values(value_buckets, batch_size, rng, tau, inverse_frequency, contrastive_batching)
        for item in range(batch_size):
            order.append(rng.choice(value_buckets[values[item % len(values)]]))
            if len(order) >= len(metadata):
                return order
    return order


def inverse_frequency_weights(bucket_sizes: Sequence[int], tau: float) -> list[float]:
    """Eq. 1: w_x = min(mean bucket size / N_x, tau)."""
    sizes = [max(1, size) for size in bucket_sizes]
    mean = sum(sizes) / len(sizes)
    return [min(mean / size, tau) for size in sizes]


def _batch_values(
    value_buckets: dict[str, list[int]],
    batch_size: int,
    rng: random.Random,
    tau: float,
    inverse_frequency: bool,
    contrastive_batching: bool,
) -> list[str]:
    """Choose the attribute values that fill one batch."""
    values = list(value_buckets)
    weights = inverse_frequency_weights([len(value_buckets[v]) for v in values], tau) if inverse_frequency else None
    if not contrastive_batching:
        return rng.choices(values, weights=weights, k=batch_size)

    weight_of = dict(zip(values, weights)) if weights is not None else None
    selected: list[str] = []
    available = values[:]
    while len(selected) < min(batch_size, len(values)):
        draw_weights = [weight_of[v] for v in available] if weight_of is not None else None
        chosen = rng.choices(available, weights=draw_weights, k=1)[0]
        selected.append(chosen)
        available.remove(chosen)
    return selected
