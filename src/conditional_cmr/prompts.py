"""
Prompt vocabulary and construction (paper §III-A).

A prompt is a fixed acquisition context followed by one phrase per metadata attribute:

    Cardiac MRI, short-axis view, end-systolic frame, mid-ventricular + 1, healthy heart,
    obese patient, male patient, patient in their 50s.

The metadata-free prompt used by MF-CFG (paper §II-A) keeps the context and drops
every metadata phrase.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

MODALITY = "Cardiac MRI"
CONTEXT = (MODALITY, "short-axis view", "end-systolic frame")

# Attribute values as stored in the metadata CSV. UK Biobank slice offsets are relative
# to the mid-ventricular slice: v = -1, 0, +1.
SLICE_POSITIONS = {"-1": "mid-ventricular - 1", "0": "mid-ventricular", "1": "mid-ventricular + 1"}
PATHOLOGIES = ("healthy", "ischemic heart disease", "atrial fibrillation", "myocardial infarction", "heart failure")
BMI_GROUPS = ("underweight", "normal BMI", "overweight", "obese")
SEXES = ("female", "male")
AGE_GROUPS = ("40s", "50s", "60s", "70s", "80s")

# Order in which attributes appear in a prompt.
PROMPT_ATTRIBUTES = ("slice_position", "pathology", "bmi_group", "sex", "age_group")

# Order in which metadata dropout visits phrases. Each visit consumes one random draw,
# so changing this order changes the dropout stream of a seeded training run.
_DROPOUT_ORDER = ("bmi_group", "slice_position", "sex", "age_group", "pathology")
_VALUES = {
    "slice_position": tuple(SLICE_POSITIONS.values()),
    "pathology": PATHOLOGIES,
    "bmi_group": BMI_GROUPS,
    "sex": SEXES,
    "age_group": AGE_GROUPS,
}

NEGATIVE_PROMPT = "blurry, noisy, artifacts"   # unconditional branch of Setup 1 (standard CFG)


def group_by_attribute(metadata: Sequence[Mapping[str, str]]) -> dict[str, dict[str, list[int]]]:
    """
    Row indices of every value of every metadata attribute: {attribute: {value: [rows]}}.

    Only the attributes in PROMPT_ATTRIBUTES are grouped; empty values are skipped.
    Values keep the order in which they first appear, which the sampler relies on.
    """
    groups: dict[str, dict[str, list[int]]] = {}
    for attribute in PROMPT_ATTRIBUTES:
        buckets: dict[str, list[int]] = {}
        for index, row in enumerate(metadata):
            value = str(row.get(attribute, "")).strip()
            if value:
                buckets.setdefault(value, []).append(index)
        if buckets:
            groups[attribute] = buckets
    return groups


def phrase(attribute: str, value: str) -> str:
    """Render one metadata value as its prompt phrase."""
    if attribute == "slice_position":
        return value
    if attribute == "pathology":
        return f"{value} heart"
    if attribute in ("bmi_group", "sex"):
        return f"{value} patient"
    if attribute == "age_group":
        return f"patient in their {value}"
    raise KeyError(f"Unknown metadata attribute: {attribute}")


def render_prompt(metadata: Mapping[str, str]) -> str:
    """Build the conditioning prompt of one sample; empty attributes are omitted."""
    phrases = [phrase(attribute, metadata[attribute]) for attribute in PROMPT_ATTRIBUTES if metadata.get(attribute)]
    return ", ".join((*CONTEXT, *phrases)) + "."


def metadata_phrases() -> tuple[str, ...]:
    """Every metadata phrase of the vocabulary, in dropout order."""
    return tuple(phrase(attribute, value) for attribute in _DROPOUT_ORDER for value in _VALUES[attribute])


def is_metadata_segment(segment: str) -> bool:
    """True when a comma-separated prompt segment was produced by a metadata phrase template."""
    normalized = segment.strip().rstrip(".").lower()
    return (
        normalized in {value.lower() for value in SLICE_POSITIONS.values()}
        or normalized.endswith((" heart", " patient"))
        or normalized.startswith("patient in their ")
    )


def metadata_free_prompt(prompt: str) -> str:
    """Drop the metadata phrases of a prompt and keep the acquisition context (MF-CFG negative branch)."""
    segments = [segment.strip().rstrip(".") for segment in prompt.split(",") if segment.strip()]
    kept = [segment for segment in segments if not is_metadata_segment(segment)]
    if not kept:
        return prompt.strip()
    return ", ".join(kept).rstrip(".") + "."


def drop_metadata(prompts: list[str], probability: float, generator: torch.Generator) -> list[str]:
    """
    Metadata dropout: remove each metadata phrase of each prompt independently with
    `probability`, so the model also learns metadata-poor conditioning.

    The last phrase keeps its trailing period, which makes it never match the phrase
    being removed; the age phrase is therefore never dropped. This reproduces the
    behaviour of the code used for the paper.
    """
    if probability <= 0.0:
        return prompts
    vocabulary = metadata_phrases()
    return [_drop_prompt_metadata(prompt, vocabulary, probability, generator) for prompt in prompts]


def _drop_prompt_metadata(
    prompt: str,
    vocabulary: tuple[str, ...],
    probability: float,
    generator: torch.Generator,
) -> str:
    updated = prompt
    removed = False
    for candidate in vocabulary:
        segments = {segment.strip().rstrip(".").lower() for segment in updated.split(",")}
        if candidate.lower() not in segments:
            continue
        if torch.rand((), generator=generator).item() >= probability:
            continue
        updated = ", ".join(part.strip() for part in updated.split(",") if part.strip().lower() != candidate.lower())
        removed = True
    if removed and updated.strip():
        return updated.strip()
    return prompt
