"""
UK Biobank preprocessing (paper §III-A).

Expected input layout (`ukbb_root`):

    prompt.json            JSON list (or JSON lines) of {"target": ..., "source": ...}
    target/<name>.png      RGB ED/ES composite; the green channel holds the ES frame
    source/<name>.png      matching segmentation composite (only checked for existence)

File names encode the metadata:
    <patient>_t_<t>_v_<v>_EDES_<Sex>_age_in_<NN>s_<bmi>_BMI_<pathology...>.png
where v in {-1, 0, 1} is the slice offset from the mid-ventricular slice.

Output (`dataset_dir`): images/<patient>/<patient>_v_<v>_es.png, plus the MINIM-style
manifest.csv (path, text, modality) and metadata.csv (path, patient_id, split, attributes).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from conditional_cmr.data.manifest import write_dataset
from conditional_cmr.prompts import MODALITY, SLICE_POSITIONS, render_prompt

log = logging.getLogger(__name__)

TIMEFRAME = "0"            # end-systolic frame index used for every sample
TEST_MODULO = 10           # patient-level 90/10 split: patient_id % 10 == 0 -> testing
ES_CHANNEL = 1             # green channel of the ED/ES composite

_FILENAME_RE = re.compile(r"^(?P<patient_id>[^_]+)_t_(?P<t>-?\d+)_v_(?P<v>-?\d+)_EDES(?:_mask)?_(?P<rest>.+)$")
_PATHOLOGY_PRIORITY = ("heart failure", "myocardial infarction", "ischemic heart disease", "atrial fibrillation")


@dataclass(frozen=True)
class Sample:
    """One ES short-axis slice and its metadata."""

    patient_id: str
    v: str
    split: str
    target_path: Path
    metadata: dict[str, str]

    @property
    def relative_image_path(self) -> str:
        return f"images/{self.patient_id}/{self.patient_id}_v_{self.v}_es.png"


def parse_filename(path: Path) -> dict[str, str] | None:
    """Read patient id, frame, slice offset, sex, age group and attributes from a file name."""
    match = _FILENAME_RE.match(path.stem)
    if match is None:
        return None
    fields = match.groupdict()
    rest = fields.pop("rest").split("_")
    if len(rest) >= 4:
        fields["sex"] = rest[0]
        fields["age_group"] = "_".join(rest[1:4])
        fields["attributes"] = "_".join(rest[4:])
    else:
        fields["attributes"] = "_".join(rest)
    return fields


def split_for_patient(patient_id: str) -> str:
    """Deterministic patient-level split, so slices of one patient never cross splits."""
    return "testing" if int(patient_id) % TEST_MODULO == 0 else "training"


def normalize_pathology(raw: str) -> str:
    """Collapse combined diagnoses into the five classes of the paper, by clinical priority."""
    value = raw.replace("_", " ").strip().lower()
    if not value or value == "healthy":
        return "healthy"
    return next((label for label in _PATHOLOGY_PRIORITY if label in value), value)


def sample_metadata(fields: dict[str, str]) -> dict[str, str]:
    """Convert parsed file-name fields into the five metadata attributes of the paper."""
    if "sex" not in fields:
        raise ValueError(f"File name lacks sex/age fields: {fields}")
    attributes = fields["attributes"].strip()
    if "_BMI_" in attributes:
        bmi, pathology = attributes.split("_BMI_", maxsplit=1)
        bmi_group = bmi.replace("_", " ").lower()
        bmi_group = "normal BMI" if bmi_group == "normal" else bmi_group
    else:
        bmi_group, pathology = "", attributes
    return {
        "slice_position": SLICE_POSITIONS[fields["v"]],
        "pathology": normalize_pathology(pathology),
        "bmi_group": bmi_group,
        "sex": fields["sex"].replace("_", " ").strip().lower(),
        "age_group": fields["age_group"].replace("_", " ").strip().removeprefix("age in "),
    }


def discover_samples(ukbb_root: Path) -> list[Sample]:
    """List every ES sample that has both its image and its segmentation file."""
    prompt_path = ukbb_root / "prompt.json"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing UK Biobank index file: {prompt_path}")
    samples: list[Sample] = []
    for row in _read_json_rows(prompt_path):
        target, source = str(row.get("target", "")).strip(), str(row.get("source", "")).strip()
        if not target or not source:
            continue
        fields = parse_filename(Path(target))
        if fields is None or fields["t"] != TIMEFRAME:
            continue
        target_path, source_path = ukbb_root / target, ukbb_root / source
        if not target_path.is_file() or not source_path.is_file():
            continue
        samples.append(Sample(
            patient_id=fields["patient_id"],
            v=fields["v"],
            split=split_for_patient(fields["patient_id"]),
            target_path=target_path,
            metadata=sample_metadata(fields),
        ))
    if not samples:
        raise FileNotFoundError(f"No UK Biobank samples were found under {ukbb_root}.")
    return sorted(samples, key=lambda sample: f"{sample.patient_id}_v_{sample.v}")


def preprocess(ukbb_root: Path, dataset_dir: Path) -> int:
    """Export ES images, manifest.csv and metadata.csv; return the number of samples."""
    samples = discover_samples(Path(ukbb_root))
    log.info("Discovered %d UK Biobank samples.", len(samples))
    manifest_rows, metadata_rows = [], []
    for sample in tqdm(samples, desc="Exporting UK Biobank", unit="image"):
        relative_path = sample.relative_image_path
        export_es_image(sample.target_path, dataset_dir / relative_path)
        manifest_rows.append({"path": relative_path, "text": render_prompt(sample.metadata), "modality": MODALITY})
        metadata_rows.append({"path": relative_path, "patient_id": sample.patient_id, "split": sample.split, **sample.metadata})
    write_dataset(dataset_dir, manifest_rows, metadata_rows)
    return len(samples)


def export_es_image(target_path: Path, output_path: Path) -> None:
    """Save the end-systolic (green) channel of an ED/ES composite as an 8-bit grayscale PNG."""
    with Image.open(target_path) as image:
        channel = np.asarray(image.convert("RGB"))[:, :, ES_CHANNEL]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(channel), mode="L").save(output_path)


def _read_json_rows(path: Path) -> list[dict]:
    """Read a JSON list or a JSON-lines file of objects."""
    text = path.read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list of rows.")
    return [row for row in rows if isinstance(row, dict)]
