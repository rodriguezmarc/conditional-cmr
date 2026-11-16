"""Read and write the preprocessed dataset: manifest.csv (MINIM format) and metadata.csv."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

MANIFEST_COLUMNS = ("path", "text", "modality")
METADATA_BASE_COLUMNS = ("path", "patient_id", "split")


@dataclass(frozen=True)
class DatasetRow:
    """One preprocessed image with its prompt and metadata attributes."""

    image_path: Path
    prompt: str
    metadata: dict[str, str]     # patient_id plus the five metadata attributes


def write_dataset(dataset_dir: Path, manifest_rows: list[dict[str, str]], metadata_rows: list[dict[str, str]]) -> None:
    """Validate and write manifest.csv and metadata.csv."""
    seen: set[str] = set()
    for index, row in enumerate(manifest_rows):
        for column in MANIFEST_COLUMNS:
            if not str(row.get(column, "")).strip():
                raise ValueError(f"Manifest row {index} has an empty {column!r}.")
        if row["path"] in seen:
            raise ValueError(f"Duplicate image path: {row['path']}")
        if not (dataset_dir / row["path"]).is_file():
            raise FileNotFoundError(f"Image path does not exist: {dataset_dir / row['path']}")
        seen.add(row["path"])
    attributes = sorted({key for row in metadata_rows for key in row} - set(METADATA_BASE_COLUMNS))
    _write_csv(dataset_dir / "manifest.csv", MANIFEST_COLUMNS, manifest_rows)
    _write_csv(dataset_dir / "metadata.csv", (*METADATA_BASE_COLUMNS, *attributes), metadata_rows)


def load_rows(dataset_dir: Path, split: str) -> list[DatasetRow]:
    """Load the rows of one split, in manifest order."""
    manifest_path, metadata_path = dataset_dir / "manifest.csv", dataset_dir / "metadata.csv"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Preprocessed dataset not found in {dataset_dir}. Run `cmr preprocess --ukbb-root PATH` first."
        )
    metadata_by_path: dict[str, dict[str, str]] = {}
    for row in _read_csv(metadata_path, METADATA_BASE_COLUMNS):
        if row["split"].strip() == split:
            metadata_by_path[row["path"].strip()] = {
                key: value.strip() for key, value in row.items() if key not in ("path", "split")
            }
    rows = [
        DatasetRow(image_path=dataset_dir / row["path"].strip(), prompt=row["text"].strip(), metadata=metadata_by_path[row["path"].strip()])
        for row in _read_csv(manifest_path, MANIFEST_COLUMNS)
        if row["path"].strip() in metadata_by_path
    ]
    if not rows:
        raise ValueError(f"No rows for split {split!r} in {dataset_dir}.")
    return rows


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
        return list(reader)
