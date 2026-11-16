"""Small JSON helpers shared by the workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(payload: Any, path: Path) -> None:
    """Write JSON atomically (temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
