"""
Workspace management for ingestion jobs.

The ingestion pipeline retains intermediate artifacts (rendered images, OCR
payloads, generated outputs) on disk for a limited period. This module manages
directory scaffolding, metadata, and retention enforcement.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

METADATA_FILENAME: Final[str] = "metadata.json"


def _utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(UTC)


@dataclass(slots=True)
class JobWorkspace:
    """Manage on-disk storage for a single ingestion job."""

    root: Path
    job_id: str
    base_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = self.root / self.job_id

    # --- Path helpers ----------------------------------------------------- #
    @property
    def images_dir(self) -> Path:
        return self.base_dir / "images"

    @property
    def ocr_dir(self) -> Path:
        return self.base_dir / "ocr"

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "outputs"

    @property
    def metadata_path(self) -> Path:
        return self.base_dir / METADATA_FILENAME

    # --- Lifecycle -------------------------------------------------------- #
    def create(self, *, subdirectories: Iterable[str] | None = None) -> None:
        """Create the workspace directory structure and metadata."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if subdirectories is None:
            directories: tuple[str, ...] = ("images", "ocr", "outputs")
        elif isinstance(subdirectories, str):
            directories = (subdirectories,)
        else:
            directories = tuple(subdirectories)
        for name in directories:
            directory = self.base_dir / name
            directory.mkdir(parents=True, exist_ok=True)

        metadata = {
            "job_id": self.job_id,
            "created_at": _utc_now().isoformat(),
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2))

    def stage_bytes(self, relative_path: str, *, data: bytes) -> Path:
        """Persist raw bytes under the workspace and return the created path."""
        destination = self.base_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination

    def cleanup(self) -> None:
        """Remove the entire workspace directory."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)

    # --- Retention -------------------------------------------------------- #
    def created_at(self) -> datetime:
        """Return the recorded creation timestamp for this workspace."""
        if not self.metadata_path.exists():
            return _utc_now()
        payload = json.loads(self.metadata_path.read_text())
        timestamp = payload.get("created_at")
        if not timestamp:
            return _utc_now()
        return datetime.fromisoformat(timestamp)


def purge_expired(*, root: Path, retention_hours: int = 72) -> None:
    """
    Remove workspaces whose metadata timestamps exceed the retention window.

    Args:
        root: Base directory containing job subdirectories.
        retention_hours: Threshold in hours before a workspace is purged.
    """

    cutoff = _utc_now() - timedelta(hours=retention_hours)
    if not root.exists():
        return

    for path in root.iterdir():
        if not path.is_dir():
            continue

        metadata_path = path / METADATA_FILENAME
        if not metadata_path.exists():
            continue

        try:
            payload = json.loads(metadata_path.read_text())
            created_at_str = payload.get("created_at")
            if not created_at_str:
                continue
            created_at = datetime.fromisoformat(created_at_str)
        except (json.JSONDecodeError, ValueError):
            continue

        if created_at < cutoff:
            shutil.rmtree(path, ignore_errors=True)
