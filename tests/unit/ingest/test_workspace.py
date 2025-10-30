from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from deepread.ingest import workspace


def test_workspace_creates_expected_structure(tmp_path: Path) -> None:
    job_workspace = workspace.JobWorkspace(root=tmp_path, job_id="job-123")

    job_workspace.create()

    assert job_workspace.base_dir.exists()
    assert job_workspace.images_dir.exists()
    assert job_workspace.ocr_dir.exists()
    assert job_workspace.outputs_dir.exists()

    metadata_path = job_workspace.base_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["job_id"] == "job-123"
    assert metadata["created_at"]  # ISO timestamp


def test_workspace_create_custom_subdirectories(tmp_path: Path) -> None:
    job_workspace = workspace.JobWorkspace(root=tmp_path, job_id="job-custom")

    job_workspace.create(subdirectories=("outputs", "submissions"))

    assert job_workspace.base_dir.exists()
    assert job_workspace.outputs_dir.exists()
    assert (job_workspace.base_dir / "submissions").exists()
    assert not job_workspace.images_dir.exists()
    assert not job_workspace.ocr_dir.exists()


def test_workspace_cleanup_removes_contents(tmp_path: Path) -> None:
    job_workspace = workspace.JobWorkspace(root=tmp_path, job_id="job-456")
    job_workspace.create()

    staged_file = job_workspace.stage_bytes("ocr/raw.txt", data=b"hello")
    assert staged_file.exists()

    job_workspace.cleanup()

    assert not job_workspace.base_dir.exists()


def test_purge_expired_deletes_old_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_workspace = workspace.JobWorkspace(root=tmp_path, job_id="old")
    old_workspace.create()

    # Backdate metadata to force expiry.
    metadata_path = old_workspace.base_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["created_at"] = (datetime.now(UTC) - timedelta(hours=80)).isoformat()
    metadata_path.write_text(json.dumps(metadata))

    # Create a fresh workspace that should remain.
    fresh_workspace = workspace.JobWorkspace(root=tmp_path, job_id="fresh")
    fresh_workspace.create()

    # Purge using retention threshold of 72 hours.
    workspace.purge_expired(root=tmp_path, retention_hours=72)

    assert not old_workspace.base_dir.exists()
    assert fresh_workspace.base_dir.exists()
