from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from deepread.api.router import create_app


def test_batch_progress_records_failures(tmp_path: Path) -> None:
    app = create_app(workspace_root=tmp_path / "workspace")
    client = TestClient(app)

    files = [
        ("documents", ("supported.md", b"# Heading\n\nContent", "text/markdown")),
        ("documents", ("unsupported.txt", b"plain text", "text/plain")),
    ]

    response = client.post("/v1/jobs", files=files, data={"requestedOutputs": ["markdown"]})
    data = response.json()

    failed = next(sub for sub in data["submissions"] if sub["originalFilename"] == "unsupported.txt")
    assert failed["status"] == "failed"
    assert "Processing failed" in failed["remediation"]

    job_id = data["jobId"]
    status_response = client.get(f"/v1/jobs/{job_id}")
    status_payload = status_response.json()
    assert status_payload["manifestPath"]
