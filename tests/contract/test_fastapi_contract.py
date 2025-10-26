from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from deepread.api.router import create_app


def test_create_job_and_fetch_status(tmp_path: Path) -> None:
    app = create_app(workspace_root=tmp_path / "api-workspace")
    client = TestClient(app)

    files = [
        ("documents", ("doc1.html", b"<h1>Title</h1><p>Content</p>", "text/html")),
        ("documents", ("doc2.html", b"<h1>Title</h1><p>Paragraph</p>", "text/html")),
    ]

    response = client.post(
        "/v1/jobs", files=files, data={"requestedOutputs": "markdown"}
    )
    assert response.status_code == 202
    payload = response.json()

    job_id = payload["jobId"]
    assert payload["status"] in {"complete", "completed_with_warnings"}
    assert len(payload["submissions"]) == 2

    status_response = client.get(f"/v1/jobs/{job_id}")
    assert status_response.status_code == 200
    status_payload = status_response.json()

    assert status_payload["jobId"] == job_id
    assert len(status_payload["submissions"]) == 2
    assert any(
        sub.get("outputs") for sub in status_payload["submissions"]
    )  # at least one successful output
