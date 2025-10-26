from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from deepread.api.router import create_app


def test_fetch_structured_outputs(tmp_path: Path) -> None:
    app = create_app(workspace_root=tmp_path / "api")
    client = TestClient(app)

    files = [
        ("documents", ("doc.md", b"# Heading\n\nContent", "text/markdown")),
    ]

    response = client.post(
        "/v1/jobs",
        files=files,
        data={"requestedOutputs": ["markdown", "json", "rich_text"]},
    )
    assert response.status_code == 202
    payload = response.json()
    submission = payload["submissions"][0]
    submission_id = submission["submissionId"]

    json_response = client.get(f"/v1/reports/{submission_id}/content", params={"format": "json"})
    assert json_response.status_code == 200
    data = json_response.json()
    assert "summary" in data

    rtf_response = client.get(f"/v1/reports/{submission_id}/content", params={"format": "rich_text"})
    assert rtf_response.status_code == 200
    assert rtf_response.text.startswith("{\\rtf1")
