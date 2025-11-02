from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from deepread.api.router import create_app
from deepread.ocr.deepseek import DeepSeekOcr


class MockOcrEngine:
    """Mock OCR engine for testing that returns predictable results."""

    def __call__(
        self, *, prompt: str, image_bytes: bytes, max_tokens: int
    ) -> tuple[str, float]:
        return "Extracted text from document", 0.95


def test_fetch_structured_outputs(tmp_path: Path) -> None:
    def create_mock_ocr() -> DeepSeekOcr:
        return DeepSeekOcr(engine=MockOcrEngine())

    app = create_app(workspace_root=tmp_path / "api", ocr_factory=create_mock_ocr)
    client = TestClient(app)

    files = [
        ("documents", ("doc.html", b"<h1>Heading</h1><p>Content</p>", "text/html")),
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

    json_response = client.get(
        f"/v1/reports/{submission_id}/content", params={"format": "json"}
    )
    assert json_response.status_code == 200
    data = json_response.json()
    assert "summary" in data

    rtf_response = client.get(
        f"/v1/reports/{submission_id}/content", params={"format": "rich_text"}
    )
    assert rtf_response.status_code == 200
    assert rtf_response.text.startswith("{\\rtf1")
