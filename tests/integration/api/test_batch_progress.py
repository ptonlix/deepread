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


def test_batch_progress_records_failures(tmp_path: Path) -> None:
    def create_mock_ocr() -> DeepSeekOcr:
        return DeepSeekOcr(engine=MockOcrEngine())

    app = create_app(workspace_root=tmp_path / "workspace", ocr_factory=create_mock_ocr)
    client = TestClient(app)

    files = [
        (
            "documents",
            ("supported.html", b"<h1>Heading</h1><p>Content</p>", "text/html"),
        ),
        ("documents", ("unsupported.txt", b"plain text", "text/plain")),
    ]

    response = client.post(
        "/v1/jobs", files=files, data={"requestedOutputs": ["markdown"]}
    )
    data = response.json()

    failed = next(
        sub
        for sub in data["submissions"]
        if sub["originalFilename"] == "unsupported.txt"
    )
    assert failed["status"] == "failed"
    assert "Processing failed" in failed["remediation"]

    job_id = data["jobId"]
    status_response = client.get(f"/v1/jobs/{job_id}")
    status_payload = status_response.json()
    assert status_payload["manifestPath"]
