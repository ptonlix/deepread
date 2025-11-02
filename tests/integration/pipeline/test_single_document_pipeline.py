from __future__ import annotations

from pathlib import Path

from deepread.ingest.pipeline import ProcessingPipeline
from deepread.ocr.deepseek import DeepSeekOcr


class MockOcrEngine:
    """Mock OCR engine for testing that returns predictable results."""

    def __call__(
        self, *, prompt: str, image_bytes: bytes, max_tokens: int
    ) -> tuple[str, float]:
        # Return a simple mock text based on the image bytes
        # In real tests, you might want to analyze the image or return specific text
        return "Extracted text from document", 0.95


def test_processing_pipeline_generates_markdown(tmp_path: Path) -> None:
    # Create a mock OCR factory for testing
    def create_mock_ocr() -> DeepSeekOcr:
        return DeepSeekOcr(engine=MockOcrEngine())

    pipeline = ProcessingPipeline(workspace_root=tmp_path, ocr_factory=create_mock_ocr)
    document = b"<html><body><h1>Heading</h1><p>This is the body of the document.</p></body></html>"

    result = pipeline.process_document(
        document=document,
        filename="input.html",
        requested_formats={"markdown"},
    )

    assert result.status == "complete"
    assert "markdown" in result.outputs

    markdown_path = result.outputs["markdown"]
    content = Path(markdown_path).read_text(encoding="utf-8")

    assert "# Insight Report" in content
    assert "Confidence" in content
