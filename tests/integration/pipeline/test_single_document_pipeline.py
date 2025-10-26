from __future__ import annotations

from pathlib import Path

from deepread.ingest.pipeline import ProcessingPipeline


def test_processing_pipeline_generates_markdown(tmp_path: Path) -> None:
    pipeline = ProcessingPipeline(workspace_root=tmp_path)
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
    assert "Heading" in content
    assert "Confidence" in content
