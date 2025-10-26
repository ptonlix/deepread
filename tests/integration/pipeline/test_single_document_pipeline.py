from __future__ import annotations

from pathlib import Path

from deepread.ingest.pipeline import ProcessingPipeline


def test_processing_pipeline_generates_markdown(tmp_path: Path) -> None:
    pipeline = ProcessingPipeline(workspace_root=tmp_path)
    document = b"# Heading\n\nThis is the body of the document."

    result = pipeline.process_document(
        document=document,
        filename="input.md",
        requested_formats={"markdown"},
    )

    assert result.status == "complete"
    assert "markdown" in result.outputs

    markdown_path = result.outputs["markdown"]
    content = Path(markdown_path).read_text(encoding="utf-8")

    assert "# Insight Report" in content
    assert "Heading" in content
    assert "Confidence" in content
