from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from deepread.insights.models import (
    ActionSuggestion,
    InsightFinding,
    InsightReport,
)
from deepread.insights.templates import render_json, render_markdown, render_rich_text


def _sample_report() -> InsightReport:
    submission_id = uuid4()
    generated_at = datetime.now(timezone.utc)
    return InsightReport(
        submission_id=submission_id,
        summary="Page 1: Summary text.",
        key_findings=[
            InsightFinding(
                title="Finding 1",
                description="Observation about the document.",
                page_refs=[1],
                confidence=0.92,
            )
        ],
        action_suggestions=[
            ActionSuggestion(instruction="Follow up with owner.", rationale="Confidence 0.92.")
        ],
        open_questions=["What is the revised timeline?"],
        generated_formats={"markdown": "insight.md", "json": "insight.json"},
        generated_at=generated_at,
        warnings=["Low confidence on page 2."],
    )


def test_render_json_includes_required_fields() -> None:
    report = _sample_report()
    payload = render_json(report)
    data = json.loads(payload)

    assert data["summary"]
    assert data["key_findings"][0]["page_refs"] == [1]
    assert data["action_suggestions"][0]["instruction"]
    assert data["warnings"][0].startswith("Low confidence")


def test_render_markdown_contains_sections() -> None:
    report = _sample_report()
    markdown = render_markdown(report)

    assert "# Insight Report" in markdown
    assert "## Key Findings" in markdown
    assert "Finding 1" in markdown
    assert "## Recommended Actions" in markdown
    assert "## Warnings" in markdown


def test_render_rich_text_returns_valid_rtf_payload() -> None:
    report = _sample_report()
    rtf = render_rich_text(report)

    assert rtf.startswith("{\\rtf1")
    assert "\\b Finding 1\\b0" in rtf
