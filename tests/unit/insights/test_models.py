from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from deepread.insights.models import (
    ActionSuggestion,
    InsightFinding,
    InsightReport,
)


def _build_finding(**kwargs) -> InsightFinding:
    base = {
        "title": "Key Finding",
        "description": "Observation about the document.",
        "page_refs": [1, 2],
        "confidence": 0.85,
    }
    base.update(kwargs)
    return InsightFinding(**base)


def test_insight_report_accepts_valid_payload() -> None:
    report = InsightReport(
        submission_id=uuid4(),
        summary="Concise overview",
        key_findings=[_build_finding()],
        action_suggestions=[
            ActionSuggestion(instruction="Follow up with client", rationale="Outstanding questions.")
        ],
        open_questions=["Need clarification on budget assumption?"],
        generated_formats={"markdown": "reports/doc.md", "json": "reports/doc.json"},
        generated_at=datetime.now(timezone.utc),
    )

    assert report.key_findings[0].page_refs == [1, 2]
    assert report.generated_formats["markdown"] == "reports/doc.md"


def test_page_refs_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _build_finding(page_refs=[0])


def test_confidence_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        _build_finding(confidence=1.2)


def test_report_requires_supported_formats() -> None:
    with pytest.raises(ValidationError):
        InsightReport(
            submission_id=uuid4(),
            summary="Summary",
            key_findings=[_build_finding()],
            action_suggestions=[ActionSuggestion(instruction="Act", rationale="Reason")],
            generated_formats={"pdf": "reports/legacy.pdf"},
            generated_at=datetime.now(timezone.utc),
        )
