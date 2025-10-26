from __future__ import annotations

from uuid import uuid4

from deepread.insights.models import InsightReport
from deepread.insights.summarizer import InsightSummarizer, PageInsight


def test_summarizer_generates_summary_and_findings() -> None:
    summarizer = InsightSummarizer()
    submission_id = uuid4()

    report = summarizer.summarize(
        submission_id=submission_id,
        page_insights=[
            PageInsight(page_index=0, text="Project goals are on track.", confidence=0.93),
            PageInsight(page_index=1, text="Risks identified on budget.", confidence=0.88),
        ],
    )

    assert isinstance(report, InsightReport)
    assert "Project goals" in report.summary
    assert len(report.key_findings) == 2
    assert report.key_findings[0].page_refs == [1]
    assert "Confidence" in report.action_suggestions[0].rationale


def test_summarizer_includes_low_confidence_warnings() -> None:
    summarizer = InsightSummarizer()
    report = summarizer.summarize(
        submission_id=uuid4(),
        page_insights=[PageInsight(page_index=0, text="Unreadable text", confidence=0.4)],
    )

    assert report.warnings, "Expected warning when confidence is low"
    assert "0.40" in report.warnings[0]
