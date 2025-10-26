"""
Insight summarization utilities.

Transforms OCR results into structured reports consumed by downstream
surfaces (CLI, API, Streamlit).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable
from uuid import UUID

from deepread.insights.models import (
    ActionSuggestion,
    InsightFinding,
    InsightReport,
)


@dataclass(slots=True)
class PageInsight:
    """Normalized OCR output for a single page."""

    page_index: int
    text: str
    confidence: float


class InsightSummarizer:
    """Aggregate page insights into an InsightReport."""

    def summarize(self, *, submission_id: UUID, page_insights: Iterable[PageInsight]) -> InsightReport:
        insights = list(page_insights)
        if not insights:
            raise ValueError("At least one page insight is required.")

        summary = self._build_summary(insights)
        findings = self._build_findings(insights)
        suggestions = self._build_suggestions(insights)
        warnings = self._build_warnings(insights)

        return InsightReport(
            submission_id=submission_id,
            summary=summary,
            key_findings=findings,
            action_suggestions=suggestions,
            open_questions=[],
            generated_formats={"markdown": "summary.md"},  # Provide default format
            generated_at=self._current_timestamp(),
            warnings=warnings,
        )

    def _build_summary(self, insights: list[PageInsight]) -> str:
        sentences = []
        for insight in insights:
            text = insight.text.strip().split(".")[0]
            if text:
                sentences.append(f"Page {insight.page_index + 1}: {text.strip()}.")
        avg_confidence = mean(insight.confidence for insight in insights)
        sentences.append(f"Average OCR confidence: {avg_confidence:.2f}.")
        return " ".join(sentences)

    def _build_findings(self, insights: list[PageInsight]) -> list[InsightFinding]:
        findings: list[InsightFinding] = []
        for insight in insights:
            snippet = insight.text.strip()[:280]
            findings.append(
                InsightFinding(
                    title=f"Page {insight.page_index + 1} insight",
                    description=snippet,
                    page_refs=[insight.page_index + 1],
                    confidence=insight.confidence,
                )
            )
        return findings

    def _build_suggestions(self, insights: list[PageInsight]) -> list[ActionSuggestion]:
        suggestions: list[ActionSuggestion] = []
        for insight in insights:
            suggestions.append(
                ActionSuggestion(
                    instruction=f"Review content on page {insight.page_index + 1}.",
                    rationale=f"Confidence recorded at {insight.confidence:.2f}.",
                )
            )
        return suggestions

    def _build_warnings(self, insights: list[PageInsight]) -> list[str]:
        warnings: list[str] = []
        for insight in insights:
            if insight.confidence < 0.6:
                warnings.append(
                    f"Page {insight.page_index + 1} has low OCR confidence ({insight.confidence:.2f})."
                )
        return warnings

    @staticmethod
    def _current_timestamp():
        from datetime import UTC, datetime

        return datetime.now(UTC)
