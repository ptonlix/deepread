"""
Pydantic models describing generated insight content.

These models enforce traceability back to page references and guarantee that
confidence scores stay within expected ranges. They also constrain supported
output formats so CLI, API, and UI surfaces remain consistent.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

SUPPORTED_OUTPUT_FORMATS = {"markdown", "json", "rich_text"}


class InsightFinding(BaseModel):
    """Key finding captured from OCR and summarization stages."""

    title: str
    description: str
    page_refs: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("page_refs")
    @classmethod
    def _validate_page_refs(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("page_refs must contain at least one reference.")
        if any(ref < 1 for ref in value):
            raise ValueError("page_refs must be positive integers.")
        return value


class ActionSuggestion(BaseModel):
    """Actionable next steps derived from the findings."""

    instruction: str
    rationale: str


class InsightReport(BaseModel):
    """Aggregate structure delivered to end users and integrations."""

    submission_id: UUID
    summary: str
    key_findings: list[InsightFinding]
    action_suggestions: list[ActionSuggestion]
    open_questions: list[str] = Field(default_factory=list)
    generated_formats: dict[str, str]
    generated_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator("key_findings")
    @classmethod
    def _validate_key_findings(
        cls, value: list[InsightFinding]
    ) -> list[InsightFinding]:
        if not value:
            raise ValueError("At least one key finding is required.")
        return value

    @field_validator("generated_formats")
    @classmethod
    def _validate_generated_formats(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("At least one generated format is required.")
        invalid = set(value.keys()) - SUPPORTED_OUTPUT_FORMATS
        if invalid:
            raise ValueError(
                f"Unsupported output formats: {', '.join(sorted(invalid))}"
            )
        for fmt, path in value.items():
            if not path:
                raise ValueError(
                    f"Generated format '{fmt}' must map to a non-empty path."
                )
        return value
