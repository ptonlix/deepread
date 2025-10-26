"""
Rendering helpers for insight outputs.
"""

from __future__ import annotations

import json
from textwrap import indent
from typing import Any

from deepread.insights.models import InsightReport


def render_markdown(report: InsightReport) -> str:
    """Return a Markdown representation of the insight report."""
    lines = [
        "# Insight Report",
        "",
        "## Summary",
        report.summary.strip(),
        "",
        "## Key Findings",
    ]

    for finding in report.key_findings:
        refs = ", ".join(f"Page {ref}" for ref in finding.page_refs)
        lines.append(
            f"- **{finding.title}** ({refs}, confidence {finding.confidence:.2f})"
        )
        lines.append(indent(finding.description.strip(), "  "))

    lines.extend(["", "## Recommended Actions"])
    for suggestion in report.action_suggestions:
        lines.append(f"- {suggestion.instruction}")
        lines.append(indent(f"Rationale: {suggestion.rationale}", "  "))

    if report.warnings:
        lines.extend(["", "## Warnings"])
        for warning in report.warnings:
            lines.append(f"- {warning}")

    if report.open_questions:
        lines.extend(["", "## Open Questions"])
        for question in report.open_questions:
            lines.append(f"- {question}")

    lines.append("")
    return "\n".join(lines)


def render_json(report: InsightReport) -> str:
    """Return a JSON string for structured integrations."""
    return json.dumps(report.model_dump(mode="json"), indent=2)


def render_manifest(submissions: list[dict[str, Any]]) -> str:
    """Render a Markdown manifest summarizing batch submissions."""
    lines = ["# Document Batch Manifest", ""]
    lines.append("| Submission ID | Filename | Status | Outputs | Remediation |")
    lines.append("|---------------|----------|--------|---------|-------------|")
    for item in submissions:
        outputs_obj = item.get("outputs")
        outputs = (
            ", ".join(sorted(outputs_obj.keys()))
            if isinstance(outputs_obj, dict)
            else "—"
        )
        remediation = item.get("remediation") or "—"
        lines.append(
            f"| {item['submission_id']} | {item['filename']} | {item['status']} | {outputs} | {remediation} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_rich_text(report: InsightReport) -> str:
    """Return an RTF representation suitable for rich-text viewers."""
    lines = [
        r"{\rtf1\ansi",
        r"\b Insight Report\b0\line",
        r"\b Summary\b0\line " + report.summary.replace("\n", r"\line "),
        r"\line\b Key Findings\b0\line",
    ]

    for finding in report.key_findings:
        refs = ", ".join(f"Page {ref}" for ref in finding.page_refs)
        lines.append(
            rf"\b {finding.title}\b0 ({refs}, confidence {finding.confidence:.2f})\line "
        )
        lines.append(finding.description.replace("\n", r"\line ") + r"\line ")

    lines.append(r"\line\b Recommended Actions\b0\line")
    for suggestion in report.action_suggestions:
        lines.append(
            rf"{suggestion.instruction}\line Rationale: {suggestion.rationale}\line "
        )

    if report.warnings:
        lines.append(r"\line\b Warnings\b0\line")
        for warning in report.warnings:
            lines.append(warning + r"\line ")

    if report.open_questions:
        lines.append(r"\line\b Open Questions\b0\line")
        for question in report.open_questions:
            lines.append(question + r"\line ")

    lines.append("}")
    return "".join(lines)
