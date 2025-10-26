# Document OCR Insight Pipeline

## Overview

This guide tracks terminology, workflows, and migration notes for the document OCR insight pipeline.

## Terminology

- **Insight Report**: Primary deliverable summarizing document content and actions.
- **Page Insight**: Page-level annotations and findings linked to original pagination.
- **Confidence Score**: Numeric indicator (0.0–1.0) denoting OCR reliability for a finding.
- **Remediation Guidance**: Actionable follow-up when conversion or OCR quality is insufficient.

## Workflows

- CLI submissions (`deepread.cli.commands`) allow document upload, status checks, and artifact retrieval.
  - `python -m deepread.cli.commands submit docs/sample.md --output-format markdown --output-format json`
  - Job metadata stored under `$DEEPREAD_STORE` (defaults to `.deepread-store/` in the current directory).
  - `status` returns lifecycle state per submission; `fetch` streams Markdown/JSON/Rich Text payloads; `manifest` prints the batch manifest.
- FastAPI endpoints (`deepread.api.router`) expose `/v1/jobs`, `/v1/jobs/{jobId}`, and `/v1/reports/{submissionId}/content` for format-specific downloads.
- Streamlit demo (`deepread.ui.app`) provides a guided upload and review experience.

## Output Formats

- **Markdown**: Generated via `deepread.insights.templates.render_markdown`, includes summary, key findings, actions, and warnings.
- **JSON**: Structured payload mirroring `InsightReport` Pydantic model for downstream integrations.
- **Rich Text (RTF)**: Produced via `deepread.insights.templates.render_rich_text` for office editors that prefer `.rtf` documents.

## Open Tasks

- Fill in detailed architectural diagram after foundational tasks.
- Document configuration for GPU provisioning and vLLM model placement.
- Capture known limitations and troubleshooting steps as features land.
- Backfill Streamlit walkthrough once UI receives interactive pipeline wiring.
