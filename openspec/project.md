# Project Context

## Purpose

Deepread orchestrates a document OCR insight pipeline that ingests office files, runs OCR, and delivers human-friendly summaries. The project exposes the pipeline through a FastAPI service, a CLI, and a Streamlit UI so teams can batch process uploads, inspect remediation guidance, and export insight reports in multiple formats.

## Tech Stack

- Python 3.12 with strict typing and Pydantic models for insight schemas
- FastAPI + Uvicorn for the HTTP surface, Streamlit for the UI, argparse-based CLI
- Asynchronous ingestion pipeline built on asyncio, PIL/Pillow images, and pandas data shaping
- DeepSeek-OCR running on a vLLM inference engine (planned integration) with retries and confidence tracking
- Document converters powered by pdf2image (Poppler), python-docx, openpyxl, html2image, imgkit/wkhtmltopdf, and docx2pdf
- Project tooling managed by `uv` (dependency resolution, build, publish) with Ruff, mypy, pytest(+pytest-asyncio), deptry, and pre-commit

## Project Conventions

### Code Style

- Target Python 3.12, keep modules small under `deepread/`, and surface public APIs through `__init__.py`.
- Maintain full typing coverage (`disallow_untyped_defs`, `no_implicit_optional`) and prefer dataclasses or Pydantic models for structured data.
- Formatting and linting are handled by Ruff (`ruff` + `ruff-format`) via pre-commit; run `make check` before pushing.
- Favor f-strings, snake_case for variables/functions, PascalCase for classes, and SCREAMING_SNAKE_CASE for constants.
- Avoid introducing non-ASCII text in code except where required for fixtures or sample content.

### Architecture Patterns

- Pipeline-first design: `ProcessingPipeline` composes conversion, OCR, summarization, and output rendering steps while persisting artifacts per-job via `JobWorkspace`.
- Conversion layer emits `PageImage` objects with image + text hints so OCR backends remain pluggable; OCR integrates with DeepSeek via an injectable `InferenceEngine`.
- Insight generation is deterministic through `InsightSummarizer` -> templates (`render_markdown/json/rich_text`) to keep API, CLI, and UI surfaces aligned.
- FastAPI router stores job + submission state in an in-memory repository per process, while CLI/Streamlit share the same pipeline and persist manifests under `.deepread-store` (configurable through `DEEPREAD_STORE`).
- Concurrency is controlled with bounded asyncio semaphores; workers stream status updates through `JobManager`.

### Testing Strategy

- Pytest with coverage (`make test` / `uv run pytest`) is the standard; results land in `coverage.xml` + `junit.xml`.
- Test layout mirrors the package (`tests/unit`, `tests/integration`, `tests/cli`, etc.) with shared fixtures in `tests/conftest.py` and sample assets under `tests/data/`.
- Async behavior is exercised with `pytest-asyncio`; prefer deterministic fixtures over network calls.
- `make check` chains pre-commit (Ruff lint/format), mypy, and deptry—CI and local pushes expect the same pipeline.

### Git Workflow

- Use feature branches per change and follow Conventional Commits (e.g., `feat: add reader batching`).
- Run `uv sync --all-extras` once, install hooks with `make githooks`, and ensure `make check test` passes before pushing (pre-push hook enforces it).
- PRs should link issues, summarize behavior changes, and list verification commands; prefer small, reviewable diffs.

## Domain Context

- Converts PDFs, DOCX, XLSX, and HTML/HTM into page images, capturing text hints to support OCR retries and deterministic tests.
- OCR leverages DeepSeek via a vLLM engine (pending final wiring) with configurable prompts, retries, and confidence thresholds.
- Insight reports synthesize OCR findings into summaries, recommended actions, warnings, and open questions and can be exported as Markdown, JSON, or RTF.
- Batch processing returns manifests and per-submission remediation guidance; outputs are stored on the filesystem for retrieval across surfaces.
- Streamlit dashboard offers manual review and downloads, while CLI/API cover automation scenarios and downstream integrations.

## Important Constraints

- Python 3.12 baseline with strict typing, Ruff linting, and mypy checks—new code must satisfy both.
- Maintain deterministic, offline-friendly processing; avoid adding network calls in core pipeline stages.
- Document conversion relies on system dependencies (Poppler for pdf2image, wkhtmltopdf for imgkit, LibreOffice on some platforms); note requirements in deployment docs when adding formats.
- Keep specs and implementation in sync via OpenSpec; large or behavior-changing work requires an approved change proposal before coding.
- Workspace writes default to the local filesystem—future object storage integrations must preserve existing contract.

## External Dependencies

- DeepSeek-OCR weights served through a vLLM inference runtime (GPU-capable deployment target).
- Local filesystem staging under `.deepread-store` and `.deepread-api/` for manifests, outputs, and cached jobs (configurable with `DEEPREAD_STORE`).
- System packages: Poppler utilities (`pdf2image`), wkhtmltopdf (`imgkit`), headless browser tooling for `html2image`, LibreOffice/Word automation for `docx2pdf`.
- Streamlit cloud or internal hosting for the UI front-end; FastAPI service typically runs with Uvicorn or another ASGI server.
