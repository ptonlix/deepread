"""
Asynchronous ingestion pipeline orchestration.

This module wires together document processing stages through a bounded worker
pool. For Phase 2 the focus is on concurrency controls and lifecycle tracking;
later phases will plug in real conversion, rendering, OCR, and summarization
logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Awaitable, Callable, Iterable
from uuid import UUID, uuid4

from deepread.insights.models import InsightReport
from deepread.insights.summarizer import InsightSummarizer, PageInsight
from deepread.insights.templates import (
    render_json,
    render_manifest,
    render_markdown,
    render_rich_text,
)
from deepread.ocr.deepseek import DeepSeekOcr
from .converters import PageImage, convert_document
from .workspace import JobWorkspace

# DPI constant for image consistency
DPI = (300, 300)
FALLBACK_TEXT_MESSAGE = (
    "Document conversion failed; generated summary from raw content."
)
FALLBACK_CONFIDENCE = 0.55

Processor = Callable[[bytes, str], Awaitable["ProcessorResult | None"]]


class JobStatus(str):
    """Lifecycle states for pipeline jobs."""

    QUEUED = "queued"
    RENDERING = "rendering"
    OCR = "ocr"
    SUMMARIZING = "summarizing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(slots=True)
class JobProgress:
    """Public information about a job's current state."""

    job_id: str
    status: str
    error: str | None = None


@dataclass(slots=True)
class _JobState:
    """Internal representation kept by the manager."""

    job_id: str
    status: str = JobStatus.QUEUED
    error: str | None = None
    result: "ProcessorResult | None" = None


@dataclass(slots=True)
class ProcessorResult:
    """Outcome returned from the worker processor."""

    status: str
    error: str | None = None
    payload: object | None = None


class JobManager:
    """Coordinate document processing with bounded concurrency."""

    def __init__(self, *, max_workers: int, processor: Processor) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._processor = processor
        self._semaphore = asyncio.Semaphore(max_workers)
        self._jobs: dict[str, _JobState] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit(self, *, document: bytes, filename: str) -> JobProgress:
        """Queue a document for processing and return initial progress."""
        job_id = uuid4().hex
        state = _JobState(job_id=job_id)
        self._jobs[job_id] = state

        task = asyncio.create_task(self._run_job(state, document, filename))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        return JobProgress(job_id=job_id, status=state.status, error=state.error)

    def get_progress(self, job_id: str) -> JobProgress:
        """Retrieve current progress for a job."""
        if job_id not in self._jobs:
            raise KeyError(f"No job with id '{job_id}'")
        state = self._jobs[job_id]
        return JobProgress(job_id=state.job_id, status=state.status, error=state.error)

    def get_result(self, job_id: str) -> ProcessorResult | None:
        if job_id not in self._jobs:
            raise KeyError(f"No job with id '{job_id}'")
        return self._jobs[job_id].result

    async def wait_for_all(self) -> None:
        """Block until all submitted tasks finish."""
        while self._tasks:
            pending = list(self._tasks)
            await asyncio.gather(*pending, return_exceptions=True)

    async def _run_job(self, state: _JobState, document: bytes, filename: str) -> None:
        try:
            async with self._semaphore:
                state.status = JobStatus.RENDERING
                result = await self._processor(document, filename)
                if isinstance(result, ProcessorResult):
                    state.status = result.status
                    state.error = result.error
                    state.result = result
                else:
                    state.status = JobStatus.COMPLETE
        except Exception as exc:  # pragma: no cover - handled in tests
            state.status = JobStatus.FAILED
            state.error = str(exc)


@dataclass(slots=True)
class SubmissionResult:
    """Return payload describing a processed submission."""

    job_id: str
    submission_id: str
    original_filename: str
    status: str
    outputs: dict[str, str]
    report: InsightReport | None
    remediation: str | None = None


@dataclass(slots=True)
class BatchResult:
    job_id: str
    status: str
    submissions: list[SubmissionResult]
    manifest_path: str


class _EchoEngine:
    """Default inference engine that echoes source text for deterministic tests."""

    def __init__(self, text: str, confidence: float = 0.92) -> None:
        self._text = text
        self._confidence = confidence

    def __call__(
        self, *, prompt: str, image_bytes: bytes, max_tokens: int
    ) -> tuple[str, float]:
        _ = (prompt, image_bytes, max_tokens)
        return self._text, self._confidence


class ProcessingPipeline:
    """End-to-end pipeline transforming documents into insight reports."""

    def __init__(
        self,
        *,
        workspace_root: Path | str,
        summarizer: InsightSummarizer | None = None,
        ocr_factory: Callable[[str], DeepSeekOcr] | None = None,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        self._summarizer = summarizer or InsightSummarizer()
        self._ocr_factory = ocr_factory or (
            lambda text: DeepSeekOcr(engine=_EchoEngine(text))
        )

    def process_document(
        self,
        *,
        document: bytes,
        filename: str,
        requested_formats: set[str],
        job_id: str | None = None,
        submission_id: UUID | None = None,
        workspace: JobWorkspace | None = None,
    ) -> SubmissionResult:
        job_id = job_id or uuid4().hex
        submission_uuid = submission_id or uuid4()
        submission_label = submission_uuid.hex[:8]
        workspace = workspace or JobWorkspace(
            root=self._workspace_root, job_id=f"{job_id}-{submission_label}"
        )
        workspace.create()

        remediation: str | None = None
        try:
            page_images = convert_document(document, filename)
        except ValueError as exc:
            remediation = f"Processing failed: {exc}"
            return SubmissionResult(
                job_id=job_id,
                submission_id=str(submission_uuid),
                original_filename=filename,
                status="failed",
                outputs={},
                report=None,
                remediation=remediation,
            )
        except Exception as exc:
            page_images = []
            remediation = f"Conversion failed: {exc}"

        if page_images:
            page_insights = self._run_render_and_ocr(workspace, page_images)
        else:
            page_insights = [
                PageInsight(
                    page_index=0,
                    text=_extract_fallback_text(document),
                    confidence=FALLBACK_CONFIDENCE,
                )
            ]
            if remediation is None:
                remediation = FALLBACK_TEXT_MESSAGE
            else:
                remediation = f"{FALLBACK_TEXT_MESSAGE} ({remediation})"

        try:
            report = self._summarizer.summarize(
                submission_id=submission_uuid, page_insights=page_insights
            )
            outputs = self._persist_outputs(workspace, report, requested_formats)
            report_with_outputs = report.model_copy(
                update={"generated_formats": outputs}
            )
            return SubmissionResult(
                job_id=job_id,
                submission_id=str(submission_uuid),
                original_filename=filename,
                status="complete",
                outputs=outputs,
                report=report_with_outputs,
                remediation=remediation,
            )
        except Exception as exc:  # pragma: no cover - exercised via integration tests
            remediation_message = (
                remediation if remediation is not None else f"Processing failed: {exc}"
            )
            return SubmissionResult(
                job_id=job_id,
                submission_id=str(submission_uuid),
                original_filename=filename,
                status="failed",
                outputs={},
                report=None,
                remediation=remediation_message,
            )

    def create_job_manager(
        self, *, max_workers: int, default_formats: set[str] | None = None
    ) -> JobManager:
        formats = default_formats or {"markdown"}

        async def processor(document: bytes, filename: str) -> ProcessorResult:
            result = await asyncio.to_thread(
                self.process_document,
                document=document,
                filename=filename,
                requested_formats=formats,
            )
            return ProcessorResult(status=result.status, error=None, payload=result)

        return JobManager(max_workers=max_workers, processor=processor)

    def process_batch(
        self,
        *,
        documents: Iterable[tuple[bytes, str]],
        requested_formats: set[str],
    ) -> BatchResult:
        documents = list(documents)
        if not documents:
            raise ValueError("At least one document must be provided")

        job_id = uuid4().hex
        batch_workspace = JobWorkspace(root=self._workspace_root, job_id=job_id)
        batch_workspace.create()

        submissions: list[SubmissionResult] = []
        for payload, filename in documents:
            submission_uuid = uuid4()
            submission_workspace = JobWorkspace(
                root=batch_workspace.base_dir, job_id=submission_uuid.hex[:8]
            )
            submission_workspace.create()
            result = self.process_document(
                document=payload,
                filename=filename,
                requested_formats=requested_formats,
                job_id=job_id,
                submission_id=submission_uuid,
                workspace=submission_workspace,
            )
            submissions.append(result)

        manifest_content = render_manifest(
            [
                {
                    "submission_id": submission.submission_id,
                    "filename": submission.original_filename,
                    "status": submission.status,
                    "outputs": submission.outputs,
                    "remediation": submission.remediation,
                }
                for submission in submissions
            ]
        )
        manifest_path = batch_workspace.outputs_dir / "manifest.md"
        manifest_path.write_text(manifest_content, encoding="utf-8")

        status = "complete"
        if any(sub.status != "complete" for sub in submissions):
            status = "completed_with_warnings"

        return BatchResult(
            job_id=job_id,
            status=status,
            submissions=submissions,
            manifest_path=str(manifest_path),
        )

    def _run_render_and_ocr(
        self, workspace: JobWorkspace, page_images: Iterable[PageImage]
    ) -> list[PageInsight]:
        from io import BytesIO

        page_insights: list[PageInsight] = []
        for page in page_images:
            buffer = BytesIO()
            page.image.save(buffer, format="PNG", dpi=DPI)
            buffer.seek(0)

            image_path = workspace.images_dir / f"page-{page.index + 1:04d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(buffer.getvalue())

            ocr = self._ocr_factory(page.text_hint)
            ocr_output = ocr.run(image_bytes=buffer.getvalue())
            page_insights.append(
                PageInsight(
                    page_index=page.index,
                    text=ocr_output.text,
                    confidence=ocr_output.confidence,
                )
            )

        return page_insights

    def _persist_outputs(
        self,
        workspace: JobWorkspace,
        report: InsightReport,
        requested_formats: set[str],
    ) -> dict[str, str]:
        outputs: dict[str, str] = {}
        if "markdown" in requested_formats:
            markdown = render_markdown(report)
            path = workspace.outputs_dir / "insight.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(markdown, encoding="utf-8")
            outputs["markdown"] = str(path)
        if "json" in requested_formats:
            payload = render_json(report)
            path = workspace.outputs_dir / "insight.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
            outputs["json"] = str(path)
        if "rich_text" in requested_formats:
            payload = render_rich_text(report)
            path = workspace.outputs_dir / "insight.rtf"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
            outputs["rich_text"] = str(path)
        return outputs


def _extract_fallback_text(document: bytes) -> str:
    """Derive a human-readable fallback snippet from raw document bytes."""
    try:
        decoded = document.decode("utf-8", errors="ignore")
    except Exception:  # pragma: no cover - extremely defensive
        decoded = ""
    text = decoded.strip() or FALLBACK_TEXT_MESSAGE
    # Limit length to keep summaries concise while preserving key context
    return shorten(text, width=1200, placeholder="…")
