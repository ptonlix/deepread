"""FastAPI application exposing the document insight pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from deepread.ingest.pipeline import BatchResult, ProcessingPipeline, SubmissionResult


@dataclass
class _JobRepository:
    jobs: dict[str, BatchResult]
    submissions: dict[str, SubmissionResult]

    def save(self, result: BatchResult) -> None:
        self.jobs[result.job_id] = result
        for submission in result.submissions:
            self.submissions[submission.submission_id] = submission

    def get_job(self, job_id: str) -> BatchResult:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def as_status_payload(self, result: BatchResult) -> dict[str, Any]:
        return {
            "jobId": result.job_id,
            "status": result.status,
            "manifestPath": result.manifest_path,
            "submissions": [
                self._serialize_submission(submission)
                for submission in result.submissions
            ],
        }

    @staticmethod
    def _serialize_submission(submission: SubmissionResult) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "submissionId": submission.submission_id,
            "originalFilename": submission.original_filename,
            "status": submission.status,
            "outputs": submission.outputs,
        }
        if submission.remediation:
            payload["remediation"] = submission.remediation
        return payload


def create_app(*, workspace_root: Path | str | None = None) -> FastAPI:
    workspace = Path(workspace_root) if workspace_root else Path.cwd() / ".deepread-api"
    workspace.mkdir(parents=True, exist_ok=True)

    repository = _JobRepository(jobs={}, submissions={})
    pipeline = ProcessingPipeline(workspace_root=workspace)

    router = APIRouter()

    @router.post("/v1/jobs", status_code=202)
    async def submit_job(
        documents: list[UploadFile] = File(...),
        requestedOutputs: list[str] | None = Form(default=None),
    ) -> JSONResponse:  # noqa: N803 (FastAPI naming convention)
        if not documents:
            raise HTTPException(
                status_code=400, detail="At least one document must be uploaded"
            )

        payloads: list[tuple[bytes, str]] = []
        for item in documents:
            name = item.filename or "untitled"
            payloads.append((await item.read(), name))

        formats = {entry.lower() for entry in requestedOutputs or []} or {"markdown"}
        batch_result = pipeline.process_batch(
            documents=payloads, requested_formats=formats
        )
        repository.save(batch_result)

        return JSONResponse(repository.as_status_payload(batch_result), status_code=202)

    @router.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str) -> JSONResponse:
        try:
            result = repository.get_job(job_id)
        except KeyError as exc:  # pragma: no cover
            raise HTTPException(
                status_code=404, detail=f"Job {job_id} not found"
            ) from exc
        return JSONResponse(repository.as_status_payload(result))

    @router.get("/v1/reports/{submission_id}/content")
    async def report_content(submission_id: str, format: str = Query(...)) -> Response:
        if submission_id not in repository.submissions:
            raise HTTPException(
                status_code=404, detail=f"Submission {submission_id} not found"
            )

        submission = repository.submissions[submission_id]
        if format not in submission.outputs:
            raise HTTPException(
                status_code=400,
                detail=f"Format '{format}' not available for submission {submission_id}",
            )

        path = Path(submission.outputs[format])
        if not path.exists():
            raise HTTPException(
                status_code=500, detail="Report content unavailable on disk"
            )

        media_types = {
            "markdown": "text/markdown",
            "json": "application/json",
            "rich_text": "application/rtf",
        }
        media_type = media_types.get(format, "text/plain")
        return Response(path.read_text(encoding="utf-8"), media_type=media_type)

    app = FastAPI(title="Document OCR Insight API", version="0.1.0")
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


# Convenience application instance for ASGI servers
app = create_app()
