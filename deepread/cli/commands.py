"""
Command-line interface for the document OCR insight pipeline.

Usage patterns:
    python -m deepread.cli.commands submit path/to/doc.pdf --output-format markdown
    python -m deepread.cli.commands status <job_id>
    python -m deepread.cli.commands fetch <job_id> --format markdown
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from deepread.ingest.pipeline import ProcessingPipeline

DEFAULT_FORMATS = {"markdown"}
SUPPORTED_FORMATS = {"markdown", "json", "rich_text"}


@dataclass(slots=True)
class SubmissionMetadata:
    submission_id: str
    filename: str
    status: str
    outputs: dict[str, str]
    remediation: str | None = None


@dataclass(slots=True)
class JobMetadata:
    job_id: str
    status: str
    manifest_path: str
    submissions: list[SubmissionMetadata]

    def to_json(self) -> str:
        return json.dumps(
            {
                "job_id": self.job_id,
                "status": self.status,
                "manifest_path": self.manifest_path,
                "submissions": [asdict(sub) for sub in self.submissions],
            },
            indent=2,
        )

    @staticmethod
    def from_json(payload: str) -> "JobMetadata":
        data = json.loads(payload)
        submissions = [
            SubmissionMetadata(**item) for item in data.get("submissions", [])
        ]
        return JobMetadata(
            job_id=data["job_id"],
            status=data["status"],
            manifest_path=data["manifest_path"],
            submissions=submissions,
        )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    store = _resolve_store()
    workspace_root = store / "workspaces"
    pipeline = ProcessingPipeline(workspace_root=workspace_root)

    if args.command == "submit":
        formats = {fmt.lower() for fmt in (args.output_format or [])} or DEFAULT_FORMATS
        _ensure_supported_formats(formats)
        document_paths = [Path(path) for path in args.paths]
        documents = [(path.read_bytes(), path.name) for path in document_paths]
        batch = pipeline.process_batch(documents=documents, requested_formats=formats)
        metadata = JobMetadata(
            job_id=batch.job_id,
            status=batch.status,
            manifest_path=batch.manifest_path,
            submissions=[
                SubmissionMetadata(
                    submission_id=sub.submission_id,
                    filename=sub.original_filename,
                    status=sub.status,
                    outputs=sub.outputs,
                    remediation=sub.remediation,
                )
                for sub in batch.submissions
            ],
        )
        _save_metadata(store, metadata)
        print(f"Job {batch.job_id}")
    elif args.command == "status":
        metadata = _load_metadata(store, args.job_id)
        display_status = (
            "completed" if metadata.status == "complete" else metadata.status
        )
        print(f"Job {metadata.job_id} status: {display_status}")
        for submission in metadata.submissions:
            remediation = (
                f" ({submission.remediation})" if submission.remediation else ""
            )
            print(f"- {submission.filename}: {submission.status}{remediation}")
    elif args.command == "fetch":
        metadata = _load_metadata(store, args.job_id)
        format_name = args.format.lower()
        _ensure_supported_formats({format_name})
        submission = _select_submission(metadata, args.submission_id)
        if format_name not in submission.outputs:
            raise SystemExit(
                f"Format '{format_name}' not available for job {args.job_id}"
            )
        content = Path(submission.outputs[format_name]).read_text(encoding="utf-8")
        print(content)
    elif args.command == "manifest":
        metadata = _load_metadata(store, args.job_id)
        manifest = Path(metadata.manifest_path).read_text(encoding="utf-8")
        print(manifest)
    else:  # pragma: no cover - argparse ensures command is valid
        parser.error(f"Unknown command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepread-cli", description="Document OCR insight pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser(
        "submit", help="Submit a document for insight generation"
    )
    submit.add_argument(
        "paths", nargs="+", help="One or more document paths to process"
    )
    submit.add_argument(
        "--output-format",
        action="append",
        choices=sorted(SUPPORTED_FORMATS),
        help="Output format to generate (can be provided multiple times)",
    )

    status = subparsers.add_parser("status", help="Show job status")
    status.add_argument("job_id", help="Identifier returned from the submit command")

    fetch = subparsers.add_parser("fetch", help="Retrieve generated output for a job")
    fetch.add_argument("job_id", help="Identifier returned from the submit command")
    fetch.add_argument(
        "--format",
        required=True,
        choices=sorted(SUPPORTED_FORMATS),
        help="Output format to read",
    )
    fetch.add_argument(
        "--submission-id",
        help="Submission identifier when multiple inputs were processed",
    )

    manifest = subparsers.add_parser("manifest", help="Show the manifest for a job")
    manifest.add_argument("job_id", help="Identifier returned from the submit command")

    return parser


def _resolve_store() -> Path:
    root = os.environ.get("DEEPREAD_STORE")
    store = Path(root) if root else Path.cwd() / ".deepread-store"
    store.mkdir(parents=True, exist_ok=True)
    return store


def _save_metadata(store: Path, metadata: JobMetadata) -> None:
    path = store / f"{metadata.job_id}.json"
    path.write_text(metadata.to_json(), encoding="utf-8")


def _load_metadata(store: Path, job_id: str) -> JobMetadata:
    path = store / f"{job_id}.json"
    if not path.exists():
        raise SystemExit(f"Job {job_id} not found")
    return JobMetadata.from_json(path.read_text(encoding="utf-8"))


def _ensure_supported_formats(formats: set[str]) -> None:
    unsupported = formats - SUPPORTED_FORMATS
    if unsupported:
        raise SystemExit(
            f"Unsupported output formats requested: {', '.join(sorted(unsupported))}"
        )


def _select_submission(
    metadata: JobMetadata, submission_id: str | None
) -> SubmissionMetadata:
    if submission_id:
        for submission in metadata.submissions:
            if submission.submission_id == submission_id:
                return submission
        raise SystemExit(
            f"Submission {submission_id} not found for job {metadata.job_id}"
        )
    if len(metadata.submissions) == 1:
        return metadata.submissions[0]
    raise SystemExit(
        "Multiple submissions present; specify --submission-id to fetch output"
    )
