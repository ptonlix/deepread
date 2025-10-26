from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from deepread.ingest.pipeline import JobManager, JobProgress, JobStatus


async def _submit_and_collect(
    manager: JobManager,
    count: int,
    payload_factory: Callable[[int], tuple[bytes, str]],
) -> list[JobProgress]:
    return [
        await manager.submit(document=document, filename=filename)
        for document, filename in (payload_factory(idx) for idx in range(count))
    ]


@pytest.mark.asyncio
async def test_job_manager_enforces_concurrency_limit() -> None:
    max_workers = 2
    active_jobs = 0
    peak_active = 0
    active_lock = asyncio.Lock()

    async def processor(_: bytes, __: str) -> None:
        nonlocal active_jobs, peak_active
        async with active_lock:
            active_jobs += 1
            peak_active = max(peak_active, active_jobs)
        await asyncio.sleep(0.05)
        async with active_lock:
            active_jobs -= 1

    manager = JobManager(max_workers=max_workers, processor=processor)

    jobs = await _submit_and_collect(
        manager=manager,
        count=5,
        payload_factory=lambda idx: (b"", f"doc-{idx}.pdf"),
    )

    await manager.wait_for_all()

    assert peak_active <= max_workers
    assert all(manager.get_progress(job.job_id).status == JobStatus.COMPLETE for job in jobs)


@pytest.mark.asyncio
async def test_job_manager_records_failure_state() -> None:
    async def processor(_: bytes, __: str) -> None:
        raise RuntimeError("synthetic failure")

    manager = JobManager(max_workers=1, processor=processor)

    job = await manager.submit(document=b"", filename="failing.docx")
    await manager.wait_for_all()

    progress = manager.get_progress(job.job_id)
    assert progress.status == JobStatus.FAILED
    assert progress.error is not None
    assert "synthetic failure" in progress.error
