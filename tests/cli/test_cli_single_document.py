from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from deepread.cli.commands import main


def _create_html_doc(tmp_path: Path) -> Path:
    doc_path = tmp_path / "sample.html"
    doc_path.write_text(
        "<html><body><h1>Sample</h1><p>This is a sample insight document.</p></body></html>",
        encoding="utf-8",
    )
    return doc_path


def test_cli_submit_status_fetch(tmp_path: Path, monkeypatch) -> None:
    store_dir = tmp_path / "store"
    monkeypatch.setenv("DEEPREAD_STORE", str(store_dir))

    doc_path = _create_html_doc(tmp_path)

    stdout = StringIO()
    with redirect_stdout(stdout):
        main(["submit", str(doc_path), "--output-format", "mardown"])
    submit_output = stdout.getvalue().strip()
    assert "Job" in submit_output
    job_id = submit_output.split()[-1]

    stdout = StringIO()
    with redirect_stdout(stdout):
        main(["status", job_id])
    status_output = stdout.getvalue().strip()
    assert "complete" in status_output.lower()

    stdout = StringIO()
    with redirect_stdout(stdout):
        main(["fetch", job_id, "--format", "markdown"])
    fetch_output = stdout.getvalue()
    assert "Sample" in fetch_output
    assert "Summary" in fetch_output
