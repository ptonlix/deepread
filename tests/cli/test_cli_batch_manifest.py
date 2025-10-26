from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from deepread.cli.commands import main


def _write_doc(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_cli_batch_manifest(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "store"
    monkeypatch.setenv("DEEPREAD_STORE", str(store))

    doc1 = _write_doc(tmp_path / "a.html", "<h1>Doc A</h1><p>Content A</p>")
    doc2 = _write_doc(tmp_path / "b.html", "<h1>Doc B</h1>")

    stdout = StringIO()
    with redirect_stdout(stdout):
        main(["submit", str(doc1), str(doc2), "--output-format", "markdown"])
    job_id = stdout.getvalue().strip().split()[-1]

    stdout = StringIO()
    with redirect_stdout(stdout):
        main(["status", job_id])
    status_output = stdout.getvalue()
    assert "completed" in status_output.lower()
    assert "a.html" in status_output

    stdout = StringIO()
    with redirect_stdout(stdout):
        main(["manifest", job_id])
    manifest_output = stdout.getvalue()
    assert "Document Batch Manifest" in manifest_output
    assert "a.html" in manifest_output and "b.html" in manifest_output
