from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from deepread.cli.commands import main


def test_cli_fetch_json_and_rich_text(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "store"
    monkeypatch.setenv("DEEPREAD_STORE", str(store))

    doc = tmp_path / "doc.md"
    doc.write_text("# Title\n\nBody text", encoding="utf-8")

    stdout = StringIO()
    with redirect_stdout(stdout):
        main(["submit", str(doc), "--output-format", "json", "--output-format", "rich_text"])
    job_id = stdout.getvalue().strip().split()[-1]

    stdout = StringIO()
    with redirect_stdout(stdout):
        main(["fetch", job_id, "--format", "json"])
    data = json.loads(stdout.getvalue())
    assert data["summary"]

    stdout = StringIO()
    with redirect_stdout(stdout):
        main(["fetch", job_id, "--format", "rich_text"])
    rtf_output = stdout.getvalue()
    assert rtf_output.startswith("{\\rtf1")
