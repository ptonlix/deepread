from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_batch_ui_renders(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPREAD_STORE", str(tmp_path / "store"))
    app = AppTest.from_file("deepread/ui/app.py")
    app.run()

    uploaders = app.get("file_uploader")
    assert uploaders, "Expected a file uploader widget"
    buttons = app.get("button")
    assert any("Process" in button.label for button in buttons)
