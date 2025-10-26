from __future__ import annotations

from io import BytesIO

import pytest
from docx import Document
from markdown_it import MarkdownIt
from openpyxl import Workbook
from pptx import Presentation

from deepread.ingest.converters import PageContent, convert_document


def _markdown_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def _html_bytes() -> bytes:
    return b"<html><body><h1>Title</h1><p>Paragraph one.</p></body></html>"


def _docx_bytes() -> bytes:
    buffer = BytesIO()
    doc = Document()
    doc.add_heading("Proposal", level=1)
    doc.add_paragraph("This document outlines the project scope.")
    doc.save(buffer)
    return buffer.getvalue()


def _pptx_bytes() -> bytes:
    buffer = BytesIO()
    presentation = Presentation()
    slide_layout = presentation.slide_layouts[0]
    slide = presentation.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Roadmap"
    slide.placeholders[1].text = "Phase 1\nPhase 2"
    presentation.save(buffer)
    return buffer.getvalue()


def _xlsx_bytes() -> bytes:
    buffer = BytesIO()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Summary"
    sheet.append(["Task", "Owner"])
    sheet.append(["Convert docs", "Team A"])
    workbook.save(buffer)
    return buffer.getvalue()


def _fake_pdf_bytes() -> bytes:
    return b"%PDF-FAKE\nPage 1 content\fPage 2 content"


@pytest.mark.parametrize(
    ("filename", "payload", "expected_text"),
    [
        ("sample.md", _markdown_bytes("# Title\n\ncontent"), "Title"),
        ("sample.html", _html_bytes(), "Title"),
        ("sample.docx", _docx_bytes(), "Proposal"),
        ("sample.pptx", _pptx_bytes(), "Roadmap"),
        ("sample.xlsx", _xlsx_bytes(), "Task"),
        ("sample.pdf", _fake_pdf_bytes(), "Page 1"),
    ],
)
def test_convert_document_extracts_text(filename: str, payload: bytes, expected_text: str) -> None:
    pages = convert_document(payload, filename)

    assert pages, "No pages returned from converter"
    assert all(isinstance(page, PageContent) for page in pages)
    assert expected_text in pages[0].text


def test_markdown_conversion_preserves_structure() -> None:
    text = "# Title\n\n- item one\n- item two"
    pages = convert_document(_markdown_bytes(text), "tasks.md")

    parsed = MarkdownIt().parse(pages[0].text)
    assert any(token.tag == "ul" for token in parsed), "Expected bullet list tokens in parsed markdown"
