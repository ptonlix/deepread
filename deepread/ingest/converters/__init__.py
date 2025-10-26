"""
Document conversion entry points.

Each supported format is transformed into a list of `PageContent` instances
containing plain text. Rendering to high-DPI images occurs in the rendering
stage; the converter stage focuses solely on extracting textual content with
minimal external dependencies.
"""

from __future__ import annotations

import io
import itertools
from dataclasses import dataclass
from typing import Iterable

from bs4 import BeautifulSoup
from docx import Document
from markdown_it import MarkdownIt
from openpyxl import load_workbook  # type: ignore[import-untyped]
from pptx import Presentation
from pypdf import PdfReader

SUPPORTED_FORMATS = {"pdf", "docx", "pptx", "xlsx", "html", "htm", "markdown", "md"}


@dataclass(slots=True)
class PageContent:
    """Text extracted from a logical page."""

    index: int
    text: str


def convert_document(payload: bytes, filename: str) -> list[PageContent]:
    """Dispatch conversion based on filename extension."""
    extension = filename.split(".")[-1].lower()
    ensure_supported(extension)

    if extension in {"markdown", "md"}:
        pages = _convert_markdown(payload)
    elif extension in {"html", "htm"}:
        pages = _convert_html(payload)
    elif extension == "docx":
        pages = _convert_docx(payload)
    elif extension == "pptx":
        pages = _convert_pptx(payload)
    elif extension == "xlsx":
        pages = _convert_xlsx(payload)
    elif extension == "pdf":
        pages = _convert_pdf(payload)
    else:  # pragma: no cover - safeguarded by ensure_supported
        raise ValueError(f"Unsupported document format: {extension}")

    return [PageContent(index=idx, text=text.strip()) for idx, text in enumerate(pages)]


def ensure_supported(extension: str) -> None:
    if extension.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported document format: {extension}")


def _convert_markdown(payload: bytes) -> list[str]:
    text = payload.decode("utf-8")
    return [text]


def _convert_html(payload: bytes) -> list[str]:
    soup = BeautifulSoup(payload, "html.parser")
    text = soup.get_text(separator="\n")
    return [text]


def _convert_docx(payload: bytes) -> list[str]:
    document = Document(io.BytesIO(payload))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return ["\n".join(paragraphs)]


def _convert_pptx(payload: bytes) -> list[str]:
    presentation = Presentation(io.BytesIO(payload))
    slides: list[str] = []
    for slide in presentation.slides:
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = shape.text.strip()
                if text:
                    texts.append(text)
        slides.append("\n".join(texts))
    return slides or [""]


def _convert_xlsx(payload: bytes) -> list[str]:
    workbook = load_workbook(io.BytesIO(payload), read_only=True)
    sheets = []
    for sheet in workbook:
        rows = []
        for row in sheet.iter_rows(values_only=True):
            cleaned = [str(cell) for cell in row if cell not in (None, "")]
            if cleaned:
                rows.append(", ".join(cleaned))
        if rows:
            sheets.append(f"{sheet.title}\n" + "\n".join(rows))
    return sheets or [""]


def _convert_pdf(payload: bytes) -> list[str]:
    buffer = io.BytesIO(payload)
    try:
        reader = PdfReader(buffer)
        texts = [page.extract_text() or "" for page in reader.pages]
        if any(texts):
            return texts
    except Exception:  # Fallback for invalid PDFs
        pass

    # Graceful fallback: treat payload as UTF-8 text with form-feed page breaks.
    decoded = payload.decode("utf-8", errors="ignore")
    pages = decoded.split("\f")
    return [page for page in pages if page.strip()] or [decoded]
