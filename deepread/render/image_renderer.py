"""
Text-to-image rendering utilities.

Renders page text to a high-DPI image so downstream OCR can consume consistent
inputs regardless of the source document format.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

PAGE_SIZE = (2480, 3508)  # A4 @ 300 DPI
DPI = (300, 300)
MARGIN = 120
LINE_HEIGHT = 40


def render_text_page(text: str) -> Image.Image:
    """
    Render text to an RGB canvas at 300 DPI.

    Args:
        text: Plain text content of the page.

    Returns:
        PIL Image with text drawn and DPI metadata applied.
    """

    image = Image.new("RGB", PAGE_SIZE, color="white")
    image.info["dpi"] = DPI

    draw = ImageDraw.Draw(image)
    # Try to load a better font, fallback to default
    font: ImageFont.ImageFont | FreeTypeFont
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()

    x, y = MARGIN, MARGIN
    max_width = PAGE_SIZE[0] - (2 * MARGIN)

    # Handle empty text case
    if not text.strip():
        text = " "

    for line in text.splitlines() or [""]:
        wrapped_lines = _wrap_line(draw, line, max_width, font)
        for segment in wrapped_lines:
            if segment.strip():  # Only draw non-empty segments
                draw.text((x, y), segment, fill="black", font=font)
            y += LINE_HEIGHT

    return image


def _wrap_line(draw: ImageDraw.ImageDraw, line: str, max_width: int, font: ImageFont.ImageFont | FreeTypeFont) -> list[str]:
    """Simple word-wrapping utility using the rendering context."""
    words = line.split()
    if not words:
        return [""]

    wrapped: list[str] = []
    current_line: list[str] = []

    for word in words:
        candidate = " ".join(current_line + [word])
        # Use textbbox instead of deprecated textsize
        bbox = draw.textbbox((0, 0), candidate, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                wrapped.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        wrapped.append(" ".join(current_line))

    return wrapped
