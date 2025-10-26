from __future__ import annotations

from statistics import mean

from PIL import Image

from deepread.render.image_renderer import render_text_page


def test_render_text_page_sets_dpi_and_dimensions() -> None:
    image = render_text_page("Sample content for rendering.")

    assert image.size == (2480, 3508)
    assert image.info.get("dpi") == (300, 300)


def test_render_text_page_draws_non_empty_pixels() -> None:
    image = render_text_page("Text that should appear on image.")
    grayscale = image.convert("L")
    pixels = list(grayscale.getdata())

    # Check that text was actually drawn by looking for non-white pixels
    non_white_pixels = [p for p in pixels if p < 255]
    assert len(non_white_pixels) > 0, "Image appears blank; expected drawn text to create non-white pixels."
    
    # Also check that the average is reasonably affected by text
    # For a large image with small text, we expect slight darkening
    assert mean(pixels) < 255, "Expected some darkening from text rendering."
