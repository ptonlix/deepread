from __future__ import annotations


from deepread.ocr.deepseek import DeepSeekOcr, OcrOutput


class FakeEngine:
    def __init__(self, responses: list[tuple[str, float]]) -> None:
        self._responses = responses
        self.invocations: list[dict[str, object]] = []

    def __call__(
        self, *, prompt: str, image_bytes: bytes, max_tokens: int
    ) -> tuple[str, float]:
        self.invocations.append(
            {"prompt": prompt, "image_bytes": image_bytes, "max_tokens": max_tokens}
        )
        if not self._responses:
            raise RuntimeError("No more responses configured")
        return self._responses.pop(0)


def make_ocr(responses: list[tuple[str, float]], **kwargs) -> DeepSeekOcr:
    engine = FakeEngine(responses=responses)
    ocr = DeepSeekOcr(engine=engine, **kwargs)
    return ocr


def test_ocr_returns_high_confidence_output() -> None:
    ocr = make_ocr([("Recognized text", 0.95)])

    output = ocr.run(image_bytes=b"image")

    assert output == OcrOutput(text="Recognized text", confidence=0.95, warnings=[])


def test_ocr_retries_until_confidence_threshold_met() -> None:
    ocr = make_ocr(
        [
            ("blurry text", 0.4),
            ("clean text", 0.88),
        ],
        max_retries=2,
        confidence_threshold=0.6,
        retry_prompt_suffix="Please clarify.",
    )

    output = ocr.run(image_bytes=b"img")

    assert output.text == "clean text"
    assert output.confidence == 0.88
    assert output.warnings == ["OCR succeeded after 1 retry."]


def test_ocr_records_warning_when_retries_exhausted() -> None:
    ocr = make_ocr(
        [
            ("blurry", 0.3),
            ("still blurry", 0.35),
        ],
        max_retries=1,
        confidence_threshold=0.6,
    )

    output = ocr.run(image_bytes=b"bytes")

    assert output.text == "still blurry"
    assert output.confidence == 0.35
    assert output.warnings == [
        "OCR confidence below threshold after retries (score=0.35)."
    ]
