"""
DeepSeek-OCR vLLM wrapper.

The real system will integrate with a vLLM inference engine. To keep early
phases testable we architect the wrapper around a pluggable callable that
accepts prompts and image bytes, returning extracted text and a confidence
score. Retries are triggered when the score falls below a configurable
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

DEFAULT_PROMPT = "Extract readable text from the supplied document page."
DEFAULT_RETRY_SUFFIX = (
    "Focus on noisy or low-contrast regions and provide your best effort transcription."
)


class InferenceEngine(Protocol):
    """Protocol capturing the pieces of vLLM we care about."""

    def __call__(
        self, *, prompt: str, image_bytes: bytes, max_tokens: int
    ) -> tuple[str, float]:
        ...


@dataclass(slots=True)
class OcrOutput:
    """Structured OCR response returned to downstream pipeline stages."""

    text: str
    confidence: float
    warnings: list[str]


@dataclass(slots=True)
class DeepSeekOcr:
    """
    Manage DeepSeek-OCR inference with retries and confidence tracking.

    Args:
        engine: Callable that encapsulates vLLM interaction.
        base_prompt: Prompt used on the first attempt.
        retry_prompt_suffix: Additional instruction appended after each retry.
        confidence_threshold: Minimum acceptable confidence score.
        max_retries: Number of additional attempts allowed when confidence is low.
        max_tokens: Generation cap supplied to the engine.
    """

    engine: InferenceEngine
    base_prompt: str = DEFAULT_PROMPT
    retry_prompt_suffix: str = DEFAULT_RETRY_SUFFIX
    confidence_threshold: float = 0.6
    max_retries: int = 1
    max_tokens: int = 2048

    def run(self, *, image_bytes: bytes) -> OcrOutput:
        """Perform OCR on the given image bytes."""
        prompt = self.base_prompt
        warnings: list[str] = []
        attempts = self.max_retries + 1
        result_text = ""
        result_confidence = 0.0

        for attempt in range(attempts):
            result_text, result_confidence = self.engine(
                prompt=prompt,
                image_bytes=image_bytes,
                max_tokens=self.max_tokens,
            )
            if result_confidence >= self.confidence_threshold:
                if attempt > 0:
                    warnings.append(f"OCR succeeded after {attempt} retry.")
                return OcrOutput(
                    text=result_text, confidence=result_confidence, warnings=warnings
                )

            if attempt < self.max_retries:
                prompt = f"{self.base_prompt}\n\n{self.retry_prompt_suffix}"
                continue

            warnings.append(
                f"OCR confidence below threshold after retries (score={result_confidence:.2f})."
            )
            break

        return OcrOutput(
            text=result_text, confidence=result_confidence, warnings=warnings
        )


def run_ocr(
    *, image_bytes: bytes, engine_factory: Callable[[], InferenceEngine] | None = None
) -> OcrOutput:
    """
    Convenience wrapper around :class:`DeepSeekOcr`.

    Consumers must currently supply an `engine_factory` that returns an
    `InferenceEngine` instance. A default implementation will be provided when
    the vLLM integration is completed in later phases.
    """

    if engine_factory is None:
        raise NotImplementedError(
            "engine_factory must be provided until vLLM wiring is implemented."
        )

    engine = engine_factory()
    return DeepSeekOcr(engine=engine).run(image_bytes=image_bytes)
