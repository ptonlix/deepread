"""
DeepSeek-OCR vLLM wrapper using official vLLM DeepSeek-OCR support.

Since vLLM 0.11.0+, DeepSeek-OCR is officially supported with built-in model
integration. This wrapper provides a clean interface for OCR inference with
retry logic and confidence tracking.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor  # type: ignore[import-not-found]
    from PIL import Image

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Type stub for when vLLM is not available
    NGramPerReqLogitsProcessor = None

DEFAULT_PROMPT = "Extract readable text from the supplied document page."
DEFAULT_RETRY_SUFFIX = (
    "Focus on noisy or low-contrast regions and provide your best effort transcription."
)


class InferenceEngine(Protocol):
    """Protocol capturing the pieces of vLLM we care about."""

    def __call__(
        self, *, prompt: str, image_bytes: bytes, max_tokens: int
    ) -> tuple[str, float]: ...


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


@dataclass
class VLLMLocalEngine:
    """Local vLLM inference engine for DeepSeek-OCR using official vLLM support.

    This engine uses vLLM's built-in DeepSeek-OCR integration available since
    vLLM 0.11.0+. No custom model registration is needed.
    """

    model_name: str = "deepseek-ai/DeepSeek-OCR"
    temperature: float = 0.0
    max_tokens: int = 8192
    ngram_size: int = 30
    window_size: int = 90
    whitelist_token_ids: tuple[int, ...] = (128821, 128822)  # <td>, </td>
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096
    enforce_eager: bool = False
    trust_remote_code: bool = True
    enable_prefix_caching: bool = False
    mm_processor_cache_gb: int = 0

    def __post_init__(self) -> None:
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM dependencies not available. Install with: "
                "uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly"
            )

        if NGramPerReqLogitsProcessor is None:
            raise ImportError(
                "NGramPerReqLogitsProcessor not found. "
                "Make sure you're using vLLM 0.11.0+ with DeepSeek-OCR support."
            )

        self._llm: LLM | None = None
        self._logger = logging.getLogger(__name__)

    def _ensure_initialized(self) -> None:
        """Lazy initialization of vLLM engine."""
        if self._llm is None:
            self._logger.info(f"Initializing vLLM engine with model: {self.model_name}")

            if not VLLM_AVAILABLE:
                raise ImportError("vLLM not available")

            self._llm = LLM(
                model=self.model_name,
                enable_prefix_caching=self.enable_prefix_caching,
                mm_processor_cache_gb=self.mm_processor_cache_gb,
                logits_processors=[NGramPerReqLogitsProcessor],
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                enforce_eager=self.enforce_eager,
                trust_remote_code=self.trust_remote_code,
            )

    def __call__(
        self, *, prompt: str, image_bytes: bytes, max_tokens: int
    ) -> tuple[str, float]:
        """Execute OCR inference using local vLLM engine."""
        self._ensure_initialized()

        try:
            # Convert bytes to PIL Image
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM not available")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Prepare prompt with image token
            ocr_prompt = f"<image>\n{prompt}"

            # Prepare input in the correct format for vLLM
            model_input = [
                {
                    "prompt": ocr_prompt,
                    "multi_modal_data": {"image": image},
                }
            ]

            # Create sampling params with ngram processor arguments
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=max_tokens,
                extra_args=dict(
                    ngram_size=self.ngram_size,
                    window_size=self.window_size,
                    whitelist_token_ids=set(self.whitelist_token_ids),
                ),
                skip_special_tokens=False,
            )

            # Generate output
            if self._llm is None:
                raise RuntimeError("LLM not initialized")
            outputs = self._llm.generate(model_input, sampling_params)  # type: ignore[arg-type]

            if not outputs or not outputs[0].outputs:
                return "", 0.0

            result_text = outputs[0].outputs[0].text.strip()

            # Calculate confidence based on text quality heuristics
            confidence = self._calculate_confidence(result_text)

            return result_text, confidence

        except Exception as e:
            self._logger.error(f"vLLM inference failed: {e}", exc_info=True)
            return "", 0.0

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on text quality heuristics."""
        if not text:
            return 0.0

        # Basic heuristics for confidence calculation
        score = 0.5  # Base score

        # Length bonus (longer text usually indicates successful OCR)
        if len(text) > 50:
            score += 0.2
        elif len(text) > 20:
            score += 0.1

        # Character variety bonus
        if re.search(r"[a-zA-Z]", text) and re.search(r"[0-9]", text):
            score += 0.1

        # Penalty for too many special characters (might indicate OCR errors)
        special_char_ratio = len(re.findall(r"[^\w\s]", text)) / len(text)
        if special_char_ratio > 0.3:
            score -= 0.2

        # Penalty for repeated characters (OCR artifacts)
        if re.search(r"(.)\1{4,}", text):
            score -= 0.1

        return max(0.0, min(1.0, score))


@dataclass
class VLLMRemoteEngine:
    """Remote vLLM inference engine for DeepSeek-OCR."""

    base_url: str
    timeout: float = 30.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def __call__(
        self, *, prompt: str, image_bytes: bytes, max_tokens: int
    ) -> tuple[str, float]:
        """Execute OCR inference using remote vLLM service."""
        # This is a placeholder for remote inference
        # In practice, you would implement HTTP/gRPC calls to a remote vLLM service
        self._logger.warning("Remote vLLM inference not yet implemented")
        return "", 0.0


def create_vllm_engine(
    mode: str = "local", model_name: str | None = None, **kwargs: Any
) -> InferenceEngine:
    """Factory function to create vLLM inference engines.

    Args:
        mode: Engine mode ("local" or "remote")
        model_name: Model name/path (defaults to "deepseek-ai/DeepSeek-OCR" for local mode)
        **kwargs: Additional configuration parameters passed to engine

    Returns:
        InferenceEngine instance
    """
    if mode == "local":
        # Extract model_name from kwargs if provided there, otherwise use default
        if model_name is None:
            model_name_str = kwargs.pop("model_name", "deepseek-ai/DeepSeek-OCR")
            model_name = (
                model_name_str
                if isinstance(model_name_str, str)
                else "deepseek-ai/DeepSeek-OCR"
            )
        return VLLMLocalEngine(model_name=model_name, **kwargs)
    elif mode == "remote":
        if "base_url" not in kwargs:
            raise ValueError("base_url required for remote mode")
        return VLLMRemoteEngine(**kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def run_ocr(
    *, image_bytes: bytes, engine_factory: Callable[[], InferenceEngine] | None = None
) -> OcrOutput:
    """
    Convenience wrapper around :class:`DeepSeekOcr`.

    If no engine_factory is provided, attempts to create a default vLLM local engine.
    """
    if engine_factory is None:
        # Try to create default vLLM engine
        try:
            engine = create_vllm_engine(mode="local")
        except Exception as e:
            raise NotImplementedError(
                f"Failed to create default vLLM engine: {e}. "
                "Please provide an engine_factory or ensure vLLM is properly installed."
            )
    else:
        engine = engine_factory()

    return DeepSeekOcr(engine=engine).run(image_bytes=image_bytes)
