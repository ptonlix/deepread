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
from typing import Any, Callable, Protocol, cast

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor  # type: ignore[import-not-found]
    from PIL import Image

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Type stub for when vLLM is not available
    NGramPerReqLogitsProcessor = None
    LLM = None  # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]
    Image = None  # type: ignore[assignment]

DEFAULT_PROMPT = "Extract readable text from the supplied document page."
DEFAULT_RETRY_SUFFIX = (
    "Focus on noisy or low-contrast regions and provide your best effort transcription."
)


class InferenceEngine(Protocol):
    """Protocol capturing the pieces of vLLM we care about."""

    def __call__(
        self, *, prompt: str, image_bytes: bytes | list[bytes], max_tokens: int
    ) -> tuple[str, float] | list[tuple[str, float]]:
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
            result = self.engine(
                prompt=prompt,
                image_bytes=image_bytes,
                max_tokens=self.max_tokens,
            )
            # For single image input, engine should return tuple[str, float]
            if isinstance(result, list):
                # Unexpected: got list for single input, use first result
                if result:
                    result_text, result_confidence = result[0]
                else:
                    result_text, result_confidence = "", 0.0
            else:
                result_text, result_confidence = result

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

    def run_batch(self, *, image_bytes_list: list[bytes]) -> list[OcrOutput]:
        """Perform OCR on multiple images in batch for better efficiency.

        Args:
            image_bytes_list: List of image bytes to process

        Returns:
            List of OcrOutput results corresponding to each input image
        """
        if not image_bytes_list:
            return []

        prompt = self.base_prompt
        results = self.engine(
            prompt=prompt,
            image_bytes=image_bytes_list,
            max_tokens=self.max_tokens,
        )

        # Type check: should be list when batch input is provided
        if not isinstance(results, list):
            # Fallback if engine returns single result unexpectedly
            single_result = results if isinstance(results, tuple) else ("", 0.0)
            return [
                OcrOutput(
                    text=single_result[0], confidence=single_result[1], warnings=[]
                )
            ]

        # Process batch results
        outputs = []
        for result_tuple in results:
            if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                outputs.append(OcrOutput(text="", confidence=0.0, warnings=[]))
                continue

            result_text, result_confidence = result_tuple
            warnings: list[str] = []

            # For batch processing, we skip per-item retries to maintain efficiency
            # If confidence is low, we still return the result but with a warning
            if result_confidence < self.confidence_threshold:
                warnings.append(
                    f"OCR confidence below threshold (score={result_confidence:.2f}). "
                    "Consider using single-image processing with retries for better results."
                )

            outputs.append(
                OcrOutput(
                    text=result_text,
                    confidence=result_confidence,
                    warnings=warnings,
                )
            )

        return outputs


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

        self._llm: Any = None
        self._logger = logging.getLogger(__name__)

    def _ensure_initialized(self) -> None:
        """Lazy initialization of vLLM engine."""
        if self._llm is None:
            self._logger.info(f"Initializing vLLM engine with model: {self.model_name}")

            if not VLLM_AVAILABLE:
                raise ImportError("vLLM not available")

            # At runtime, VLLM_AVAILABLE check ensures LLM and NGramPerReqLogitsProcessor are available
            assert (
                LLM is not None
            ), "LLM should be available when VLLM_AVAILABLE is True"
            assert (
                NGramPerReqLogitsProcessor is not None
            ), "NGramPerReqLogitsProcessor should be available when VLLM_AVAILABLE is True"

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
        self, *, prompt: str, image_bytes: bytes | list[bytes], max_tokens: int
    ) -> tuple[str, float] | list[tuple[str, float]]:
        """Execute OCR inference using local vLLM engine.

        Args:
            prompt: OCR prompt text
            image_bytes: Single image bytes or list of image bytes for batch processing
            max_tokens: Maximum tokens for generation

        Returns:
            For single image: tuple of (text, confidence)
            For batch images: list of tuples (text, confidence)
        """
        self._ensure_initialized()

        # Handle batch processing
        is_batch = isinstance(image_bytes, list)
        if is_batch:
            # Type narrowing: when is_batch is True, image_bytes is list[bytes]
            image_list = cast(list[bytes], image_bytes)
        else:
            # Type narrowing: when is_batch is False, image_bytes is bytes
            single_bytes = cast(bytes, image_bytes)
            image_list = [single_bytes]

        try:
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM not available")

            # Convert all image bytes to PIL Images
            # At runtime, VLLM_AVAILABLE check ensures Image is available
            assert (
                Image is not None
            ), "Image should be available when VLLM_AVAILABLE is True"
            images = [
                Image.open(io.BytesIO(img_bytes)).convert("RGB")
                for img_bytes in image_list
            ]

            # Prepare prompt with image token
            ocr_prompt = f"<image>\n{prompt}"

            # Prepare batched input in the correct format for vLLM
            # vLLM's generate accepts this multi-modal format for DeepSeek-OCR
            model_input: list[dict[str, Any]] = [
                {
                    "prompt": ocr_prompt,
                    "multi_modal_data": {"image": image},
                }
                for image in images
            ]

            # Create sampling params with ngram processor arguments
            # At runtime, VLLM_AVAILABLE check ensures SamplingParams is available
            assert (
                SamplingParams is not None
            ), "SamplingParams should be available when VLLM_AVAILABLE is True"
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
            # vLLM's generate method accepts multi-modal input format for DeepSeek-OCR
            # The type checker may not recognize this format, but it's correct at runtime
            # Cast to Any to work around vLLM's type definitions for multi-modal inputs
            outputs = self._llm.generate(cast(Any, model_input), sampling_params)

            if not outputs:
                return [] if is_batch else ("", 0.0)

            # Process all outputs
            results = []
            for output in outputs:
                if not output.outputs:
                    results.append(("", 0.0))
                    continue

                result_text = output.outputs[0].text.strip()
                confidence = self._calculate_confidence(result_text)
                results.append((result_text, confidence))

            # Return single result for backward compatibility or list for batch
            return results if is_batch else results[0]

        except Exception as e:
            self._logger.error(f"vLLM inference failed: {e}", exc_info=True)
            return [] if is_batch else ("", 0.0)

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
        self, *, prompt: str, image_bytes: bytes | list[bytes], max_tokens: int
    ) -> tuple[str, float] | list[tuple[str, float]]:
        """Execute OCR inference using remote vLLM service."""
        # This is a placeholder for remote inference
        # In practice, you would implement HTTP/gRPC calls to a remote vLLM service
        is_batch = isinstance(image_bytes, list)
        self._logger.warning("Remote vLLM inference not yet implemented")
        if is_batch:
            return [("", 0.0) for _ in image_bytes]
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
