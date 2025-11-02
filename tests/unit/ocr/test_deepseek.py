from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from deepread.ocr.deepseek import (
    DeepSeekOcr,
    OcrOutput,
    VLLMLocalEngine,
    VLLMRemoteEngine,
    create_vllm_engine,
)


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


class TestVLLMEngines:
    """Test vLLM engine creation and inference."""

    @patch("deepread.ocr.deepseek.VLLM_AVAILABLE", True)
    @patch("deepread.ocr.deepseek.NGramPerReqLogitsProcessor", Mock())
    def test_create_vllm_engine_local(self):
        """Test creating local vLLM engine."""
        engine = create_vllm_engine(mode="local", model_name="test-model")
        assert isinstance(engine, VLLMLocalEngine)
        assert engine.model_name == "test-model"

    @patch("deepread.ocr.deepseek.VLLM_AVAILABLE", True)
    def test_create_vllm_engine_remote(self):
        """Test creating remote vLLM engine."""
        engine = create_vllm_engine(
            mode="remote", base_url="http://localhost:8000", timeout=60.0
        )
        assert isinstance(engine, VLLMRemoteEngine)
        assert engine.base_url == "http://localhost:8000"
        assert engine.timeout == 60.0

    def test_create_vllm_engine_invalid_mode(self):
        """Test error handling for invalid mode."""
        with pytest.raises(ValueError, match="Unsupported mode"):
            create_vllm_engine(mode="invalid")

    def test_create_vllm_engine_remote_missing_base_url(self):
        """Test error handling when base_url is missing for remote mode."""
        with pytest.raises(ValueError, match="base_url required"):
            create_vllm_engine(mode="remote")

    @patch("deepread.ocr.deepseek.VLLM_AVAILABLE", False)
    def test_vllm_local_engine_unavailable(self):
        """Test VLLMLocalEngine when vLLM is not available."""
        with pytest.raises(ImportError, match="vLLM dependencies not available"):
            VLLMLocalEngine()

    @patch("deepread.ocr.deepseek.VLLM_AVAILABLE", True)
    @patch("deepread.ocr.deepseek.NGramPerReqLogitsProcessor", Mock())
    def test_vllm_local_engine_inference(self):
        """Test VLLMLocalEngine inference call."""
        # Create mock Image module
        mock_image_module = Mock()
        mock_image = Mock()
        mock_image_module.open.return_value.convert.return_value = mock_image

        with (
            patch("deepread.ocr.deepseek.LLM") as mock_llm_class,
            patch("deepread.ocr.deepseek.Image", mock_image_module, create=True),
        ):
            # Setup mocks
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm

            mock_output = Mock()
            mock_output.outputs = [Mock()]
            mock_output.outputs[0].text = "extracted text"
            mock_llm.generate.return_value = [mock_output]

            # Test inference
            engine = VLLMLocalEngine(model_name="test-model")
            result_text, confidence = engine(
                prompt="Extract text", image_bytes=b"fake_image", max_tokens=100
            )

            assert result_text == "extracted text"
            assert 0.0 <= confidence <= 1.0
            mock_llm.generate.assert_called_once()

    @patch("deepread.ocr.deepseek.VLLM_AVAILABLE", True)
    @patch("deepread.ocr.deepseek.NGramPerReqLogitsProcessor", Mock())
    def test_vllm_local_engine_inference_error(self):
        """Test VLLMLocalEngine error handling during inference."""
        # Create mock Image module
        mock_image_module = Mock()

        with (
            patch("deepread.ocr.deepseek.LLM") as mock_llm_class,
            patch("deepread.ocr.deepseek.Image", mock_image_module, create=True),
        ):
            # Setup mocks
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm
            mock_llm.generate.side_effect = Exception("Inference failed")

            # Test error handling
            engine = VLLMLocalEngine()
            result_text, confidence = engine(
                prompt="Extract text", image_bytes=b"fake_image", max_tokens=100
            )

            assert result_text == ""
            assert confidence == 0.0

    def test_vllm_remote_engine_inference(self):
        """Test VLLMRemoteEngine inference call (placeholder implementation)."""
        engine = VLLMRemoteEngine(base_url="http://localhost:8000")
        result_text, confidence = engine(
            prompt="Extract text", image_bytes=b"fake_image", max_tokens=100
        )

        # Current implementation returns empty results
        assert result_text == ""
        assert confidence == 0.0
