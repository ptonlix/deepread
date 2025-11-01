"""Tests for configuration management."""

from __future__ import annotations

import os
from unittest.mock import patch

from deepread.config import DeepReadConfig, OCRConfig, get_config, set_config


def test_ocr_config_defaults() -> None:
    """Test OCR configuration default values."""
    config = OCRConfig()

    assert config.mode == "fallback"
    assert config.model_path == "deepseek-ai/DeepSeek-OCR"
    assert config.tensor_parallel_size == 1
    assert config.gpu_memory_utilization == 0.8
    assert config.max_model_len == 4096
    assert config.api_url == "http://localhost:8000/v1"
    assert config.api_key is None
    assert config.timeout == 30
    assert config.max_tokens == 2048
    assert config.temperature == 0.0
    assert config.confidence_threshold == 0.7
    assert config.max_retries == 3


def test_ocr_config_from_env() -> None:
    """Test OCR configuration from environment variables."""
    env_vars = {
        "DEEPREAD_OCR_MODE": "vllm_local",
        "DEEPREAD_OCR_MODEL_PATH": "custom/model",
        "DEEPREAD_OCR_TENSOR_PARALLEL_SIZE": "2",
        "DEEPREAD_OCR_GPU_MEMORY_UTILIZATION": "0.9",
        "DEEPREAD_OCR_MAX_MODEL_LEN": "8192",
        "DEEPREAD_OCR_API_URL": "http://custom:8000/v1",
        "DEEPREAD_OCR_API_KEY": "secret-key",
        "DEEPREAD_OCR_TIMEOUT": "60",
        "DEEPREAD_OCR_MAX_TOKENS": "4096",
        "DEEPREAD_OCR_TEMPERATURE": "0.1",
        "DEEPREAD_OCR_CONFIDENCE_THRESHOLD": "0.8",
        "DEEPREAD_OCR_MAX_RETRIES": "5",
    }

    with patch.dict(os.environ, env_vars):
        config = OCRConfig.from_env()

    assert config.mode == "vllm_local"
    assert config.model_path == "custom/model"
    assert config.tensor_parallel_size == 2
    assert config.gpu_memory_utilization == 0.9
    assert config.max_model_len == 8192
    assert config.api_url == "http://custom:8000/v1"
    assert config.api_key == "secret-key"
    assert config.timeout == 60
    assert config.max_tokens == 4096
    assert config.temperature == 0.1
    assert config.confidence_threshold == 0.8
    assert config.max_retries == 5


def test_ocr_config_to_vllm_local() -> None:
    """Test OCR config conversion to vLLM local configuration."""
    config = OCRConfig(
        mode="vllm_local",
        model_path="test/model",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        max_tokens=4096,
        temperature=0.1,
    )

    vllm_config = config.to_vllm_config()

    expected = {
        "model_name": "test/model",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "max_tokens": 4096,
        "temperature": 0.1,
        "ngram_size": 30,
        "window_size": 90,
        "whitelist_token_ids": (128821, 128822),
    }
    assert vllm_config == expected


def test_ocr_config_to_vllm_remote() -> None:
    """Test OCR config conversion to vLLM remote configuration."""
    config = OCRConfig(
        mode="vllm_remote",
        api_url="http://remote:8000/v1",
        api_key="test-key",
        timeout=60,
        max_tokens=4096,
        temperature=0.1,
    )

    vllm_config = config.to_vllm_config()

    expected = {
        "api_url": "http://remote:8000/v1",
        "api_key": "test-key",
        "timeout": 60,
        "max_tokens": 4096,
        "temperature": 0.1,
    }
    assert vllm_config == expected


def test_ocr_config_to_vllm_fallback() -> None:
    """Test OCR config conversion for fallback mode."""
    config = OCRConfig(mode="fallback")

    vllm_config = config.to_vllm_config()

    assert vllm_config == {}


def test_deepread_config_defaults() -> None:
    """Test DeepRead configuration default values."""
    config = DeepReadConfig()

    assert config.store_root == ".deepread-store"
    assert isinstance(config.ocr, OCRConfig)


def test_deepread_config_from_env() -> None:
    """Test DeepRead configuration from environment variables."""
    env_vars = {
        "DEEPREAD_STORE": "/custom/store",
        "DEEPREAD_OCR_MODE": "vllm_local",
    }

    with patch.dict(os.environ, env_vars):
        config = DeepReadConfig.from_env()

    assert config.store_root == "/custom/store"
    assert config.ocr.mode == "vllm_local"


def test_global_config_singleton() -> None:
    """Test global configuration singleton behavior."""
    # Reset global config
    set_config(None)  # type: ignore

    # First call should create config from environment
    with patch.dict(os.environ, {"DEEPREAD_OCR_MODE": "vllm_local"}):
        config1 = get_config()

    # Second call should return same instance
    config2 = get_config()

    assert config1 is config2
    assert config1.ocr.mode == "vllm_local"


def test_set_global_config() -> None:
    """Test setting global configuration."""
    custom_config = DeepReadConfig(
        store_root="/custom",
        ocr=OCRConfig(mode="vllm_remote"),
    )

    set_config(custom_config)
    retrieved_config = get_config()

    assert retrieved_config is custom_config
    assert retrieved_config.store_root == "/custom"
    assert retrieved_config.ocr.mode == "vllm_remote"
