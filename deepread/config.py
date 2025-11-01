"""
Configuration management for deepread.

Provides centralized configuration for OCR engines, inference modes,
and deployment parameters through environment variables and defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class OCRConfig:
    """Configuration for OCR inference engines."""
    
    # OCR mode: "fallback", "vllm_local", "vllm_remote"
    mode: str = "fallback"
    
    # vLLM local configuration
    model_path: str = "deepseek-ai/DeepSeek-OCR"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096
    
    # vLLM remote configuration
    api_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    timeout: int = 30
    
    # OCR processing parameters
    max_tokens: int = 2048
    temperature: float = 0.0
    confidence_threshold: float = 0.7
    max_retries: int = 3
    
    # DeepSeek-OCR specific parameters (using vLLM built-in support)
    ngram_size: int = 30
    window_size: int = 90
    whitelist_token_ids: tuple[int, ...] = (128821, 128822)  # <td>, </td>
    
    @classmethod
    def from_env(cls) -> OCRConfig:
        """Create OCR configuration from environment variables."""
        return cls(
            mode=os.environ.get("DEEPREAD_OCR_MODE", "fallback"),
            model_path=os.environ.get("DEEPREAD_OCR_MODEL_PATH", "deepseek-ai/DeepSeek-OCR"),
            tensor_parallel_size=int(os.environ.get("DEEPREAD_OCR_TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.environ.get("DEEPREAD_OCR_GPU_MEMORY_UTILIZATION", "0.8")),
            max_model_len=int(os.environ.get("DEEPREAD_OCR_MAX_MODEL_LEN", "4096")),
            api_url=os.environ.get("DEEPREAD_OCR_API_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("DEEPREAD_OCR_API_KEY"),
            timeout=int(os.environ.get("DEEPREAD_OCR_TIMEOUT", "30")),
            max_tokens=int(os.environ.get("DEEPREAD_OCR_MAX_TOKENS", "2048")),
            temperature=float(os.environ.get("DEEPREAD_OCR_TEMPERATURE", "0.0")),
            confidence_threshold=float(os.environ.get("DEEPREAD_OCR_CONFIDENCE_THRESHOLD", "0.7")),
            max_retries=int(os.environ.get("DEEPREAD_OCR_MAX_RETRIES", "3")),
            ngram_size=int(os.environ.get("DEEPREAD_OCR_NGRAM_SIZE", "30")),
            window_size=int(os.environ.get("DEEPREAD_OCR_WINDOW_SIZE", "90")),
            whitelist_token_ids=tuple(
                int(x) for x in os.environ.get("DEEPREAD_OCR_WHITELIST_TOKEN_IDS", "128821,128822").split(",")
            ),
        )
    
    def to_vllm_config(self) -> dict[str, Any]:
        """Convert to vLLM engine configuration."""
        if self.mode == "vllm_local":
            return {
                "model_name": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "ngram_size": self.ngram_size,
                "window_size": self.window_size,
                "whitelist_token_ids": self.whitelist_token_ids,
            }
        elif self.mode == "vllm_remote":
            return {
                "api_url": self.api_url,
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        else:
            return {}


@dataclass
class DeepReadConfig:
    """Main configuration for deepread application."""
    
    # Storage configuration
    store_root: str = ".deepread-store"
    
    # OCR configuration
    ocr: OCRConfig = None  # type: ignore
    
    def __post_init__(self) -> None:
        if self.ocr is None:
            self.ocr = OCRConfig()
    
    @classmethod
    def from_env(cls) -> DeepReadConfig:
        """Create configuration from environment variables."""
        return cls(
            store_root=os.environ.get("DEEPREAD_STORE", ".deepread-store"),
            ocr=OCRConfig.from_env(),
        )


# Global configuration instance
_config: DeepReadConfig | None = None


def get_config() -> DeepReadConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = DeepReadConfig.from_env()
    return _config


def set_config(config: DeepReadConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config