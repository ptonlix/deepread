"""DeepSeek-OCR processing utilities.

This module contains utilities adapted from DeepSeek-OCR-vllm repository
(https://github.com/deepseek-ai/DeepSeek-OCR) under MIT License.

Copyright (c) 2025 DeepSeek
"""

from deepread.ocr.process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from deepread.ocr.process.image_process import DeepseekOCRProcessor

__all__ = ["NoRepeatNGramLogitsProcessor", "DeepseekOCRProcessor"]

