## Why

当前 OCR 实现 (`deepread/ocr/deepseek.py`) 中的 `VLLMLocalEngine` 未按照 DeepSeek-OCR 官方仓库的正确方式使用 vLLM，导致：

- 使用了不存在的 `NGramPerReqLogitsProcessor`，导致运行时错误
- 未注册自定义模型类 `DeepseekOCRForCausalLM` 到 vLLM ModelRegistry
- 未使用正确的图像处理器 `DeepseekOCRProcessor` 处理输入图像
- 缺少 vLLM 初始化所需的 `hf_overrides` 参数
- 图像输入格式不正确，无法与 DeepSeek-OCR 模型兼容

这些问题导致 OCR 功能无法正常工作，需要参考官方实现 (`tmp/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/`) 进行修复。

## What Changes

- 修复 `VLLMLocalEngine` 实现，按照官方方式正确初始化和使用 vLLM
- 添加模型注册逻辑，将 `DeepseekOCRForCausalLM` 注册到 vLLM ModelRegistry
- 集成 `DeepseekOCRProcessor` 用于正确的图像预处理和 tokenization
- 使用 `NoRepeatNGramLogitsProcessor` 替代错误的 logits processor
- 更新 vLLM 初始化参数，添加 `hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]}`
- 修正图像输入格式，使用 `multi_modal_data` 格式而非简单的 PIL Image
- 更新配置类以支持新增的 OCR 参数（如 `ngram_size`, `window_size`, `crop_mode` 等）
- 更新测试以反映新的实现方式

## Impact

- **受影响的规范**: `specs/ocr/spec.md` - 需要更新实现细节
- **受影响的代码**:
  - `deepread/ocr/deepseek.py` - 主要修复文件
  - `deepread/config.py` - 配置参数扩展
  - `tests/unit/ocr/test_deepseek.py` - 测试更新
- **依赖变化**: 需要将 DeepSeek-OCR-vllm 中的关键组件（如 `DeepseekOCRProcessor`, `NoRepeatNGramLogitsProcessor`）集成到项目中，或作为依赖引用
- **BREAKING**: 无，这是内部实现修复，不改变公共 API
