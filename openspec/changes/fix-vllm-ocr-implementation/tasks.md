## 1. 代码分析和准备

- [ ] 1.1 深入分析 DeepSeek-OCR-vllm 官方实现，理解关键组件的作用

  - [ ] 阅读 `deepseek_ocr.py` 了解模型结构
  - [ ] 阅读 `run_dpsk_ocr_pdf.py` 和 `run_dpsk_ocr_image.py` 了解使用模式
  - [ ] 分析 `DeepseekOCRProcessor` 的图像处理流程
  - [ ] 理解 `NoRepeatNGramLogitsProcessor` 的工作原理

- [x] 1.2 检查许可证兼容性

  - [x] 确认 DeepSeek-OCR 仓库的许可证（MIT，已确认兼容）
  - [x] 确认我们是否可以复制/修改代码（可以，MIT 许可证允许）

- [x] 1.3 确定代码集成策略
  - [x] 决定是复制代码还是创建依赖（复制代码，MIT 许可证允许）
  - [ ] 确定需要的最小代码集
  - [ ] 标记需要保留的注释和来源引用（保留 MIT 许可证声明）

## 2. 实现修复

- [ ] 2.1 添加必要的依赖组件（如果需要）

  - [ ] 集成或复制 `DeepseekOCRProcessor` 相关代码
  - [ ] 集成或复制 `NoRepeatNGramLogitsProcessor`
  - [ ] 确保所有导入路径正确

- [ ] 2.2 修复 `VLLMLocalEngine` 实现

  - [ ] 在 `__init__` 中添加模型注册：`ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)`
  - [ ] 更新 `LLM` 初始化，添加 `hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]}`
  - [ ] 添加其他必要的 vLLM 参数（`block_size`, `max_num_seqs`, 等）
  - [ ] 在 `__call__` 中使用 `DeepseekOCRProcessor().tokenize_with_images()` 处理图像
  - [ ] 使用 `NoRepeatNGramLogitsProcessor` 替代错误的 processor
  - [ ] 修正图像输入格式为 `{"prompt": str, "multi_modal_data": {"image": ...}}`
  - [ ] 更新 `SamplingParams` 配置

- [ ] 2.3 更新配置类

  - [ ] 在 `OCRConfig` 中添加 OCR 特定参数：
    - [ ] `ngram_size: int = 30`
    - [ ] `window_size: int = 90`
    - [ ] `crop_mode: bool = True`
    - [ ] `base_size: int = 1024`
    - [ ] `image_size: int = 640`
    - [ ] `min_crops: int = 2`
    - [ ] `max_crops: int = 6`
    - [ ] `block_size: int = 256`
    - [ ] `max_num_seqs: int = 100`
  - [ ] 更新 `from_env` 方法以读取新参数
  - [ ] 更新 `to_vllm_config` 方法以传递新参数

- [ ] 2.4 修复置信度计算
  - [ ] 审查当前的启发式置信度计算方法
  - [ ] 考虑是否需要改进或使用模型输出的置信度（如果可用）
  - [ ] 确保置信度值在合理范围内

## 3. 测试和验证

- [ ] 3.1 更新单元测试

  - [ ] 修复 `test_vllm_local_engine_inference` 测试，使其匹配新的实现
  - [ ] 添加测试覆盖模型注册逻辑
  - [ ] 添加测试覆盖图像处理器调用
  - [ ] 确保所有 mock 正确设置

- [ ] 3.2 运行现有测试

  - [ ] 运行 `make test` 确保没有回归
  - [ ] 修复任何失败的测试
  - [ ] 确保类型检查通过 (`make typecheck`)

- [ ] 3.3 GPU 环境验证（暂缓，待后续手工测试）
  - [ ] ~~在实际 GPU 环境测试 OCR 功能~~（后续手工测试）
  - [ ] ~~验证图像处理正确性~~（后续手工测试）
  - [ ] ~~验证输出质量和置信度~~（后续手工测试）
  - [ ] ~~记录性能指标（可选）~~（后续手工测试）

## 4. 文档和清理

- [ ] 4.1 代码文档

  - [ ] 更新 `VLLMLocalEngine` 的文档字符串
  - [ ] 添加配置参数的文档说明
  - [ ] 添加来源引用（如果复制了代码）

- [ ] 4.2 更新项目文档

  - [ ] 更新 README 或相关文档，说明 OCR 配置选项
  - [ ] 如果添加了新依赖，更新安装说明

- [ ] 4.3 代码清理
  - [ ] 运行 `make check` 确保代码风格正确
  - [ ] 移除未使用的导入
  - [ ] 确保所有 TODO 和 FIXME 已处理或记录

## 5. 提案验证

- [ ] 5.1 验证 OpenSpec 提案
  - [ ] 运行 `openspec validate fix-vllm-ocr-implementation --strict`
  - [ ] 修复所有验证错误
  - [ ] 确保 spec delta 正确描述了更改
