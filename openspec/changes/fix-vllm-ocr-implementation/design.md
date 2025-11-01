## Context

当前 `VLLMLocalEngine` 的实现尝试使用 vLLM 进行 OCR 推理，但实现方式与 DeepSeek-OCR 官方仓库的标准用法不匹配。官方仓库提供了完整的 vLLM 集成示例，包括：

- 自定义模型类注册 (`DeepseekOCRForCausalLM`)
- 专用的图像处理器 (`DeepseekOCRProcessor`)
- 重复文本抑制的 logits processor (`NoRepeatNGramLogitsProcessor`)
- 特定的 vLLM 配置参数

## Goals

1. 修复 `VLLMLocalEngine` 实现，使其能够正确运行 DeepSeek-OCR 模型
2. 遵循官方实现模式，确保与 DeepSeek-OCR 模型兼容
3. 保持现有 API 接口不变，仅修复内部实现
4. 添加必要的配置选项以支持 OCR 特定参数

## Non-Goals

- 不改变 `DeepSeekOcr` 和 `OcrOutput` 的公共接口
- 不重构整体 OCR 架构（仅修复实现）
- 不实现远程 OCR 服务的完整功能（保留占位符即可）

## Decisions

### Decision 1: 模型类注册方式

**选择**: 在 `VLLMLocalEngine` 初始化时注册 `DeepseekOCRForCausalLM` 到 vLLM ModelRegistry

**理由**:

- 官方示例在脚本级别注册，但我们需要在引擎初始化时注册以确保正确性
- 可以多次注册而不影响功能（vLLM Registry 支持覆盖）

**替代方案**:

- 在模块级别注册 - 可能导致导入时依赖问题
- 延迟注册 - 增加复杂度

### Decision 2: 图像处理器集成

**选择**: 将 `DeepseekOCRProcessor` 的核心功能集成到项目中，或创建包装类

**理由**:

- 官方处理器包含图像裁剪、tokenization 等关键逻辑
- 直接复制代码可以避免外部依赖，但需要处理许可证
- 创建包装类可以保持代码清洁，但需要维护兼容性

**替代方案**:

- 作为外部依赖引入 - 需要确保可用性和版本兼容
- 实现简化版本 - 可能丢失重要功能

### Decision 3: Logits Processor

**选择**: 使用官方 `NoRepeatNGramLogitsProcessor` 实现

**理由**:

- 官方实现已经验证有效
- 支持 ngram 抑制和 token whitelist，适合 OCR 场景

**替代方案**:

- 使用 vLLM 内置的 logits processor - 可能功能不匹配

### Decision 4: vLLM 初始化参数

**选择**: 添加完整的官方推荐配置，包括：

- `hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]}`
- `block_size=256`
- `max_num_seqs` (可配置)
- `disable_mm_preprocessor_cache=True` (可选)

**理由**:

- 官方示例证明这些参数对模型运行至关重要
- 通过配置类暴露可调参数，保持灵活性

### Decision 5: 图像输入格式

**选择**: 使用官方格式：`{"prompt": str, "multi_modal_data": {"image": processed_image_data}}`

**理由**:

- 与 DeepSeek-OCR 模型期望的输入格式匹配
- 官方示例已经验证此格式有效

## Risks / Trade-offs

### Risk 1: 代码重复

**风险**: 复制官方代码可能导致维护负担

**缓解**:

- 优先考虑将关键组件作为依赖引入
- 如果必须复制，保留原始注释和来源引用
- 考虑创建适配层，未来易于替换

### Risk 2: 配置复杂度

**风险**: 新增配置参数可能增加用户配置难度

**缓解**:

- 提供合理的默认值（基于官方示例）
- 仅在必要时暴露高级参数
- 完善文档说明

### Risk 3: 测试覆盖

**风险**: 新的实现可能需要 GPU 环境才能完整测试

**缓解**:

- 保留现有的 mock 测试覆盖核心逻辑
- 添加集成测试标记，标记为需要 GPU
- 在 CI 中可选运行 GPU 测试

## Migration Plan

### Phase 1: 准备

1. 从官方仓库提取必要的处理器和工具类（MIT 许可证已确认，可直接复制）
2. 保留 MIT 许可证声明和版权信息
3. 更新依赖（如需要）

### Phase 2: 实现

1. 修复 `VLLMLocalEngine.__init__` 和 `__call__` 方法
2. 添加模型注册逻辑
3. 更新配置类
4. 修复测试

### Phase 3: 验证

1. 运行单元测试
2. 在 GPU 环境进行集成测试
3. 验证与现有 pipeline 的兼容性

### Rollback

如果新实现有问题，可以：

- 回退到之前的 commit
- 或临时禁用 vLLM 模式，使用 fallback 引擎

## Open Questions

1. **依赖管理**: 是否应该将 `DeepseekOCRProcessor` 等组件打包为单独的包，还是直接集成？（优先直接集成，MIT 许可证允许）
2. **默认配置**: 哪些配置参数应该暴露给用户，哪些应该硬编码为推荐值？
3. **错误处理**: 当模型注册失败或处理器初始化失败时，如何优雅降级？
