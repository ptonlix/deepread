## Why
- The ingestion流水线目前仅使用占位OCR引擎，无法利用DeepSeek-OCR提供的高精度识别能力。
- 产品目标是提供真实可用的OCR汇总，必须集成官方DeepSeek-OCR模型与推理流程。

## What Changes
- 引入基于vLLM Inference的DeepSeek-OCR推理能力，可选择本地部署或远程服务，统一封装在`DeepSeekOcr`。
- 在pipeline中复用真实OCR输出来生成InsightReport，包括置信度与警告信息。
- 更新依赖、配置和文档，说明GPU/CUDA、模型权重等部署要求，并提供降级策略。
- 新增端到端与单元测试，覆盖成功识别与失败回退场景。

## Impact
- 需要GPU环境、vLLM运行时及额外Python依赖（PyTorch、transformers、vllm等）；部署成本提升。
- 首次加载模型耗时较长，需要缓存/单例管理，并处理多进程资源竞争。
- 需要完善日志、错误处理和重试策略，以确保服务稳定性。
- Streamlit/CLI/API用户可获得真实OCR输出；若部署条件不足将继续使用降级路径。*** End Patch***]
